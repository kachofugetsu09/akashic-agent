from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from agent.plugins.fs_watch import InotifyTreeWatcher
from agent.plugins.input_preparation import SOURCE_EXCLUDED_NAMES
from agent.plugins.manager import PluginManager

logger = logging.getLogger(__name__)

_MAX_RECONCILE_ATTEMPTS = 3


class PluginWatcher:
    def __init__(
        self,
        manager: PluginManager,
        *,
        baseline_revision: dict[str, str] | None = None,
        interval_seconds: float = 1.0,
        backstop_seconds: float = 60.0,
        debounce_seconds: float = 0.25,
        after_reconcile: Callable[[], Awaitable[None]] | None = None,
        accepting: Callable[[], bool] = lambda: True,
        fs_events: bool = True,
    ) -> None:
        self._manager = manager
        self._baseline_revision = baseline_revision
        self._interval_seconds = interval_seconds
        self._backstop_seconds = backstop_seconds
        self._debounce_seconds = debounce_seconds
        self._after_reconcile = after_reconcile
        self._accepting = accepting
        self._fs_events = fs_events
        self._events: InotifyTreeWatcher | None = None
        self._debounce_handle: asyncio.TimerHandle | None = None
        self._wake = asyncio.Event()
        self._forced = False
        self._manual_wake_pending = False
        self._confirmation_pending = False
        self._notification_pending = False
        self._running = True
        self._run_started = False
        self._stopped = asyncio.Event()

    def _on_fs_event(self) -> None:
        """文件系统事件只做写稳定合并；变化判定仍由指纹探测完成。"""
        loop = asyncio.get_running_loop()
        if self._debounce_handle is not None:
            self._debounce_handle.cancel()
        self._debounce_handle = loop.call_later(self._debounce_seconds, self._wake.set)

    async def run(self) -> None:
        """事件驱动监听插件文件状态，慢速兜底探测兜住事件缺口，变化后热重载。"""

        revision = self._baseline_revision
        failed_revision: dict[str, str] | None = None
        failed_attempts = 0
        blocked_revision: dict[str, str] | None = None
        pending_ids: frozenset[str] = frozenset()
        full_pending = False
        if self._fs_events:
            events = InotifyTreeWatcher(
                self._on_fs_event,
                exclude=lambda name: name in SOURCE_EXCLUDED_NAMES,
            )
            # 平台不支持或监听预算耗尽时回退到纯轮询。
            self._events = events if events.start() else None
            if self._events is not None:
                # 首次扫描前先用源码树目标武装事件来源；配置文件目标在扫描后补齐。
                trees, files = self._manager.watch_targets()
                self._events.set_targets(trees, files)
        self._run_started = True
        # 首个扫描不等兜底间隔：无基线时立即建立基线，有基线时立即验证一次指纹。
        self._wake.set()
        try:
            # 1. 启动前已停止时，不再触碰 manager
            if not self._running:
                return
            while self._running:
                if not self._accepting():
                    await asyncio.sleep(self._interval_seconds)
                    continue
                # 2. 事件来源存活时只留慢速兜底；事件唤醒或兜底超时都走同一扫描。
                events_live = self._events is not None and self._events.active
                timeout = self._interval_seconds if not events_live else self._backstop_seconds
                try:
                    _ = await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=timeout,
                    )
                except TimeoutError:
                    pass
                self._wake.clear()
                if not self._running:
                    break
                forced = self._forced
                manual_wake = self._manual_wake_pending
                self._forced = False
                self._manual_wake_pending = False
                # 3. 读取最新状态；单次文件竞争交给下一轮恢复
                try:
                    current_revision = await asyncio.to_thread(
                        self._manager.watch_revision
                    )
                except (OSError, ValueError, RuntimeError):
                    self._forced = self._forced or forced or manual_wake
                    self._manual_wake_pending = self._manual_wake_pending or manual_wake
                    logger.exception("插件热重载状态扫描失败")
                    continue
                if self._events is not None:
                    # 扫描后刷新监听目标：安装/卸载改变根集合与 data_dir 配置。
                    trees, files = self._manager.watch_targets()
                    self._events.set_targets(trees, files)
                # 启动只记下磁盘基线，不把关机期间留下的候选重新当成更新。
                # 明确的手动唤醒仍可请求处理当前输入。
                if revision is None:
                    revision = current_revision
                    if not forced:
                        continue
                if manual_wake:
                    full_pending = True
                if manual_wake or (
                    failed_revision is not None and current_revision != failed_revision
                ):
                    failed_revision = None
                    failed_attempts = 0
                    blocked_revision = None
                changed = forced or current_revision != revision
                changed_ids = frozenset(
                    plugin_id for plugin_id in current_revision.keys() | revision.keys()
                    if current_revision.get(plugin_id) != revision.get(plugin_id)
                ) | pending_ids
                if blocked_revision == current_revision and not manual_wake:
                    changed = False
                if not changed and not self._notification_pending:
                    continue
                # 4. 同 revision 失败有界重试；通知失败只重试通知，不重复 reconcile
                needs_confirmation = False
                if changed:
                    if failed_revision != current_revision:
                        failed_revision = current_revision
                        failed_attempts = 0
                        blocked_revision = None
                    failed_attempts += 1
                    try:
                        results = await self._manager.reconcile_changed(
                            plugin_ids=None if full_pending else changed_ids,
                        )
                    except Exception:
                        pending_ids = changed_ids
                        logger.exception("插件热重载失败")
                        if failed_attempts >= _MAX_RECONCILE_ATTEMPTS:
                            blocked_revision = current_revision
                            # 失败输入已观察并报告；无关变化不能重新授权它们。
                            revision = current_revision
                            pending_ids = frozenset()
                            full_pending = False
                            if self._confirmation_pending:
                                self._notification_pending = False
                        else:
                            # 保留旧 revision；下一轮按轮询间隔自动重试。
                            self._forced = True
                        continue
                    else:
                        # 安装器原子替换目录时，单次 discover 可能只看到短暂缺口。
                        # 禁用结果先确认一次，只向移动端发布稳定后的最终目录。
                        needs_confirmation = any(
                            result.get("publication_state") == "disabled"
                            for result in results
                        )
                        if needs_confirmation:
                            pending_ids = changed_ids
                            self._confirmation_pending = True
                            self._forced = True
                        else:
                            self._confirmation_pending = False
                            pending_ids = frozenset()
                            full_pending = False
                        revision = current_revision
                        self._notification_pending = self._after_reconcile is not None
                        failed_revision = None
                        failed_attempts = 0
                        blocked_revision = None
                if needs_confirmation:
                    continue
                if self._notification_pending:
                    try:
                        assert self._after_reconcile is not None
                        await self._after_reconcile()
                    except Exception:
                        logger.exception("插件热重载后置通知失败")
                    else:
                        self._notification_pending = False
        finally:
            if self._events is not None:
                self._events.close()
                self._events = None
            if self._debounce_handle is not None:
                self._debounce_handle.cancel()
                self._debounce_handle = None
            self._stopped.set()

    def wake(self) -> None:
        self._forced = True
        self._manual_wake_pending = True
        self._wake.set()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        if not self._run_started:
            self._stopped.set()

    async def wait_stopped(self) -> None:
        _ = await self._stopped.wait()
