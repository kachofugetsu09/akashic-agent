from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from agent.plugins.manager import PluginManager

logger = logging.getLogger(__name__)

_MAX_RECONCILE_ATTEMPTS = 3


class PluginWatcher:
    def __init__(
        self,
        manager: PluginManager,
        *,
        baseline_revision: str,
        interval_seconds: float = 1.0,
        after_reconcile: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._manager = manager
        self._baseline_revision = baseline_revision
        self._interval_seconds = interval_seconds
        self._after_reconcile = after_reconcile
        self._wake = asyncio.Event()
        self._forced = False
        self._manual_wake_pending = False
        self._confirmation_pending = False
        self._notification_pending = False
        self._running = True
        self._run_started = False
        self._stopped = asyncio.Event()

    async def run(self) -> None:
        """轮询插件文件状态，并在变化后执行一次热重载。"""

        revision = self._baseline_revision
        failed_revision: str | None = None
        failed_attempts = 0
        blocked_revision: str | None = None
        self._run_started = True
        try:
            # 1. 启动前已停止时，不再触碰 manager
            if not self._running:
                return
            while self._running:
                # 2. 等待定时轮询或外部唤醒
                try:
                    _ = await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=self._interval_seconds,
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
                except OSError:
                    self._forced = self._forced or forced or manual_wake
                    self._manual_wake_pending = self._manual_wake_pending or manual_wake
                    logger.exception("插件热重载状态扫描失败")
                    continue
                if manual_wake:
                    failed_revision = None
                    failed_attempts = 0
                    blocked_revision = None
                elif failed_revision is not None and current_revision != failed_revision:
                    failed_revision = None
                    failed_attempts = 0
                    blocked_revision = None
                changed = forced or current_revision != revision
                if blocked_revision == current_revision and not manual_wake:
                    changed = False
                if not changed and not self._notification_pending:
                    continue
                # 4. 同 revision 失败有界重试；通知失败只重试通知，不重复 reconcile
                confirming = self._confirmation_pending
                needs_confirmation = False
                if changed:
                    if failed_revision != current_revision:
                        failed_revision = current_revision
                        failed_attempts = 0
                        blocked_revision = None
                    failed_attempts += 1
                    try:
                        results = await self._manager.reconcile_changed()
                    except Exception:
                        logger.exception("插件热重载失败")
                        if failed_attempts >= _MAX_RECONCILE_ATTEMPTS:
                            blocked_revision = current_revision
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
                            self._confirmation_pending = True
                            self._forced = True
                        elif confirming:
                            self._confirmation_pending = False
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
