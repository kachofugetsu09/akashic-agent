from __future__ import annotations

import asyncio
import logging

from agent.plugins.manager import PluginManager

logger = logging.getLogger(__name__)


class PluginWatcher:
    def __init__(self, manager: PluginManager, *, interval_seconds: float = 1.0) -> None:
        self._manager = manager
        self._interval_seconds = interval_seconds
        self._wake = asyncio.Event()
        self._forced = False
        self._running = True
        self._run_started = False
        self._stopped = asyncio.Event()

    async def run(self) -> None:
        """轮询插件文件状态，并在变化后执行一次热重载。"""

        revision: str | None = None
        self._run_started = True
        try:
            # 1. 启动前已停止时，不再触碰 manager
            if not self._running:
                return
            try:
                revision = self._manager.watch_revision()
            except OSError:
                logger.exception("插件热重载状态扫描失败")
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
                self._forced = False
                # 3. 读取最新状态；单次文件竞争交给下一轮恢复
                try:
                    current_revision = self._manager.watch_revision()
                except OSError:
                    self._forced = self._forced or forced
                    logger.exception("插件热重载状态扫描失败")
                    continue
                if revision is None:
                    revision = current_revision
                    forced = True
                if not forced and current_revision == revision:
                    continue
                # 4. 失败版本只尝试一次，后续文件变化仍可恢复热重载
                try:
                    _ = await self._manager.reconcile_changed()
                except Exception:
                    logger.exception("插件热重载失败")
                revision = current_revision
        finally:
            self._stopped.set()

    def wake(self) -> None:
        self._forced = True
        self._wake.set()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        if not self._run_started:
            self._stopped.set()

    async def wait_stopped(self) -> None:
        _ = await self._stopped.wait()
