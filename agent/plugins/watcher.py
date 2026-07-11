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
        self._stopped = asyncio.Event()

    async def run(self) -> None:
        revision: str | None = None
        scan_failed = False
        try:
            try:
                revision = self._manager.watch_revision()
            except Exception:
                scan_failed = True
                logger.exception("插件热重载状态扫描失败")
            while self._running:
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=self._interval_seconds,
                    )
                except TimeoutError:
                    pass
                self._wake.clear()
                if not self._running:
                    break
                forced = self._forced or scan_failed
                self._forced = False
                try:
                    current_revision = self._manager.watch_revision()
                except Exception:
                    scan_failed = True
                    logger.exception("插件热重载状态扫描失败")
                    continue
                scan_failed = False
                if revision is None:
                    revision = current_revision
                    if not forced:
                        continue
                if not forced and current_revision == revision:
                    continue
                try:
                    await self._manager.reconcile_changed()
                    revision = current_revision
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("插件热重载扫描失败")
        finally:
            self._stopped.set()

    def wake(self) -> None:
        self._forced = True
        self._wake.set()

    def stop(self) -> None:
        self._running = False
        self._wake.set()

    async def wait_stopped(self) -> None:
        await self._stopped.wait()
