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
        self._running = True
        self._stopped = asyncio.Event()

    async def run(self) -> None:
        try:
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
                try:
                    await self._manager.reconcile_changed()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("插件热重载扫描失败")
        finally:
            self._stopped.set()

    def wake(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._running = False
        self._wake.set()

    async def wait_stopped(self) -> None:
        await self._stopped.wait()
