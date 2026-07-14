from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from agent.mcp.declarations import (
    declarations_input_revision,
    load_workspace_mcp_declarations,
)
from agent.plugins.manager import PluginManager

logger = logging.getLogger(__name__)


class WorkspaceMcpWatcher:
    """轮询 workspace MCP 声明并串行发布内容变化。"""

    def __init__(
        self,
        manager: PluginManager,
        declarations_dir: Path,
        *,
        mcp_root: Path | None = None,
        interval_seconds: float = 1.0,
    ) -> None:
        self._manager = manager
        self._declarations_dir = declarations_dir
        self._mcp_root = mcp_root or declarations_dir.parent
        self._interval_seconds = interval_seconds
        self._active_revision: str | None = None
        self._running = True
        self._wake = asyncio.Event()
        self._forced = False
        self._last_input_revision: str | None = None
        self._stopped = asyncio.Event()
        self._reconcile_lock = asyncio.Lock()
        self.last_error: str | None = None
        self._active_generation_id: str | None = None
        self._active_servers: tuple[str, ...] = ()
        self._active_tools: tuple[str, ...] = ()

    async def reconcile(self) -> bool:
        """发布变化后的完整声明；失败时调用方获得原始异常。"""

        async with self._reconcile_lock:
            desired = load_workspace_mcp_declarations(
                self._declarations_dir,
                mcp_root=self._mcp_root,
            )
            if desired.revision == self._active_revision:
                return False
            await self._manager.prepare_workspace_mcp(
                desired.specs,
                revision=desired.revision,
            )
            if not self._running:
                await self._manager.discard_workspace_mcp_candidate()
                return False
            generation = await self._manager.publish_workspace_mcp()
            self._active_revision = desired.revision
            self._active_generation_id = generation.generation_id
            self._active_servers = tuple(sorted(generation.catalog.servers))
            self._active_tools = generation.catalog.tool_names
            self.last_error = None
            return True

    def status(self) -> dict[str, Any]:
        """返回当前已发布 workspace MCP 代际的只读状态。"""

        return {
            "generationId": self._active_generation_id,
            "revision": self._active_revision,
            "servers": list(self._active_servers),
            "tools": list(self._active_tools),
            "lastError": self.last_error,
        }

    async def run(self) -> None:
        """持续检查内容 revision，并允许失败版本在修复后恢复。"""

        try:
            while self._running:
                try:
                    _ = await asyncio.wait_for(
                        self._wake.wait(), timeout=self._interval_seconds
                    )
                except TimeoutError:
                    pass
                self._wake.clear()
                if not self._running:
                    break
                forced = self._forced
                self._forced = False
                try:
                    input_revision = declarations_input_revision(
                        self._declarations_dir,
                        mcp_root=self._mcp_root,
                    )
                except OSError as error:
                    self.last_error = str(error)
                    logger.error("workspace MCP 声明扫描失败: %s", error)
                    continue
                if not forced and input_revision == self._last_input_revision:
                    continue
                self._last_input_revision = input_revision
                try:
                    await self.reconcile()
                except (OSError, ValueError, RuntimeError) as error:
                    self.last_error = str(error)
                    logger.error("workspace MCP 热重载失败: %s", error)
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
