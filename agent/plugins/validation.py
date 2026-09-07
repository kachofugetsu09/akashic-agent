"""一次业务验证的隔离数据与资源；没有恢复队列或自动重跑。"""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agent.plugin_composition.context import CompositionRoot
from bus.event_bus import EventBus
from session.artifact_store import ArtifactStore
from session.log import MessageLog

if TYPE_CHECKING:
    from agent.plugins.manager import PluginManager
    from agent.plugins.snapshot import RuntimeSnapshotLease


@dataclass
class ValidationHost:
    """清理失败时保留实际宿主及连接，允许原 owner 重试。"""

    identity: str
    workspace: Path
    manager: PluginManager
    messages: MessageLog
    artifacts: ArtifactStore
    bus: EventBus
    modules: ExitStack
    task: asyncio.Task[object]
    parent_lease: RuntimeSnapshotLease
    root: CompositionRoot | None = None
    active: bool = True
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def close(self) -> None:
        """先核对运行资源已退出，再关闭模块和持久数据连接。"""
        async with self.lock:
            if self.closed:
                return
            await self.manager.stop_validation_resources()
            if self.root is not None:
                await self.root.dispose()
            await self.bus.aclose()
            self.modules.close()
            self.artifacts.close()
            self.messages.close()
            await self.parent_lease.release()
            self.closed = True
