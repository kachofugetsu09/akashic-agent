"""一次业务验证的隔离数据与资源；没有恢复队列或自动重跑。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agent.plugin_composition.context import CompositionRoot
from agent.plugin_composition.channels import CHANNELS, RawInbound
from agent.plugin_composition.context import RuntimeScope
from bus.event_bus import EventBus
from bus.queue import MessageBus
from session.admissions import SessionAdmissions
from session.identities import ChannelIdentities
from session.inbound_store import InboundHandoffStore
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
    task: asyncio.Task[object]
    parent_lease: RuntimeSnapshotLease
    message_bus: MessageBus
    admissions: SessionAdmissions
    identities: ChannelIdentities
    inbound_store: InboundHandoffStore
    root: CompositionRoot | None = None
    active: bool = True
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def recover_input(self, raw: RawInbound) -> bool:
        """只交回验证 Root 的当前 provider；不借主宿主或恢复特权。"""
        snapshot = self.manager.current_snapshot
        if snapshot is None or not snapshot.accepting_leases:
            return False
        lease = self.manager.snapshot_store.lease(snapshot.snapshot_id)
        async with RuntimeScope(lease):
            root = lease.snapshot.composition_root
            if root is None:
                raise RuntimeError("验证输入恢复需要当前 Root")
            channels = root.context.get(CHANNELS)
            return False if channels is None else await channels.recover_inbound(raw)

    async def close(self) -> None:
        """先核对运行资源已退出，再关闭模块和持久数据连接。"""
        async with self.lock:
            if self.closed:
                return
            await self.manager.stop_validation_resources()
            if self.root is not None:
                await self.root.dispose()
            # Bus 必须先释放输入租约；失败时全部存储和原宿主仍在。
            await self.message_bus.aclose()
            await self.bus.aclose()
            self.inbound_store.close()
            self.identities.close()
            self.admissions.close()
            self.artifacts.close()
            self.messages.close()
            await self.parent_lease.release()
            self.closed = True
