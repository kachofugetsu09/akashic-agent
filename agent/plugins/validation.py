"""一次隔离调用的真实资源；不持有安装、更新或 stable 控制面。"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from agent.control.frame_book import FrameBook
from agent.restart import RestartGate
from agent.plugin_composition.admission import SOURCE_ADMISSION
from agent.plugin_composition.artifacts import ArtifactImport, ArtifactRead
from agent.plugin_composition.channel_io import InputCustody, ChannelIdentity
from agent.plugin_composition.channels import CHANNELS, RawInbound
from agent.plugin_composition.context import CompositionRoot, RuntimeScope
from agent.plugin_composition.model import FiberState
from agent.plugin_composition.processes import PluginProcesses
from agent.plugin_composition.tasks import PluginTasks
from agent.plugins.generation import PluginGeneration
from agent.plugins.snapshot import RuntimeSnapshot, RuntimeSnapshotLease, RuntimeSnapshotStore
from agent.plugins.archive import PluginArchive
from bus.event_bus import EventBus
from bus.queue import MessageBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.channels.attachment_import import ChannelOutboundAttachmentImporter
from session.admissions import SessionAdmissions
from session.identities import ChannelIdentities, ChannelIdentityWriteReceipt
from session.inbound_store import InboundHandoffStore
from session.artifact_store import ArtifactStore
from session.log import MessageLog


@dataclass
class ValidationHost:
    """从分配 Root 起保留实际资源，失败关闭仍由原宿主重试。"""

    identity: str
    workspace: Path
    messages: MessageLog
    artifacts: ArtifactStore
    bus: EventBus
    task: asyncio.Task[object]
    parent_lease: RuntimeSnapshotLease
    message_bus: MessageBus
    admissions: SessionAdmissions
    identities: ChannelIdentities
    inbound_store: InboundHandoffStore
    archive: PluginArchive
    attachments: ChannelAttachmentArtifactStore
    root: CompositionRoot | None = None
    generations: tuple[PluginGeneration, ...] = ()
    tasks: PluginTasks = field(default_factory=PluginTasks)
    processes: PluginProcesses = field(default_factory=PluginProcesses)
    control_frames: FrameBook = field(default_factory=FrameBook)
    snapshot_store: RuntimeSnapshotStore = field(init=False)
    restart_gate: RestartGate = field(init=False)
    workspace_id: str = field(init=False)
    input_custody: InputCustody = field(init=False)
    channel_identity: ChannelIdentity = field(init=False)
    artifact_read: ArtifactRead = field(init=False)
    artifact_import: ArtifactImport = field(init=False)
    active: bool = True
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        """固定本次宿主端口；实例发布后不替换服务或数据连接。"""
        self.snapshot_store = RuntimeSnapshotStore(self._close_snapshot)
        self.bus.bind_runtime_snapshot_store(self.snapshot_store)
        self.restart_gate = RestartGate(boot_id=self.identity, supervised=False)
        self.workspace_id = hashlib.sha256(
            str(self.workspace.resolve(strict=False)).encode("utf-8")
        ).hexdigest()[:16]
        bus = self.message_bus
        self.input_custody = InputCustody(
            bus.prepare_channel_input, bus.complete_channel_input, bus.retain_channel_input,
            bus.reserve_durable_inbound, bus.defer_durable_inbound,
            bus.settle_rejected_inbound, bus.has_pending_durable_inbound,
            bus.pending_durable_attachment_refs, bus.recover_durable_inbounds,
        )
        self.channel_identity = ChannelIdentity(
            self.identities.resolve, self._remember_identity, self._rollback_identity,
        )
        self.artifact_read = ArtifactRead(self.attachments.acquire)
        self.artifact_import = ArtifactImport(
            ChannelOutboundAttachmentImporter(self.attachments).import_source,
        )

    async def _remember_identity(self, channel: str, identity: str, recipient: str) -> object:
        return self.identities.remember(channel, identity, recipient)

    async def _rollback_identity(self, receipt: object) -> bool:
        if not isinstance(receipt, ChannelIdentityWriteReceipt):
            raise TypeError("channel identity rollback receipt 类型无效")
        return self.identities.rollback(receipt)

    async def recover_input(self, raw: RawInbound) -> bool:
        """只交回本次 Root 的当前 provider；不借主宿主或恢复特权。"""
        snapshot = self.snapshot_store.current
        if snapshot is None or not snapshot.accepting_leases:
            return False
        lease = self.snapshot_store.lease(snapshot.snapshot_id)
        async with RuntimeScope(lease):
            root = lease.snapshot.composition_root
            if root is None:
                raise RuntimeError("验证输入恢复需要当前 Root")
            channels = root.context.get(CHANNELS)
            return False if channels is None else await channels.recover_inbound(raw)

    async def _close_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        """Store 排空 lease 后才释放实际 Root，失败保留 Store 与模块 owner。"""
        root = snapshot.composition_root
        assert root is not None
        await root.dispose()
        for generation in snapshot.generations.values():
            generation.state = "retired"

    async def stop_resources(self) -> None:
        """停止接纳和工作，再由 Store 或构建 Root 回收作用域。"""
        # 1. 本宿主从未发启动事件；没有对应的 stopping 事件可重放。
        self.snapshot_store.pause_admission()
        if self.root is not None:
            admission = self.root.context.get(SOURCE_ADMISSION)
            if admission is not None:
                admission.close()
        await self.tasks.close()
        await self.processes.close()
        # 2. Store 拒绝尚有 lease 的关闭；未发布或失败 Root 也保留实际句柄。
        await self.snapshot_store.close()
        if self.root is not None and self.root.root_fiber.state != FiberState.DISPOSED:
            await self.root.dispose()
        for generation in self.generations:
            generation.state = "retired"
        self.control_frames.close()

    async def close(self) -> None:
        """先确认运行资源退出，再关闭输入 owner 和持久连接；失败不移除句柄。"""
        async with self.lock:
            if self.closed:
                return
            await self.stop_resources()
            await self.message_bus.aclose()
            await self.bus.aclose()
            self.inbound_store.close()
            self.identities.close()
            self.admissions.close()
            self.artifacts.close()
            self.messages.close()
            await self.parent_lease.release()
            self.closed = True
