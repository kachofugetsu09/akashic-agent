from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from agent.restart import RestartGate
from agent.control.frame_book import FrameBook

if TYPE_CHECKING:
    from agent.plugins.manager import PluginManager
    from infra.channels.artifacts import ChannelAttachmentArtifactStore

logger = logging.getLogger(__name__)

from agent.config_models import Config
from agent.plugin_composition.channels import (
    ChannelCommitRole,
    ChannelDeliveryReceipt,
    ChannelTerminalStatus,
    DeliveryStatus,
    JsonValue,
    OutboundEnvelope,
)
from agent.plugin_composition.durable_deliveries import (
    DurableBindingAttempt,
    DurableDeliveryRequest,
    ProviderStarted,
)
from agent.plugins.manifest import plugins_root
from agent.plugins.snapshot import lease_current_runtime_snapshot
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.workspace_lock import PluginPublicationLock
from bus.event_bus import EventBus
from bus.events import (
    ChannelMessage,
)
from infra.channels.attachment_import import (
    import_channel_attachments,
)
from bus.queue import MessageBus
from core.net.http import SharedHttpResources
from session.artifact_store import ArtifactStore
from session.manager import SessionManager
from session.log import MessageLog
from session.admissions import SessionAdmissions
from session.identities import ChannelIdentities
from session.inbound_store import InboundHandoffStore


async def _noop_async() -> None:
    return None


async def _dispatch_v3_channel_push(
    plugin_manager: PluginManager,
    bus: MessageBus,
    message: ChannelMessage,
    passive: bool,
    attachment_store: ChannelAttachmentArtifactStore | None = None,
    session_manager: SessionManager | None = None,
) -> ChannelDeliveryReceipt:
    """Dispatch one direct push through the exact public stable Channel binding."""

    # 1. 当前 turn 优先复用 exact snapshot；独立调用才租用公开 stable。
    source = lease_current_runtime_snapshot()
    if source is None:
        source = await plugin_manager.snapshot_store.acquire()
    binding = None
    try:
        catalog = source.snapshot.channel_catalog
        registry = (
            catalog.registry
            if catalog is not None
            else source.snapshot.channel_registry
        )
        descriptor = (
            None
            if registry is None
            else next(
                (item for item in registry.descriptors if item.name == message.channel),
                None,
            )
        )
        if descriptor is None:
            raise RuntimeError(
                f"committed Channel catalog 缺少目标渠道: {message.channel!r}"
            )
        binding = plugin_manager.channel_generation_host.acquire_binding(
            source,
            message.channel,
        )
    finally:
        try:
            await source.release()
        except BaseException:
            if binding is not None:
                await binding.aclose()
            raise

    # 2. 在 provider 调用前把授权来源冻结为 Core-owned opaque refs。
    if binding is None:
        raise RuntimeError("v3 Channel catalog 存在但 exact binding 未建立")
    delivery_id = uuid4().hex
    try:
        if message.attachments and message.attachment_refs:
            raise RuntimeError("ChannelMessage 不得同时携带 path 与 opaque refs")
        if message.attachments and attachment_store is None:
            raise RuntimeError("Channel attachment store 尚未绑定")
        attachment_refs = message.attachment_refs
        if message.attachments:
            attachment_refs = await import_channel_attachments(
                cast("ChannelAttachmentArtifactStore", attachment_store),
                message.attachments,
            )

        # 3. Akashic direct push 先成为目标 Session 的正文；adapter 只负责通知。
        if message.channel == "akashic" and not passive:
            if session_manager is None:
                raise RuntimeError("Akashic direct push 缺少 Session owner")
            control_turn_id = message.control_turn_id or f"turn:{delivery_id}"
            session_message_id = await session_manager.append_durable_delivery(
                session_key=f"akashic:{message.chat_id}",
                content=message.content,
                delivery_id=delivery_id,
                control_turn_id=control_turn_id,
                metadata={
                    **message.metadata,
                    "effects": {"post_commit": "suppress"},
                    "attachment_ids": [ref.artifact_id for ref in attachment_refs],
                },
            )
            message = replace(
                message,
                session_message_id=session_message_id,
                control_turn_id=control_turn_id,
            )

        # 4. exact binding 保留到唯一一次 typed receipt 收束。
        envelope = OutboundEnvelope(
            logical_delivery_id=delivery_id,
            delivery_id=delivery_id,
            attempt_sequence=1,
            snapshot_id=binding.snapshot_id,
            generation_id=binding.generation_id,
            binding_token=binding.binding_token,
            channel=message.channel,
            recipient=message.chat_id,
            body=message.content,
            metadata=cast(Mapping[str, JsonValue], message.metadata),
            attachments=attachment_refs,
            commit_role=(
                ChannelCommitRole.PASSIVE if passive else ChannelCommitRole.DIRECT
            ),
            thinking=message.thinking,
            reply_to=message.reply_to,
            session_message_id=message.session_message_id,
            control_turn_id=message.control_turn_id,
            execution_attempt_id=message.execution_attempt_id,
            terminal_status=(
                ChannelTerminalStatus(message.terminal_status.value)
                if message.terminal_status is not None
                else None
            ),
        )
        try:
            receipt = await bus.publish_channel_outbound_awaited(
                envelope,
                binding,
                passive=passive,
            )
        except BaseException as error:
            if message.channel != "akashic" or passive:
                raise
            logger.warning(
                "Akashic Session 已提交，客户端通知抛错: delivery_id=%s error=%s",
                delivery_id,
                error,
                exc_info=True,
            )
            return ChannelDeliveryReceipt(
                delivery_id=delivery_id,
                status=DeliveryStatus.DELIVERED,
            )
        if message.channel == "akashic" and not passive:
            if receipt.status is not DeliveryStatus.DELIVERED:
                logger.warning(
                    "Akashic Session 已提交，客户端通知未确认: delivery_id=%s status=%s error=%s",
                    delivery_id,
                    receipt.status.value,
                    receipt.error,
                )
            return ChannelDeliveryReceipt(
                delivery_id=delivery_id,
                status=DeliveryStatus.DELIVERED,
                provider_ids=receipt.provider_ids,
            )
        return receipt
    finally:
        await binding.aclose()


async def _dispatch_v3_durable_delivery(
    plugin_manager: PluginManager,
    bus: MessageBus,
    request: DurableDeliveryRequest,
    provider_started: ProviderStarted,
    session_manager: SessionManager | None = None,
) -> ChannelDeliveryReceipt:
    """Persist one exact binding attempt before its direct provider dispatch."""

    # 1. Resolve and retain the same committed Channel boundary as ordinary sends.
    source = lease_current_runtime_snapshot()
    if source is None:
        source = await plugin_manager.snapshot_store.acquire()
    binding = None
    try:
        catalog = source.snapshot.channel_catalog
        registry = (
            catalog.registry
            if catalog is not None
            else source.snapshot.channel_registry
        )
        descriptor = (
            None
            if registry is None
            else next(
                (item for item in registry.descriptors if item.name == request.channel),
                None,
            )
        )
        if descriptor is None:
            raise RuntimeError(
                f"committed Channel catalog 缺少目标渠道: {request.channel!r}"
            )
        binding = plugin_manager.channel_generation_host.acquire_binding(
            source, request.channel
        )
    finally:
        try:
            await source.release()
        except BaseException:
            if binding is not None:
                await binding.aclose()
            raise

    # 2. Freeze the exact binding; MessageBus commits provider_started at its adapter edge.
    if binding is None:
        raise RuntimeError("durable delivery exact Channel binding 未建立")
    try:
        attempt = DurableBindingAttempt(
            attempt_id=request.logical_delivery_id,
            snapshot_id=binding.snapshot_id,
            generation_id=binding.generation_id,
            binding_token=binding.binding_token,
        )
        provider_attempt_started = False

        def mark_provider_started() -> None:
            nonlocal provider_attempt_started
            if provider_attempt_started:
                return
            provider_started(attempt)
            provider_attempt_started = True

        session_message_id = None
        if request.channel == "akashic":
            if session_manager is None:
                raise RuntimeError("Akashic durable delivery 缺少 Session owner")
            session_message_id = await session_manager.append_durable_delivery(
                session_key=request.projection_session_id,
                content=request.body,
                delivery_id=request.logical_delivery_id,
                control_turn_id=request.accepted_turn.turn_id,
                metadata={
                    **request.metadata,
                    "effects": {"post_commit": "suppress"},
                },
            )
        envelope = OutboundEnvelope(
            logical_delivery_id=request.logical_delivery_id,
            delivery_id=request.logical_delivery_id,
            attempt_sequence=1,
            snapshot_id=binding.snapshot_id,
            generation_id=binding.generation_id,
            binding_token=binding.binding_token,
            channel=request.channel,
            recipient=request.recipient,
            body=request.body,
            metadata=cast(Mapping[str, JsonValue], request.metadata),
            commit_role=ChannelCommitRole.DIRECT,
            control_turn_id=request.accepted_turn.turn_id,
            session_message_id=session_message_id,
        )
        try:
            receipt = await bus.publish_channel_outbound_awaited(
                envelope,
                binding,
                passive=False,
                before_provider=mark_provider_started,
            )
        except BaseException as error:
            if request.channel != "akashic":
                raise
            logger.warning(
                "Akashic durable Session 已提交，客户端通知抛错: delivery_id=%s error=%s",
                request.logical_delivery_id,
                error,
                exc_info=True,
            )
            mark_provider_started()
            return ChannelDeliveryReceipt(
                delivery_id=request.logical_delivery_id,
                status=DeliveryStatus.DELIVERED,
            )
        if request.channel == "akashic":
            if receipt.status is not DeliveryStatus.DELIVERED:
                logger.warning(
                    "Akashic durable Session 已提交，客户端通知未确认: delivery_id=%s status=%s error=%s",
                    request.logical_delivery_id,
                    receipt.status.value,
                    receipt.error,
                )
            return ChannelDeliveryReceipt(
                delivery_id=request.logical_delivery_id,
                status=DeliveryStatus.DELIVERED,
                provider_ids=receipt.provider_ids,
            )
        return receipt
    finally:
        await binding.aclose()


@dataclass
class CoreRuntime:
    """只装配消息、资源与插件；回复和来源由普通插件运行。"""

    config: Config
    workspace: Path
    http_resources: SharedHttpResources
    bus: MessageBus
    event_bus: EventBus
    message_log: MessageLog
    admissions: SessionAdmissions
    identities: ChannelIdentities
    inbound_store: InboundHandoffStore
    artifact_metadata: ArtifactStore
    channel_attachment_store: ChannelAttachmentArtifactStore
    plugin_manager: PluginManager
    plugin_publication_lock: PluginPublicationLock
    restart_gate: "RestartGate"
    control_frames: FrameBook
    _plugin_publication_locked: bool = False

    def _lock_plugin_publication(self) -> None:
        if not self._plugin_publication_locked:
            self.plugin_publication_lock.acquire()
            self._plugin_publication_locked = True

    async def start(self) -> None:
        """取得插件目录独占权，再发布插件及其 Skill 投影。"""
        from agent.plugins.skill_links import PluginSkillLinker

        # 1. 正式插件加载只建立 Root；后台消费者由 runtime lifecycle 启动。
        self._lock_plugin_publication()
        await self.plugin_manager.load_all()
        # 2. 既有安装 owner 同步投影，不由消息消费者管理文件。
        result = PluginSkillLinker(
            workspace=self.workspace,
            plugin_roots=self.plugin_manager.skill_projection_roots,
        ).sync(self.plugin_manager.active_plugins())
        logger.info("插件 skill 同步完成: %s", result)
        self.plugin_manager.sync_manifest()

    async def inspect_modules(self) -> str:
        """展示实际发布的组合图，不再构造旧回复 Pipeline。"""
        self._lock_plugin_publication()
        await self.plugin_manager.load_all()
        snapshot = self.plugin_manager.current_snapshot
        assert snapshot is not None and snapshot.composition_topology is not None
        topology = snapshot.composition_topology
        parts = [f"identity: {topology.identity}", f"revision: {topology.composition_revision}"]
        parts.extend(f"fiber: {fiber.parent or '<root>'} -> {fiber.name}" for fiber in topology.fibers)
        parts.extend(f"listener: {listener}" for listener in topology.listeners)
        return "\n".join(parts)

    async def stop(self) -> None:
        """先排空插件资源，再关闭各自拥有的数据库连接。"""
        async def close_control_frames() -> None:
            self.control_frames.close()

        async def close_storage() -> None:
            # 每个连接都尝试关闭；前一项失败不能泄漏后续 owner。
            errors: list[Exception] = []
            for store in (self.inbound_store, self.identities, self.admissions,
                          self.artifact_metadata, self.message_log):
                try:
                    store.close()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise ExceptionGroup("Core storage close 失败", errors)

        await run_cleanup_steps(
            ("plugin_manager.terminate_all", self.plugin_manager.terminate_all),
            ("control_frames.close", close_control_frames),
            ("event_bus.aclose", self.event_bus.aclose),
            ("plugin_publication_lock.release", self._release_plugin_publication),
            ("storage.close", close_storage),
        )

    async def _release_plugin_publication(self) -> None:
        if self._plugin_publication_locked:
            self.plugin_publication_lock.release()
            self._plugin_publication_locked = False


def build_core_runtime(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
    restart_gate: RestartGate | None = None,
    *,
    clear_stale_session_admissions: bool = False,
    plugin_dirs: Iterable[Path] | None = None,
) -> CoreRuntime:
    """从已迁移消息库装配窄 owner；构造失败关闭此前取得的连接。"""
    from contextlib import ExitStack
    from agent.plugins.manager import PluginManager
    from infra.channels.artifacts import ChannelAttachmentArtifactStore

    # 1. MessageLog 先核对 schema，旧库不能借普通启动绕过 yoyo。
    bus = MessageBus()
    event_bus = EventBus()
    with ExitStack() as cleanup:
        message_log = MessageLog(workspace / "sessions.db")
        _ = cleanup.callback(message_log.close)
        artifact_metadata = ArtifactStore(workspace / "sessions.db")
        _ = cleanup.callback(artifact_metadata.close)
        admissions = SessionAdmissions(workspace / "sessions.db")
        _ = cleanup.callback(admissions.close)
        identities = ChannelIdentities(workspace / "sessions.db")
        _ = cleanup.callback(identities.close)
        inbound_store = InboundHandoffStore(workspace / "sessions.db")
        _ = cleanup.callback(inbound_store.close)
        if clear_stale_session_admissions:
            admissions.clear_stale()
        bus.bind_mobile_session_admission_owner(admissions)
        bus.bind_durable_inbound_store(inbound_store)
        attachments = ChannelAttachmentArtifactStore(
            workspace=workspace, metadata_store=artifact_metadata,
        )
        # 2. PluginManager 分配日志、归档和资源能力，不持有旧 SessionManager。
        if restart_gate is None:
            restart_gate = RestartGate(boot_id="unmanaged", supervised=False)
        control_frames = FrameBook()
        resolved_plugin_dirs = (
            _resolve_plugin_dirs(workspace)
            if plugin_dirs is None
            else _resolve_plugin_dirs(workspace, plugin_dirs=plugin_dirs)
        )
        manager = PluginManager(
            plugin_dirs=resolved_plugin_dirs, event_bus=event_bus,
            workspace=workspace, message_log=message_log, channel_identities=identities,
            installed_cache_root=plugins_root() / "cache",
            channel_attachment_store=attachments,
            disabled_builtin_plugins=_disabled_builtin_plugins_for_runtime(
                config, resolved_plugin_dirs
            ),
            restart_gate=restart_gate,
            control_frames=control_frames,
        )
        manager.channel_generation_host.bind_input_custody(bus)
        bus.bind_channel_outbound_dispatcher(manager.channel_generation_host.dispatch_outbound)
        runtime = CoreRuntime(
            config=config, workspace=workspace, http_resources=http_resources,
            bus=bus, event_bus=event_bus, message_log=message_log,
            admissions=admissions, identities=identities, inbound_store=inbound_store,
            artifact_metadata=artifact_metadata, channel_attachment_store=attachments,
            plugin_manager=manager, plugin_publication_lock=PluginPublicationLock(plugins_root()),
            restart_gate=restart_gate,
            control_frames=control_frames,
        )
        _ = cleanup.pop_all()
        return runtime


def _resolve_plugin_dirs(
    workspace: Path,
    *,
    plugin_dirs: Iterable[Path] = (),
) -> list[Path]:
    """Return only explicitly requested development plugin roots."""

    _ = workspace
    roots = [Path(item).expanduser() for item in plugin_dirs]
    extra = os.environ.get("AKASHIC_EXTRA_PLUGIN_DIRS", "")
    roots.extend(
        Path(item).expanduser() for item in extra.split(os.pathsep) if item.strip()
    )
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        normalized = root.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(root)
    return result


def _disabled_builtin_plugins_for_runtime(
    config: Config,
    plugin_dirs: Iterable[Path] = (),
) -> frozenset[str]:
    """Apply generic disabled/Workload rules to explicit development roots."""

    disabled = set(config.disabled_builtin_plugins)
    roots = tuple(plugin_dirs)
    if not roots:
        return frozenset(disabled)

    from agent.plugins.source_resolver import resolve_plugin_sources

    known = {
        source.plugin_name or source.plugin_root.name
        for source in resolve_plugin_sources(list(roots))
    }
    unknown = sorted(disabled - known)
    if unknown:
        raise ValueError(
            "agent.plugins.disabled_builtin 包含未知内置插件: " + ", ".join(unknown)
        )
    if os.environ.get("AKASHIC_WORKLOAD_SOCKET", "").strip():
        return frozenset(disabled)

    from agent.plugins.static_manifest import load_static_plugin_manifest

    unavailable = {
        manifest.name
        for root in roots
        for path in root.glob("*/akashic.plugin.toml")
        if (manifest := load_static_plugin_manifest(path.parent)).workloads
    }
    if unavailable:
        logger.warning(
            "当前部署没有 Workload Controller，未启用内置 Workload 插件: %s",
            ", ".join(sorted(unavailable)),
        )
    disabled.update(unavailable)
    return frozenset(disabled)
