from __future__ import annotations

import logging
import inspect
import os
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast
from uuid import uuid4

if TYPE_CHECKING:
    from agent.plugins.manager import PluginManager
    from agent.restart import RestartCoordinator
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
from agent.plugin_composition import TextEmbeddingSettings
from agent.plugin_composition.durable_deliveries import (
    DurableBindingAttempt,
    DurableDeliveryRequest,
    ProviderStarted,
)
from agent.plugins.manifest import plugins_root
from agent.plugins.snapshot import lease_current_runtime_snapshot
from agent.context import ContextBuilder
from agent.looping.core import AgentLoop
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig,
    LLMServices,
    SessionServices,
)
from agent.provider import LLMProvider
from agent.model_runtime.registry import ModelRegistry
from agent.tools.base import ToolExecutionContext, get_current_tool_context
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.turns.outbound import OutboundPort, PushToolOutboundPort
from bootstrap.toolsets.meta import build_readonly_tools
from bootstrap.toolsets.protocol import ToolsetDeps
from bootstrap.wiring import (
    wire_turn_lifecycle,
    resolve_context_factory,
    resolve_memory_toolset_provider,
    resolve_toolset_provider,
)
from agent.lifecycle.facade import TurnLifecycle
from bootstrap.providers import build_model_registry
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.workspace_lock import PluginPublicationLock
from bus.event_bus import EventBus
from bus.events import (
    ChannelMessage,
)
from bootstrap.channel_attachment_import import (
    ChannelOutboundAttachmentImporter,
    import_channel_attachments,
)
from bus.processing import ProcessingState
from bus.queue import MessageBus
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources
from session.activity import PresenceStore
from session.manager import SessionManager


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
    config: Config
    http_resources: SharedHttpResources
    loop: AgentLoop
    bus: MessageBus
    event_bus: EventBus
    tools: ToolRegistry
    push_tool: MessagePushTool
    session_manager: SessionManager
    provider: LLMProvider
    light_provider: LLMProvider | None
    memory_runtime: MemoryRuntime
    presence: PresenceStore
    channel_attachment_store: "ChannelAttachmentArtifactStore | None" = None
    model_registry: ModelRegistry | None = None
    agent_provider: LLMProvider | None = None
    plugin_manager: "PluginManager | None" = None
    workspace: Path | None = None
    background_job_host: object | None = None
    plugin_publication_lock: PluginPublicationLock | None = None
    _plugin_publication_locked: bool = False

    def _lock_plugin_publication(self) -> None:
        """在首次消费 plugin-home 前取得整个 Core 生命周期的独占权。"""

        if self._plugin_publication_locked or self.plugin_publication_lock is None:
            return
        self.plugin_publication_lock.acquire()
        self._plugin_publication_locked = True

    def bind_conversation_runtime(self, runtime: object) -> None:
        """Bind the unique ConversationRuntime before plugin job admission."""

        session_creator = getattr(
            self.session_manager.control_store,
            "create_session",
            None,
        )
        if not callable(session_creator):
            raise RuntimeError("Core SessionManager 缺少 programmatic session creator")
        session_reader = getattr(
            self.session_manager.control_store,
            "get_session_meta",
            None,
        )
        if not callable(session_reader):
            raise RuntimeError("Core SessionManager 缺少 programmatic session reader")
        manager = self.plugin_manager
        if manager is not None:
            manager.bind_conversation_runtime(
                runtime,
                programmatic_session_creator=session_creator,
                programmatic_session_reader=session_reader,
            )
        host = self.background_job_host
        if host is not None:
            bind = getattr(host, "bind_conversation_runtime", None)
            if not callable(bind):
                raise RuntimeError(
                    "BackgroundJob Host 缺少 ConversationRuntime binding"
                )
            bind(
                runtime,
                programmatic_session_creator=session_creator,
                programmatic_session_reader=session_reader,
            )

    async def start(self) -> None:
        """启动外部连接和插件扩展。"""

        # 1. 加载插件后同步 skill，再绑定工具 hook。
        if self.plugin_manager is not None:
            self._lock_plugin_publication()
            await self.plugin_manager.load_all()
            if self.workspace is not None:
                from agent.plugins.skill_links import PluginSkillLinker

                link_result = PluginSkillLinker(
                    workspace=self.workspace,
                    plugin_roots=self.plugin_manager.plugin_dirs,
                ).sync(self.plugin_manager.active_plugins())
                logger.info(
                    "插件 skill 同步完成: expected=%d created=%d repaired=%d removed=%d skipped=%d",
                    link_result.expected,
                    link_result.created,
                    link_result.repaired,
                    link_result.removed,
                    link_result.skipped,
                )
            sync_manifest = getattr(self.plugin_manager, "sync_manifest", None)
            if callable(sync_manifest):
                manifest_path = sync_manifest()
                logger.info("插件清单已同步: %s", manifest_path)
            logger.info("插件加载完成: %d 个", self.plugin_manager.loaded_count)

    async def inspect_modules(self) -> str:
        """按实际运行时依赖生成各阶段模块图。"""

        # 1. 先加载插件，确保展示的是当前快照。
        if self.plugin_manager is not None:
            self._lock_plugin_publication()
            await self.plugin_manager.load_all()

        from agent.lifecycle.phase import inspect_phase
        from agent.lifecycle.phases.after_reasoning import (
            default_after_reasoning_modules,
        )
        from agent.lifecycle.phases.after_step import default_after_step_modules
        from agent.lifecycle.phases.after_turn import default_after_turn_modules
        from agent.lifecycle.phases.before_reasoning import (
            default_before_reasoning_modules,
        )
        from agent.lifecycle.phases.before_step import default_before_step_modules
        from agent.lifecycle.phases.before_turn import default_before_turn_modules
        from agent.lifecycle.phases.prompt_render import default_prompt_render_modules

        # 2. 从 AgentLoop 的构造不变量取得固定 Core 阶段依赖。
        pipeline = self.loop._passive_pipeline
        context = self.loop.context

        phases = [
            (
                "before_turn",
                default_before_turn_modules(
                    self.event_bus,
                    self.session_manager,
                    pipeline._context_store,
                ),
            ),
            (
                "before_reasoning",
                default_before_reasoning_modules(
                    self.event_bus,
                    self.tools,
                    self.session_manager,
                    context,
                ),
            ),
            (
                "prompt_render",
                default_prompt_render_modules(
                    self.event_bus,
                    context,
                ),
            ),
            (
                "before_step",
                default_before_step_modules(self.event_bus),
            ),
            (
                "after_step",
                default_after_step_modules(self.event_bus),
            ),
            (
                "after_reasoning",
                default_after_reasoning_modules(
                    self.event_bus,
                    pipeline._session,
                ),
            ),
            (
                "after_turn",
                default_after_turn_modules(
                    self.event_bus,
                    pipeline._outbound_port,
                    context,
                ),
            ),
        ]

        # 3. 分开渲染 Core phase DAG 与 committed v3 Root 拓扑。
        parts: list[str] = []
        for phase_name, modules in phases:
            parts.append("=" * 60)
            parts.append(phase_name)
            parts.append("=" * 60)
            parts.append(inspect_phase(modules))
        snapshot = (
            self.plugin_manager.current_snapshot
            if self.plugin_manager is not None
            else None
        )
        topology = None if snapshot is None else snapshot.composition_topology
        if topology is not None:
            parts.extend(("=" * 60, "composition", "=" * 60))
            parts.append(f"identity: {topology.identity}")
            parts.append(f"revision: {topology.composition_revision}")
            for fiber in topology.fibers:
                parent = fiber.parent or "<root>"
                parts.append(f"fiber: {parent} -> {fiber.name}")
            for listener in topology.listeners:
                parts.append(f"listener: {listener}")
        return "\n".join(parts)

    async def stop(self) -> None:
        """按所有权逆序关闭核心运行时资源。"""

        # 1. 将同步 session close 和 shell cleanup 适配为异步清理步骤。
        async def _stop_shell() -> None:
            shell_tool = self.tools.get_tool("shell")
            shutdown = getattr(shell_tool, "shutdown", None)
            if callable(shutdown):
                result = shutdown()
                if inspect.isawaitable(result):
                    await cast(Awaitable[object], result)

        async def _close_session_manager() -> None:
            self.session_manager.close()

        # 2. 由统一 cleanup runner 完成全部步骤并保留失败。
        await run_cleanup_steps(
            ("shell.shutdown", _stop_shell),
            ("compaction.shutdown", self.loop.shutdown_compaction),
            ("event_bus.aclose", self.event_bus.aclose),
            (
                "plugin_manager.terminate_all",
                (
                    self.plugin_manager.terminate_all
                    if self.plugin_manager is not None
                    else _noop_async
                ),
            ),
            ("plugin_publication_lock.release", self._release_plugin_publication),
            ("session_manager.close", _close_session_manager),
        )

    async def _release_plugin_publication(self) -> None:
        lock = self.plugin_publication_lock
        was_locked = self._plugin_publication_locked
        self._plugin_publication_locked = False
        if was_locked and lock is not None:
            lock.release()


def build_registered_tools(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
    *,
    bus: MessageBus,
    provider,
    light_provider,
    vl_provider=None,
    session_store=None,
    tools: ToolRegistry | None = None,
    event_publisher=None,
    agent_loop_provider: Callable[[], Any] | None = None,
    tool_context_provider: Callable[
        [], ToolExecutionContext | None
    ] = get_current_tool_context,
    restart_coordinator: "RestartCoordinator | None" = None,
) -> tuple[ToolRegistry, MessagePushTool, MemoryRuntime]:
    """按配置顺序构造并注册核心工具资源。"""

    from session.store import SessionStore

    # 1. 构造共享服务；外部传入的 session_store 和 http_resources 不转移 ownership。
    wiring = config.wiring
    _ = agent_loop_provider
    tools = tools if tools is not None else ToolRegistry()
    multimodal = config.multimodal
    vl_available = not multimodal and config.vl_model != ""
    readonly_tools = build_readonly_tools(
        http_resources,
        workspace=workspace,
        multimodal=multimodal,
        vl_available=vl_available,
        context_provider=tool_context_provider,
    )
    store = (
        session_store
        if session_store is not None
        else SessionStore(workspace / "sessions.db")
    )
    push_tool = MessagePushTool(chat_lane=bus.chat_lane)
    memory_result = resolve_memory_toolset_provider(wiring.memory).register(
        tools,
        ToolsetDeps(
            config=config,
            workspace=workspace,
            provider=provider,
            light_provider=light_provider,
            http_resources=http_resources,
            event_publisher=event_publisher,
        ),
    )
    memory_runtime = memory_result.extras["memory_runtime"]
    # 2. 保持 wiring.toolsets 顺序注册。
    for name in wiring.toolsets:
        provider_obj = resolve_toolset_provider(
            name,
            readonly_tools=readonly_tools if name == "meta_common" else None,
        )
        result = provider_obj.register(
            tools,
            ToolsetDeps(
                config=config,
                workspace=workspace,
                session_store=store,
                push_tool=push_tool,
                http_resources=http_resources,
                provider=provider,
                light_provider=light_provider,
                vl_provider=vl_provider,
                vl_model=config.vl_model,
                bus=bus,
                event_publisher=event_publisher,
            ),
        )

    # 3. 自重启只在 supervisor 与 tool_search 两个边界都成立时注册。
    if (
        restart_coordinator is not None
        and restart_coordinator.supervised
        and config.tool_search_enabled
    ):
        from agent.tools.agent_restart import AgentRestartTool

        tools.register(
            AgentRestartTool(restart_coordinator),
            risk="external-side-effect",
            always_on=False,
            preloadable=False,
            requires_turn_search=True,
            search_hint="重启 akashic agent 服务 重新加载核心配置",
        )

    return tools, push_tool, memory_runtime


def _build_loop_deps(
    *,
    config: Config,
    workspace: Path,
    bus: MessageBus,
    provider: LLMProvider,
    fallback_provider: LLMProvider | None,
    fallback_model: str,
    light_provider: LLMProvider | None,
    tools: ToolRegistry,
    session_manager: SessionManager,
    presence: PresenceStore,
    processing_state: ProcessingState,
    event_bus: EventBus,
    memory_runtime: MemoryRuntime,
    outbound_port: OutboundPort | None = None,
) -> AgentLoopDeps:
    """将已构造的 runtime 资源装配成 AgentLoop 依赖。"""

    # 1. 按 typed wiring 解析 context，并注入配置声明的媒体能力。
    wiring = config.wiring
    context = resolve_context_factory(wiring.context)(
        workspace,
        memory_runtime.markdown.store,
    )
    if isinstance(context, ContextBuilder):
        context.set_media_capabilities(
            multimodal=config.multimodal,
            vl_available=config.vl_model != "",
        )

    # 2. 绑定模型与 session；动态上下文由 Prompt lifecycle 插件负责。
    light = light_provider or provider
    llm_services = LLMServices(
        provider=provider,
        light_provider=light,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
    )
    session_services = SessionServices(
        session_manager=session_manager, presence=presence
    )

    return AgentLoopDeps(
        bus=bus,
        event_bus=event_bus,
        provider=provider,
        tools=tools,
        session_manager=session_manager,
        workspace=workspace,
        presence=presence,
        light_provider=light_provider,
        processing_state=processing_state,
        memory_runtime=memory_runtime,
        context=context,
        llm_services=llm_services,
        session_services=session_services,
        outbound_port=outbound_port,
    )


def build_core_runtime(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
    restart_coordinator: "RestartCoordinator | None" = None,
    *,
    clear_stale_session_admissions: bool = False,
) -> CoreRuntime:
    """构造核心运行时及其插件快照依赖。"""

    # 1. 创建总线、provider 和由 CoreRuntime.stop 负责关闭的 session owner。
    bus = MessageBus()
    event_bus = EventBus()
    model_registry = build_model_registry(config)
    provider = model_registry.provider("default")
    fallback_provider = model_registry.provider(
        "default",
        honor_session_selection=False,
    )
    light_provider = model_registry.provider("fast")
    agent_provider = model_registry.provider("agent")
    vl_provider = model_registry.provider("vision") if config.vl_model else None
    # 2. agent_provider 供 AgentLoop 使用，provider 供 consolidation 事件提取使用。
    loop_provider = agent_provider
    loop_model = config.agent_model or config.model
    session_manager = SessionManager(workspace)
    if clear_stale_session_admissions:
        session_manager.clear_stale_admissions()
    bus.bind_mobile_session_admission_owner(session_manager)
    loop_ref: dict[str, AgentLoop] = {}
    tools, push_tool, memory_runtime = build_registered_tools(
        config,
        workspace,
        http_resources,
        bus=bus,
        provider=provider,
        light_provider=light_provider,
        vl_provider=vl_provider,
        session_store=session_manager._store,
        event_publisher=event_bus,
        agent_loop_provider=lambda: loop_ref.get("loop"),
        restart_coordinator=restart_coordinator,
    )
    presence = PresenceStore(session_manager._store)
    processing_state = ProcessingState()
    loop_deps = _build_loop_deps(
        config=config,
        workspace=workspace,
        bus=bus,
        provider=loop_provider,
        fallback_provider=fallback_provider,
        fallback_model=config.model,
        light_provider=light_provider,
        tools=tools,
        session_manager=session_manager,
        presence=presence,
        processing_state=processing_state,
        event_bus=event_bus,
        memory_runtime=memory_runtime,
        outbound_port=PushToolOutboundPort(push_tool, commit_role="passive"),
    )
    loop = AgentLoop(
        loop_deps,
        AgentLoopConfig(
            llm=LLMConfig(
                model=loop_model,
                light_model=config.light_model,
                max_iterations=config.max_iterations,
                max_tokens=0,
                tool_search_enabled=config.tool_search_enabled,
                multimodal=config.multimodal,
                vl_available=config.vl_model != "",
            ),
            context_compaction=config.context_compaction,
        ),
    )
    loop_ref["loop"] = loop
    wire_turn_lifecycle(
        lifecycle=TurnLifecycle(event_bus),
        active_turn_states=loop.active_turn_states,
    )

    from agent.plugins.manager import PluginManager as _PluginManager
    from infra.channels.artifacts import ChannelAttachmentArtifactStore

    # 3. 创建插件 manager，并把 snapshot store 绑定到 loop。
    channel_attachment_store = ChannelAttachmentArtifactStore(
        workspace=workspace,
        session_store=session_manager.control_store,
    )
    session_services = loop_deps.session_services
    if session_services is None:
        raise RuntimeError("AgentLoop 缺少 SessionServices")
    session_services.outbound_attachment_importer = ChannelOutboundAttachmentImporter(
        channel_attachment_store
    )
    plugin_manager = _PluginManager(
        plugin_dirs=_resolve_plugin_dirs(workspace),
        event_bus=event_bus,
        tool_registry=tools,
        workspace=workspace,
        session_manager=session_manager,
        installed_cache_root=plugins_root() / "cache",
        channel_attachment_store=channel_attachment_store,
        disabled_builtin_plugins=config.disabled_builtin_plugins,
        text_embedding_settings=TextEmbeddingSettings(
            base_url=(
                config.memory.embedding.base_url
                or config.light_base_url
                or config.base_url
                or ""
            ),
            api_key=(
                config.memory.embedding.api_key
                or config.light_api_key
                or config.api_key
            ),
            model=config.memory.embedding.model,
            output_dimensionality=config.memory.embedding.output_dimensionality,
        ),
    )
    plugin_manager.bind_continuation_publisher(bus.publish_inbound)
    plugin_manager.bind_delivery_sender(push_tool.dispatch)
    plugin_manager.bind_durable_delivery_sender(
        lambda request, provider_started: _dispatch_v3_durable_delivery(
            plugin_manager,
            bus,
            request,
            provider_started,
            session_manager=session_manager,
        )
    )
    from agent.plugins.generation_activity_host import ActivityHost
    from agent.plugins.generation_job_host import BackgroundJobActivityAdapter

    background_jobs = BackgroundJobActivityAdapter(
        plugin_manager.snapshot_store,
        model_provider=provider,
        model_registry=model_registry,
        workspace=str(workspace),
    )
    plugin_manager.bind_activity_host(ActivityHost((background_jobs,)))
    bus.bind_channel_outbound_dispatcher(
        plugin_manager.channel_generation_host.dispatch_outbound
    )
    push_tool.bind_v3_channel_dispatcher(
        lambda message, passive: _dispatch_v3_channel_push(
            plugin_manager,
            bus,
            message,
            passive,
            channel_attachment_store,
            session_manager=session_manager,
        )
    )
    plugin_manager.channel_generation_host.bind_inbound_publisher(
        bus.publish_channel_inbound
    )
    loop.bind_runtime_snapshot_store(plugin_manager.snapshot_store)
    return CoreRuntime(
        config=config,
        workspace=workspace,
        http_resources=http_resources,
        loop=loop,
        bus=bus,
        event_bus=event_bus,
        tools=tools,
        push_tool=push_tool,
        session_manager=session_manager,
        provider=provider,
        light_provider=light_provider,
        agent_provider=agent_provider,
        memory_runtime=memory_runtime,
        presence=presence,
        channel_attachment_store=channel_attachment_store,
        model_registry=model_registry,
        plugin_manager=plugin_manager,
        background_job_host=background_jobs,
        plugin_publication_lock=PluginPublicationLock(plugins_root()),
    )


def _resolve_plugin_dirs(workspace: Path) -> list[Path]:
    project_root = Path(__file__).resolve().parent.parent
    roots = [project_root / "plugins"]
    extra = os.environ.get("AKASHIC_EXTRA_PLUGIN_DIRS", "")
    roots.extend(
        Path(item).expanduser() for item in extra.split(os.pathsep) if item.strip()
    )
    return roots
