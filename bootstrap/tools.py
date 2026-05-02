from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agent.plugins.manager import PluginManager

logger = logging.getLogger(__name__)

from agent.config_models import Config, WiringConfig
from agent.context import ContextBuilder
from agent.looping.consolidation import ConsolidationService
from agent.peer_agent.process_manager import PeerProcessManager
from agent.peer_agent.poller import PeerAgentPoller
from agent.peer_agent.registry import PeerAgentRegistry
from agent.looping.core import AgentLoop
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig,
    LLMServices,
    MemoryConfig,
    MemoryServices,
    SessionServices,
    TurnScheduler,
)
from agent.mcp.registry import McpServerRegistry
from agent.provider import LLMProvider
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.scheduler import SchedulerService
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.turns.outbound import BusOutboundPort
from bootstrap.toolsets.mcp import McpToolsetProvider
from bootstrap.toolsets.memory import MemoryToolsetProvider
from bootstrap.toolsets.meta import (
    CommonMetaToolsetProvider,
    SpawnToolsetProvider,
    build_readonly_tools,
)
from bootstrap.toolsets.peer import build_peer_agent_resources
from bootstrap.toolsets.protocol import ToolsetDeps
from bootstrap.toolsets.schedule import (
    SchedulerToolsetProvider,
    build_scheduler,
)
from bootstrap.wiring import (
    wire_turn_lifecycle,
    resolve_context_factory,
    resolve_memory_toolset_provider,
    resolve_toolset_provider,
)
from agent.lifecycle.facade import TurnLifecycle
from bootstrap.providers import build_providers, build_vl_provider
from bus.event_bus import EventBus
from bus.processing import ProcessingState
from bus.queue import MessageBus
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources
from memory2.profile_extractor import ProfileFactExtractor
from memory2.query_rewriter import QueryRewriter
from memory2.sufficiency_checker import SufficiencyChecker
from proactive_v2.presence import PresenceStore
from session.manager import SessionManager


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
    scheduler: SchedulerService
    provider: LLMProvider
    light_provider: LLMProvider | None
    mcp_registry: McpServerRegistry
    memory_runtime: MemoryRuntime
    presence: PresenceStore
    peer_process_manager: PeerProcessManager | None
    peer_poller: PeerAgentPoller | None
    agent_provider: LLMProvider | None = None
    plugin_manager: "PluginManager | None" = None

    async def start(self) -> None:
        self.mcp_registry.start_connect_all_background()

        if (
            self.peer_poller is not None
            and self.peer_process_manager is not None
            and self.config.peer_agents
        ):
            peer_registry = PeerAgentRegistry(
                process_manager=self.peer_process_manager,
                poller=self.peer_poller,
                requester=self.http_resources.local_service,
            )
            peer_tools = await peer_registry.discover_all(self.config.peer_agents)
            for t in peer_tools:
                self.tools.register(
                    t,
                    always_on=False,
                    risk="external-side-effect",
                )
            self.peer_poller.start()
        if self.plugin_manager is not None:
            await self.plugin_manager.load_all()
            logger.info("插件加载完成: %d 个", self.plugin_manager.loaded_count)
            self.loop.add_before_turn_plugin_modules(
                self.plugin_manager.before_turn_modules_early,
                self.plugin_manager.before_turn_modules_late,
            )
            self.loop.add_before_reasoning_plugin_modules(
                self.plugin_manager.before_reasoning_modules_before_emit,
                self.plugin_manager.before_reasoning_modules_after_emit,
            )
            self.loop.add_prompt_render_plugin_modules(
                self.plugin_manager.prompt_render_modules_top,
                self.plugin_manager.prompt_render_modules_bottom,
            )
            self.loop.add_before_step_plugin_modules(
                self.plugin_manager.before_step_modules_before_emit,
                self.plugin_manager.before_step_modules_after_emit,
            )
            self.loop.add_after_step_plugin_modules(
                self.plugin_manager.after_step_modules_before_fanout,
                self.plugin_manager.after_step_modules_after_fanout,
            )
            self.loop.add_after_reasoning_plugin_modules(
                self.plugin_manager.after_reasoning_modules_before_emit,
                self.plugin_manager.after_reasoning_modules_before_persist,
            )
            self.loop.add_after_turn_plugin_modules(
                self.plugin_manager.after_turn_modules_before_commit,
                self.plugin_manager.after_turn_modules_before_fanout,
            )
            if self.plugin_manager.tool_hooks:
                self.loop.add_tool_hooks(self.plugin_manager.tool_hooks)
                spawn_tool = self.tools.get_tool("spawn")
                if spawn_tool is not None and hasattr(spawn_tool, "add_tool_hooks"):
                    spawn_tool.add_tool_hooks(self.plugin_manager.tool_hooks)

    async def stop(self) -> None:
        if self.plugin_manager is not None:
            await self.plugin_manager.terminate_all()
        await self.mcp_registry.shutdown()
        await self.event_bus.aclose()
        if self.peer_poller is not None:
            await self.peer_poller.stop()
        if self.peer_process_manager is not None:
            await self.peer_process_manager.shutdown_all()


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
) -> tuple[
    ToolRegistry,
    MessagePushTool,
    SchedulerService,
    McpServerRegistry,
    MemoryRuntime,
    PeerProcessManager | None,
    PeerAgentPoller | None,
]:
    from session.store import SessionStore

    # ── 第一阶段：建服务（依赖无顺序陷阱）────────────────────────────────────
    wiring = getattr(config, "wiring", WiringConfig())
    tools = tools or ToolRegistry()
    multimodal = getattr(config, "multimodal", True)
    vl_available = (not multimodal) and bool(getattr(config, "vl_model", ""))
    readonly_tools = build_readonly_tools(
        http_resources, multimodal=multimodal, vl_available=vl_available
    )
    store = session_store or SessionStore(workspace / "sessions.db")
    push_tool = MessagePushTool()
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
    scheduler = build_scheduler(
        workspace,
        push_tool,
        agent_loop_provider=agent_loop_provider,
    )
    peer_process_manager, peer_poller = build_peer_agent_resources(
        config, bus, http_resources
    )

    # ── 第二阶段：注册工具（所有服务已就绪）──────────────────────────────────
    mcp_registry = None
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
                vl_model=getattr(config, "vl_model", ""),
                bus=bus,
                memory_port=memory_runtime.port,
                scheduler=scheduler,
                event_publisher=event_publisher,
            ),
        )
        maybe_mcp = result.extras.get("mcp_registry")
        if maybe_mcp is not None:
            mcp_registry = maybe_mcp
    if mcp_registry is None:
        from agent.mcp.registry import McpServerRegistry

        mcp_registry = McpServerRegistry(
            config_path=workspace / "mcp_servers.json",
            tool_registry=tools,
        )

    return (
        tools,
        push_tool,
        scheduler,
        mcp_registry,
        memory_runtime,
        peer_process_manager,
        peer_poller,
    )


def _build_loop_deps(
    *,
    config: Config,
    workspace: Path,
    bus: MessageBus,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    tools: ToolRegistry,
    session_manager: SessionManager,
    presence: PresenceStore,
    processing_state: ProcessingState,
    event_bus: EventBus,
    memory_runtime: MemoryRuntime,
) -> AgentLoopDeps:
    wiring = getattr(config, "wiring", WiringConfig())
    llm_config = LLMConfig(
        model=config.model,
        light_model=config.light_model,
        max_iterations=config.max_iterations,
        max_tokens=config.max_tokens,
        tool_search_enabled=config.tool_search_enabled,
        multimodal=bool(getattr(config, "multimodal", True)),
        vl_available=bool(getattr(config, "vl_model", "")),
    )
    memory_config = MemoryConfig(
        window=config.memory_window,
        top_k_procedure=config.memory_v2.top_k_procedure,
        top_k_history=config.memory_v2.top_k_history,
        route_intention_enabled=config.memory_v2.route_intention_enabled,
        procedure_guard_enabled=config.memory_v2.procedure_guard_enabled,
        gate_llm_timeout_ms=config.memory_v2.gate_llm_timeout_ms,
        gate_max_tokens=config.memory_v2.gate_max_tokens,
        hyde_enabled=config.memory_v2.hyde_enabled,
        hyde_timeout_ms=config.memory_v2.hyde_timeout_ms,
    )
    resolved_memory_config = MemoryConfig(
        window=memory_config.window,
        top_k_procedure=min(3, max(1, int(memory_config.top_k_procedure))),
        top_k_history=max(1, int(memory_config.top_k_history)),
        route_intention_enabled=memory_config.route_intention_enabled,
        procedure_guard_enabled=memory_config.procedure_guard_enabled,
        gate_llm_timeout_ms=max(100, int(memory_config.gate_llm_timeout_ms)),
        gate_max_tokens=max(32, int(memory_config.gate_max_tokens)),
        hyde_enabled=memory_config.hyde_enabled,
        hyde_timeout_ms=memory_config.hyde_timeout_ms,
    )

    light = light_provider or provider
    query_rewriter = (
        QueryRewriter(
            llm_client=light,
            model=config.light_model or config.model,
            max_tokens=config.memory_v2.gate_max_tokens,
            timeout_ms=config.memory_v2.gate_llm_timeout_ms,
        )
        if config.memory_v2.route_intention_enabled
        else None
    )
    sufficiency_checker = SufficiencyChecker(
        llm_client=light,
        model=config.light_model or config.model,
    )
    profile_extractor = ProfileFactExtractor(
        llm_client=light,
        model=config.light_model or config.model,
    )
    hyde_enhancer = None
    if memory_config.hyde_enabled and llm_config.light_model:
        from memory2.hyde_enhancer import HyDEEnhancer

        hyde_enhancer = HyDEEnhancer(
            light_provider=light,
            light_model=llm_config.light_model,
            timeout_s=memory_config.hyde_timeout_ms / 1000.0,
        )

    context = resolve_context_factory(wiring.context)(
        workspace,
        getattr(memory_runtime, "profile_reader", None) or memory_runtime.port,
    )
    if isinstance(context, ContextBuilder):
        context.set_media_capabilities(
            multimodal=bool(getattr(config, "multimodal", True)),
            vl_available=bool(getattr(config, "vl_model", "")),
        )
    memory_engine = getattr(memory_runtime, "engine", None)
    memory_facade = getattr(memory_runtime, "facade", None)
    llm_services = LLMServices(provider=provider, light_provider=light)
    memory_services = MemoryServices(
        engine=memory_engine,
        facade=memory_facade,
        query_rewriter=query_rewriter,
        hyde_enhancer=hyde_enhancer,
        sufficiency_checker=sufficiency_checker,
    )
    session_services = SessionServices(
        session_manager=session_manager, presence=presence
    )
    consolidation = ConsolidationService(
        memory_port=memory_runtime.port,
        profile_maint=getattr(memory_runtime, "profile_maint", None)
        or memory_runtime.port,
        provider=provider,
        model=config.model,
        keep_count=memory_config.keep_count,
        profile_extractor=profile_extractor,
        recent_context_provider=light or provider,
        recent_context_model=config.light_model or config.model,
    )
    if memory_facade is not None:
        memory_facade.bind_consolidation_runner(
            lambda session, archive_all: consolidation.consolidate(
                session,
                archive_all=archive_all,
            )
        )

    async def _consolidate_and_save(session: object) -> None:
        # scheduler 只负责起后台任务；真正的工作是“consolidate + save session”这两步。
        if memory_facade is not None:
            await memory_facade.run_consolidation(session)  # type: ignore[arg-type]
        else:
            await consolidation.consolidate(session)  # type: ignore[arg-type]
        await session_manager.save_async(session)  # type: ignore[arg-type]

    turn_scheduler = TurnScheduler(
        consolidation_runner=_consolidate_and_save,
        keep_count=memory_config.keep_count,
    )
    retrieval_pipeline = DefaultMemoryRetrievalPipeline(
        memory=memory_services,
        memory_config=resolved_memory_config,
        llm=llm_services,
        workspace=workspace,
        light_model=llm_config.light_model or llm_config.model,
        event_publisher=event_bus,
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
        query_rewriter=query_rewriter,
        sufficiency_checker=sufficiency_checker,
        profile_extractor=profile_extractor,
        retrieval_pipeline=retrieval_pipeline,
        context=context,
        llm_services=llm_services,
        memory_services=memory_services,
        session_services=session_services,
        hyde_enhancer=hyde_enhancer,
        consolidation_service=consolidation,
        scheduler=turn_scheduler,
    )


def build_core_runtime(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
) -> CoreRuntime:
    bus = MessageBus()
    event_bus = EventBus()
    provider, light_provider, agent_provider = build_providers(config)
    vl_provider = build_vl_provider(config)
    # agent_provider is used for the AgentLoop (QA / tool calling).
    # provider (llm.main) is used for consolidation event extraction.
    loop_provider = agent_provider or provider
    loop_model = config.agent_model or config.model
    session_manager = SessionManager(workspace)
    loop_ref: dict[str, AgentLoop] = {}
    tools, push_tool, scheduler, mcp_registry, memory_runtime, peer_pm, peer_poller = (
        build_registered_tools(
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
        )
    )
    presence = PresenceStore(session_manager._store)
    processing_state = ProcessingState()
    loop_deps = _build_loop_deps(
        config=config,
        workspace=workspace,
        bus=bus,
        provider=loop_provider,
        light_provider=light_provider,
        tools=tools,
        session_manager=session_manager,
        presence=presence,
        processing_state=processing_state,
        event_bus=event_bus,
        memory_runtime=memory_runtime,
    )
    loop = AgentLoop(
        loop_deps,
        AgentLoopConfig(
            llm=LLMConfig(
                model=loop_model,
                light_model=config.light_model,
                max_iterations=config.max_iterations,
                max_tokens=config.max_tokens,
                tool_search_enabled=config.tool_search_enabled,
                multimodal=bool(getattr(config, "multimodal", True)),
                vl_available=bool(getattr(config, "vl_model", "")),
            ),
            memory=MemoryConfig(
                window=config.memory_window,
                top_k_procedure=config.memory_v2.top_k_procedure,
                top_k_history=config.memory_v2.top_k_history,
                route_intention_enabled=config.memory_v2.route_intention_enabled,
                procedure_guard_enabled=config.memory_v2.procedure_guard_enabled,
                gate_llm_timeout_ms=config.memory_v2.gate_llm_timeout_ms,
                gate_max_tokens=config.memory_v2.gate_max_tokens,
                hyde_enabled=config.memory_v2.hyde_enabled,
                hyde_timeout_ms=config.memory_v2.hyde_timeout_ms,
            ),
        ),
    )
    loop_ref["loop"] = loop
    wire_turn_lifecycle(
        lifecycle=TurnLifecycle(event_bus),
        active_turn_states=loop.active_turn_states,
    )

    from agent.plugins.manager import PluginManager as _PluginManager
    plugin_manager = _PluginManager(
        plugin_dirs=_resolve_plugin_dirs(workspace),
        event_bus=event_bus,
        tool_registry=tools,
        workspace=workspace,
    )

    return CoreRuntime(
        config=config,
        http_resources=http_resources,
        loop=loop,
        bus=bus,
        event_bus=event_bus,
        tools=tools,
        push_tool=push_tool,
        session_manager=session_manager,
        scheduler=scheduler,
        provider=provider,
        light_provider=light_provider,
        agent_provider=agent_provider,
        mcp_registry=mcp_registry,
        memory_runtime=memory_runtime,
        presence=presence,
        peer_process_manager=peer_pm,
        peer_poller=peer_poller,
        plugin_manager=plugin_manager,
    )


def _resolve_plugin_dirs(workspace: Path) -> list[Path]:
    project_root = Path(__file__).resolve().parent.parent
    return [project_root / "plugins"]
