from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.config_models import Config
from agent.peer_agent.process_manager import PeerProcessManager
from agent.peer_agent.poller import PeerAgentPoller
from agent.peer_agent.registry import PeerAgentRegistry
from agent.looping.core import AgentLoop, AgentLoopConfig, AgentLoopDeps
from agent.mcp.registry import McpServerRegistry
from agent.provider import LLMProvider
from agent.scheduler import SchedulerService
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from bootstrap.toolsets.fitbit import register_fitbit_tools
from bootstrap.toolsets.mcp import register_mcp_tools
from bootstrap.toolsets.memory import build_memory_toolset
from bootstrap.toolsets.meta import (
    build_readonly_tools,
    register_meta_and_common_tools,
    register_spawn_tool,
)
from bootstrap.toolsets.peer import build_peer_agent_resources
from bootstrap.toolsets.schedule import build_scheduler, register_scheduler_tools
from bootstrap.toolsets.skill_actions import register_skill_action_tools
from bootstrap.providers import build_providers
from bus.processing import ProcessingState
from bus.queue import MessageBus
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources
from memory2.profile_extractor import ProfileFactExtractor
from memory2.query_rewriter import QueryRewriter
from memory2.sufficiency_checker import SufficiencyChecker
from proactive.presence import PresenceStore
from session.manager import SessionManager


@dataclass
class CoreRuntime:
    config: Config
    http_resources: SharedHttpResources
    loop: AgentLoop
    bus: MessageBus
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

    async def start(self) -> None:
        await self.mcp_registry.load_and_connect_all()

        if self.peer_poller is not None and self.config.peer_agents:
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
                    tags=["peer", "delegate"],
                    risk="external-side-effect",
                    search_keywords=["agent", "专家"],
                )
            self.peer_poller.start()

    async def stop(self) -> None:
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
    session_store=None,
    tools: ToolRegistry | None = None,
    observe_writer=None,
) -> tuple[ToolRegistry, MessagePushTool, SchedulerService, McpServerRegistry, MemoryRuntime, PeerProcessManager | None, PeerAgentPoller | None]:
    from session.store import SessionStore
    tools = tools or ToolRegistry()
    readonly_tools = build_readonly_tools(http_resources)
    store = session_store or SessionStore(workspace / "sessions.db")
    push_tool = register_meta_and_common_tools(tools, readonly_tools, store)
    register_skill_action_tools(tools, workspace)
    register_fitbit_tools(tools, config, http_resources)
    subagent_manager = register_spawn_tool(
        tools,
        config,
        workspace,
        bus,
        provider,
        http_resources,
    )
    memory_runtime = build_memory_toolset(
        config,
        workspace,
        tools,
        provider,
        light_provider,
        http_resources,
        observe_writer=observe_writer,
    )
    subagent_manager.set_memory_port(memory_runtime.port)
    scheduler = build_scheduler(workspace, push_tool)
    register_scheduler_tools(tools, scheduler)
    mcp_registry = register_mcp_tools(tools, workspace)

    # Peer agent 工具（异步注册，需在 event loop 中运行）
    peer_process_manager, peer_poller = build_peer_agent_resources(
        config, bus, http_resources
    )
    return tools, push_tool, scheduler, mcp_registry, memory_runtime, peer_process_manager, peer_poller


def build_core_runtime(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
    observe_writer=None,
) -> CoreRuntime:
    bus = MessageBus()
    provider, light_provider = build_providers(config)
    session_manager = SessionManager(workspace)
    tools, push_tool, scheduler, mcp_registry, memory_runtime, peer_pm, peer_poller = build_registered_tools(
        config,
        workspace,
        http_resources,
        bus=bus,
        provider=provider,
        light_provider=light_provider,
        session_store=session_manager._store,
        observe_writer=observe_writer,
    )
    presence = PresenceStore(workspace / "presence.json")
    processing_state = ProcessingState()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=bus,
            provider=provider,
            tools=tools,
            session_manager=session_manager,
            workspace=workspace,
            presence=presence,
            light_provider=light_provider,
            processing_state=processing_state,
            memory_runtime=memory_runtime,
            observe_writer=observe_writer,
            query_rewriter=(
                QueryRewriter(
                    llm_client=light_provider or provider,
                    model=config.light_model or config.model,
                    max_tokens=config.memory_v2.gate_max_tokens,
                    timeout_ms=config.memory_v2.gate_llm_timeout_ms,
                )
                if config.memory_v2.route_intention_enabled
                else None
            ),
            sufficiency_checker=(
                SufficiencyChecker(
                    llm_client=light_provider or provider,
                    model=config.light_model or config.model,
                )
                if config.memory_v2.sufficiency_check_enabled
                else None
            ),
            profile_extractor=(
                ProfileFactExtractor(
                    llm_client=light_provider or provider,
                    model=config.light_model or config.model,
                )
                if config.memory_v2.profile_extraction_enabled
                else None
            ),
        ),
        AgentLoopConfig(
            model=config.model,
            light_model=config.light_model,
            max_iterations=config.max_iterations,
            max_tokens=config.max_tokens,
            memory_window=config.memory_window,
            memory_top_k_procedure=config.memory_v2.top_k_procedure,
            memory_top_k_history=config.memory_v2.top_k_history,
            memory_route_intention_enabled=config.memory_v2.route_intention_enabled,
            memory_sop_guard_enabled=config.memory_v2.sop_guard_enabled,
            memory_gate_llm_timeout_ms=config.memory_v2.gate_llm_timeout_ms,
            memory_gate_max_tokens=config.memory_v2.gate_max_tokens,
            tool_search_enabled=config.tool_search_enabled,
            memory_hyde_enabled=config.memory_v2.hyde_enabled,
            memory_hyde_timeout_ms=config.memory_v2.hyde_timeout_ms,
        ),
    )

    scheduler.agent_loop = loop

    return CoreRuntime(
        config=config,
        http_resources=http_resources,
        loop=loop,
        bus=bus,
        tools=tools,
        push_tool=push_tool,
        session_manager=session_manager,
        scheduler=scheduler,
        provider=provider,
        light_provider=light_provider,
        mcp_registry=mcp_registry,
        memory_runtime=memory_runtime,
        presence=presence,
        peer_process_manager=peer_pm,
        peer_poller=peer_poller,
    )
