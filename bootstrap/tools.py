from __future__ import annotations

import asyncio
import logging
import inspect
import os
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:
    from agent.plugins.manager import PluginManager

logger = logging.getLogger(__name__)

from agent.config_models import Config
from agent.context import ContextBuilder
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
)
from agent.mcp.watcher import WorkspaceMcpWatcher
from agent.provider import LLMProvider
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.scheduler import SchedulerService
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from bootstrap.toolsets.meta import build_readonly_tools
from bootstrap.toolsets.peer import build_peer_agent_resources
from bootstrap.toolsets.protocol import ToolsetDeps
from bootstrap.toolsets.schedule import build_scheduler
from bootstrap.wiring import (
    wire_turn_lifecycle,
    resolve_context_factory,
    resolve_memory_toolset_provider,
    resolve_toolset_provider,
)
from agent.lifecycle.facade import TurnLifecycle
from agent.plugins.jobs import ProviderPluginLlmService
from bootstrap.providers import build_providers, build_vl_provider
from bootstrap.cleanup import run_cleanup_steps
from bus.event_bus import EventBus
from bus.processing import ProcessingState
from bus.queue import MessageBus
from core.memory.markdown import MemoryLifecycleBindRequest, MarkdownMemoryMaintenance
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources
from proactive_v2.presence import PresenceStore
from session.manager import Session, SessionManager


async def _noop_async() -> None:
    return None


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
    workspace_mcp_watcher: WorkspaceMcpWatcher
    workspace_mcp_watcher_task: asyncio.Task[None] | None
    memory_runtime: MemoryRuntime
    presence: PresenceStore
    peer_process_manager: PeerProcessManager | None
    peer_poller: PeerAgentPoller | None
    agent_provider: LLMProvider | None = None
    plugin_manager: "PluginManager | None" = None
    workspace: Path | None = None

    async def start(self) -> None:
        """启动外部连接、peer 资源和插件扩展。"""

        # 1. 仅在 peer 配置真实启用时发现工具并启动轮询。
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

        # 2. workspace MCP 必须先原子发布，插件同名声明随后 fail-loud
        await self.workspace_mcp_watcher.reconcile()

        # 3. 加载插件后同步 skill，再绑定工具 hook。
        if self.plugin_manager is not None:
            await self.plugin_manager.load_all()
            self.plugin_manager.assert_no_workspace_mcp_plugin_conflicts()
            if self.workspace is not None:
                from agent.plugins.skill_links import PluginSkillLinker

                link_result = PluginSkillLinker(
                    workspace=self.workspace,
                    plugin_roots=self.plugin_manager.plugin_dirs,
                    memory_engine=self.memory_runtime.engine,
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
            if self.plugin_manager.tool_hooks:
                self.loop.add_tool_hooks(self.plugin_manager.tool_hooks)
                spawn_tool = self.tools.get_tool("spawn")
                if spawn_tool is not None and hasattr(spawn_tool, "add_tool_hooks"):
                    spawn_tool.add_tool_hooks(self.plugin_manager.tool_hooks)

        # 4. 首次启动全部成功后才启动容错热重载 watcher
        self.workspace_mcp_watcher_task = asyncio.create_task(
            self.workspace_mcp_watcher.run(), name="workspace_mcp_watcher"
        )

    async def inspect_modules(self) -> str:
        """按实际运行时依赖生成各阶段模块图。"""

        # 1. 先加载插件，确保展示的是当前快照。
        if self.plugin_manager is not None:
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

        # 2. 收集各阶段插件贡献。
        manager = self.plugin_manager
        before_turn_modules = manager.before_turn_modules if manager is not None else []
        before_reasoning_modules = (
            manager.before_reasoning_modules if manager is not None else []
        )
        prompt_render_modules = manager.prompt_render_modules if manager is not None else []
        before_step_modules = manager.before_step_modules if manager is not None else []
        after_step_modules = manager.after_step_modules if manager is not None else []
        after_reasoning_modules = (
            manager.after_reasoning_modules if manager is not None else []
        )
        after_turn_modules = manager.after_turn_modules if manager is not None else []

        # 3. 从 AgentLoop 的构造不变量取得阶段依赖。
        agent_core = self.loop._agent_core
        pipeline = agent_core.pipeline
        context = self.loop.context

        phases = [
            (
                "before_turn",
                default_before_turn_modules(
                    self.event_bus,
                    self.session_manager,
                    pipeline._context_store,
                    plugin_modules=cast(Any, before_turn_modules),
                ),
            ),
            (
                "before_reasoning",
                default_before_reasoning_modules(
                    self.event_bus,
                    self.tools,
                    self.session_manager,
                    context,
                    plugin_modules=cast(Any, before_reasoning_modules),
                ),
            ),
            (
                "prompt_render",
                default_prompt_render_modules(
                    self.event_bus,
                    context,
                    plugin_modules=cast(Any, prompt_render_modules),
                ),
            ),
            (
                "before_step",
                default_before_step_modules(
                    self.event_bus,
                    plugin_modules=cast(Any, before_step_modules),
                ),
            ),
            (
                "after_step",
                default_after_step_modules(
                    self.event_bus,
                    plugin_modules=cast(Any, after_step_modules),
                ),
            ),
            (
                "after_reasoning",
                default_after_reasoning_modules(
                    self.event_bus,
                    pipeline._session,
                    plugin_modules=cast(Any, after_reasoning_modules),
                ),
            ),
            (
                "after_turn",
                default_after_turn_modules(
                    self.event_bus,
                    pipeline._outbound_port,
                    context,
                    pipeline._history_window,
                    plugin_modules=cast(Any, after_turn_modules),
                ),
            ),
        ]

        # 4. 统一渲染执行顺序和依赖树。
        parts: list[str] = []
        for phase_name, modules in phases:
            parts.append("=" * 60)
            parts.append(phase_name)
            parts.append("=" * 60)
            parts.append(inspect_phase(modules))
        return "\n".join(parts)

    async def stop(self) -> None:
        """按所有权逆序关闭核心运行时资源。"""

        # 1. 将动态 spawn 工具和同步 session close 适配为异步清理步骤。
        async def _stop_spawn() -> None:
            spawn_tool = self.tools.get_tool("spawn")
            shutdown = getattr(spawn_tool, "shutdown", None)
            if callable(shutdown):
                result = shutdown()
                if inspect.isawaitable(result):
                    await cast(Awaitable[object], result)

        async def _close_session_manager() -> None:
            self.session_manager.close()

        async def _stop_workspace_mcp_watcher() -> None:
            self.workspace_mcp_watcher.stop()
            task = self.workspace_mcp_watcher_task
            if task is not None:
                _ = task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if self.plugin_manager is not None:
                await self.plugin_manager.discard_workspace_mcp_candidate()

        # 2. 由统一 cleanup runner 完成全部步骤并保留失败。
        await run_cleanup_steps(
            ("workspace_mcp_watcher.stop", _stop_workspace_mcp_watcher),
            ("spawn.shutdown", _stop_spawn),
            ("event_bus.aclose", self.event_bus.aclose),
            (
                "plugin_manager.terminate_all",
                self.plugin_manager.terminate_all
                if self.plugin_manager is not None
                else _noop_async,
            ),
            (
                "peer_poller.stop",
                self.peer_poller.stop if self.peer_poller is not None else _noop_async,
            ),
            (
                "peer_process_manager.shutdown_all",
                self.peer_process_manager.shutdown_all
                if self.peer_process_manager is not None
                else _noop_async,
            ),
            ("session_manager.close", _close_session_manager),
        )


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
    MemoryRuntime,
    PeerProcessManager | None,
    PeerAgentPoller | None,
]:
    """按配置顺序构造并注册核心工具资源。"""

    from session.store import SessionStore

    # 1. 构造共享服务；外部传入的 session_store 和 http_resources 不转移 ownership。
    wiring = config.wiring
    tools = tools if tools is not None else ToolRegistry()
    multimodal = config.multimodal
    vl_available = not multimodal and config.vl_model != ""
    readonly_tools = build_readonly_tools(
        http_resources, multimodal=multimodal, vl_available=vl_available
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
    scheduler = build_scheduler(
        workspace,
        push_tool,
        agent_loop_provider=agent_loop_provider,
    )
    peer_process_manager, peer_poller = build_peer_agent_resources(
        config, bus, http_resources
    )

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
                memory_engine=memory_runtime.engine,
                scheduler=scheduler,
                event_publisher=event_publisher,
            ),
        )

    return (
        tools,
        push_tool,
        scheduler,
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

    # 2. 绑定 memory/session service 与 retrieval pipeline。
    memory_engine = memory_runtime.engine
    light = light_provider or provider
    llm_services = LLMServices(provider=provider, light_provider=light)
    memory_services = MemoryServices(engine=memory_engine)
    session_services = SessionServices(
        session_manager=session_manager, presence=presence
    )
    _bind_memory_lifecycle_if_supported(
        markdown=memory_runtime.markdown.maintenance,
        session_manager=session_manager,
    )
    retrieval_pipeline = DefaultMemoryRetrievalPipeline(
        memory=memory_services,
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
        retrieval_pipeline=retrieval_pipeline,
        context=context,
        llm_services=llm_services,
        memory_services=memory_services,
        session_services=session_services,
    )


def _bind_memory_lifecycle_if_supported(
    *,
    markdown: MarkdownMemoryMaintenance,
    session_manager: SessionManager,
) -> None:
    async def _save_session(session: object) -> None:
        await session_manager.save_async(cast(Session, session))

    markdown.bind_lifecycle(
        MemoryLifecycleBindRequest(
            get_session=session_manager.get_or_create,
            save_session=_save_session,
        )
    )


def build_core_runtime(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
) -> CoreRuntime:
    """构造核心运行时及其插件快照依赖。"""

    # 1. 创建总线、provider 和由 CoreRuntime.stop 负责关闭的 session owner。
    bus = MessageBus()
    event_bus = EventBus()
    provider, light_provider, agent_provider = build_providers(config)
    vl_provider = build_vl_provider(config)
    # 2. agent_provider 供 AgentLoop 使用，provider 供 consolidation 事件提取使用。
    loop_provider = agent_provider or provider
    loop_model = config.agent_model or config.model
    session_manager = SessionManager(workspace)
    loop_ref: dict[str, AgentLoop] = {}
    tools, push_tool, scheduler, memory_runtime, peer_pm, peer_poller = (
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
                multimodal=config.multimodal,
                vl_available=config.vl_model != "",
            ),
            memory=MemoryConfig(
                window=config.memory_window,
            ),
        ),
    )
    loop_ref["loop"] = loop
    wire_turn_lifecycle(
        lifecycle=TurnLifecycle(event_bus),
        active_turn_states=loop.active_turn_states,
    )

    from agent.plugins.manager import PluginManager as _PluginManager

    # 3. 创建插件 manager，并把 snapshot store 绑定到 loop。
    plugin_manager = _PluginManager(
        plugin_dirs=_resolve_plugin_dirs(workspace),
        event_bus=event_bus,
        tool_registry=tools,
        workspace=workspace,
        session_manager=session_manager,
        memory_engine=memory_runtime.engine,
        llm=ProviderPluginLlmService(
            provider=provider,
            model=config.model,
            max_tokens=config.max_tokens,
        ),
        installed_cache_root=_resolve_installed_plugin_cache_root(),
    )
    loop.bind_runtime_snapshot_store(plugin_manager.snapshot_store)
    workspace_mcp_watcher = WorkspaceMcpWatcher(
        plugin_manager,
        workspace / "mcp" / "servers",
        mcp_root=workspace / "mcp",
    )

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
        scheduler=scheduler,
        provider=provider,
        light_provider=light_provider,
        agent_provider=agent_provider,
        workspace_mcp_watcher=workspace_mcp_watcher,
        workspace_mcp_watcher_task=None,
        memory_runtime=memory_runtime,
        presence=presence,
        peer_process_manager=peer_pm,
        peer_poller=peer_poller,
        plugin_manager=plugin_manager,
    )


def _resolve_plugin_dirs(workspace: Path) -> list[Path]:
    project_root = Path(__file__).resolve().parent.parent
    roots = [project_root / "plugins"]
    extra = os.environ.get("AKASHIC_EXTRA_PLUGIN_DIRS", "")
    roots.extend(
        Path(item).expanduser()
        for item in extra.split(os.pathsep)
        if item.strip()
    )
    return roots


def _resolve_installed_plugin_cache_root() -> Path:
    return Path.home() / ".akashic-plugin" / "cache"
