from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from agent.loop import AgentLoop
from agent.mcp.manage_tools import McpAddTool, McpListTool, McpRemoveTool
from agent.mcp.registry import McpServerRegistry
from agent.scheduler import LatencyTracker, SchedulerService
from agent.tools.filesystem import ListDirTool, ReadFileTool
from agent.tools.fitbit import FitbitHealthSnapshotTool, FitbitSleepReportTool
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.schedule import CancelScheduleTool, ListSchedulesTool, ScheduleTool
from agent.tools.shell import ShellTool
from agent.tools.skill_action_tool import (
    SkillActionListTool,
    SkillActionRegisterTool,
    SkillActionResetTool,
    SkillActionRestartTool,
    SkillActionRewriteTool,
    SkillActionStatusTool,
    SkillActionUnregisterTool,
    SkillActionUpdateTool,
)
from agent.tools.update_now import UpdateNowTool
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from bootstrap.memory import build_memory_runtime
from bootstrap.providers import build_providers
from bus.processing import ProcessingState
from bus.queue import MessageBus
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources
from feeds.novel import NovelKBFeedSource
from feeds.registry import FeedRegistry
from feeds.rss import RSSFeedSource
from feeds.store import FeedStore
from feeds.tools import FeedManageTool, FeedQueryTool
from proactive.presence import PresenceStore
from session.manager import SessionManager


def build_core_runtime(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
) -> tuple:
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(ShellTool())
    tools.register(WebSearchTool())
    tools.register(WebFetchTool(http_resources.external_default))
    tools.register(ReadFileTool())
    tools.register(ListDirTool())
    push_tool = MessagePushTool()
    tools.register(push_tool)

    skill_actions_path = workspace / "skill_actions.json"
    agent_tasks_dir = workspace / "agent-tasks"
    db_path = agent_tasks_dir / "task_notes.db"
    tools.register(
        SkillActionRegisterTool(skill_actions_path, agent_tasks_dir=agent_tasks_dir)
    )
    tools.register(SkillActionUnregisterTool(skill_actions_path))
    tools.register(
        SkillActionListTool(skill_actions_path, agent_tasks_dir=agent_tasks_dir)
    )
    tools.register(SkillActionStatusTool(agent_tasks_dir))
    tools.register(SkillActionUpdateTool(agent_tasks_dir))
    tools.register(SkillActionRestartTool(agent_tasks_dir, db_path=db_path))
    tools.register(SkillActionResetTool(agent_tasks_dir))
    tools.register(SkillActionRewriteTool(agent_tasks_dir, db_path=db_path))

    fitbit_url = getattr(config.proactive, "fitbit_url", "http://127.0.0.1:18765")
    if getattr(config.proactive, "fitbit_enabled", False):
        tools.register(
            FitbitHealthSnapshotTool(
                fitbit_url,
                requester=http_resources.local_service,
            )
        )
        tools.register(
            FitbitSleepReportTool(
                fitbit_url,
                requester=http_resources.local_service,
            )
        )

    provider, light_provider = build_providers(config)
    memory_runtime: MemoryRuntime = build_memory_runtime(
        config,
        workspace,
        tools,
        provider,
        light_provider,
        http_resources,
    )
    tools.register(UpdateNowTool(memory_runtime.port))

    scheduler = SchedulerService(
        store_path=workspace / "schedules.json",
        push_tool=push_tool,
        agent_loop=None,
        tracker=LatencyTracker(),
    )

    session_manager = SessionManager(workspace)
    presence = PresenceStore(workspace / "presence.json")
    processing_state = ProcessingState()
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        tools=tools,
        session_manager=session_manager,
        workspace=workspace,
        model=config.model,
        max_iterations=config.max_iterations,
        max_tokens=config.max_tokens,
        presence=presence,
        light_model=config.light_model,
        light_provider=light_provider,
        processing_state=processing_state,
        memory_top_k_procedure=config.memory_v2.top_k_procedure,
        memory_top_k_history=config.memory_v2.top_k_history,
        memory_route_intention_enabled=config.memory_v2.route_intention_enabled,
        memory_sop_guard_enabled=config.memory_v2.sop_guard_enabled,
        memory_gate_llm_timeout_ms=config.memory_v2.gate_llm_timeout_ms,
        memory_gate_max_tokens=config.memory_v2.gate_max_tokens,
        memory_runtime=memory_runtime,
    )

    scheduler.agent_loop = loop
    tools.register(ScheduleTool(scheduler))
    tools.register(ListSchedulesTool(scheduler))
    tools.register(CancelScheduleTool(scheduler))

    mcp_registry = McpServerRegistry(
        config_path=workspace / "mcp_servers.json",
        tool_registry=tools,
    )
    tools.register(McpAddTool(mcp_registry))
    tools.register(McpRemoveTool(mcp_registry))
    tools.register(McpListTool(mcp_registry))

    return (
        loop,
        bus,
        tools,
        push_tool,
        session_manager,
        scheduler,
        provider,
        light_provider,
        mcp_registry,
        memory_runtime,
        presence,
    )


def build_feed_runtime(
    workspace: Path,
    tools: ToolRegistry,
    http_resources: SharedHttpResources,
) -> tuple[FeedRegistry, FeedStore, FeedManageTool]:
    feed_store = FeedStore(workspace / "feeds.json")
    feed_registry = FeedRegistry(feed_store)
    feed_registry.register_source_type(
        "rss",
        lambda sub: RSSFeedSource(sub, requester=http_resources.feed_fetcher),
    )
    feed_registry.register_source_type("novel-kb", lambda sub: NovelKBFeedSource(sub))

    feed_manage_tool = FeedManageTool(feed_store)
    tools.register(feed_manage_tool)
    tools.register(FeedQueryTool(feed_store, feed_registry))
    return feed_registry, feed_store, feed_manage_tool
