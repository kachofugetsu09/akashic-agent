from __future__ import annotations

from pathlib import Path

from agent.background.subagent_manager import SubagentManager
from agent.config_models import Config
from agent.policies.delegation import DelegationPolicy
from agent.looping.core import AgentLoop
from agent.mcp.manage_tools import McpAddTool, McpListTool, McpRemoveTool
from agent.mcp.registry import McpServerRegistry
from agent.scheduler import LatencyTracker, SchedulerService
from agent.tool_bundles import build_fitbit_tools, build_readonly_research_tools
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.schedule import CancelScheduleTool, ListSchedulesTool, ScheduleTool
from agent.tools.shell import ShellTool
from agent.tools.spawn import SpawnTool
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
from agent.tools.list_tools import ListToolsTool
from agent.tools.tool_search import ToolSearchTool
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
    readonly_tools = {
        tool.name: tool
        for tool in build_readonly_research_tools(
            fetch_requester=http_resources.external_default,
            include_list_dir=True,
        )
    }

    # 元工具：始终可见，不暴露在搜索结果里
    tools.register(
        ToolSearchTool(tools),
        always_on=True,
        tags=["meta"],
        risk="read-only",
    )
    tools.register(
        ListToolsTool(tools),
        always_on=True,
        tags=["meta"],
        risk="read-only",
    )
    tools.register(
        ShellTool(),
        tags=["system"],
        risk="external-side-effect",
        search_keywords=["终端", "命令", "bash", "运行命令", "执行脚本", "shell"],
    )
    tools.register(
        readonly_tools["web_search"],
        always_on=True,
        tags=["web"],
        risk="read-only",
        search_keywords=["搜索", "网络搜索", "谷歌", "bing", "查资料"],
    )
    tools.register(
        readonly_tools["web_fetch"],
        always_on=True,
        tags=["web"],
        risk="read-only",
        search_keywords=["网页", "抓取网页", "读取网址", "fetch", "浏览网页"],
    )
    tools.register(
        readonly_tools["read_file"],
        always_on=True,
        tags=["filesystem"],
        risk="read-only",
        search_keywords=["读文件", "查看文件", "文件内容", "read"],
    )
    tools.register(
        readonly_tools["list_dir"],
        always_on=True,
        tags=["filesystem"],
        risk="read-only",
        search_keywords=["查看目录", "列出文件", "ls", "目录内容", "浏览目录", "dir"],
    )
    push_tool = MessagePushTool()
    tools.register(
        push_tool,
        tags=["message"],
        risk="external-side-effect",
        search_keywords=["推送消息", "发送消息", "通知用户", "给用户发消息", "push"],
    )

    skill_actions_path = workspace / "skill_actions.json"
    agent_tasks_dir = workspace / "agent-tasks"
    db_path = agent_tasks_dir / "task_notes.db"
    tools.register(
        SkillActionRegisterTool(skill_actions_path, agent_tasks_dir=agent_tasks_dir),
        tags=["skill", "task"],
        risk="write",
        search_keywords=["注册技能", "创建技能", "添加skill", "新建技能"],
    )
    tools.register(
        SkillActionUnregisterTool(skill_actions_path),
        tags=["skill", "task"],
        risk="write",
        search_keywords=["删除技能", "注销技能", "移除skill"],
    )
    tools.register(
        SkillActionListTool(skill_actions_path, agent_tasks_dir=agent_tasks_dir),
        tags=["skill", "task"],
        risk="read-only",
        search_keywords=["技能列表", "查看技能", "skill列表", "有哪些技能"],
    )
    tools.register(
        SkillActionStatusTool(agent_tasks_dir),
        tags=["skill", "task"],
        risk="read-only",
        search_keywords=["技能状态", "任务进度", "skill状态", "任务运行情况"],
    )
    tools.register(
        SkillActionUpdateTool(agent_tasks_dir),
        tags=["skill", "task"],
        risk="write",
        search_keywords=["更新技能", "修改技能", "skill更新"],
    )
    tools.register(
        SkillActionRestartTool(agent_tasks_dir, db_path=db_path),
        tags=["skill", "task"],
        risk="write",
        search_keywords=["重启技能", "重新运行skill", "skill重启"],
    )
    tools.register(
        SkillActionResetTool(agent_tasks_dir),
        tags=["skill", "task"],
        risk="write",
        search_keywords=["重置技能", "清空技能状态", "skill重置"],
    )
    tools.register(
        SkillActionRewriteTool(agent_tasks_dir, db_path=db_path),
        tags=["skill", "task"],
        risk="write",
        search_keywords=["重写技能", "重构skill", "skill重写"],
    )

    fitbit_url = getattr(config.proactive, "fitbit_url", "http://127.0.0.1:18765")
    if getattr(config.proactive, "fitbit_enabled", False):
        fitbit_tools = {
            tool.name: tool
            for tool in build_fitbit_tools(
                fitbit_url=fitbit_url,
                requester=http_resources.local_service,
            )
        }
        tools.register(
            fitbit_tools["fitbit_health_snapshot"],
            tags=["health", "fitbit"],
            risk="read-only",
            search_keywords=[
                "健康数据",
                "运动数据",
                "fitbit",
                "心率",
                "步数",
                "卡路里",
            ],
        )
        tools.register(
            fitbit_tools["fitbit_sleep_report"],
            tags=["health", "fitbit"],
            risk="read-only",
            search_keywords=["睡眠报告", "睡眠数据", "睡眠质量", "fitbit", "sleep"],
        )

    provider, light_provider = build_providers(config)
    subagent_manager = SubagentManager(
        provider=provider,
        workspace=workspace,
        bus=bus,
        model=config.model,
        max_tokens=config.max_tokens,
        fetch_requester=http_resources.external_default,
    )
    if config.spawn_enabled:
        tools.register(
            SpawnTool(subagent_manager, tools, policy=DelegationPolicy()),
            always_on=True,
            tags=["meta", "background"],
            risk="write",
            search_keywords=[
                "后台",
                "长任务",
                "异步",
                "继续处理",
                "spawn",
                "阻塞",
                "后台执行",
            ],
        )
    memory_runtime: MemoryRuntime = build_memory_runtime(
        config,
        workspace,
        tools,
        provider,
        light_provider,
        http_resources,
    )
    subagent_manager.set_memory_port(memory_runtime.port)
    tools.register(
        UpdateNowTool(memory_runtime.port),
        tags=["memory"],
        risk="write",
        search_keywords=["更新记忆", "同步记忆", "刷新知识库", "memory更新"],
    )

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
        tool_search_enabled=config.tool_search_enabled,
        memory_hyde_enabled=config.memory_v2.hyde_enabled,
        memory_hyde_timeout_ms=config.memory_v2.hyde_timeout_ms,
    )

    scheduler.agent_loop = loop
    tools.register(
        ScheduleTool(scheduler),
        tags=["scheduling"],
        risk="write",
        search_keywords=[
            "定时任务",
            "设置提醒",
            "计划任务",
            "cron",
            "延时执行",
            "timer",
        ],
    )
    tools.register(
        ListSchedulesTool(scheduler),
        tags=["scheduling"],
        risk="read-only",
        search_keywords=["查看定时任务", "定时列表", "提醒列表", "有哪些计划"],
    )
    tools.register(
        CancelScheduleTool(scheduler),
        tags=["scheduling"],
        risk="write",
        search_keywords=["取消定时", "删除提醒", "取消任务", "cancel schedule"],
    )

    mcp_registry = McpServerRegistry(
        config_path=workspace / "mcp_servers.json",
        tool_registry=tools,
    )
    tools.register(
        McpAddTool(mcp_registry),
        tags=["mcp", "system"],
        risk="external-side-effect",
        search_keywords=["添加MCP", "连接MCP", "注册MCP服务器", "mcp add"],
    )
    tools.register(
        McpRemoveTool(mcp_registry),
        tags=["mcp", "system"],
        risk="write",
        search_keywords=["删除MCP", "移除MCP服务器", "mcp remove"],
    )
    tools.register(
        McpListTool(mcp_registry),
        tags=["mcp", "system"],
        risk="read-only",
        search_keywords=["MCP列表", "查看MCP服务器", "mcp list"],
    )

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
    tools.register(
        feed_manage_tool,
        tags=["feed"],
        risk="write",
        search_keywords=["RSS订阅", "订阅管理", "添加订阅", "删除订阅", "feed管理"],
    )
    tools.register(
        FeedQueryTool(feed_store, feed_registry),
        tags=["feed"],
        risk="read-only",
        search_keywords=["查询订阅", "读取订阅", "RSS内容", "feed查询", "订阅内容"],
    )
    return feed_registry, feed_store, feed_manage_tool
