from __future__ import annotations

from pathlib import Path

from agent.background.subagent_manager import SubagentManager
from agent.config_models import Config
from agent.policies.delegation import DelegationPolicy
from agent.tool_bundles import build_readonly_research_tools
from agent.tools.list_tools import ListToolsTool
from agent.tools.message_lookup import FetchMessagesTool, SearchMessagesTool
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.shell import ShellTool
from agent.tools.spawn import SpawnTool
from agent.tools.tool_search import ToolSearchTool
from bus.queue import MessageBus
from core.net.http import SharedHttpResources


def build_readonly_tools(http_resources: SharedHttpResources) -> dict[str, object]:
    return {
        tool.name: tool
        for tool in build_readonly_research_tools(
            fetch_requester=http_resources.external_default,
            include_list_dir=True,
        )
    }


def register_meta_and_common_tools(
    tools: ToolRegistry,
    readonly_tools: dict[str, object],
    session_store,
) -> MessagePushTool:
    tools.register(ToolSearchTool(tools), always_on=True, tags=["meta"], risk="read-only")
    tools.register(ListToolsTool(tools), always_on=True, tags=["meta"], risk="read-only")
    tools.register(
        ShellTool(),
        always_on=True,
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
    tools.register(
        FetchMessagesTool(session_store),
        always_on=True,
        tags=["memory", "session"],
        risk="read-only",
        search_keywords=["消息回溯", "按ID查消息", "fetch messages", "source_ref"],
    )
    tools.register(
        SearchMessagesTool(session_store),
        always_on=True,
        tags=["memory", "session"],
        risk="read-only",
        search_keywords=["搜索历史消息", "全文检索消息", "search messages", "原始对话", "你之前说", "聊过什么", "具体内容"],
    )
    push_tool = MessagePushTool()
    tools.register(
        push_tool,
        tags=["message"],
        risk="external-side-effect",
        search_keywords=["推送消息", "发送消息", "通知用户", "给用户发消息", "push"],
    )
    return push_tool


def register_spawn_tool(
    tools: ToolRegistry,
    config: Config,
    workspace: Path,
    bus: MessageBus,
    provider,
    http_resources: SharedHttpResources,
) -> SubagentManager:
    subagent_manager = SubagentManager(
        provider=provider,
        workspace=workspace,
        bus=bus,
        model=config.model,
        max_tokens=config.max_tokens,
        fetch_requester=http_resources.external_default,
    )
    if config.spawn_enabled:
        # 暂时注释掉 spawn 工具注册，仅保留 subagent_manager 初始化，避免影响其他依赖链路。
        # tools.register(
        #     SpawnTool(subagent_manager, tools, policy=DelegationPolicy()),
        #     always_on=True,
        #     tags=["meta", "background"],
        #     risk="write",
        #     search_keywords=["后台", "长任务", "异步", "继续处理", "spawn", "阻塞", "后台执行"],
        # )
        pass
    return subagent_manager
