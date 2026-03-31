from __future__ import annotations

from pathlib import Path

from agent.background.subagent_manager import SubagentManager
from agent.config_models import Config
from agent.policies.delegation import DelegationPolicy
from agent.tool_bundles import build_readonly_research_tools
from agent.tools.meta import register_common_meta_tools
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.spawn import SpawnTool
from bus.queue import MessageBus
from core.memory.port import MemoryPort
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
    push_tool: MessagePushTool | None = None,
) -> MessagePushTool:
    return register_common_meta_tools(
        tools,
        readonly_tools,
        session_store,
        push_tool=push_tool,
    )


def register_spawn_tool(
    tools: ToolRegistry,
    config: Config,
    workspace: Path,
    bus: MessageBus,
    provider,
    http_resources: SharedHttpResources,
    memory_port: MemoryPort | None = None,
) -> SubagentManager:
    subagent_manager = SubagentManager(
        provider=provider,
        workspace=workspace,
        bus=bus,
        model=config.model,
        max_tokens=config.max_tokens,
        fetch_requester=http_resources.external_default,
        memory=memory_port,
    )
    if config.spawn_enabled:
        # 暂时注释掉 spawn 工具注册，仅保留 subagent_manager 初始化，避免影响其他依赖链路。
        # tools.register(
        #     SpawnTool(subagent_manager, tools, policy=DelegationPolicy()),
        #     always_on=True,
        #     risk="write",
        #     search_hint="后台执行 长任务 异步",
        # )
        pass
    return subagent_manager
