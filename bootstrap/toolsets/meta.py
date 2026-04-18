from __future__ import annotations

from pathlib import Path

from agent.background.subagent_manager import SubagentManager
from agent.config_models import Config
from agent.policies.delegation import DelegationPolicy
from agent.tool_bundles import build_readonly_research_tools
from agent.tools.base import Tool
from agent.tools.meta import register_common_meta_tools
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.spawn import SpawnTool
from bus.queue import MessageBus
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetProvider,
    build_registration_result,
)
from core.memory.port import MemoryPort
from core.net.http import SharedHttpResources


class CommonMetaToolsetProvider(ToolsetProvider):
    def __init__(self, readonly_tools: dict[str, Tool]) -> None:
        self._readonly_tools = readonly_tools

    def register(self, registry: ToolRegistry, deps: ToolsetDeps):
        before = set(registry._tools.keys())
        push_tool = register_common_meta_tools(
            registry,
            self._readonly_tools,
            deps.session_store,
            push_tool=deps.push_tool,
        )
        return build_registration_result(
            registry=registry,
            source_name="meta_common",
            before=before,
            extras={"push_tool": push_tool},
        )


class SpawnToolsetProvider(ToolsetProvider):
    def register(self, registry: ToolRegistry, deps: ToolsetDeps):
        before = set(registry._tools.keys())
        config = deps.config
        bus = deps.bus
        http_resources = deps.http_resources
        if config is None or bus is None or http_resources is None:
            raise ValueError("spawn toolset 缺少必要依赖")
        subagent_manager = SubagentManager(
            provider=deps.provider,
            workspace=deps.workspace,
            bus=bus,
            model=config.model,
            max_tokens=config.max_tokens,
            fetch_requester=http_resources.external_default,
        )
        if config.spawn_enabled:
            registry.register(
                SpawnTool(subagent_manager, registry, policy=DelegationPolicy()),
                always_on=True,
                risk="write",
                search_hint="后台执行 子任务 多步调研 独立任务",
            )
        return build_registration_result(
            registry=registry,
            source_name="spawn",
            before=before,
            extras={"subagent_manager": subagent_manager},
        )


def build_readonly_tools(http_resources: SharedHttpResources) -> dict[str, Tool]:
    return {
        tool.name: tool
        for tool in build_readonly_research_tools(
            fetch_requester=http_resources.external_default,
            include_list_dir=True,
        )
    }


def register_meta_and_common_tools(
    tools: ToolRegistry,
    readonly_tools: dict[str, Tool],
    session_store,
    push_tool: MessagePushTool | None = None,
) -> MessagePushTool:
    result = CommonMetaToolsetProvider(readonly_tools).register(
        tools,
        ToolsetDeps(
            config=None,
            workspace=Path("."),
            session_store=session_store,
            push_tool=push_tool,
        ),
    )
    return result.extras["push_tool"]


def register_spawn_tool(
    tools: ToolRegistry,
    config: Config,
    workspace: Path,
    bus: MessageBus,
    provider,
    http_resources: SharedHttpResources,
    memory_port: MemoryPort | None = None,
) -> SubagentManager:
    result = SpawnToolsetProvider().register(
        tools,
        ToolsetDeps(
            config=config,
            workspace=workspace,
            provider=provider,
            http_resources=http_resources,
            bus=bus,
            memory_port=memory_port,
        ),
    )
    return result.extras["subagent_manager"]
