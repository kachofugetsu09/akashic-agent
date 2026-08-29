from __future__ import annotations

from pathlib import Path
from typing import Callable

from agent.skills import SkillsLoader
from agent.tool_bundles import build_readonly_research_tools
from agent.tools.base import Tool, ToolExecutionContext
from agent.tools.meta import register_common_meta_tools
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.skill_loader import LoadSkillTool
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetProvider,
    ToolsetRegistrationResult,
    build_registration_result,
)
from core.net.http import SharedHttpResources
from session.store import SessionStore


class CommonMetaToolsetProvider(ToolsetProvider):
    _REQUIRED_READONLY_TOOLS = frozenset(
        {"web_search", "web_fetch", "read_file", "list_dir"}
    )

    def __init__(self, readonly_tools: dict[str, Tool]) -> None:
        self._readonly_tools = readonly_tools

    def register(
        self,
        registry: ToolRegistry,
        deps: ToolsetDeps,
    ) -> ToolsetRegistrationResult:
        """校验 common meta 依赖并注册共享元工具。"""

        # 1. 在注册边界报告缺失的共享依赖。
        missing_readonly = sorted(
            self._REQUIRED_READONLY_TOOLS - self._readonly_tools.keys()
        )
        if missing_readonly:
            raise ValueError(
                "meta_common toolset 缺少只读工具: " + ", ".join(missing_readonly)
            )
        if deps.session_store is None:
            raise ValueError("meta_common toolset 缺少必要依赖: session_store")

        # 2. 注册历史查询、skill 和可选视觉工具。
        before = registry.get_registered_names()
        push_tool = register_common_meta_tools(
            registry,
            self._readonly_tools,
            deps.session_store,
            push_tool=deps.push_tool,
        )
        registry.register(
            LoadSkillTool(SkillsLoader(deps.workspace, runtime_catalog="normal")),
            always_on=True,
            risk="read-only",
            search_hint="技能 skill SKILL.md 使用能力 先 load_skill 不要 read_file 猜路径",
        )

        # 主模型不支持多模态时，注册视觉工具供模型调用
        from agent.tools.vision import ReadImageVisionTool

        registry.register(
            ReadImageVisionTool(),
            always_on=True,
            risk="read-only",
            search_hint="看图 识图 图片内容 视觉识别 VL",
        )

        return build_registration_result(
            registry=registry,
            source_name="meta_common",
            before=before,
            extras={"push_tool": push_tool},
        )


def build_readonly_tools(
    http_resources: SharedHttpResources,
    *,
    workspace: Path | None = None,
    context_provider: Callable[[], ToolExecutionContext | None] | None = None,
) -> dict[str, Tool]:
    _ = context_provider
    return {
        tool.name: tool
        for tool in build_readonly_research_tools(
            fetch_requester=http_resources.external_default,
            allowed_dir=workspace,
            include_list_dir=True,
        )
    }


def register_meta_and_common_tools(
    tools: ToolRegistry,
    readonly_tools: dict[str, Tool],
    session_store: SessionStore,
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
