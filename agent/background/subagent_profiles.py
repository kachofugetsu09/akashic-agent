from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from agent.provider import LLMProvider
from agent.subagent import SubAgent
from agent.tool_hooks.base import ToolHook
from agent.tool_bundles import build_readonly_research_tools
from agent.tools.base import Tool
from agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from agent.tools.shell import ShellTool
from core.net.http import HttpRequester

PROFILE_RESEARCH = "research"
PROFILE_SCRIPTING = "scripting"
PROFILE_GENERAL = "general"


@dataclass(frozen=True)
class SubagentRuntime:
    provider: LLMProvider
    model: str
    max_tokens: int
    tool_hooks: list[ToolHook] = field(default_factory=list)


@dataclass
class SubagentSpec:
    tools: list[Tool]
    system_prompt: str = ""
    max_iterations: int = 30
    mandatory_exit_tools: Sequence[str] = field(default_factory=tuple)

    def build(self, runtime: SubagentRuntime) -> SubAgent:
        agent = SubAgent(
            provider=runtime.provider,
            model=runtime.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
            max_tokens=runtime.max_tokens,
            mandatory_exit_tools=self.mandatory_exit_tools,
        )
        if runtime.tool_hooks:
            agent.add_tool_hooks(runtime.tool_hooks)
        return agent


def build_research_spec(
    *,
    workspace: Path,
    task_dir: Path,
    fetch_requester: HttpRequester,
    system_prompt: str,
    max_iterations: int = 20,
    multimodal: bool = True,
) -> SubagentSpec:
    """只读调研：搜索、读文件、抓网页；禁止执行命令和写文件。"""
    tools = build_readonly_research_tools(
        fetch_requester=fetch_requester,
        allowed_dir=workspace,
        include_list_dir=True,
        multimodal=multimodal,
    )
    return SubagentSpec(
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )


def build_scripting_spec(
    *,
    workspace: Path,
    task_dir: Path,
    fetch_requester: HttpRequester,
    system_prompt: str,
    max_iterations: int = 20,
    multimodal: bool = True,
) -> SubagentSpec:
    """执行型：运行命令、读写文件（仅限 task_dir）；禁止网络访问。"""
    tools: list[Tool] = [
        ReadFileTool(allowed_dir=workspace, multimodal=multimodal),
        ListDirTool(allowed_dir=workspace),
        WriteFileTool(allowed_dir=task_dir),
        EditFileTool(allowed_dir=task_dir),
        ShellTool(
            allow_network=False,
            working_dir=task_dir,
        ),
    ]
    return SubagentSpec(
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )


def build_general_spec(
    *,
    workspace: Path,
    task_dir: Path,
    fetch_requester: HttpRequester,
    system_prompt: str,
    max_iterations: int = 20,
    multimodal: bool = True,
) -> SubagentSpec:
    """通用型：调研与执行兼有；仅在任务明确需要两者时使用。"""
    tools = build_readonly_research_tools(
        fetch_requester=fetch_requester,
        allowed_dir=workspace,
        include_list_dir=True,
        multimodal=multimodal,
    ) + [
        WriteFileTool(allowed_dir=task_dir),
        EditFileTool(allowed_dir=task_dir),
        ShellTool(
            allow_network=True,
            working_dir=task_dir,
        ),
    ]
    return SubagentSpec(
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )


_PROFILE_BUILDERS = {
    PROFILE_RESEARCH: build_research_spec,
    PROFILE_SCRIPTING: build_scripting_spec,
    PROFILE_GENERAL: build_general_spec,
}


def build_spawn_spec(
    *,
    workspace: Path,
    task_dir: Path,
    fetch_requester: HttpRequester,
    system_prompt: str,
    max_iterations: int = 20,
    profile: str = PROFILE_RESEARCH,
    multimodal: bool = True,
) -> SubagentSpec:
    """根据 profile 选择对应的工具集构建 SubagentSpec。

    profile:
        research  — 只读调研（默认，最小权限）
        scripting — 执行型，可运行命令和写文件，禁止网络
        general   — 两者兼有，仅在明确需要时使用
    """
    builder = _PROFILE_BUILDERS.get(profile, build_research_spec)
    return builder(
        workspace=workspace,
        task_dir=task_dir,
        fetch_requester=fetch_requester,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        multimodal=multimodal,
    )
