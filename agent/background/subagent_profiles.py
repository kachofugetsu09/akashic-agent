from __future__ import annotations

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
    """构建可联网但禁止命令执行与写文件的只读调研配置。"""
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
    """构建可执行命令并仅向任务目录写入、但禁止联网的配置。"""
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
    """构建同时允许联网调研和任务目录写入的通用配置。"""
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
    """按 profile 选择权限边界并构建子任务配置。"""
    builder = _PROFILE_BUILDERS.get(profile)
    if builder is None:
        raise ValueError(f"未知 subagent profile: {profile!r}")
    return builder(
        workspace=workspace,
        task_dir=task_dir,
        fetch_requester=fetch_requester,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        multimodal=multimodal,
    )
