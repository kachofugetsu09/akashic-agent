from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from agent.provider import LLMProvider
from agent.subagent import SubAgent
from agent.tool_bundles import build_readonly_research_tools
from agent.tools.base import Tool
from agent.tools.filesystem import EditFileTool, WriteFileTool
from agent.tools.message_push import MessagePushTool
from agent.tools.notify_owner import NotifyOwnerTool
from agent.tools.shell import ShellTool
from agent.tools.task_note import TaskDoneTool, TaskNoteTool, TaskRecallTool
from core.net.http import HttpRequester


@dataclass(frozen=True)
class SubagentRuntime:
    provider: LLMProvider
    model: str
    max_tokens: int


@dataclass
class SubagentSpec:
    tools: list[Tool]
    system_prompt: str = ""
    max_iterations: int = 30
    mandatory_exit_tools: Sequence[str] = field(default_factory=tuple)

    def build(self, runtime: SubagentRuntime) -> SubAgent:
        return SubAgent(
            provider=runtime.provider,
            model=runtime.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
            max_tokens=runtime.max_tokens,
            mandatory_exit_tools=self.mandatory_exit_tools,
        )


def build_spawn_spec(
    *,
    workspace: Path,
    fetch_requester: HttpRequester,
    system_prompt: str,
    max_iterations: int = 20,
) -> SubagentSpec:
    tools = build_readonly_research_tools(
        fetch_requester=fetch_requester,
        allowed_dir=workspace,
        include_list_dir=True,
    ) + [
        WriteFileTool(allowed_dir=workspace),
        EditFileTool(allowed_dir=workspace),
        ShellTool(),
    ]
    return SubagentSpec(
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )


def build_skill_action_spec(
    *,
    agent_tasks_dir: Path,
    action_dir: Path,
    fetch_requester: HttpRequester,
    shell_tool: Tool,
    db_path: Path,
    push_tool: MessagePushTool,
    channel: str,
    chat_id: str,
    system_prompt: str,
    max_iterations: int = 40,
) -> SubagentSpec:
    tools = build_readonly_research_tools(
        fetch_requester=fetch_requester,
        allowed_dir=agent_tasks_dir,
        include_list_dir=True,
    ) + [
        WriteFileTool(allowed_dir=agent_tasks_dir),
        shell_tool,
        TaskNoteTool(db_path),
        TaskRecallTool(db_path),
        TaskDoneTool(action_dir),
        NotifyOwnerTool(push_tool, channel, chat_id),
    ]
    return SubagentSpec(
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        mandatory_exit_tools=("task_note", "notify_owner"),
    )
