from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from agent.provider import LLMProvider
from agent.subagent import SubAgent
from agent.tool_bundles import build_readonly_research_tools
from agent.tools.base import Tool
from agent.tools.filesystem import EditFileTool, WriteFileTool
from agent.tools.shell import ShellTool
from core.memory.port import MemoryPort
from core.net.http import HttpRequester


@dataclass(frozen=True)
class SubagentRuntime:
    provider: LLMProvider
    model: str
    max_tokens: int
    memory: MemoryPort | None = None


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
            memory=runtime.memory,
        )


def build_spawn_spec(
    *,
    workspace: Path,
    task_dir: Path,
    fetch_requester: HttpRequester,
    system_prompt: str,
    max_iterations: int = 20,
) -> SubagentSpec:
    tools = build_readonly_research_tools(
        fetch_requester=fetch_requester,
        allowed_dir=workspace,
        include_list_dir=True,
    ) + [
        WriteFileTool(allowed_dir=task_dir),
        EditFileTool(allowed_dir=task_dir),
        ShellTool(),
    ]
    return SubagentSpec(
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )
