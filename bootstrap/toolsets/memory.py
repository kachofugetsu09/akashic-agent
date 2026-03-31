from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from agent.tools.registry import ToolRegistry
from agent.tools.update_now import UpdateNowTool
from bootstrap.memory import build_memory_runtime
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources


def build_memory_toolset(
    config: Config,
    workspace: Path,
    tools: ToolRegistry,
    provider,
    light_provider,
    http_resources: SharedHttpResources,
    *,
    observe_writer=None,
) -> MemoryRuntime:
    memory_runtime = build_memory_runtime(
        config,
        workspace,
        tools,
        provider,
        light_provider,
        http_resources,
        observe_writer=observe_writer,
    )
    tools.register(
        UpdateNowTool(memory_runtime.port),
        risk="write",
    )
    return memory_runtime
