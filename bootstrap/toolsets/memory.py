from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from agent.tools.registry import ToolRegistry
from agent.tools.update_now import UpdateNowTool
from bootstrap.memory import build_memory_runtime
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetProvider,
    build_registration_result,
)
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources


class MemoryToolsetProvider(ToolsetProvider):
    def register(
        self,
        registry: ToolRegistry,
        deps: ToolsetDeps,
    ):
        before = set(registry._tools.keys())
        config = deps.config
        http_resources = deps.http_resources
        if config is None or http_resources is None:
            raise ValueError("memory toolset 缺少必要依赖")
        memory_runtime = build_memory_runtime(
            config,
            deps.workspace,
            registry,
            deps.provider,
            deps.light_provider,
            http_resources,
            observe_writer=deps.observe_writer,
        )
        registry.register(
            UpdateNowTool(memory_runtime.profile_maint or memory_runtime.port),
            risk="write",
        )
        return build_registration_result(
            registry=registry,
            source_name="memory",
            before=before,
            extras={"memory_runtime": memory_runtime},
        )


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
    result = MemoryToolsetProvider().register(
        tools,
        ToolsetDeps(
            config=config,
            workspace=workspace,
            provider=provider,
            light_provider=light_provider,
            http_resources=http_resources,
            observe_writer=observe_writer,
        ),
    )
    return result.extras["memory_runtime"]
