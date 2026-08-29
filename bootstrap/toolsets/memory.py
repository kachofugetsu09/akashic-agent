from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agent.config_models import Config
from agent.plugins.snapshot import RuntimeSnapshotStore
from agent.tools.registry import ToolRegistry
from bootstrap.memory import build_memory_runtime
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetProvider,
    ToolsetRegistrationResult,
    build_registration_result,
)
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources

if TYPE_CHECKING:
    from bus.event_bus import EventBus


class MemoryToolsetProvider(ToolsetProvider):
    def register(
        self,
        registry: ToolRegistry,
        deps: ToolsetDeps,
    ) -> ToolsetRegistrationResult:
        """Build the retained privileged Markdown runtime."""

        # 1. Validate the bootstrap boundary.
        before = registry.get_registered_names()
        if deps.config is None:
            raise ValueError("memory toolset 缺少必要依赖: config")
        if deps.runtime_snapshot_store is None:
            raise ValueError("memory toolset 缺少必要依赖: runtime_snapshot_store")
        if deps.http_resources is None:
            raise ValueError("memory toolset 缺少必要依赖: http_resources")

        # 2. Build only Markdown ownership; embedded memory belongs to plugins.
        runtime = build_memory_runtime(
            deps.config,
            deps.workspace,
            registry,
            deps.runtime_snapshot_store,
            deps.http_resources,
            event_publisher=deps.event_publisher,
        )
        return build_registration_result(
            registry=registry,
            source_name="markdown_memory",
            before=before,
            extras={"memory_runtime": runtime},
        )


def build_memory_toolset(
    config: Config,
    workspace: Path,
    tools: ToolRegistry,
    runtime_snapshot_store: RuntimeSnapshotStore,
    http_resources: SharedHttpResources,
    *,
    event_publisher: "EventBus | None" = None,
) -> MemoryRuntime:
    result = MemoryToolsetProvider().register(
        tools,
        ToolsetDeps(
            config=config,
            workspace=workspace,
            runtime_snapshot_store=runtime_snapshot_store,
            http_resources=http_resources,
            event_publisher=event_publisher,
        ),
    )
    return result.extras["memory_runtime"]
