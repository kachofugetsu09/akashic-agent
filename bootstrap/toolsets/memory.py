from __future__ import annotations

from agent.tools.registry import ToolRegistry
from bootstrap.memory import build_memory_runtime
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetProvider,
    ToolsetRegistrationResult,
    build_registration_result,
)


class MemoryToolsetProvider(ToolsetProvider):
    def register(
        self,
        registry: ToolRegistry,
        deps: ToolsetDeps,
    ) -> ToolsetRegistrationResult:
        """Build the retained privileged Markdown runtime."""

        before = registry.get_registered_names()
        if deps.runtime_snapshot_store is None:
            raise ValueError("memory toolset 缺少必要依赖: runtime_snapshot_store")

        runtime = build_memory_runtime(
            deps.workspace,
            deps.runtime_snapshot_store,
            event_publisher=deps.event_publisher,
        )
        return build_registration_result(
            registry=registry,
            source_name="markdown_memory",
            before=before,
            extras={"memory_runtime": runtime},
        )
