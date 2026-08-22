from __future__ import annotations

from agent.tools.registry import ToolRegistry
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetProvider,
    build_registration_result,
)


class SchedulerToolsetProvider(ToolsetProvider):
    """Retain the legacy wiring position while v3 owns all Scheduler Tools."""

    def register(self, registry: ToolRegistry, deps: ToolsetDeps):
        before = registry.get_registered_names()
        return build_registration_result(
            registry=registry,
            source_name="schedule",
            before=before,
            extras={},
        )
