from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent.config_models import Config
    from agent.scheduler import SchedulerService
    from agent.tools.message_push import MessagePushTool
    from agent.tools.registry import ToolRegistry
    from bus.queue import MessageBus
    from core.net.http import SharedHttpResources


@dataclass
class ToolsetDeps:
    config: "Config | None"
    workspace: Path
    provider: Any = None
    light_provider: Any = None
    http_resources: "SharedHttpResources | None" = None
    session_store: object | None = None
    push_tool: "MessagePushTool | None" = None
    bus: "MessageBus | None" = None
    scheduler: "SchedulerService | None" = None
    memory_port: object | None = None
    observe_writer: object | None = None


@dataclass
class ToolsetRegistrationResult:
    source_name: str
    tool_names: list[str] = field(default_factory=list)
    always_on_names: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolsetProvider(Protocol):
    def register(
        self,
        registry: "ToolRegistry",
        deps: ToolsetDeps,
    ) -> ToolsetRegistrationResult: ...


def build_registration_result(
    *,
    registry: "ToolRegistry",
    source_name: str,
    before: set[str],
    extras: dict[str, Any] | None = None,
) -> ToolsetRegistrationResult:
    tool_names = sorted(registry.get_registered_names() - before)
    always_on = sorted(set(tool_names) & registry.get_always_on_names())
    return ToolsetRegistrationResult(
        source_name=source_name,
        tool_names=tool_names,
        always_on_names=always_on,
        extras=dict(extras or {}),
    )
