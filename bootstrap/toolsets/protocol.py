from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent.config_models import Config
    from agent.tools.message_push import MessagePushTool
    from agent.tools.registry import ToolRegistry
    from bus.event_bus import EventBus
    from bus.queue import MessageBus
    from core.net.http import SharedHttpResources
    from agent.plugins.snapshot import RuntimeSnapshotStore
    from session.store import SessionStore


@dataclass
class ToolsetDeps:
    config: "Config | None"
    workspace: Path
    runtime_snapshot_store: "RuntimeSnapshotStore | None" = None
    http_resources: "SharedHttpResources | None" = None
    session_store: "SessionStore | None" = None
    push_tool: "MessagePushTool | None" = None
    bus: "MessageBus | None" = None
    event_publisher: "EventBus | None" = None


@dataclass
class ToolsetRegistrationResult:
    source_name: str
    tool_names: list[str] = field(default_factory=list[str])
    always_on_names: list[str] = field(default_factory=list[str])
    extras: dict[str, Any] = field(default_factory=dict[str, Any])


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
        extras={} if extras is None else dict(extras),
    )
