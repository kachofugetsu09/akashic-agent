from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MobileUiSlot = Literal[
    "turn.before_reasoning",
    "turn.before_tool",
    "turn.after_answer",
    "drawer.panel",
]


@dataclass(frozen=True)
class MobileUiNavigation:
    label: str
    description: str


@dataclass(frozen=True)
class MobileUiContribution:
    module: str
    stylesheet: str | None = None
    navigation: MobileUiNavigation | None = None
    slots: tuple[MobileUiSlot, ...] = ()


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    command: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = "."


@dataclass(frozen=True)
class ManagedServiceSpec:
    id: str
    command: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = "."
    readiness_url: str = ""
    startup_timeout_seconds: float = 15


@dataclass(frozen=True)
class ProactiveSourceSpec:
    id: str
    channels: tuple[Literal["alert", "content", "context"], ...]
    server: str
    fetch_tool: str
    ack_tool: str = ""
    fetch_page_size: int = 0


@dataclass(frozen=True)
class RegisteredProactiveSource:
    plugin_id: str
    spec: ProactiveSourceSpec


def proactive_source_key(source: RegisteredProactiveSource) -> str:
    return f"{source.plugin_id}:{source.spec.id}"
