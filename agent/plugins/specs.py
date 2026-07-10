from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    command: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = "."


@dataclass(frozen=True)
class ProactiveSourceSpec:
    id: str
    channels: tuple[Literal["alert", "content", "context"], ...]
    server: str
    fetch_tool: str
    ack_tool: str = ""
    poll_tool: str = ""
    poll_interval_seconds: int = 0


@dataclass(frozen=True)
class RegisteredProactiveSource:
    plugin_id: str
    spec: ProactiveSourceSpec
