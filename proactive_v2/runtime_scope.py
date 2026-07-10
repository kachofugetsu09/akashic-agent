from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent.plugins.specs import RegisteredProactiveSource
from agent.tool_hooks import ToolHook
from agent.tools.registry import ToolRegistry
from agent.turns.orchestrator import TurnOrchestrator
from bus.event_bus import EventBus
from proactive_v2.mcp_sources import McpGateway


@dataclass
class ProactiveRuntimeScope:
    cfg: Any
    sense: Any
    presence: Any | None
    provider: Any
    model: str
    max_tokens: int
    memory: Any | None
    state_store: Any
    any_action_gate: Any | None
    passive_busy_fn: Any | None
    deduper: Any | None
    rng: Any
    workspace_context_fn: Callable[[], str]
    mcp_gateway: McpGateway
    proactive_sources: list[RegisteredProactiveSource] = field(default_factory=list)
    shared_tools: ToolRegistry | None = None
    turn_orchestrator: TurnOrchestrator | None = None
    event_bus: EventBus | None = None
    tool_hooks: list[ToolHook] = field(default_factory=list)
    schedule_fn: Callable[[float | None], int] | None = None
