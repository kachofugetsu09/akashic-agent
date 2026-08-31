from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.passive_turn import Reasoner
    from agent.core.runtime_support import ToolDiscoveryState
    from agent.turns.outbound import OutboundPort
    from agent.plugin_composition.channels import AttachmentRef
    from agent.tools.registry import ToolRegistry
    from bus.event_bus import EventBus
    from bus.processing import ProcessingState
    from bus.queue import MessageBus
    from session.activity import PresenceStore
    from session.manager import SessionManager


# ── Config dataclasses（参数，不含服务对象）───────────────────────────────────


@dataclass
class LLMConfig:
    max_iterations: int = 10
    max_tokens: int = 0
    tool_search_enabled: bool = False


class OutboundAttachmentImporter(Protocol):
    async def import_media(
        self,
        media: tuple[str, ...],
    ) -> tuple["AttachmentRef", ...]: ...


@dataclass
class SessionServices:
    session_manager: SessionManager
    presence: PresenceStore | None = None
    outbound_attachment_importer: OutboundAttachmentImporter | None = None


@dataclass
class AgentLoopDeps:
    bus: "MessageBus"
    tools: "ToolRegistry"
    session_manager: "SessionManager"
    workspace: Path
    event_bus: "EventBus | None" = None
    presence: "PresenceStore | None" = None
    processing_state: "ProcessingState | None" = None
    context: "ContextBuilder | None" = None
    session_services: SessionServices | None = None
    tool_discovery: "ToolDiscoveryState | None" = None
    reasoner: "Reasoner | None" = None
    outbound_port: "OutboundPort | None" = None


@dataclass
class AgentLoopConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
