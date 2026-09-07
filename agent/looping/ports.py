from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from agent.plugin_composition.channels import AttachmentRef
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
