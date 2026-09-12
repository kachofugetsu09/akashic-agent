"""Independent composition keys consumed by the Akashic client plugin.

The client plugin names the capabilities it uses by their stable ServiceKey
identity.  It does not import another plugin's implementation.  The channel
host resolves these keys inside each request scope.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from agent.plugin_composition import MODEL_CALL_STATS, MODEL_CATALOG, ServiceKey
from agent.plugin_composition.commands import COMMANDS
from agent.plugin_composition.channels import AttachmentRef
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugin_composition.rpc import rpc_method_key
from agent.plugin_composition.message_view import MessageDisplayReader

from .services import MobileUiProvider, WebUiProvider


class ReplyStatusReader(Protocol):
    """Read-only reply activity for one session."""

    async def follow(self, session_id: str): ...


class ModelSelectionReader(Protocol):
    """Read one saved model selection from session metadata."""

    def read_saved(self, metadata: Mapping[str, object]) -> Any: ...


# These identities belong to the providers that publish the capabilities.  A
# duplicate local ServiceKey is intentional: ServiceKey connects by its stable
# name, so this module remains independent from the provider plugin package.
REPLY_STATUS = ServiceKey[ReplyStatusReader]("reply.status.v2")
MODEL_SELECTION = ServiceKey[ModelSelectionReader]("models.selection.v1")
MESSAGE_DISPLAY = ServiceKey[MessageDisplayReader]("core.message_display.v1")
ARTIFACT_RESOLVE = ServiceKey[
    Callable[[tuple[str, ...]], tuple[AttachmentRef, ...]]
]("core.artifact_resolve.v1")
MOBILE_UI = ServiceKey[MobileUiProvider]("core.mobile_ui.v1")
WEB_UI = ServiceKey[WebUiProvider]("core.web_ui.v1")

INSPECTION_DOCUMENTS_LIST = rpc_method_key("inspection/documents.list")
INSPECTION_DOCUMENTS_GET = rpc_method_key("inspection/documents.get")
INSPECTION_JOBS_LIST = rpc_method_key("inspection/jobs.list")
INSPECTION_JOBS_GET = rpc_method_key("inspection/jobs.get")
INSPECTION_SKILLS_LIST = rpc_method_key("inspection/skills.list")

MODEL_CALL = rpc_method_key("models/call_stats")
MODEL_CATALOG_RPC = rpc_method_key("models/catalog")
MODEL_DISCOVER = rpc_method_key("models/discover")
MODEL_COMMAND = rpc_method_key("models/command")

INSPECTION_RPC_KEYS = (
    INSPECTION_DOCUMENTS_LIST,
    INSPECTION_DOCUMENTS_GET,
    INSPECTION_JOBS_LIST,
    INSPECTION_JOBS_GET,
    INSPECTION_SKILLS_LIST,
)
MODEL_RPC_KEYS = (MODEL_CALL, MODEL_CATALOG_RPC, MODEL_DISCOVER, MODEL_COMMAND)

# The required dependency set is deliberately explicit and flat.  It is used
# by the manifest importer to activate the ordinary channel only when the
# exact providers are present.
CLIENT_CAPABILITIES = (
    MESSAGE_CATALOG,
    COMMANDS,
    MESSAGE_DISPLAY,
    ARTIFACT_RESOLVE,
    MOBILE_UI,
    WEB_UI,
    MODEL_CATALOG,
    MODEL_CALL_STATS,
    MODEL_SELECTION,
    REPLY_STATUS,
    *INSPECTION_RPC_KEYS,
    *MODEL_RPC_KEYS,
)


__all__ = [
    "CLIENT_CAPABILITIES",
    "ARTIFACT_RESOLVE",
    "INSPECTION_DOCUMENTS_GET",
    "INSPECTION_DOCUMENTS_LIST",
    "INSPECTION_JOBS_GET",
    "INSPECTION_JOBS_LIST",
    "INSPECTION_RPC_KEYS",
    "INSPECTION_SKILLS_LIST",
    "MODEL_CALL",
    "MODEL_CALL_STATS",
    "MODEL_CATALOG",
    "MODEL_CATALOG_RPC",
    "MODEL_COMMAND",
    "MODEL_DISCOVER",
    "MODEL_RPC_KEYS",
    "MODEL_SELECTION",
    "MESSAGE_DISPLAY",
    "MOBILE_UI",
    "WEB_UI",
    "ReplyStatusReader",
    "ModelSelectionReader",
    "REPLY_STATUS",
]
