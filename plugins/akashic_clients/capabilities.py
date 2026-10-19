"""Independent composition keys consumed by the Akashic client plugin.

The client plugin names the capabilities it uses by their stable ServiceKey
identity.  It does not import another plugin's implementation.  The channel
host resolves these keys inside each request scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plugins.models.contract import MODEL_CATALOG
from plugins.ledger.contract import MESSAGE_CATALOG, SESSION_ADMIN
from plugins.gateway.contract import RpcMethod
from agent.plugin_composition.runtime_catalog import (
    RUNTIME_CATALOG as RUNTIME_CATALOG,
)
from plugins.mcp.contract import MCP_DETAIL, MCP_SERVERS
from plugins.ui.contract import (
    WEB_UI as WEB_UI,
)
from plugins.models.contract import (
    MODEL_SELECTION as MODEL_SELECTION,
    ModelSelection as ModelSelectionReader,
)
from plugins.reply.contract import (
    REPLY_STATUS as REPLY_STATUS,
    ReplyStatus as ReplyStatusReader,
)
from plugins.ui.contract import (
    MESSAGE_DISPLAY as MESSAGE_DISPLAY,
    PLUGIN_UI as PLUGIN_UI,
)

if TYPE_CHECKING:
    pass


# These identities belong to the providers that publish the capabilities.  A
# duplicate local ServiceKey is intentional: ServiceKey connects by its stable
# name, so this module remains independent from the provider plugin package.


INSPECTION_DOCUMENTS_LIST = RpcMethod.key("inspection/documents.list")
INSPECTION_DOCUMENTS_GET = RpcMethod.key("inspection/documents.get")
INSPECTION_JOBS_LIST = RpcMethod.key("inspection/jobs.list")
INSPECTION_JOBS_GET = RpcMethod.key("inspection/jobs.get")
INSPECTION_SKILLS_LIST = RpcMethod.key("inspection/skills.list")

INSPECTION_RPC_KEYS = (
    INSPECTION_DOCUMENTS_LIST,
    INSPECTION_DOCUMENTS_GET,
    INSPECTION_JOBS_LIST,
    INSPECTION_JOBS_GET,
    INSPECTION_SKILLS_LIST,
)

# 聊天启动只等待基础能力；诊断 RPC 在请求中借用，未接线的旧模型 RPC 不阻塞启动。
CLIENT_CAPABILITIES = (
    RUNTIME_CATALOG,
    MESSAGE_CATALOG,
    MESSAGE_DISPLAY,
    PLUGIN_UI,
    WEB_UI,
    MODEL_CATALOG,
    MODEL_SELECTION,
    SESSION_ADMIN,
)


__all__ = [
    "CLIENT_CAPABILITIES",
    "INSPECTION_DOCUMENTS_GET",
    "INSPECTION_DOCUMENTS_LIST",
    "INSPECTION_JOBS_GET",
    "INSPECTION_JOBS_LIST",
    "INSPECTION_RPC_KEYS",
    "INSPECTION_SKILLS_LIST",
    "MODEL_CATALOG",
    "MODEL_SELECTION",
    "MESSAGE_DISPLAY",
    "PLUGIN_UI",
    "SESSION_ADMIN",
    "WEB_UI",
    "ReplyStatusReader",
    "ModelSelectionReader",
    "REPLY_STATUS",
    "RUNTIME_CATALOG",
    "MCP_DETAIL",
    "MCP_SERVERS",
]
