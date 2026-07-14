from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.mcp.host import PreparedMcpCatalog
from agent.plugins.scope import PluginScope

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshot


@dataclass
class WorkspaceMcpGeneration:
    generation_id: str
    revision: str
    scope: PluginScope
    catalog: PreparedMcpCatalog
    runtime_snapshot: RuntimeSnapshot | None = None
    state: str = "prepared"
    lease_count: int = 0
