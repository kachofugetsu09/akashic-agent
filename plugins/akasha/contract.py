"""akasha 发布的只读查询与结算合同。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agent.plugin_composition import ServiceKey


class SemanticInterest(Protocol):
    def decision(self) -> bool | None: ...
    def status(self) -> str | None: ...
    async def score(
        self, texts: Sequence[str], *, cutoff: str
    ) -> tuple[float, ...]: ...


SEMANTIC_INTEREST = ServiceKey[SemanticInterest]("akasha.semantic-interest.v1")



from typing import Any

from agent.plugin_composition.model import ServiceKey

# Marker only: vector-backed providers declare a mutually exclusive role.
EMBEDDING_MEMORY_PLUGIN = ServiceKey[Any]("plugin.claim.embedding_memory")
