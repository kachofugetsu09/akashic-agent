"""Wake 读取和结算普通来源的合同；来源继续拥有持久状态。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Protocol

from agent.plugin_composition import ServiceKey


class SemanticInterest(Protocol):
    def decision(self) -> bool | None: ...
    def status(self) -> str | None: ...
    async def score(
        self, texts: Sequence[str], *, cutoff: str
    ) -> tuple[float, ...]: ...


SEMANTIC_INTEREST = ServiceKey[SemanticInterest]("akasha.semantic-interest.v1")
