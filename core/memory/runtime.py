from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.memory.engine import (
        ExplicitRetrievalRequest,
        ExplicitRetrievalResult,
        InterestRetrievalRequest,
        InterestRetrievalResult,
        MemoryEngine,
        MemoryEngineRetrieveRequest,
        MemoryEngineRetrieveResult,
    )
    from core.memory.markdown import MarkdownMemoryRuntime

logger = logging.getLogger(__name__)


@dataclass
class MemoryRuntime:
    markdown: "MarkdownMemoryRuntime"
    engine: "MemoryEngine"
    closeables: list[Any] = field(default_factory=list)

    def read_long_term(self) -> str:
        return self.markdown.store.read_long_term()

    def read_self(self) -> str:
        return self.markdown.store.read_self()

    def read_recent_context(self) -> str:
        return self.markdown.store.read_recent_context()

    def read_recent_history(self, *, max_chars: int = 0) -> str:
        return self.markdown.store.read_recent_history(max_chars=max_chars)

    def get_memory_context(self) -> str:
        return self.markdown.store.get_memory_context()

    def has_long_term_memory(self) -> bool:
        return bool(self.read_long_term().strip())

    async def retrieve(
        self,
        request: "MemoryEngineRetrieveRequest",
    ) -> "MemoryEngineRetrieveResult":
        return await self.engine.retrieve(request)

    async def retrieve_explicit(
        self,
        request: "ExplicitRetrievalRequest",
    ) -> "ExplicitRetrievalResult":
        return await self.engine.retrieve_explicit(request)

    async def retrieve_interest_block(
        self,
        request: "InterestRetrievalRequest",
    ) -> "InterestRetrievalResult":
        return await self.engine.retrieve_interest_block(request)

    async def aclose(self) -> None:
        first_error: Exception | None = None
        for closeable in reversed(self.closeables):
            try:
                if hasattr(closeable, "aclose"):
                    result = closeable.aclose()
                    if inspect.isawaitable(result):
                        await result
                elif hasattr(closeable, "close"):
                    closeable.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logger.warning(
                    "memory runtime close failed for %s: %s",
                    type(closeable).__name__,
                    exc,
                )
        if first_error is not None:
            raise first_error
