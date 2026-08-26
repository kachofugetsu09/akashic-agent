from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from core.memory.markdown import MarkdownMemoryRuntime

logger = logging.getLogger(__name__)


class _AsyncCloseable(Protocol):
    def aclose(self) -> object: ...


class _Closeable(Protocol):
    def close(self) -> object: ...


@dataclass
class MemoryRuntime:
    """Own the privileged Markdown runtime until its plugin migration."""

    markdown: "MarkdownMemoryRuntime"
    closeables: list[object] = field(default_factory=list[object])

    def read_long_term(self) -> str:
        return self.markdown.store.read_long_term()

    def read_self(self) -> str:
        return self.markdown.store.read_self()

    def get_memory_context(self) -> str:
        return self.markdown.store.get_memory_context()

    def has_long_term_memory(self) -> bool:
        return bool(self.read_long_term().strip())

    async def aclose(self) -> None:
        first_error: BaseException | None = None
        for closeable in reversed(self.closeables):
            try:
                if hasattr(closeable, "aclose"):
                    result = cast(_AsyncCloseable, closeable).aclose()
                    if inspect.isawaitable(result):
                        await result
                elif hasattr(closeable, "close"):
                    _ = cast(_Closeable, closeable).close()
            except asyncio.CancelledError as exc:
                if first_error is None:
                    first_error = exc
                logger.warning(
                    "memory runtime close cancelled for %s: %s",
                    type(closeable).__name__,
                    exc,
                )
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
