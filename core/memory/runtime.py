from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.memory.engine import MemoryEngine

logger = logging.getLogger(__name__)


@dataclass
class MemoryRuntime:
    engine: "MemoryEngine"
    closeables: list[Any] = field(default_factory=list)

    # TODO(memory-engine-cleanup): 旧调用方完成迁移后删除这些 MemoryRuntime 兼容属性。
    @property
    def port(self) -> "MemoryEngine":
        return self.engine

    @property
    def facade(self) -> "MemoryEngine":
        return self.engine

    @property
    def profile_reader(self) -> "MemoryEngine":
        return self.engine

    @property
    def profile_maint(self) -> "MemoryEngine":
        return self.engine

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
