from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from memory2.post_response_worker import PostResponseMemoryWorker

if TYPE_CHECKING:
    from core.memory.engine import MemoryEngine
    from core.memory.engine import PassiveMemoryEngine
    from core.memory.port import MemoryPort
    from core.memory.profile import ProfileMaintenanceStore, ProfileReader


logger = logging.getLogger(__name__)


@dataclass
class MemoryRuntime:
    """Runtime holder for all memory-related dependencies."""

    port: "MemoryPort"
    engine: "MemoryEngine | None" = None
    passive_engine: "PassiveMemoryEngine | None" = None
    profile_reader: "ProfileReader | None" = None
    profile_maint: "ProfileMaintenanceStore | None" = None
    post_response_worker: PostResponseMemoryWorker | None = None
    closeables: list[Any] = field(default_factory=list)

    async def aclose(self) -> None:
        """Close owned resources in reverse creation order."""
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
