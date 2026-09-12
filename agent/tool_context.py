"""Source-neutral execution provenance for the legacy tool registry."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Immutable runtime provenance captured for one tool execution."""

    origin_channel: str = ""
    origin_chat_id: str = ""
    origin_session_key: str = ""
    turn_id: str = ""
    current_timestamp: str = ""
    current_user_source_ref: str = ""
    execution_id: str = ""

    @property
    def timestamp(self) -> datetime | None:
        if not self.current_timestamp:
            return None
        return datetime.fromisoformat(self.current_timestamp)


_CURRENT_TOOL_CONTEXT: ContextVar[ToolExecutionContext | None] = ContextVar(
    "akashic_current_tool_execution_context",
    default=None,
)


def get_current_tool_context() -> ToolExecutionContext | None:
    """Return the immutable provenance for the current async execution."""

    return _CURRENT_TOOL_CONTEXT.get()


@contextmanager
def tool_execution_context_scope(
    context: ToolExecutionContext | None,
) -> Any:
    """Bind one execution context and restore the caller context on exit."""

    token: Token[ToolExecutionContext | None] = _CURRENT_TOOL_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_TOOL_CONTEXT.reset(token)


__all__ = [
    "ToolExecutionContext",
    "get_current_tool_context",
    "tool_execution_context_scope",
]
