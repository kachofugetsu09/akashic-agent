from __future__ import annotations

from contextvars import ContextVar

current_turn_id: ContextVar[str] = ContextVar("current_turn_id", default="")
