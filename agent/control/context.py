from __future__ import annotations

from contextvars import ContextVar

running_turn_id: ContextVar[str] = ContextVar("running_turn_id", default="")
