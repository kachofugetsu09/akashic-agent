from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_DIAG_FIELDS = (
    "event",
    "flow",
    "phase",
    "session",
    "turn",
    "tick",
    "action",
    "reason",
    "duration_ms",
    "counts",
    "error_type",
    "error_fp",
    "note",
)

diagnostic_session: ContextVar[str | None] = ContextVar("diagnostic_session", default=None)
diagnostic_flow: ContextVar[str | None] = ContextVar("diagnostic_flow", default=None)
diagnostic_phase: ContextVar[str | None] = ContextVar("diagnostic_phase", default=None)
diagnostic_turn: ContextVar[str | None] = ContextVar("diagnostic_turn", default=None)
diagnostic_tick: ContextVar[str | None] = ContextVar("diagnostic_tick", default=None)


def diagnostic_line(method: str, **fields: object) -> str:
    parts = [f"[{method}]"]
    for key in _DIAG_FIELDS:
        value = _clean(fields.get(key, "-"))
        if key == "note":
            value = f'"{value}"'
        parts.append(f"{key}={value}")
    return " ".join(parts)


@contextmanager
def diagnostic_context(
    *,
    session: str | None = None,
    flow: str | None = None,
    phase: str | None = None,
    turn: str | None = None,
    tick: str | None = None,
) -> Iterator[None]:
    tokens = []
    if session is not None:
        tokens.append((diagnostic_session, diagnostic_session.set(session)))
    if flow is not None:
        tokens.append((diagnostic_flow, diagnostic_flow.set(flow)))
    if phase is not None:
        tokens.append((diagnostic_phase, diagnostic_phase.set(phase)))
    if turn is not None:
        tokens.append((diagnostic_turn, diagnostic_turn.set(turn)))
    if tick is not None:
        tokens.append((diagnostic_tick, diagnostic_tick.set(tick)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current_diagnostic_context() -> dict[str, str]:
    return {
        "session": diagnostic_session.get() or "",
        "flow": diagnostic_flow.get() or "",
        "phase": diagnostic_phase.get() or "",
        "turn": diagnostic_turn.get() or "",
        "tick": diagnostic_tick.get() or "",
    }


def _clean(value: object) -> str:
    text = str(value if value is not None else "-").replace("\n", " ").strip()
    if not text:
        return "-"
    return text.replace('"', "'")
