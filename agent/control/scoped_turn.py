from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TurnAcceptedReceipt:
    """Identify the Turn after Core accepts custody."""

    session_id: str
    turn_id: str
