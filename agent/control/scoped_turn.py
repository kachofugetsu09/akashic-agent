from __future__ import annotations

from dataclasses import dataclass


class TurnAdmissionRetiredError(RuntimeError):
    """Report that an unaccepted child Turn must hand off to a newer Root."""


@dataclass(frozen=True, slots=True)
class TurnAcceptedReceipt:
    """Identify the Turn after Core accepts custody."""

    session_id: str
    turn_id: str
