from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping


Presence = Literal[
    "active",
    "idle",
    "sleeping",
    "in_game",
    "offline",
    "unknown",
]
ContextSignal = Literal["refresh", "reevaluate"]


@dataclass(frozen=True, slots=True)
class NormalizedContext:
    presence: Presence
    interruptibility: float
    confidence: float
    transition: str
    observed_at: datetime | None = None
    expires_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


@dataclass(frozen=True, slots=True)
class ContextDriveResult:
    context: NormalizedContext
    signal: ContextSignal
    should_contact: bool
    changed_fields: tuple[str, ...]


def evaluate_context(
    snapshot: Mapping[str, Any],
    *,
    previous: NormalizedContext | None = None,
    transition_confidence: float = 0.55,
) -> ContextDriveResult:
    presence = _presence(snapshot)
    confidence = _bounded(snapshot.get("confidence", snapshot.get("presence_confidence", 0.5)))
    interruptibility = _interruptibility(snapshot, presence)
    observed_at = _optional_time(snapshot.get("observed_at") or snapshot.get("changed_at"))
    expires_at = _optional_time(snapshot.get("expires_at"))
    changed_fields = _changed_fields(previous, presence, interruptibility)
    transition = str(snapshot.get("transition") or "").strip()
    if not transition and previous is not None and changed_fields:
        transition = f"{previous.presence}->{presence}"
    meaningful_transition = (
        bool(transition)
        and confidence >= transition_confidence
        and (previous is None or bool(changed_fields))
    )
    return ContextDriveResult(
        context=NormalizedContext(
            presence=presence,
            interruptibility=interruptibility,
            confidence=confidence,
            transition=transition,
            observed_at=observed_at,
            expires_at=expires_at,
            raw=dict(snapshot),
        ),
        signal="reevaluate" if meaningful_transition else "refresh",
        should_contact=False,
        changed_fields=changed_fields,
    )


def _presence(snapshot: Mapping[str, Any]) -> Presence:
    raw = str(snapshot.get("presence") or "").strip().lower().replace("-", "_")
    aliases: dict[str, Presence] = {
        "active": "active",
        "awake": "active",
        "online": "active",
        "idle": "idle",
        "away": "idle",
        "sleeping": "sleeping",
        "asleep": "sleeping",
        "in_game": "in_game",
        "playing": "in_game",
        "offline": "offline",
        "unknown": "unknown",
    }
    if raw in aliases:
        return aliases[raw]
    if snapshot.get("sleeping") is True:
        return "sleeping"
    if snapshot.get("in_game") is True or str(snapshot.get("current_game") or "").strip():
        return "in_game"
    if snapshot.get("online") is False:
        return "offline"
    return "unknown"


def _interruptibility(snapshot: Mapping[str, Any], presence: Presence) -> float:
    explicit = snapshot.get("interruptibility")
    if isinstance(explicit, str):
        values = {"high": 0.85, "medium": 0.5, "low": 0.15, "none": 0.0}
        if explicit.strip().lower() in values:
            return values[explicit.strip().lower()]
    if explicit is not None:
        return _bounded(explicit)
    defaults = {
        "active": 0.8,
        "idle": 0.65,
        "sleeping": 0.0,
        "in_game": 0.15,
        "offline": 0.0,
        "unknown": 0.5,
    }
    value = defaults[presence]
    return min(value, 0.1) if snapshot.get("busy") is True else value


def _changed_fields(
    previous: NormalizedContext | None,
    presence: Presence,
    interruptibility: float,
) -> tuple[str, ...]:
    if previous is None:
        return ()
    changed: list[str] = []
    if previous.presence != presence:
        changed.append("presence")
    if abs(previous.interruptibility - interruptibility) >= 0.2:
        changed.append("interruptibility")
    return tuple(changed)


def _bounded(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(str(value))))
    except (TypeError, ValueError):
        return 0.0


def _optional_time(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
