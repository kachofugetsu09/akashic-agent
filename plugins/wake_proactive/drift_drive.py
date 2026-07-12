from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


DriftDecision = Literal["attempt", "idle"]
_HAZARD_HALF_LIFE_HOURS = 12.0


@dataclass(frozen=True, slots=True)
class DriftDriveResult:
    decision: DriftDecision
    hazard_before: float
    hazard_after: float
    threshold: float
    rate: float
    idle_hours: float
    idle_drive: float
    content_suppression: float
    context_suppression: float
    recent_drift_suppression: float
    repetition_suppression: float
    reasons: tuple[str, ...]


def advance_drift_drive(
    *,
    now: datetime,
    hazard: float,
    threshold: float,
    updated_at: datetime | None,
    last_user_at: datetime | None,
    last_drift_at: datetime | None,
    content_evidence: float,
    busy: bool = False,
    sleeping: bool = False,
    in_game: bool = False,
    context_suppression: float | None = None,
    repetition: float = 0.0,
    max_rate_per_hour: float = 0.3,
) -> DriftDriveResult:
    content = _bounded(content_evidence)
    repetition_score = _bounded(repetition)
    idle_hours = (
        max(0.0, (now - last_user_at).total_seconds() / 3600)
        if last_user_at is not None
        else 0.0
    )
    idle_drive = 1.0 - math.exp(-idle_hours / 4.0)
    content_suppression = content
    context_suppression_score = (
        _bounded(context_suppression)
        if context_suppression is not None
        else _combined_context_suppression(
            busy=busy,
            sleeping=sleeping,
            in_game=in_game,
        )
    )
    recent_drift_suppression = (
        math.exp(-max(0.0, (now - last_drift_at).total_seconds()) / (6 * 3600))
        if last_drift_at is not None
        else 0.0
    )
    repetition_suppression = repetition_score
    rate = (
        max_rate_per_hour
        * idle_drive
        * (1.0 - 0.95 * content_suppression)
        * (1.0 - 0.98 * context_suppression_score)
        * (1.0 - 0.9 * recent_drift_suppression)
        * (1.0 - 0.9 * repetition_suppression)
    )
    elapsed_hours = (
        max(0.0, (now - updated_at).total_seconds() / 3600)
        if updated_at is not None
        else 5 / 60
    )
    before = max(0.0, hazard)
    time_constant = _HAZARD_HALF_LIFE_HOURS / math.log(2.0)
    retention = math.exp(-elapsed_hours / time_constant)
    after = (
        before * retention
        + max(0.0, rate) * time_constant * (1.0 - retention)
    )
    attempt = after >= threshold
    return DriftDriveResult(
        decision="attempt" if attempt else "idle",
        hazard_before=before,
        hazard_after=after,
        threshold=threshold,
        rate=rate,
        idle_hours=idle_hours,
        idle_drive=idle_drive,
        content_suppression=content_suppression,
        context_suppression=context_suppression_score,
        recent_drift_suppression=recent_drift_suppression,
        repetition_suppression=repetition_suppression,
        reasons=_reasons(
            content=content,
            busy=busy,
            sleeping=sleeping,
            in_game=in_game,
            recent_drift=recent_drift_suppression,
            repetition=repetition_score,
            attempt=attempt,
        ),
    )


def _combined_context_suppression(
    *,
    busy: bool,
    sleeping: bool,
    in_game: bool,
) -> float:
    suppressions = (
        0.9 if busy else 0.0,
        0.98 if sleeping else 0.0,
        0.8 if in_game else 0.0,
    )
    remaining = math.prod(1.0 - value for value in suppressions)
    return 1.0 - remaining


def _reasons(
    *,
    content: float,
    busy: bool,
    sleeping: bool,
    in_game: bool,
    recent_drift: float,
    repetition: float,
    attempt: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if content >= 0.5:
        reasons.append("content_evidence")
    if busy:
        reasons.append("busy")
    if sleeping:
        reasons.append("sleeping")
    if in_game:
        reasons.append("in_game")
    if recent_drift >= 0.5:
        reasons.append("recent_drift")
    if repetition >= 0.5:
        reasons.append("repetition")
    if attempt:
        reasons.append("leisure_ready")
    return tuple(reasons)


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
