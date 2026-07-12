from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class HazardResult:
    should_wake: bool
    hazard_before: float
    hazard_after: float
    threshold: float
    evidence: float
    refractory: float
    rate: float
    driver_item_id: str


def _parse_time(value: object, fallback: datetime) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return fallback
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=fallback.tzinfo)
    return parsed


def advance_hazard(
    events: list[dict[str, Any]],
    *,
    now: datetime,
    hazard: float,
    threshold: float,
    updated_at: datetime | None,
    last_wake_at: datetime | None,
    lambda_max: float = 0.790506,
) -> HazardResult:
    if not events:
        return HazardResult(False, hazard, hazard, threshold, 0.0, 0.0, 0.0, "")

    contributions: list[tuple[str, float]] = []
    for event in events:
        raw_probability = event.get("_wake_interest_score")
        if raw_probability is None:
            raw_probability = event.get("preprocess_score")
        probability = min(
            0.999,
            max(0.0, _as_float(raw_probability)),
        )
        log_evidence = -math.log1p(-probability)
        published_at = _parse_time(event.get("published_at"), now)
        age_hours = max(0.0, (now - published_at).total_seconds() / 3600)
        contribution = log_evidence * (1 + math.log1p(age_hours / 6.0))
        contributions.append((str(event.get("id") or ""), contribution))

    evidence = sum(value for _, value in contributions)
    refractory = (
        math.exp(-max(0.0, (now - last_wake_at).total_seconds()) / (2 * 3600))
        if last_wake_at is not None
        else 0.0
    )
    advantage = evidence - 1.0 - refractory
    scaled = max(-60.0, min(60.0, advantage / 0.5))
    rate = lambda_max / (1 + math.exp(-scaled))
    elapsed_hours = (
        max(0.0, (now - updated_at).total_seconds() / 3600)
        if updated_at is not None
        else 5 / 60
    )
    before = hazard
    after = hazard + rate * elapsed_hours
    driver = max(contributions, key=lambda pair: pair[1])[0]
    return HazardResult(
        should_wake=after >= threshold,
        hazard_before=before,
        hazard_after=after,
        threshold=threshold,
        evidence=evidence,
        refractory=refractory,
        rate=rate,
        driver_item_id=driver,
    )


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
