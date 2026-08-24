from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

WAKE_ADMISSION_FLOOR = 0.02


@dataclass(frozen=True, slots=True)
class HazardResult:
    should_wake: bool
    hazard_before: float
    hazard_after: float
    threshold: float
    evidence: float
    refractory: float
    rate: float
    preference_pressure: float
    driver_item_id: str


def advance_hazard(
    events: list[dict[str, Any]],
    *,
    now: datetime,
    new_item_ids: set[str],
    random_draw: float,
    last_wake_at: datetime | None,
    pool_mass: float | None = None,
) -> HazardResult:
    """用新事件推动全池概率抽签，并返回可审计的触发结果。"""

    if not events or not new_item_ids:
        return HazardResult(False, 0.0, 0.0, random_draw, 0.0, 0.0, 0.0, 0.0, "")
    ranked = rank_events(events, now=now)
    contributions: list[tuple[str, str, float]] = []
    preference_pressure = 0.0
    new_mass = 0.0
    for event in ranked:
        features = event["_wake_rank_features"]
        probability = float(features["interest"])
        semantic_interest = float(features["semantic_interest"])
        freshness = float(features["freshness"])
        confidence = float(features["publication_confidence"])
        preference_pressure = max(
            preference_pressure,
            semantic_interest * probability * freshness * confidence,
        )
        item_id = str(event.get("id") or "")
        admission_identity = str(event.get("_wake_admission_identity") or item_id)
        contribution = max(
            0.0, float(event["_wake_rank_score"]) - WAKE_ADMISSION_FLOOR
        )
        contributions.append((item_id, admission_identity, contribution))
        if admission_identity in new_item_ids:
            new_mass += contribution
    evidence = max(
        sum(value for _, _, value in contributions),
        max(0.0, float(pool_mass or 0.0)),
    )
    refractory = (
        1.0
        - math.exp(-max(0.0, (now - last_wake_at).total_seconds()) / (2 * 3600))
        if last_wake_at is not None
        else 1.0
    )
    new_signal = 1.0 - math.exp(-new_mass / 0.35)
    pool_signal = 1.0 - math.exp(-evidence / 1.5)
    probability = 1.0 - math.exp(
        -3.0 * new_signal * (0.25 + 0.75 * pool_signal) * refractory
    )
    return HazardResult(
        random_draw < probability,
        new_mass,
        probability,
        random_draw,
        evidence,
        refractory,
        probability,
        preference_pressure,
        max(contributions, key=lambda contribution: contribution[2])[0],
    )


def rank_events(
    events: list[dict[str, Any]], *, now: datetime
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for event in events:
        raw = event.get("_wake_interest_score")
        probability = min(
            0.999,
            max(0.0, _as_float(event.get("preprocess_score") if raw is None else raw)),
        )
        semantic = min(0.999, max(0.0, _as_float(event.get("_wake_semantic_interest"))))
        published = event.get("published_at")
        reference = _parse_time(published or event.get("first_seen_at") or now, now)
        age_hours = max(0.0, (now - reference).total_seconds() / 3600)
        freshness = math.exp(-math.log(2.0) * age_hours / 36.0)
        confidence = 1.0 if published else 0.03
        if event.get("wake_eligible") is False:
            confidence *= 0.01
        copied = dict(event)
        copied["_wake_rank_score"] = -math.log1p(-probability) * freshness * confidence
        copied["_wake_rank_features"] = {
            "interest": probability,
            "semantic_interest": semantic,
            "freshness": freshness,
            "age_hours": age_hours,
            "publication_confidence": confidence,
            "admission_mass": copied["_wake_rank_score"],
            "source_diversity": 1.0,
        }
        scored.append(copied)
    scored.sort(key=_rank_key, reverse=True)
    counts: dict[str, int] = {}
    for event in scored:
        source = str(
            event.get("_reservoir_original_source_id")
            or event.get("source_id")
            or event.get("source")
            or "unknown"
        )
        position = counts.get(source, 0)
        multiplier = 0.5**position
        counts[source] = position + 1
        event["_wake_rank_score"] = float(event["_wake_rank_score"]) * multiplier
        event["_wake_rank_features"]["source_diversity"] = multiplier
    return sorted(scored, key=_rank_key, reverse=True)


def _rank_key(event: dict[str, Any]) -> tuple[float, str]:
    return (
        float(event["_wake_rank_score"]),
        str(event.get("published_at") or event.get("first_seen_at") or ""),
    )


def _parse_time(value: object, fallback: datetime) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=fallback.tzinfo)


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
