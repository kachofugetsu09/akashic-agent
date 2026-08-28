from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

WAKE_ADMISSION_FLOOR = 0.02
WAKE_POOL_THRESHOLD = 1.0

_FRESHNESS_HALF_LIFE_HOURS = 36.0
_MISSING_PUBLICATION_CONFIDENCE = 0.03
_INELIGIBLE_CONFIDENCE_MULTIPLIER = 0.01


@dataclass(frozen=True, slots=True)
class PoolResult:
    should_wake: bool
    new_mass: float
    pool_mass: float
    threshold: float
    below_floor: int
    driver_item_id: str


def measure_pool(
    events: list[dict[str, Any]],
    *,
    now: datetime,
    new_item_ids: set[str] | None = None,
) -> PoolResult:
    """Measure one fixed-score pool after time decay and floor filtering."""

    ranked = rank_events(events, now=now)
    new_identities = new_item_ids or set()
    contributions: list[tuple[str, str, float]] = []
    below_floor = 0
    for event in ranked:
        item_id = str(event.get("id") or "")
        identity = str(event.get("_wake_admission_identity") or item_id)
        features = event["_wake_rank_features"]
        mass = float(features["admission_mass"])
        contribution = mass if mass >= WAKE_ADMISSION_FLOOR else 0.0
        below_floor += contribution == 0.0
        contributions.append((item_id, identity, contribution))
    pool_mass = sum(value for _, _, value in contributions)
    new_mass = sum(
        value for _, identity, value in contributions if identity in new_identities
    )
    driver = max(contributions, key=lambda item: item[2])[0] if contributions else ""
    return PoolResult(
        should_wake=bool(new_identities) and pool_mass > WAKE_POOL_THRESHOLD,
        new_mass=new_mass,
        pool_mass=pool_mass,
        threshold=WAKE_POOL_THRESHOLD,
        below_floor=below_floor,
        driver_item_id=driver,
    )


def rank_events(events: list[dict[str, Any]], *, now: datetime) -> list[dict[str, Any]]:
    """Decay fixed scores, then apply source diversity only to page order."""

    scored: list[dict[str, Any]] = []
    for event in events:
        raw_initial = event.get("_wake_initial_score")
        if raw_initial is None:
            raw_interest = event.get("_wake_interest_score")
            if raw_interest is None:
                raw_interest = event.get("preprocess_score")
            initial = build_initial_score(
                _as_float(raw_interest),
                has_published_at=bool(event.get("published_at")),
                wake_eligible=event.get("wake_eligible") is not False,
            )
        else:
            initial = min(7.0, max(0.0, _as_float(raw_initial)))
        semantic = min(
            0.999, max(0.0, _as_float(event.get("_wake_semantic_interest")))
        )
        reference = _parse_time(
            event.get("published_at") or event.get("first_seen_at") or now,
            now,
        )
        age_hours = max(0.0, (now - reference).total_seconds() / 3600)
        freshness = math.exp(
            -math.log(2.0) * age_hours / _FRESHNESS_HALF_LIFE_HOURS
        )
        copied = dict(event)
        admission_mass = initial * freshness
        copied["_wake_rank_score"] = admission_mass
        copied["_wake_rank_features"] = {
            "initial_score": initial,
            "semantic_interest": semantic,
            "freshness": freshness,
            "age_hours": age_hours,
            "admission_mass": admission_mass,
            "source_diversity": 1.0,
        }
        scored.append(copied)
    scored.sort(key=_rank_key, reverse=True)
    source_counts: dict[str, int] = {}
    for event in scored:
        source = str(event.get("source_id") or event.get("source") or "unknown")
        position = source_counts.get(source, 0)
        multiplier = 0.5**position
        source_counts[source] = position + 1
        event["_wake_rank_score"] = float(event["_wake_rank_score"]) * multiplier
        event["_wake_rank_features"]["source_diversity"] = multiplier
    return sorted(scored, key=_rank_key, reverse=True)


def build_initial_score(
    interest: float,
    *,
    has_published_at: bool,
    wake_eligible: bool,
) -> float:
    """Convert one 0..1 interest probability to the legacy admission-mass scale."""

    probability = min(0.999, max(0.0, interest))
    confidence = 1.0 if has_published_at else _MISSING_PUBLICATION_CONFIDENCE
    if not wake_eligible:
        confidence *= _INELIGIBLE_CONFIDENCE_MULTIPLIER
    return -math.log1p(-probability) * confidence


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
    return (
        parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=fallback.tzinfo)
    )


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
