from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any


_FRESHNESS_HALF_LIFE_HOURS = 36.0
_MISSING_PUBLICATION_CONFIDENCE = 0.03
_INELIGIBLE_CONFIDENCE_MULTIPLIER = 0.01
_SOURCE_DIVERSITY_DECAY = 0.5
WAKE_ADMISSION_FLOOR = 0.02
_NEW_MASS_SCALE = 0.35
_POOL_MASS_SCALE = 1.5
_CONTENT_TRIGGER_GAIN = 3.0
_REFRACTORY_HOURS = 2.0


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
    new_item_ids: set[str],
    random_draw: float,
    last_wake_at: datetime | None,
) -> HazardResult:
    """用新事件推动全池概率抽签，并返回可审计的触发结果。"""

    # 1. 计算所有存活内容的压力和本轮新增质量
    if not events or not new_item_ids:
        return HazardResult(
            False, 0.0, 0.0, random_draw, 0.0, 0.0, 0.0, 0.0, ""
        )

    ranked = rank_events(events, now=now)
    contributions: list[tuple[str, float]] = []
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
        contribution = max(
            0.0,
            float(event["_wake_rank_score"]) - WAKE_ADMISSION_FLOOR,
        )
        contributions.append((item_id, contribution))
        if item_id in new_item_ids:
            new_mass += contribution

    if not contributions:
        return HazardResult(
            False, 0.0, 0.0, random_draw, 0.0, 0.0, 0.0, 0.0, ""
        )

    # 2. 新事件提供 kick，旧池只放大本次抽签
    evidence = sum(value for _, value in contributions)
    refractory = (
        1.0
        - math.exp(
            -max(0.0, (now - last_wake_at).total_seconds())
            / (_REFRACTORY_HOURS * 3600)
        )
        if last_wake_at is not None
        else 1.0
    )
    new_signal = 1.0 - math.exp(-new_mass / _NEW_MASS_SCALE)
    pool_signal = 1.0 - math.exp(-evidence / _POOL_MASS_SCALE)
    event_drive = new_signal * (0.25 + 0.75 * pool_signal) * refractory
    probability = 1.0 - math.exp(-_CONTENT_TRIGGER_GAIN * event_drive)
    driver = max(contributions, key=lambda pair: pair[1])[0]

    # 3. 概率抽签只在新事件到达时发生
    return HazardResult(
        should_wake=random_draw < probability,
        hazard_before=new_mass,
        hazard_after=probability,
        threshold=random_draw,
        evidence=evidence,
        refractory=refractory,
        rate=probability,
        preference_pressure=preference_pressure,
        driver_item_id=driver,
    )


def rank_events(
    events: list[dict[str, Any]],
    *,
    now: datetime,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for event in events:
        raw_probability = event.get("_wake_interest_score")
        if raw_probability is None:
            raw_probability = event.get("preprocess_score")
        probability = min(0.999, max(0.0, _as_float(raw_probability)))
        semantic_interest = min(
            0.999,
            max(0.0, _as_float(event.get("_wake_semantic_interest"))),
        )
        raw_published_at = event.get("published_at")
        raw_first_seen_at = event.get("first_seen_at")
        reference_time = _parse_time(
            raw_published_at or raw_first_seen_at or now,
            now,
        )
        age_hours = max(0.0, (now - reference_time).total_seconds() / 3600)
        freshness = math.exp(
            -math.log(2.0) * age_hours / _FRESHNESS_HALF_LIFE_HOURS
        )
        publication_confidence = (
            1.0 if raw_published_at else _MISSING_PUBLICATION_CONFIDENCE
        )
        if event.get("wake_eligible") is False:
            publication_confidence *= _INELIGIBLE_CONFIDENCE_MULTIPLIER
        evidence = -math.log1p(-probability) * freshness * publication_confidence
        copied = dict(event)
        copied["_wake_rank_score"] = evidence
        copied["_wake_rank_features"] = {
            "interest": probability,
            "semantic_interest": semantic_interest,
            "freshness": freshness,
            "age_hours": age_hours,
            "publication_confidence": publication_confidence,
            "admission_mass": evidence,
            "source_diversity": 1.0,
        }
        scored.append(copied)

    scored.sort(
        key=lambda event: (
            float(event["_wake_rank_score"]),
            str(event.get("published_at") or event.get("first_seen_at") or ""),
        ),
        reverse=True,
    )
    source_counts: dict[str, int] = {}
    for event in scored:
        source_id = str(
            event.get("_reservoir_original_source_id")
            or event.get("source_id")
            or event.get("source")
            or "unknown"
        )
        position = source_counts.get(source_id, 0)
        multiplier = _SOURCE_DIVERSITY_DECAY ** position
        source_counts[source_id] = position + 1
        event["_wake_rank_score"] = float(event["_wake_rank_score"]) * multiplier
        event["_wake_rank_features"]["source_diversity"] = multiplier

    return sorted(
        scored,
        key=lambda event: (
            float(event["_wake_rank_score"]),
            str(event.get("published_at") or event.get("first_seen_at") or ""),
        ),
        reverse=True,
    )


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
