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
        recent_drift_suppression=recent_drift_suppression,
        repetition_suppression=repetition_suppression,
        reasons=_reasons(
            content=content,
            recent_drift=recent_drift_suppression,
            repetition=repetition_score,
            attempt=attempt,
        ),
    )


def sample_drift_delay_hours(
    *,
    random_draw: float,
    idle_hours: float,
    recent_drift_suppression: float,
    repetition_suppression: float,
    max_rate_per_hour: float = 0.08,
) -> float:
    """从递增的空闲 hazard 采样下一次一次性 Drift 到期时间。"""

    # 1. 将上下文和近期重复转成连续速率
    scale = (
        max_rate_per_hour
        * (1.0 - 0.9 * _bounded(recent_drift_suppression))
        * (1.0 - 0.9 * _bounded(repetition_suppression))
    )
    target = -math.log1p(-min(1.0 - 1e-12, max(0.0, random_draw)))
    start_mass = _integrated_idle_drive(max(0.0, idle_hours), scale)

    # 2. 单调求解剩余累计 hazard，避免周期轮询积累
    low = max(0.0, idle_hours)
    high = low + 1.0
    while _integrated_idle_drive(high, scale) - start_mass < target:
        high = low + 2.0 * (high - low)
    for _ in range(64):
        middle = (low + high) / 2.0
        if _integrated_idle_drive(middle, scale) - start_mass < target:
            low = middle
        else:
            high = middle
    return high - max(0.0, idle_hours)


def _integrated_idle_drive(idle_hours: float, scale: float) -> float:
    return scale * (
        idle_hours - 4.0 * (1.0 - math.exp(-idle_hours / 4.0))
    )


def _reasons(
    *,
    content: float,
    recent_drift: float,
    repetition: float,
    attempt: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if content >= 0.5:
        reasons.append("content_evidence")
    if recent_drift >= 0.5:
        reasons.append("recent_drift")
    if repetition >= 0.5:
        reasons.append("repetition")
    if attempt:
        reasons.append("leisure_ready")
    return tuple(reasons)


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
