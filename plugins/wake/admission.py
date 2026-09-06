from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from plugins.akasha.interest import SemanticInterest

from .api import ContentWakeServices, DriftWakeServices
from .content import (_content_text, _datetime, _integer, _mapping, _pool_detail,
                      _preprocess_interest, _semantic_score, _sequence, _string)
from .pool import build_initial_score
from .state import ContentScore, WakeState


@dataclass(frozen=True, slots=True)
class Pool:
    snapshot_seq: int
    items: tuple[Mapping[str, object], ...]
    active_count: int
    due_count: int
    expired_count: int
    scored_count: int

    @property
    def detail(self) -> str:
        return (f"Content 池 active={self.active_count}, due={self.due_count}, "
                f"expired={self.expired_count}, scored={self.scored_count}")


@dataclass(frozen=True, slots=True)
class Admission:
    owner: Literal["alert", "content", "drift"] | None
    detail: str
    pool: Pool
    proposals: tuple[Mapping[str, object], ...] = ()


class Duties:
    """维护 Content 原有评分与到期规则，只提出一份本轮固定的业务选择。"""

    def __init__(self, content: ContentWakeServices, drift: DriftWakeServices,
                 state: WakeState, interest: SemanticInterest):
        self.content, self.drift, self.state, self.interest = content, drift, state, interest
        self._maintenance = asyncio.Lock()

    def deadline(self, now: datetime) -> datetime | None:
        content = self.content.snapshot(now)
        items = _sequence(content.get("items"), "Content items")
        values = [self.state.unseen_deadline(items), self.content.alert_deadline(now)]
        drift = self.drift.snapshot(now).get("next_due")
        if drift is not None:
            values.append(_datetime(drift))
        return min((value for value in values if value is not None), default=None)

    async def check(self, now: datetime) -> Admission:
        """先维护池，再按 Alert、通过阈值的 Content、到期 Drift 依次选择。"""
        pool = await self.maintain(now)
        count = self.state.unseen_due_count(pool.items, now)
        audit = self.state.audit_pool(pool.items, now=now)
        detail = _pool_detail(pool.detail, count, audit)
        alert = self.content.alert_deadline(now)
        if alert is not None and alert <= now:
            return Admission("alert", detail + "；Alert 已到期", pool)
        if self.state.has_unseen_due(pool.items, now):
            result = self.state.evaluate(pool.items, snapshot_seq=pool.snapshot_seq, now=now)
            detail = _pool_detail(pool.detail, count, result)
            if result.should_wake:
                return Admission("content", detail, pool)
        proposals = _sequence(self.drift.snapshot(now).get("proposals"), "Drift proposals")
        if any(item.get("due") is True for item in proposals):
            return Admission("drift", detail + "；Drift 已到期", pool, tuple(proposals))
        return Admission(None, detail + "；没有到期职责或新 Content 不足", pool)

    async def maintain(self, now: datetime) -> Pool:
        """沿原规则只评分一次，低质量 Content 满最短停留期后由 EventMail 过期。"""
        async with self._maintenance:
            snapshot = self.content.snapshot(now)
            items = _sequence(snapshot.get("items"), "Content items")
            scored = len(self.state.unscored_due_items(items))
            items = await self._score(items, now)
            refs = self.state.expired_content_refs(items, now=now, minimum_residence=timedelta(hours=24))
            expired = 0
            if refs:
                result = self.content.expire(refs, now)
                expired = len(_sequence(result.get("expired"), "expired Content"))
                snapshot = self.content.snapshot(now)
                items = _sequence(snapshot.get("items"), "Content items")
                scored += len(self.state.unscored_due_items(items))
                items = await self._score(items, now)
            return Pool(_integer(snapshot.get("snapshot_seq"), "snapshot_seq"), items,
                        sum(item.get("status") in {"pending", "deferred"} for item in items),
                        sum(item.get("due") is True for item in items), expired, scored)

    async def _score(self, items: Sequence[Mapping[str, object]], now: datetime) -> tuple[Mapping[str, object], ...]:
        """原 preprocess 和语义兴趣合成后保存不可变初分，不以缺依赖冒充零分。"""
        unscored = self.state.unscored_due_items(items)
        if not unscored:
            return self.state.scored_items(items)
        scores = await self.interest.score([_content_text(item) for item in unscored], cutoff=now.isoformat())
        if len(scores) != len(unscored):
            raise ValueError("兴趣分数数量与 Content 候选不一致")
        records: list[ContentScore] = []
        for item, value in zip(unscored, scores, strict=True):
            semantic = _semantic_score(value)
            payload = _mapping(item.get("payload"), "Content payload")
            interest = 1 - (1 - _preprocess_interest(payload)) * (1 - semantic)
            ref = _mapping(item.get("ref"), "Content ref")
            records.append(ContentScore(
                source_id=_string(ref.get("source_id"), "source_id"),
                item_id=_string(ref.get("item_id"), "item_id"),
                revision=_string(ref.get("revision"), "revision"),
                initial_score=build_initial_score(interest, has_published_at=bool(payload.get("published_at")),
                                                 wake_eligible=payload.get("wake_eligible") is not False),
                semantic_interest=semantic, scored_at=now,
            ))
        self.state.record_content_scores(records)
        return self.state.scored_items(items)
