from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Callable

from feeds.base import FeedItem


class Decider:
    def __init__(
        self,
        *,
        randomize_fn: Callable[[Any], tuple[Any, float]],
        source_key_fn: Callable[[FeedItem], str],
        item_id_fn: Callable[[FeedItem], str],
        semantic_text_fn: Callable[[FeedItem, int], str],
        semantic_text_max_chars: int,
        composer: Any | None = None,
        judge: Any | None = None,
    ) -> None:
        self._randomize_fn = randomize_fn
        self._source_key = source_key_fn
        self._item_id = item_id_fn
        self._semantic_text = semantic_text_fn
        self._semantic_text_max_chars = semantic_text_max_chars
        self._composer = composer
        self._judge = judge

    async def compose_for_judge(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        preference_block: str = "",
        no_content_token: str = "<no_content/>",
    ) -> str:
        if self._composer is not None:
            return await self._composer.compose_for_judge(
                items=items,
                recent=recent,
                preference_block=preference_block,
                no_content_token=no_content_token,
            )
        if self._judge is None:
            return ""
        return await self._judge.compose_for_judge(
            items=items,
            recent=recent,
            preference_block=preference_block,
            no_content_token=no_content_token,
        )

    async def judge_message(
        self,
        *,
        message: str,
        recent: list[dict],
        recent_proactive_text: str,
        preference_block: str = "",
        age_hours: float,
        sent_24h: int,
        interrupt_factor: float,
    ) -> Any:
        if self._judge is None:
            return None
        return await self._judge.judge_message(
            message=message,
            recent=recent,
            recent_proactive_text=recent_proactive_text,
            preference_block=preference_block,
            age_hours=age_hours,
            sent_24h=sent_24h,
            interrupt_factor=interrupt_factor,
        )

    def pre_compose_veto(
        self,
        *,
        age_hours: float,
        sent_24h: int,
        interrupt_factor: float,
    ) -> str | None:
        if self._judge is None:
            return None
        return self._judge.pre_compose_veto(
            age_hours=age_hours,
            sent_24h=sent_24h,
            interrupt_factor=interrupt_factor,
        )

    def randomize_decision(self, decision: Any) -> tuple[Any, float]:
        return self._randomize_fn(decision)

    def item_id_for(self, item: FeedItem) -> str:
        return self._item_id(item)

    def resolve_evidence_item_ids(
        self, decision: Any, items: list[FeedItem]
    ) -> list[str]:
        valid_order = [self._item_id(item) for item in items]
        valid = set(valid_order)
        selected: list[str] = []
        seen: set[str] = set()
        for raw in getattr(decision, "evidence_item_ids", []) or []:
            item_id = str(raw)
            if item_id in valid and item_id not in seen:
                selected.append(item_id)
                seen.add(item_id)
        if selected:
            return selected[:3]
        return valid_order[:3]

    def build_delivery_key(self, item_ids: list[str], message: str) -> str:
        # 1. 有证据时仅按证据去重，换措辞不改变 dedupe。
        if item_ids:
            raw = "|".join(sorted(set(item_ids)))
        else:
            # 2. 无证据时退化到消息前缀 + 4 小时时间桶，避免空证据消息互相踩掉。
            now = datetime.now(timezone.utc)
            time_bucket = f"{now.year}-{now.month:02d}-{now.day:02d}-h{now.hour // 4}"
            prefix = re.sub(r"\s+", " ", (message or "").strip().lower())[:40]
            raw = f"no_evidence::{time_bucket}::{prefix}"
        # 3. 最后统一收口成稳定 key。
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def semantic_entries(self, items: list[FeedItem]) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for item in items:
            entries.append(
                {
                    "source_key": self._source_key(item),
                    "item_id": self._item_id(item),
                    "text": self._semantic_text(item, self._semantic_text_max_chars),
                }
            )
        return entries


DefaultDecidePort = Decider

