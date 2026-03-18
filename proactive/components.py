from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from agent.tools.web_fetch import WebFetchTool
from core.net.http import get_default_http_requester
from feeds.base import FeedItem
from prompts.proactive import build_compose_prompt_messages
from proactive.composer import (
    Composer,
    _build_proactive_prompt_context,
    build_proactive_memory_query,
    build_proactive_preference_hyde_prompt,
    build_proactive_preference_query,
    classify_content_quality,
)
from proactive.judge import (
    Judge,
    MessageDeduper,
    ProactiveJudgeResult,
)
from proactive.sender import ProactiveSendMeta, ProactiveSender, ProactiveSourceRef
from proactive.state import ProactiveStateStore

logger = logging.getLogger(__name__)


class ProactiveJudge:
    """兼容旧接口：对外仍暴露 compose + judge，同步桥接到拆分后的服务。"""

    def __init__(
        self,
        *,
        provider,
        model: str,
        max_tokens: int,
        format_items: Callable[[list[FeedItem]], str],
        format_recent: Callable[[list[dict]], str],
        cfg: Any,
    ) -> None:
        self._composer = Composer(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            format_items=format_items,
            format_recent=format_recent,
        )
        self._judge = Judge(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            format_recent=format_recent,
            cfg=cfg,
        )

    async def compose_for_judge(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        preference_block: str = "",
        no_content_token: str = "<no_content/>",
    ):
        items = await self._enrich_items_for_compose(items)
        prompt_context = _build_proactive_prompt_context(
            items=items,
            recent=recent,
            format_items=self._composer._format_items,
            format_recent=self._composer._format_recent,
        )
        system_msg, user_msg = build_compose_prompt_messages(
            prompt_context=prompt_context,
            preference_block=preference_block,
            no_content_token=no_content_token,
        )
        response = await self._composer._provider.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            tools=[],
            model=self._composer._model,
            max_tokens=min(512, self._composer._max_tokens),
        )
        text = (response.content or "").strip()
        return no_content_token if text.startswith(no_content_token) else text

    async def judge_message(self, **kwargs):
        return await self._judge.judge_message(**kwargs)

    def pre_compose_veto(self, **kwargs):
        return self._judge.pre_compose_veto(**kwargs)

    async def _enrich_items_for_compose(self, items: list[FeedItem]) -> list[FeedItem]:
        candidates = [
            item
            for item in items[:2]
            if item.url and classify_content_quality(item) != "full"
        ]
        if not candidates:
            return items
        fetcher = WebFetchTool(get_default_http_requester("external_default"))
        for item in candidates:
            try:
                raw = await fetcher.execute(url=item.url, format="text", timeout=8)
                data = json.loads(raw or "{}")
                text = str(data.get("text", "") or "").strip()
            except Exception as exc:
                logger.info("[compose] enrich_item_failed url=%s err=%s", item.url, exc)
                setattr(item, "content_status", "fetch_failed")
                continue
            if len(text) <= 400:
                setattr(item, "content_status", "fetch_failed")
                continue
            item.content = text[:4000]
            setattr(item, "content_status", "fetched")
        return items


class ProactiveItemFilter:
    """兼容旧测试的去重组件；主链路已不再依赖。"""

    def __init__(
        self,
        *,
        cfg: Any,
        state: ProactiveStateStore,
        source_key_fn: Callable[[FeedItem], str],
        item_id_fn: Callable[[FeedItem], str],
        semantic_text_fn: Callable[[FeedItem, int], str],
        build_tfidf_vectors_fn: Callable[[list[str], int], list[dict[str, float]]],
        cosine_fn: Callable[[dict[str, float], dict[str, float]], float],
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._source_key = source_key_fn
        self._item_id = item_id_fn
        self._semantic_text = semantic_text_fn
        self._build_tfidf_vectors = build_tfidf_vectors_fn
        self._cosine = cosine_fn

    def filter_new_items(
        self,
        items: list[FeedItem],
    ) -> tuple[list[FeedItem], list[tuple[str, str]], list[tuple[str, str]]]:
        if not items:
            return [], [], []
        now = datetime.now(timezone.utc)
        source_fresh: list[FeedItem] = []
        source_entries: list[tuple[str, str]] = []
        cooldown_hours = getattr(self._cfg, "llm_reject_cooldown_hours", 0)
        for item in items:
            source_key = self._source_key(item)
            item_id = self._item_id(item)
            if self._state.is_item_seen(
                source_key=source_key,
                item_id=item_id,
                ttl_hours=self._cfg.dedupe_seen_ttl_hours,
                now=now,
            ):
                continue
            if cooldown_hours > 0 and self._state.is_rejection_cooled(
                source_key=source_key,
                item_id=item_id,
                ttl_hours=cooldown_hours,
                now=now,
            ):
                continue
            source_fresh.append(item)
            source_entries.append((source_key, item_id))
        if not self._cfg.semantic_dedupe_enabled or not source_fresh:
            return source_fresh, source_entries, []
        return self._semantic_dedupe(source_fresh, source_entries, now)

    def _semantic_dedupe(
        self,
        source_fresh: list[FeedItem],
        source_entries: list[tuple[str, str]],
        now: datetime,
    ) -> tuple[list[FeedItem], list[tuple[str, str]], list[tuple[str, str]]]:
        history = self._state.get_semantic_items(
            window_hours=self._cfg.semantic_dedupe_window_hours,
            max_candidates=self._cfg.semantic_dedupe_max_candidates,
            now=now,
        )
        payload = [
            {
                "item": item,
                "source_key": source_key,
                "item_id": item_id,
                "text": self._semantic_text(item, self._cfg.semantic_dedupe_text_max_chars),
            }
            for item, (source_key, item_id) in zip(source_fresh, source_entries)
        ]
        if not payload:
            return [], [], []
        docs = [row["text"] for row in history] + [row["text"] for row in payload]
        vectors = self._build_tfidf_vectors(docs, self._cfg.semantic_dedupe_ngram)
        history_vectors = vectors[: len(history)]
        payload_vectors = vectors[len(history) :]
        keep_items: list[FeedItem] = []
        keep_entries: list[tuple[str, str]] = []
        duplicate_entries: list[tuple[str, str]] = []
        accepted_vectors: list[dict[str, float]] = []
        accepted_meta: list[dict[str, str]] = []
        threshold = self._cfg.semantic_dedupe_threshold
        for index, row in enumerate(payload):
            vec = payload_vectors[index]
            best_sim = 0.0
            for history_vec in history_vectors:
                best_sim = max(best_sim, self._cosine(vec, history_vec))
            for accepted_vec in accepted_vectors:
                best_sim = max(best_sim, self._cosine(vec, accepted_vec))
            if best_sim >= threshold:
                duplicate_entries.append((row["source_key"], row["item_id"]))
                continue
            keep_items.append(row["item"])
            keep_entries.append((row["source_key"], row["item_id"]))
            accepted_vectors.append(vec)
            accepted_meta.append(
                {"source_key": row["source_key"], "item_id": row["item_id"]}
            )
        return keep_items, keep_entries, duplicate_entries


ProactiveMessageDeduper = MessageDeduper
ComposerService = Composer

__all__ = [
    "Composer",
    "ComposerService",
    "MessageDeduper",
    "ProactiveItemFilter",
    "ProactiveJudge",
    "ProactiveJudgeResult",
    "ProactiveMessageDeduper",
    "ProactiveSendMeta",
    "ProactiveSender",
    "ProactiveSourceRef",
    "build_proactive_memory_query",
    "build_proactive_preference_hyde_prompt",
    "build_proactive_preference_query",
    "classify_content_quality",
]
