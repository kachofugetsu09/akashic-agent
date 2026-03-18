from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable

from feeds.base import FeedItem
from memory2.hyde_enhancer import HyDEEnhancer
from memory2.injection_planner import retrieve_history_items, retrieve_procedure_items
from proactive.composer import (
    build_proactive_memory_query,
    build_proactive_preference_hyde_prompt,
)

if TYPE_CHECKING:
    from agent.provider import LLMProvider
    from core.memory.port import MemoryPort

logger = logging.getLogger(__name__)
_MAX_PROCEDURE_RETRIEVE_K = 3


class MemoryRetrievalService:
    """Event-only proactive retrieval with fail-open behavior."""

    def __init__(
        self,
        *,
        cfg: Any,
        memory: "MemoryPort | None",
        item_id_fn: Callable[[FeedItem], str],
        trace_writer: Callable[[dict[str, Any]], None] | None = None,
        observe_writer: Any | None = None,
        light_provider: "LLMProvider | None" = None,
        light_model: str = "",
    ) -> None:
        self._cfg = cfg
        self._memory = memory
        self._item_id = item_id_fn
        self._trace_writer = trace_writer
        self._observe_writer = observe_writer
        self._preference_hyde = self._build_preference_hyde_enhancer(
            light_provider=light_provider,
            light_model=light_model,
        )

    async def retrieve_proactive_context(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
        is_crisis: bool,
        tick_id: str = "",
    ):
        from proactive.ports import ProactiveRetrievedMemory

        # 1. 先处理关闭场景，统一 fail-open。
        if (
            not getattr(self._cfg, "memory_retrieval_enabled", True)
            or self._memory is None
        ):
            result = ProactiveRetrievedMemory.empty("retrieval_disabled")
            self._trace(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                result=result,
                candidate_items=items,
                tick_id=tick_id,
            )
            return result
        try:
            # 2. 再并发检索 procedure/history/preference 三路。
            query = build_proactive_memory_query(
                items=items,
                recent=recent,
                decision_signals=decision_signals,
                is_crisis=is_crisis,
                max_items=max(1, int(getattr(self._cfg, "memory_query_max_items", 3))),
                max_recent=max(
                    1,
                    int(getattr(self._cfg, "memory_query_max_recent_messages", 3)),
                ),
            )
            history_open, history_reason = self._decide_history_gate(
                items=items,
                recent=recent,
                decision_signals=decision_signals,
                is_crisis=is_crisis,
            )
            p_items, h_items, history_scope_mode, raw_pref_items, pref_queries = (
                await self._retrieve_parallel(
                    channel=channel,
                    chat_id=chat_id,
                    items=items,
                    query=query,
                    history_open=history_open,
                )
            )
            # 3. 最后统一组装返回对象并落 trace。
            result = self._build_result(
                query=query,
                history_open=history_open,
                history_reason=history_reason,
                history_scope_mode=history_scope_mode,
                p_items=p_items,
                h_items=h_items,
                raw_pref_items=raw_pref_items,
            )
            pref_hit_count = len(raw_pref_items)
            self._log_injected_preview(result, pref_hit_count=pref_hit_count)
            self._trace(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                result=result,
                candidate_items=items,
                preference_query="\n".join(pref_queries),
                preference_hit_count=pref_hit_count,
                p_items_raw=p_items,
                h_items_raw=h_items,
                pref_items_raw=raw_pref_items,
                tick_id=tick_id,
            )
            return result
        except Exception:
            logger.exception("[proactive.memory] retrieve_proactive_context failed")
            result = ProactiveRetrievedMemory.empty("retrieve_exception")
            self._trace(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                result=result,
                candidate_items=items,
                tick_id=tick_id,
            )
            return result

    async def _retrieve_parallel(
        self,
        *,
        channel: str,
        chat_id: str,
        items: list[FeedItem],
        query: str,
        history_open: bool,
    ) -> tuple[list[dict], list[dict], str, list[dict], list[str]]:
        top_k_proc = min(
            _MAX_PROCEDURE_RETRIEVE_K,
            max(1, int(getattr(self._cfg, "memory_top_k_procedure", 4))),
        )
        top_k_hist = max(1, int(getattr(self._cfg, "memory_top_k_history", 6)))
        top_k_pref = max(1, int(getattr(self._cfg, "preference_per_source_top_k", 2)))
        pref_enabled = bool(
            getattr(self._cfg, "preference_retrieval_enabled", True) and items
        )
        pref_queries_used: list[str] = []

        async def _safe_pref() -> list[dict]:
            if not pref_enabled:
                return []
            try:
                pref_items, used_queries = await self._retrieve_preference_by_sources(
                    items=items,
                    top_k_per_source=top_k_pref,
                )
                pref_queries_used.extend(used_queries)
                return pref_items
            except Exception as exc:
                logger.warning("[proactive.memory] preference 检索失败: %s", exc)
                return []

        p_items, h_result, raw_pref_items = await asyncio.gather(
            retrieve_procedure_items(self._memory, query, top_k=top_k_proc),
            (
                retrieve_history_items(
                    self._memory,
                    query,
                    memory_types=["event"],
                    top_k=top_k_hist,
                    prefer_scoped=True,
                    scope_channel=channel,
                    scope_chat_id=chat_id,
                    allow_global=bool(
                        getattr(self._cfg, "memory_scope_fallback_to_global", False)
                    ),
                )
                if history_open
                else asyncio.sleep(0, result=([], "disabled"))
            ),
            _safe_pref(),
        )
        h_items, history_scope_mode = h_result
        return p_items, h_items, history_scope_mode, raw_pref_items, pref_queries_used

    def _build_result(
        self,
        *,
        query: str,
        history_open: bool,
        history_reason: str,
        history_scope_mode: str,
        p_items: list[dict],
        h_items: list[dict],
        raw_pref_items: list[dict],
    ):
        from proactive.ports import ProactiveRetrievedMemory

        merged_items = _merge_memory_items(p_items + h_items)
        selected_items = self._memory.select_for_injection(merged_items)
        block, item_ids = _build_injection_block(self._memory, merged_items)
        preference_block = ""
        if raw_pref_items:
            pref_items = _merge_memory_items(raw_pref_items)
            preference_block, _pref_ids = _build_injection_block(self._memory, pref_items)
        return ProactiveRetrievedMemory(
            query=query,
            block=block,
            item_ids=item_ids,
            items=selected_items,
            procedure_hits=len(p_items),
            history_hits=len(h_items),
            history_channel_open=history_open,
            history_gate_reason=history_reason,
            history_scope_mode=history_scope_mode,
            preference_block=preference_block,
        )

    def _log_injected_preview(self, result: Any, *, pref_hit_count: int) -> None:
        if not result.items and not result.preference_block:
            return
        injected_preview = " | ".join(
            f"{str(item.get('memory_type', ''))}:{str(item.get('summary', ''))[:40]}"
            for item in result.items[:4]
            if isinstance(item, dict)
        )
        logger.info(
            "[proactive.memory] query=%r p=%d h=%d pref=%d injected=%s",
            result.query[:50],
            result.procedure_hits,
            result.history_hits,
            pref_hit_count,
            injected_preview or "preference_only",
        )

    def _decide_history_gate(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
        is_crisis: bool,
    ) -> tuple[bool, str]:
        if not bool(getattr(self._cfg, "memory_history_gate_enabled", True)):
            return True, "gate_disabled"
        if is_crisis:
            return True, "crisis"
        if any((item.title or "").strip() for item in items):
            return True, "has_topic_items"
        alert_signal = decision_signals.get("alert_events") or decision_signals.get(
            "health_events"
        )
        if isinstance(alert_signal, list) and alert_signal:
            return True, "alert_events"
        recent_texts = [
            str(message.get("content", "")).strip()
            for message in recent[-3:]
            if str(message.get("content", "")).strip()
        ]
        if len(recent_texts) >= 2 or sum(len(text) for text in recent_texts) >= 40:
            return True, "recent_continuity"
        return False, "insufficient_topic_signal"

    async def _retrieve_preference_by_sources(
        self,
        *,
        items: list[FeedItem],
        top_k_per_source: int,
    ) -> tuple[list[dict], list[str]]:
        unique_items = self._unique_items_by_topic(items)
        max_sources = max(1, int(getattr(self._cfg, "preference_max_sources", 5)))
        selected_items = unique_items[:max_sources]
        if not selected_items:
            return [], []
        tasks = [
            self._retrieve_preference_hits_for_item(item, top_k_per_source)
            for item in selected_items
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        merged: list[dict] = []
        queries: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("[proactive.memory] preference 分路检索失败: %s", result)
                continue
            hit_items, used_queries = result
            merged.extend(hit_items or [])
            queries.extend(used_queries or [])
        return _merge_memory_items(merged), queries

    def _build_preference_hyde_enhancer(
        self,
        *,
        light_provider: "LLMProvider | None",
        light_model: str,
    ) -> HyDEEnhancer | None:
        if (
            not getattr(self._cfg, "preference_hyde_enabled", False)
            or light_provider is None
            or not light_model
        ):
            return None
        return HyDEEnhancer(
            light_provider=light_provider,
            light_model=light_model,
            timeout_s=max(
                0.2,
                float(getattr(self._cfg, "preference_hyde_timeout_ms", 2000)) / 1000.0,
            ),
            prompt_builder=build_proactive_preference_hyde_prompt,
        )

    async def _retrieve_preference_hits_for_item(
        self,
        item: FeedItem,
        top_k_per_source: int,
    ) -> tuple[list[dict], list[str]]:
        query = self._build_preference_query_for_item(item)
        if self._preference_hyde is None:
            hits = await self._memory.retrieve_related(
                query,
                memory_types=["preference", "profile"],
                top_k=top_k_per_source,
            )
            return hits or [], [query]
        result = await self._preference_hyde.augment(
            raw_query=query,
            context=self._build_preference_hyde_context(item),
            retrieve_fn=self._memory.retrieve_related,
            top_k=top_k_per_source,
            memory_types=["preference", "profile"],
        )
        queries = [query]
        if result.hypothesis:
            queries.append(result.hypothesis)
        return result.items, queries

    @staticmethod
    def _build_preference_query_for_item(item: FeedItem) -> str:
        source = item.source_name or item.source_type or "该来源"
        title = re.sub(r"\s+", " ", (item.title or "").strip())[:80]
        return f"用户对 {source} 的偏好和态度；相关话题：{title}"

    @staticmethod
    def _build_preference_hyde_context(item: FeedItem) -> str:
        lines: list[str] = []
        title = re.sub(r"\s+", " ", (item.title or "").strip())
        source = (item.source_name or item.source_type or "").strip()
        content = re.sub(r"\s+", " ", (item.content or "").strip())[:160]
        if title:
            lines.append(f"候选内容：{title}")
        if source:
            lines.append(f"来源：{source}")
        if content and content != title:
            lines.append(f"摘要：{content}")
        return "\n".join(lines)

    @staticmethod
    def _unique_items_by_topic(items: list[FeedItem]) -> list[FeedItem]:
        seen: set[str] = set()
        unique: list[FeedItem] = []
        for item in items:
            title = re.sub(r"\s+", " ", (item.title or "").strip()).lower()
            url = (item.url or "").strip().lower()
            if title or url:
                key = f"{title}|{url}"
            else:
                source = (item.source_name or "").strip().lower()
                source_type = (item.source_type or "").strip().lower()
                if source or source_type:
                    key = f"{source_type}:{source}"
                else:
                    fallback = f"{(item.content or '').strip()[:80]}"
                    key = "fallback:" + hashlib.sha1(
                        fallback.encode("utf-8")
                    ).hexdigest()[:12]
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _trace(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        result: Any,
        candidate_items: list[FeedItem],
        preference_query: str = "",
        preference_hit_count: int = 0,
        p_items_raw: list[dict] | None = None,
        h_items_raw: list[dict] | None = None,
        pref_items_raw: list[dict] | None = None,
        tick_id: str = "",
    ) -> None:
        if not bool(getattr(self._cfg, "memory_trace_enabled", True)):
            return
        if self._trace_writer:
            payload = {
                "session_key": session_key,
                "channel": channel,
                "chat_id": chat_id,
                "memory_query": result.query,
                "history_channel_open": result.history_channel_open,
                "history_gate_reason": result.history_gate_reason,
                "history_scope_mode": result.history_scope_mode,
                "procedure_hits": result.procedure_hits,
                "history_hits": result.history_hits,
                "injected_item_ids": result.item_ids,
                "injected_block_preview": (result.block or "")[:240],
                "candidate_item_ids": [
                    self._item_id(item) for item in candidate_items[:5]
                ],
                "fallback_reason": result.fallback_reason,
                "preference_query": preference_query,
                "preference_hit_count": preference_hit_count,
                "preference_block_preview": (result.preference_block or "")[:120],
            }
            try:
                self._trace_writer(payload)
            except Exception:
                logger.exception("[proactive.memory] trace writer failed")
        if self._observe_writer is not None:
            try:
                self._emit_observe_rag(
                    session_key=session_key,
                    result=result,
                    preference_query=preference_query,
                    p_items_raw=p_items_raw or [],
                    h_items_raw=h_items_raw or [],
                    pref_items_raw=pref_items_raw or [],
                    tick_id=tick_id,
                )
            except Exception:
                logger.exception("[proactive.memory] observe emit failed")

    def _emit_observe_rag(
        self,
        *,
        session_key: str,
        result: Any,
        preference_query: str,
        p_items_raw: list[dict],
        h_items_raw: list[dict],
        pref_items_raw: list[dict],
        tick_id: str = "",
    ) -> None:
        from core.observe.events import RagItemTrace, RagTrace

        injected_id_set = set(result.item_ids)

        def _to_trace(item: dict, path: str) -> RagItemTrace:
            raw_extra = item.get("extra_json")
            extra_str = json.dumps(raw_extra, ensure_ascii=False) if raw_extra else None
            return RagItemTrace(
                item_id=str(item.get("id", "")),
                memory_type=str(item.get("memory_type", "")),
                score=float(item.get("score", 0.0)),
                summary=str(item.get("summary", "")),
                happened_at=item.get("happened_at"),
                extra_json=extra_str,
                retrieval_path=path,
                injected=str(item.get("id", "")) in injected_id_set,
            )

        trace_items: list[RagItemTrace] = []
        for item in p_items_raw:
            trace_items.append(_to_trace(item, "procedure"))
        for item in h_items_raw:
            trace_items.append(_to_trace(item, "history_raw"))
        for item in pref_items_raw:
            trace_items.append(_to_trace(item, "preference"))
        rag = RagTrace(
            source="proactive",
            session_key=session_key,
            original_query=result.query,
            query=result.query,
            gate_type=None,
            route_decision=None,
            route_latency_ms=None,
            hyde_hypothesis=None,
            history_scope_mode=result.history_scope_mode,
            history_gate_reason=result.history_gate_reason,
            items=trace_items,
            injected_block=result.block,
            preference_block=result.preference_block,
            preference_query=preference_query or None,
            sufficiency_check_json=None,
            fallback_reason=result.fallback_reason,
            tick_id=tick_id or None,
        )
        self._observe_writer.emit(rag)


DefaultMemoryRetrievalPort = MemoryRetrievalService


def _merge_memory_items(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for item in items:
        item_id = str(item.get("id", "") or "")
        if item_id and item_id in seen:
            continue
        if item_id:
            seen.add(item_id)
        merged.append(item)
    return merged


def _build_injection_block(
    memory: "MemoryPort",
    items: list[dict],
) -> tuple[str, list[str]]:
    if callable(getattr(memory, "build_injection_block", None)):
        return memory.build_injection_block(items)
    return memory.format_injection_with_ids(items)

