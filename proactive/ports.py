from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random as _random_module
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)
_MAX_PROCEDURE_RETRIEVE_K = 3

if TYPE_CHECKING:
    from agent.provider import LLMProvider
    from core.memory.port import MemoryPort

from feeds.base import FeedItem
from memory2.hyde_enhancer import HyDEEnhancer
from memory2.injection_planner import (
    retrieve_history_items,
    retrieve_procedure_items,
)
from proactive.components import (
    build_proactive_memory_query,
    build_proactive_preference_hyde_prompt,
)
from proactive.energy import compute_energy, d_recent
from proactive.presence import PresenceStore
from proactive.schedule import ScheduleStore
from proactive.state import ProactiveStateStore
from session.manager import SessionManager


@dataclass
class ProactiveSourceRef:
    item_id: str
    source_type: str
    source_name: str
    title: str
    url: str | None = None
    published_at: str | None = None


@dataclass
class ProactiveSendMeta:
    evidence_item_ids: list[str] = field(default_factory=list)
    source_refs: list[ProactiveSourceRef] = field(default_factory=list)
    state_summary_tag: str = "none"


@dataclass
class RecentProactiveMessage:
    content: str
    timestamp: datetime | None = None
    state_summary_tag: str = "none"
    source_refs: list[ProactiveSourceRef] = field(default_factory=list)


class SensePort(Protocol):
    def compute_energy(self) -> float: ...
    def collect_recent(self) -> list[dict]: ...
    def collect_recent_proactive(self, n: int = 5) -> list[RecentProactiveMessage]: ...
    def compute_interruptibility(
        self,
        *,
        now_hour: int,
        now_utc: datetime,
        recent_msg_count: int,
    ) -> tuple[float, dict[str, float]]: ...
    async def fetch_items(self, limit_per_source: int) -> list[FeedItem]: ...
    def filter_new_items(
        self, items: list[FeedItem]
    ) -> tuple[list[FeedItem], list[tuple[str, str]], list[tuple[str, str]]]: ...
    def read_memory_text(self) -> str: ...
    def has_global_memory(self) -> bool: ...
    def last_user_at(self) -> datetime | None: ...
    def refresh_sleep_context(self) -> bool: ...
    def target_session_key(self) -> str: ...


class DecidePort(Protocol):
    async def score_features(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
        retrieved_memory_block: str = "",
        preference_block: str = "",
    ) -> dict[str, float | str] | None: ...
    async def compose_message(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
        retrieved_memory_block: str = "",
        preference_block: str = "",
    ) -> str: ...
    async def compose_for_judge(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        preference_block: str = "",
        no_content_token: str = "<no_content/>",
    ) -> str: ...
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
    ) -> Any: ...
    def pre_compose_veto(
        self,
        *,
        age_hours: float,
        sent_24h: int,
        interrupt_factor: float,
    ) -> str | None: ...
    def randomize_decision(self, decision: Any) -> tuple[Any, float]: ...
    def resolve_evidence_item_ids(
        self, decision: Any, items: list[FeedItem]
    ) -> list[str]: ...
    def build_delivery_key(self, item_ids: list[str], message: str) -> str: ...
    def semantic_entries(self, items: list[FeedItem]) -> list[dict[str, str]]: ...
    def item_id_for(self, item: FeedItem) -> str: ...


class ActPort(Protocol):
    async def send(
        self,
        message: str,
        meta: ProactiveSendMeta | None = None,
    ) -> bool: ...


@dataclass
class ProactiveRetrievedMemory:
    query: str = ""
    block: str = ""
    item_ids: list[str] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)
    procedure_hits: int = 0
    history_hits: int = 0
    history_channel_open: bool = False
    history_gate_reason: str = "disabled"
    history_scope_mode: str = "disabled"
    fallback_reason: str = ""
    preference_block: str = ""  # 偏好专项 RAG 结果，独立于 procedure+event 的 block

    @classmethod
    def empty(cls, fallback_reason: str = "") -> "ProactiveRetrievedMemory":
        return cls(fallback_reason=fallback_reason)


class MemoryRetrievalPort(Protocol):
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
    ) -> ProactiveRetrievedMemory: ...


class DefaultMemoryRetrievalPort:
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
    ) -> ProactiveRetrievedMemory:
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

            # 同步准备各路参数
            p_query = query
            top_k_proc = min(
                _MAX_PROCEDURE_RETRIEVE_K,
                max(1, int(getattr(self._cfg, "memory_top_k_procedure", 4))),
            )
            top_k_hist = max(1, int(getattr(self._cfg, "memory_top_k_history", 6)))
            top_k_pref = max(
                1, int(getattr(self._cfg, "preference_per_source_top_k", 2))
            )
            pref_enabled = bool(
                getattr(self._cfg, "preference_retrieval_enabled", True) and items
            )
            pref_queries_used: list[str] = []

            # preference 是可选路，失败时降级为空并记录，不影响主路径的异常语义。
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
                except Exception as _e:
                    logger.warning("[proactive.memory] preference 检索失败: %s", _e)
                    return []

            # 三路检索并发：procedure / history / preference 彼此无数据依赖。
            # procedure 和 history 失败时直接抛出，由外层 except 统一处理。
            p_items, h_result, raw_pref_items = await asyncio.gather(
                retrieve_procedure_items(self._memory, p_query, top_k=top_k_proc),
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
            h_items: list[dict]
            h_items, history_scope_mode = h_result

            merged_items = _merge_memory_items(p_items + h_items)
            selected_items = self._memory.select_for_injection(merged_items)
            block, item_ids = _build_injection_block(self._memory, merged_items)

            # 偏好专项 RAG：针对候选 item 来源/话题独立查询 preference 类型记忆。
            # 目的：检索"用户只关注 TeamAtlas/PlayerNova 不关心其他战队"之类的明确偏好，
            # 用于引擎层的偏好否决门（engine preference_veto_enabled）。
            preference_block = ""
            pref_hit_count = len(raw_pref_items)
            if raw_pref_items:
                pref_items = _merge_memory_items(raw_pref_items)
                preference_block, _pref_ids = _build_injection_block(
                    self._memory,
                    pref_items,
                )

            result = ProactiveRetrievedMemory(
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
            if result.items or preference_block:
                injected_preview = " | ".join(
                    f"{str(item.get('memory_type', ''))}:{str(item.get('summary', ''))[:40]}"
                    for item in result.items[:4]
                    if isinstance(item, dict)
                )
                logger.info(
                    "[proactive.memory] query=%r p=%d h=%d pref=%d injected=%s",
                    query[:50],
                    result.procedure_hits,
                    result.history_hits,
                    pref_hit_count,
                    injected_preview or "preference_only",
                )
            self._trace(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                result=result,
                candidate_items=items,
                preference_query="\n".join(pref_queries_used),
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
        if any((it.title or "").strip() for it in items):
            return True, "has_topic_items"
        # alert_events 涵盖所有告警类型；兼容旧快照中只有 health_events 的情况。
        _alert_signal = decision_signals.get("alert_events") or decision_signals.get("health_events")
        if isinstance(_alert_signal, list) and _alert_signal:
            return True, "alert_events"
        recent_texts = [
            str(m.get("content", "")).strip()
            for m in recent[-3:]
            if str(m.get("content", "")).strip()
        ]
        if len(recent_texts) >= 2 or sum(len(t) for t in recent_texts) >= 40:
            return True, "recent_continuity"
        return False, "insufficient_topic_signal"

    async def _retrieve_preference_by_sources(
        self,
        *,
        items: list[FeedItem],
        top_k_per_source: int,
    ) -> tuple[list[dict], list[str]]:
        # 1. 先按条目/topic 去重，而不是按 source 去重。
        #    同一来源下可能同时混有用户高度关心与完全无关的内容；
        unique_items = self._unique_items_by_topic(items)
        max_sources = max(1, int(getattr(self._cfg, "preference_max_sources", 5)))
        selected_items = unique_items[:max_sources]
        if not selected_items:
            return [], []
        # 2. 并发跑逐条偏好检索；每条先 raw，再按需追加 HyDE。
        tasks = [
            self._retrieve_preference_hits_for_item(item, top_k_per_source)
            for item in selected_items
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # 3. 最后合并去重，单路异常按 fail-open 忽略。
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
        # 1. 先跑原始偏好 query，保持现有命中路径可用。
        query = self._build_preference_query_for_item(item)
        if self._preference_hyde is None:
            hits = await self._memory.retrieve_related(
                query,
                memory_types=["preference", "profile"],
                top_k=top_k_per_source,
            )
            return hits or [], [query]
        # 2. 再复用 HyDEEnhancer 生成偏好风格假想记忆，并追加第二路检索。
        result = await self._preference_hyde.augment(
            raw_query=query,
            context=self._build_preference_hyde_context(item),
            retrieve_fn=self._memory.retrieve_related,
            top_k=top_k_per_source,
            memory_types=["preference", "profile"],
        )
        # 3. 最后返回合并结果，并把 raw/hyde 两路 query 都带回 trace。
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
                    key = "fallback:" + hashlib.sha1(
                        f"{(item.content or '').strip()[:80]}".encode("utf-8")
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
        result: ProactiveRetrievedMemory,
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
        # JSONL trace（保留原有行为）
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
                "candidate_item_ids": [self._item_id(item) for item in candidate_items[:5]],
                "fallback_reason": result.fallback_reason,
                "preference_query": preference_query,
                "preference_hit_count": preference_hit_count,
                "preference_block_preview": (result.preference_block or "")[:120],
            }
            try:
                self._trace_writer(payload)
            except Exception:
                logger.exception("[proactive.memory] trace writer failed")
        # observe DB trace（新增）
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
        result: ProactiveRetrievedMemory,
        preference_query: str,
        p_items_raw: list[dict],
        h_items_raw: list[dict],
        pref_items_raw: list[dict],
        tick_id: str = "",
    ) -> None:
        import json as _json

        from core.observe.events import RagItemTrace, RagTrace

        injected_id_set = set(result.item_ids)

        def _to_trace(item: dict, path: str) -> RagItemTrace:
            raw_extra = item.get("extra_json")
            extra_str = _json.dumps(raw_extra, ensure_ascii=False) if raw_extra else None
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


class DefaultSensePort:
    def __init__(
        self,
        *,
        cfg: Any,
        sessions: SessionManager,
        state: ProactiveStateStore,
        item_filter: Any,
        memory: "MemoryPort | None",
        presence: PresenceStore | None,
        schedule: ScheduleStore | None,
        rng: Any,
        fitbit: Any | None = None,
    ) -> None:
        self._cfg = cfg
        self._sessions = sessions
        self._state = state
        self._item_filter = item_filter
        self._memory = memory
        self._presence = presence
        self._schedule = schedule
        self._rng = rng
        self._fitbit = fitbit

    def sleep_context(self) -> Any:
        """返回最新的 SleepContext，未配置时返回 None。"""
        if self._fitbit is None:
            return None
        return self._fitbit.get()

    def refresh_sleep_context(self) -> bool:
        """主动刷新一次 Fitbit SleepContext（本地 /api/agent），失败时返回 False。"""
        if self._fitbit is None:
            return False
        refresh = getattr(self._fitbit, "refresh_now", None)
        if not callable(refresh):
            return False
        try:
            return bool(refresh())
        except Exception:
            return False


    def target_session_key(self) -> str:
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        return f"{channel}:{chat_id}" if channel and chat_id else ""

    def read_memory_text(self) -> str:
        if not self._memory:
            return ""
        try:
            return self._memory.read_long_term().strip()
        except Exception:
            return ""

    def has_global_memory(self) -> bool:
        if not self._memory:
            return False
        try:
            return bool(self._memory.read_long_term().strip())
        except Exception:
            return False

    def last_user_at(self) -> datetime | None:
        if not self._presence:
            return None
        return self._presence.get_last_user_at(self.target_session_key())

    def compute_energy(self) -> float:
        if not self._presence:
            return 0.0
        session_key = self.target_session_key()
        last_target = self._presence.get_last_user_at(session_key)
        last_global = self._presence.most_recent_user_at()
        energy_target = compute_energy(last_target)
        energy_global = compute_energy(last_global) * 0.6
        return max(energy_target, energy_global)

    def collect_recent(self) -> list[dict]:
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        if not channel or not chat_id:
            return []
        key = f"{channel}:{chat_id}"
        try:
            session = self._sessions.get_or_create(key)
            msgs = session.messages[-self._cfg.recent_chat_messages :]
            return [
                {
                    "role": m["role"],
                    "content": str(m.get("content", ""))[:200],
                    "timestamp": str(m.get("timestamp", "")),
                }
                for m in msgs
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
        except Exception:
            return []

    def compute_interruptibility(
        self,
        *,
        now_hour: int,
        now_utc: datetime,
        recent_msg_count: int,
    ) -> tuple[float, dict[str, float]]:
        session_key = self.target_session_key()
        if not self._presence or not session_key:
            return 1.0, {
                "f_reply": 1.0,
                "f_activity": 1.0,
                "f_fatigue": 1.0,
                "random_delta": 0.0,
            }

        last_user = self._presence.get_last_user_at(session_key)
        last_proactive = self._presence.get_last_proactive_at(session_key)
        last_global_user = self._presence.most_recent_user_at()

        if last_proactive is None:
            f_reply = 0.6
        elif last_user is not None and last_user > last_proactive:
            lag_min = max(0.0, (last_user - last_proactive).total_seconds() / 60.0)
            decay = max(self._cfg.interrupt_reply_decay_minutes, 1.0)
            f_reply = math.exp(-lag_min / decay)
        else:
            silence_min = max(0.0, (now_utc - last_proactive).total_seconds() / 60.0)
            decay = max(self._cfg.interrupt_no_reply_decay_minutes, 1.0)
            f_reply = 0.15 + 0.35 * math.exp(-silence_min / decay)

        if last_global_user is None:
            f_live = 0.2
        else:
            idle_min = max(0.0, (now_utc - last_global_user).total_seconds() / 60.0)
            decay = max(self._cfg.interrupt_activity_decay_minutes, 1.0)
            f_live = math.exp(-idle_min / decay)
        f_recent = d_recent(recent_msg_count, self._cfg.score_recent_scale)
        f_activity = 0.5 * f_live + 0.5 * f_recent

        sent_24h = self._state.count_deliveries_in_window(
            session_key,
            self._cfg.interrupt_fatigue_window_hours,
            now=now_utc,
        )
        soft_cap = max(self._cfg.interrupt_fatigue_soft_cap, 0.1)
        f_fatigue = 1.0 / (1.0 + sent_24h / soft_cap)

        w_sum = (
            self._cfg.interrupt_weight_reply
            + self._cfg.interrupt_weight_activity
            + self._cfg.interrupt_weight_fatigue
        )
        raw = (
            self._cfg.interrupt_weight_reply * f_reply
            + self._cfg.interrupt_weight_activity * f_activity
            + self._cfg.interrupt_weight_fatigue * f_fatigue
        ) / (w_sum if w_sum > 0 else 1.0)
        random_delta = (self._rng or _random_module).uniform(
            -self._cfg.interrupt_random_strength,
            self._cfg.interrupt_random_strength,
        )
        score = max(self._cfg.interrupt_min_floor, min(1.0, raw + random_delta))
        return score, {
            "f_reply": f_reply,
            "f_activity": f_activity,
            "f_fatigue": f_fatigue,
            "random_delta": random_delta,
        }

    def collect_recent_proactive(self, n: int = 5) -> list[RecentProactiveMessage]:
        """从目标 session 取最近 n 条 proactive=True 的结构化助手消息（按时间升序）。"""
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        if not channel or not chat_id:
            return []
        key = f"{channel}:{chat_id}"
        try:
            session = self._sessions.get_or_create(key)
            results: list[RecentProactiveMessage] = []
            for m in reversed(session.messages):
                if (
                    m.get("role") == "assistant"
                    and m.get("proactive")
                    and m.get("content")
                ):
                    source_refs: list[ProactiveSourceRef] = []
                    for raw in m.get("source_refs") or []:
                        if not isinstance(raw, dict):
                            continue
                        source_refs.append(
                            ProactiveSourceRef(
                                item_id=str(raw.get("item_id", "") or ""),
                                source_type=str(raw.get("source_type", "") or ""),
                                source_name=str(raw.get("source_name", "") or ""),
                                title=str(raw.get("title", "") or ""),
                                url=(
                                    str(raw.get("url")).strip()
                                    if raw.get("url") is not None
                                    else None
                                ),
                                published_at=(
                                    str(raw.get("published_at")).strip()
                                    if raw.get("published_at") is not None
                                    else None
                                ),
                            )
                        )
                    ts = None
                    raw_ts = str(m.get("timestamp", "") or "").strip()
                    if raw_ts:
                        try:
                            ts = datetime.fromisoformat(raw_ts)
                            if ts.tzinfo is None:
                                ts = ts.replace(
                                    tzinfo=datetime.now().astimezone().tzinfo
                                )
                        except Exception:
                            ts = None
                    results.append(
                        RecentProactiveMessage(
                            content=str(m["content"]),
                            timestamp=ts,
                            state_summary_tag=str(
                                m.get("state_summary_tag", "none") or "none"
                            ),
                            source_refs=source_refs,
                        )
                    )
                    if len(results) >= n:
                        break
            return list(reversed(results))
        except Exception:
            return []

    def filter_new_items(
        self, items: list[FeedItem]
    ) -> tuple[list[FeedItem], list[tuple[str, str]], list[tuple[str, str]]]:
        return self._item_filter.filter_new_items(items)


class DefaultDecidePort:
    def __init__(
        self,
        *,
        randomize_fn: Callable[[Any], tuple[Any, float]],
        source_key_fn: Callable[[FeedItem], str],
        item_id_fn: Callable[[FeedItem], str],
        semantic_text_fn: Callable[[FeedItem, int], str],
        semantic_text_max_chars: int,
        feature_scorer: Any | None = None,
        message_composer: Any | None = None,
        judge: Any | None = None,
    ) -> None:
        self._randomize_fn = randomize_fn
        self._source_key = source_key_fn
        self._item_id = item_id_fn
        self._semantic_text = semantic_text_fn
        self._semantic_text_max_chars = semantic_text_max_chars
        self._feature_scorer = feature_scorer
        self._message_composer = message_composer
        self._judge = judge

    async def score_features(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
        retrieved_memory_block: str = "",
        preference_block: str = "",
    ) -> dict[str, float | str] | None:
        if not self._feature_scorer:
            return None
        return await self._feature_scorer.score_features(
            items=items,
            recent=recent,
            decision_signals=decision_signals,
            retrieved_memory_block=retrieved_memory_block,
            preference_block=preference_block,
        )

    async def compose_message(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
        retrieved_memory_block: str = "",
        preference_block: str = "",
    ) -> str:
        if not self._message_composer:
            return ""
        return await self._message_composer.compose_message(
            items=items,
            recent=recent,
            decision_signals=decision_signals,
            retrieved_memory_block=retrieved_memory_block,
            preference_block=preference_block,
        )

    async def compose_for_judge(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        preference_block: str = "",
        no_content_token: str = "<no_content/>",
    ) -> str:
        if not self._judge:
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
        if not self._judge:
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
        if not self._judge:
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
        valid_order = [self._item_id(i) for i in items]
        valid = set(valid_order)
        seen: set[str] = set()
        selected: list[str] = []
        for raw in getattr(decision, "evidence_item_ids", []) or []:
            item_id = str(raw)
            if item_id in valid and item_id not in seen:
                selected.append(item_id)
                seen.add(item_id)
        if selected:
            return selected[:3]
        return valid_order[:3]

    def build_delivery_key(self, item_ids: list[str], message: str) -> str:
        if item_ids:
            # 有证据：仅基于 item_ids 去重，换措辞不影响结果
            raw = "|".join(sorted(set(item_ids)))
        else:
            # 无证据：退化防护——用消息前缀(40字) + 4小时时间桶，
            # 避免所有空证据消息都被同一 hash 压住
            now = datetime.now(timezone.utc)
            time_bucket = f"{now.year}-{now.month:02d}-{now.day:02d}-h{now.hour // 4}"
            prefix = re.sub(r"\s+", " ", (message or "").strip().lower())[:40]
            raw = f"no_evidence::{time_bucket}::{prefix}"
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
