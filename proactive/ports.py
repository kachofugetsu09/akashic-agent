from __future__ import annotations

import hashlib
import logging
import math
import random as _random_module
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)

from agent.memory import MemoryStore
from feeds.base import FeedItem
from feeds.buffer import FeedBuffer
from feeds.registry import FeedRegistry
from feeds.store import FeedStore
from proactive.energy import compute_energy, d_recent, time_weight
from proactive.presence import PresenceStore
from proactive.schedule import ScheduleStore
from proactive.source_scorer import SourceScorer
from proactive.state import ProactiveStateStore
from session.manager import SessionManager


class SensePort(Protocol):
    def compute_energy(self) -> float: ...
    def collect_recent(self) -> list[dict]: ...
    def collect_recent_proactive(self, n: int = 5) -> list[str]: ...
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
    def target_session_key(self) -> str: ...
    def quiet_hours(self) -> tuple[int, int, float]: ...


class DecidePort(Protocol):
    async def score_features(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
    ) -> dict[str, float | str] | None: ...
    async def compose_message(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
    ) -> str: ...
    async def reflect(
        self,
        items: list[FeedItem],
        recent: list[dict],
        energy: float = 0.0,
        urge: float = 0.0,
        is_crisis: bool = False,
        decision_signals: dict[str, object] | None = None,
    ) -> Any: ...
    def randomize_decision(self, decision: Any) -> tuple[Any, float]: ...
    def resolve_evidence_item_ids(
        self, decision: Any, items: list[FeedItem]
    ) -> list[str]: ...
    def build_delivery_key(self, item_ids: list[str], message: str) -> str: ...
    def semantic_entries(self, items: list[FeedItem]) -> list[dict[str, str]]: ...
    def item_id_for(self, item: FeedItem) -> str: ...


class ActPort(Protocol):
    async def send(self, message: str) -> bool: ...


class DefaultSensePort:
    def __init__(
        self,
        *,
        cfg: Any,
        feeds: FeedRegistry,
        sessions: SessionManager,
        state: ProactiveStateStore,
        item_filter: Any,
        memory: MemoryStore | None,
        presence: PresenceStore | None,
        schedule: ScheduleStore | None,
        rng: Any,
        feed_buffer: FeedBuffer | None = None,
        source_scorer: SourceScorer | None = None,
        feed_store: FeedStore | None = None,
    ) -> None:
        self._cfg = cfg
        self._feeds = feeds
        self._sessions = sessions
        self._state = state
        self._item_filter = item_filter
        self._memory = memory
        self._presence = presence
        self._schedule = schedule
        self._rng = rng
        self._feed_buffer = feed_buffer
        self._source_scorer = source_scorer
        self._feed_store = feed_store

    def target_session_key(self) -> str:
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        return f"{channel}:{chat_id}" if channel and chat_id else ""

    def quiet_hours(self) -> tuple[int, int, float]:
        if self._schedule:
            return (
                self._schedule.quiet_hours_start(self._cfg.quiet_hours_start),
                self._schedule.quiet_hours_end(self._cfg.quiet_hours_end),
                self._schedule.quiet_hours_weight(self._cfg.quiet_hours_weight),
            )
        return (
            self._cfg.quiet_hours_start,
            self._cfg.quiet_hours_end,
            self._cfg.quiet_hours_weight,
        )

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
        q_start, q_end, q_weight = self.quiet_hours()
        w_time = time_weight(now_hour, q_start, q_end, q_weight)
        f_time = max(0.0, min(1.0, w_time))

        session_key = self.target_session_key()
        if not self._presence or not session_key:
            return 1.0, {
                "f_time": f_time,
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
            self._cfg.interrupt_weight_time
            + self._cfg.interrupt_weight_reply
            + self._cfg.interrupt_weight_activity
            + self._cfg.interrupt_weight_fatigue
        )
        raw = (
            self._cfg.interrupt_weight_time * f_time
            + self._cfg.interrupt_weight_reply * f_reply
            + self._cfg.interrupt_weight_activity * f_activity
            + self._cfg.interrupt_weight_fatigue * f_fatigue
        ) / (w_sum if w_sum > 0 else 1.0)
        random_delta = (self._rng or _random_module).uniform(
            -self._cfg.interrupt_random_strength,
            self._cfg.interrupt_random_strength,
        )
        score = max(self._cfg.interrupt_min_floor, min(1.0, raw + random_delta))
        return score, {
            "f_time": f_time,
            "f_reply": f_reply,
            "f_activity": f_activity,
            "f_fatigue": f_fatigue,
            "random_delta": random_delta,
        }

    async def fetch_items(self, limit_per_source: int) -> list[FeedItem]:
        if self._feed_buffer is not None:
            n = getattr(self._cfg, "feed_poller_read_limit", 50)
            items = self._feed_buffer.get_all(n=n)
            logger.debug(
                "[sense] fetch_items from buffer items=%d read_limit=%d", len(items), n
            )
            return items

        # direct mode：尝试用 source_scorer 动态分配配额
        per_source_limits: dict[str, int] | None = None
        scorer_enabled = getattr(self._cfg, "source_scorer_enabled", False)
        if (
            scorer_enabled
            and self._source_scorer is not None
            and self._feed_store is not None
        ):
            try:
                subs = self._feed_store.list_enabled()
                memory_text = self.read_memory_text()
                total_budget = getattr(self._cfg, "source_scorer_total_budget", 60)
                min_per = getattr(self._cfg, "source_scorer_min_per_source", 2)
                max_per = getattr(self._cfg, "source_scorer_max_per_source", 20)
                per_source_limits = await self._source_scorer.get_limits(
                    subscriptions=subs,
                    memory_text=memory_text,
                    total_budget=total_budget,
                    min_per_source=min_per,
                    max_per_source=max_per,
                )
            except Exception as e:
                logger.warning("[sense] source_scorer 失败，回退均等分配: %s", e)
                per_source_limits = None

        return await self._feeds.fetch_all(
            limit_per_source=limit_per_source,
            per_source_limits=per_source_limits,
        )

    def collect_recent_proactive(self, n: int = 5) -> list[str]:
        """从目标 session 取最近 n 条 proactive=True 的助手消息内容（按时间升序）。"""
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        if not channel or not chat_id:
            return []
        key = f"{channel}:{chat_id}"
        try:
            session = self._sessions.get_or_create(key)
            results: list[str] = []
            for m in reversed(session.messages):
                if (
                    m.get("role") == "assistant"
                    and m.get("proactive")
                    and m.get("content")
                ):
                    results.append(str(m["content"]))
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
        reflector: Any,
        randomize_fn: Callable[[Any], tuple[Any, float]],
        source_key_fn: Callable[[FeedItem], str],
        item_id_fn: Callable[[FeedItem], str],
        semantic_text_fn: Callable[[FeedItem, int], str],
        semantic_text_max_chars: int,
        feature_scorer: Any | None = None,
        message_composer: Any | None = None,
    ) -> None:
        self._reflector = reflector
        self._randomize_fn = randomize_fn
        self._source_key = source_key_fn
        self._item_id = item_id_fn
        self._semantic_text = semantic_text_fn
        self._semantic_text_max_chars = semantic_text_max_chars
        self._feature_scorer = feature_scorer
        self._message_composer = message_composer

    async def score_features(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
    ) -> dict[str, float | str] | None:
        if not self._feature_scorer:
            return None
        return await self._feature_scorer.score_features(
            items=items,
            recent=recent,
            decision_signals=decision_signals,
        )

    async def compose_message(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
    ) -> str:
        if not self._message_composer:
            return ""
        return await self._message_composer.compose_message(
            items=items,
            recent=recent,
            decision_signals=decision_signals,
        )

    async def reflect(
        self,
        items: list[FeedItem],
        recent: list[dict],
        energy: float = 0.0,
        urge: float = 0.0,
        is_crisis: bool = False,
        decision_signals: dict[str, object] | None = None,
    ) -> Any:
        return await self._reflector.reflect(
            items=items,
            recent=recent,
            energy=energy,
            urge=urge,
            is_crisis=is_crisis,
            decision_signals=decision_signals,
        )

    def randomize_decision(self, decision: Any) -> tuple[Any, float]:
        return self._randomize_fn(decision)

    def item_id_for(self, item: FeedItem) -> str:
        return self._item_id(item)

    def resolve_evidence_item_ids(
        self, decision: Any, items: list[FeedItem]
    ) -> list[str]:
        valid = {self._item_id(i) for i in items}
        selected = [x for x in getattr(decision, "evidence_item_ids", []) if x in valid]
        if selected:
            return sorted(set(selected))
        fallback = sorted(valid)
        return fallback[:5]

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
