"""
ProactiveLoop — 主动触达核心循环。

独立于 AgentLoop，定期：
  1. 拉取所有订阅信息流的最新内容
  2. 获取用户最近聊天上下文
  3. 调用 LLM 反思：有没有值得主动说的
  4. 高于阈值时通过 MessagePushTool 发送消息
"""
from __future__ import annotations

import asyncio
import math
import hashlib
import json
import logging
import random as _random_module
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from agent.provider import LLMProvider
from agent.memory import MemoryStore
from agent.tools.message_push import MessagePushTool
from feeds.base import FeedItem
from feeds.registry import FeedRegistry
from proactive.energy import (
    compute_energy,
    composite_score,
    d_content,
    d_energy,
    d_recent,
    next_tick_from_score,
    random_weight,
    time_weight,
)
from proactive.memory_sampler import sample_memory_chunks
from proactive.presence import PresenceStore
from proactive.state import ProactiveStateStore
from session.manager import SessionManager

logger = logging.getLogger(__name__)


@dataclass
class ProactiveConfig:
    enabled: bool = False
    interval_seconds: int = 1800    # 无 presence 时的固定间隔（秒）
    threshold: float = 0.70         # score 高于此值才发送
    items_per_source: int = 3       # 每个信息源取几条
    recent_chat_messages: int = 20  # 回顾最近 N 条对话
    model: str = ""                 # 留空则继承全局 model
    default_channel: str = "telegram"
    default_chat_id: str = ""
    dedupe_seen_ttl_hours: int = 24 * 14
    delivery_dedupe_hours: int = 24
    only_new_items_trigger: bool = True
    semantic_dedupe_enabled: bool = True
    semantic_dedupe_threshold: float = 0.90
    semantic_dedupe_window_hours: int = 72
    semantic_dedupe_max_candidates: int = 200
    semantic_dedupe_ngram: int = 3
    semantic_dedupe_text_max_chars: int = 240
    use_global_memory: bool = True
    global_memory_max_chars: int = 3000
    # ── 多维打分 ──
    score_weight_energy: float = 0.40    # D_energy 权重（互动饥渴度）
    score_weight_content: float = 0.40   # D_content 权重（信息流新鲜度）
    score_weight_recent: float = 0.20    # D_recent 权重（对话语境丰富度）
    score_content_halfsat: float = 3.0   # D_content 半饱和点（新条目数）
    score_recent_scale: float = 10.0     # D_recent 消息数归一化基数
    score_llm_threshold: float = 0.40    # draw_score 超过此值才调 LLM
    score_pre_threshold: float = 0.05    # pre_score 低于此值直接跳过（省 feed 拉取）
    # ── Interruptibility（非硬拦截，作为软权重）──
    interrupt_weight_time: float = 0.25
    interrupt_weight_reply: float = 0.35
    interrupt_weight_activity: float = 0.25
    interrupt_weight_fatigue: float = 0.15
    interrupt_activity_decay_minutes: float = 180.0
    interrupt_reply_decay_minutes: float = 120.0
    interrupt_no_reply_decay_minutes: float = 360.0
    interrupt_fatigue_window_hours: int = 24
    interrupt_fatigue_soft_cap: float = 6.0
    interrupt_random_strength: float = 0.12
    interrupt_min_floor: float = 0.08
    # ── 昼夜节律 ──
    quiet_hours_start: int = 23           # 静默开始（本地时间）
    quiet_hours_end: int = 8              # 静默结束（本地时间）
    quiet_hours_weight: float = 0.0      # 静默时段权重，0=完全不发，0.1=低概率仍可发
    # ── tick 间隔（由 base_score 驱动）──
    tick_interval_s0: int = 4800         # base_score ≤ 0.20 → ~80 min
    tick_interval_s1: int = 2400         # base_score > 0.20 → ~40 min
    tick_interval_s2: int = 1080         # base_score > 0.40 → ~18 min
    tick_interval_s3: int = 420          # base_score > 0.70 → ~7 min
    tick_jitter: float = 0.3            # 随机抖动幅度，0.3=±30%，0=关闭
    # ── 旧参数兼容（当前主流程不再使用）──
    energy_cool_threshold: float = 0.20
    energy_crisis_threshold: float = 0.05
    energy_min_urge: float = 0.10


@dataclass
class _Decision:
    score: float
    should_send: bool
    message: str
    reasoning: str
    evidence_item_ids: list[str] = field(default_factory=list)


class ProactiveLoop:
    def __init__(
        self,
        feed_registry: FeedRegistry,
        session_manager: SessionManager,
        provider: LLMProvider,
        push_tool: MessagePushTool,
        config: ProactiveConfig,
        model: str,
        max_tokens: int = 1024,
        state_store: ProactiveStateStore | None = None,
        state_path: Path | None = None,
        memory_store: MemoryStore | None = None,
        presence: PresenceStore | None = None,
        rng: _random_module.Random | None = None,
    ) -> None:
        self._feeds = feed_registry
        self._sessions = session_manager
        self._provider = provider
        self._push = push_tool
        self._cfg = config
        self._model = config.model or model
        self._max_tokens = max_tokens
        self._state = state_store or ProactiveStateStore(state_path or Path("proactive_state.json"))
        self._memory = memory_store
        self._presence = presence
        self._rng = rng
        self._running = False
        logger.info(
            "[proactive] 去重配置 seen_ttl=%dh delivery_window=%dh only_new_items_trigger=%s semantic_enabled=%s semantic_threshold=%.2f semantic_window=%dh ngram=%d use_global_memory=%s memory_max_chars=%d",
            self._cfg.dedupe_seen_ttl_hours,
            self._cfg.delivery_dedupe_hours,
            self._cfg.only_new_items_trigger,
            self._cfg.semantic_dedupe_enabled,
            self._cfg.semantic_dedupe_threshold,
            self._cfg.semantic_dedupe_window_hours,
            self._cfg.semantic_dedupe_ngram,
            self._cfg.use_global_memory,
            self._cfg.global_memory_max_chars,
        )

    async def run(self) -> None:
        self._running = True
        logger.info(
            f"ProactiveLoop 已启动  阈值={self._cfg.threshold}  "
            f"目标={self._cfg.default_channel}:{self._cfg.default_chat_id}"
        )
        last_base_score: float | None = None
        while self._running:
            interval = self._next_interval(last_base_score)
            logger.info("[proactive] 下次 tick 间隔=%ds", interval)
            await asyncio.sleep(interval)
            try:
                last_base_score = await self._tick()
            except Exception:
                logger.exception("ProactiveLoop tick 异常")
                last_base_score = None

    def _next_interval(self, base_score: float | None = None) -> int:
        """根据 base_score 返回自适应等待秒数。无 presence 时回退固定间隔。"""
        if not self._presence:
            return self._cfg.interval_seconds
        # base_score 由 _tick 传入；首次启动时用电量估算一个初始值
        if base_score is None:
            session_key = self._target_session_key()
            last_user_at = self._presence.get_last_user_at(session_key)
            energy = compute_energy(last_user_at)
            base_score = d_energy(energy) * self._cfg.score_weight_energy
        return next_tick_from_score(
            base_score,
            tick_s3=self._cfg.tick_interval_s3,
            tick_s2=self._cfg.tick_interval_s2,
            tick_s1=self._cfg.tick_interval_s1,
            tick_s0=self._cfg.tick_interval_s0,
            tick_jitter=self._cfg.tick_jitter,
            rng=self._rng,
        )

    def _target_session_key(self) -> str:
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        return f"{channel}:{chat_id}" if channel and chat_id else ""

    def stop(self) -> None:
        self._running = False

    def _sample_random_memory(self, n: int = 2) -> list[str]:
        """随机抽取 n 条记忆片段，无记忆时返回 []。"""
        if not self._memory:
            return []
        try:
            raw = self._memory.read_long_term().strip()
            return sample_memory_chunks(raw, n=n)
        except Exception as e:
            logger.warning("[proactive] 随机记忆抽取失败: %s", e)
            return []

    def _has_global_memory(self) -> bool:
        if not self._memory:
            return False
        try:
            return bool(self._memory.read_long_term().strip())
        except Exception:
            return False

    def _compute_energy(self) -> float:
        """计算目标 session 的当前电量（取目标与全局较高值）。"""
        if not self._presence:
            return 0.0  # 无 presence 时视作完全放电
        session_key = self._target_session_key()
        last_target = self._presence.get_last_user_at(session_key)
        last_global = self._presence.most_recent_user_at()
        energy_target = compute_energy(last_target)
        energy_global = compute_energy(last_global) * 0.6
        return max(energy_target, energy_global)

    def _compute_interruptibility(
        self,
        *,
        now_hour: int,
        now_utc: datetime,
        recent_msg_count: int,
    ) -> tuple[float, dict[str, float]]:
        """计算软打扰系数（0~1），并注入随机探索，避免长期锁死。"""
        w_time = time_weight(
            now_hour,
            self._cfg.quiet_hours_start,
            self._cfg.quiet_hours_end,
            self._cfg.quiet_hours_weight,
        )
        f_time = max(0.0, min(1.0, w_time))

        session_key = self._target_session_key()
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

        # 回复信号：有回复且回复快 -> 高；长期不回复 -> 低，但保留探索地板
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

        # 活跃信号：最近消息越近越高；并结合当前上下文消息量
        if last_global_user is None:
            f_live = 0.2
        else:
            idle_min = max(0.0, (now_utc - last_global_user).total_seconds() / 60.0)
            decay = max(self._cfg.interrupt_activity_decay_minutes, 1.0)
            f_live = math.exp(-idle_min / decay)
        f_recent = d_recent(recent_msg_count, self._cfg.score_recent_scale)
        f_activity = 0.5 * f_live + 0.5 * f_recent

        # 疲劳信号：最近主动次数越多，系数越低（软约束）
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

    # ── internal ──────────────────────────────────────────────────

    async def _tick(self) -> float:
        """执行一次主动判断循环，返回本轮 base_score（供调整下次 tick 间隔）。"""
        logger.info("[proactive] tick 开始")
        self._state.cleanup(
            seen_ttl_hours=self._cfg.dedupe_seen_ttl_hours,
            delivery_ttl_hours=self._cfg.delivery_dedupe_hours,
            semantic_ttl_hours=max(self._cfg.dedupe_seen_ttl_hours, self._cfg.semantic_dedupe_window_hours),
        )

        # ── 第一阶段：pre_score（不拉 feed，纯本地计算）──────────────
        energy = self._compute_energy()
        now_hour = datetime.now().hour
        now_utc = datetime.now(timezone.utc)

        recent = self._collect_recent()
        de = d_energy(energy)
        dr = d_recent(len(recent), self._cfg.score_recent_scale)
        interruptibility, interrupt_detail = self._compute_interruptibility(
            now_hour=now_hour,
            now_utc=now_utc,
            recent_msg_count=len(recent),
        )
        interrupt_factor = 0.6 + 0.4 * interruptibility
        # pre_score：只用 D_energy 和 D_recent，权重重新归一化（排除 D_content）
        w_sum = self._cfg.score_weight_energy + self._cfg.score_weight_recent
        pre_score = (
            (self._cfg.score_weight_energy * de + self._cfg.score_weight_recent * dr) / w_sum
            if w_sum > 0 else 0.0
        ) * interrupt_factor

        logger.info(
            "[proactive] pre_score=%.3f interrupt=%.3f factor=%.3f"
            " (time=%.2f reply=%.2f activity=%.2f fatigue=%.2f rand=%+.2f)"
            " D_energy=%.3f D_recent=%.3f energy=%.3f msg_count=%d",
            pre_score,
            interruptibility,
            interrupt_factor,
            interrupt_detail["f_time"],
            interrupt_detail["f_reply"],
            interrupt_detail["f_activity"],
            interrupt_detail["f_fatigue"],
            interrupt_detail["random_delta"],
            de,
            dr,
            energy,
            len(recent),
        )

        if pre_score < self._cfg.score_pre_threshold:
            logger.info("[proactive] pre_score 过低（%.3f < %.2f），跳过本轮", pre_score, self._cfg.score_pre_threshold)
            return pre_score

        # ── 第二阶段：拉 feed，计算完整 base_score ──────────────────
        items = await self._feeds.fetch_all(self._cfg.items_per_source)
        logger.info("[proactive] 拉取到 %d 条信息", len(items))
        new_items, new_entries, semantic_duplicate_entries = self._filter_new_items(items)
        logger.info(
            "[proactive] 去重后剩余新信息 %d 条（过滤重复 %d 条）",
            len(new_items),
            len(items) - len(new_items),
        )
        if semantic_duplicate_entries:
            self._state.mark_items_seen(semantic_duplicate_entries)
            logger.info(
                "[proactive] 已标记语义重复条目为 seen count=%d",
                len(semantic_duplicate_entries),
            )

        has_memory = self._has_global_memory()
        if self._cfg.only_new_items_trigger and not new_items and not self._presence and not has_memory:
            logger.info("[proactive] 无新信息且 only_new_items_trigger=true（无 presence），跳过本轮反思")
            return pre_score

        dc = d_content(len(new_items), self._cfg.score_content_halfsat)
        base_score = composite_score(
            de, dc, dr,
            self._cfg.score_weight_energy,
            self._cfg.score_weight_content,
            self._cfg.score_weight_recent,
        ) * interrupt_factor

        w_random = random_weight(rng=self._rng)
        draw_score = base_score * w_random
        session_key = self._target_session_key()
        target_last_user = self._presence.get_last_user_at(session_key) if self._presence and session_key else None
        force_reflect = (
            energy < 0.05
            or (self._presence is not None and bool(session_key) and target_last_user is None)
            or (energy < 0.20 and has_memory)
        )

        logger.info(
            "[proactive] base_score=%.3f  D_energy=%.3f D_content=%.3f D_recent=%.3f"
            "  interrupt=%.3f W_random=%.2f → draw_score=%.3f 阈值=%.2f force_reflect=%s",
            base_score, de, dc, dr, interruptibility, w_random, draw_score, self._cfg.score_llm_threshold, force_reflect,
        )

        if draw_score < self._cfg.score_llm_threshold and not force_reflect:
            logger.info("[proactive] draw_score 未过门槛，跳过本轮反思")
            return base_score
        if draw_score < self._cfg.score_llm_threshold and force_reflect:
            logger.info("[proactive] draw_score 未过门槛，但命中兜底条件，继续反思")

        # ── 第三阶段：LLM 反思 ────────────────────────────────────
        is_crisis = energy < 0.05
        decision = await self._reflect(
            new_items, recent,
            energy=energy, urge=draw_score,
            is_crisis=is_crisis,
        )
        logger.info(
            f"[proactive] score={decision.score:.2f}  "
            f"send={decision.should_send}  "
            f"reasoning={decision.reasoning[:80]!r}"
        )

        # 4. 阈值判断
        if decision.should_send and decision.score >= self._cfg.threshold:
            channel = (self._cfg.default_channel or "").strip()
            chat_id = self._cfg.default_chat_id.strip()
            session_key = f"{channel}:{chat_id}" if channel and chat_id else ""
            evidence_ids = _resolve_evidence_item_ids(decision, new_items if new_items else items)
            delivery_key = _build_delivery_key(evidence_ids, decision.message)
            logger.info(
                "[proactive] 发送前去重检查 session=%s evidence_count=%d delivery_key=%s",
                session_key or "（未配置）",
                len(evidence_ids),
                delivery_key[:16],
            )
            if session_key and self._state.is_delivery_duplicate(
                session_key=session_key,
                delivery_key=delivery_key,
                window_hours=self._cfg.delivery_dedupe_hours,
            ):
                logger.info("[proactive] 命中发送去重，跳过发送")
                self._state.mark_items_seen(new_entries)
                self._state.mark_semantic_items(_semantic_entries(new_items, self._cfg.semantic_dedupe_text_max_chars))
                logger.info("[proactive] 已按去重命中标记本轮条目为 seen（视为已送达过同等内容）")
                return base_score
            sent = await self._send(decision.message)
            if sent and session_key:
                self._state.mark_delivery(session_key, delivery_key)
                self._state.mark_items_seen(new_entries)
                self._state.mark_semantic_items(_semantic_entries(new_items, self._cfg.semantic_dedupe_text_max_chars))
                logger.info("[proactive] 已发送成功并标记本轮条目为 seen")
            else:
                logger.info("[proactive] 本轮发送未成功，不标记 seen，后续可再次尝试")
        else:
            logger.info("[proactive] 决定不主动发送")
            logger.info("[proactive] 本轮未发送，不标记 seen，后续可再次尝试")
        return base_score

    def _filter_new_items(
        self, items: list[FeedItem]
    ) -> tuple[list[FeedItem], list[tuple[str, str]], list[tuple[str, str]]]:
        if not items:
            logger.info("[proactive] 本轮无 item，去重过滤跳过")
            return [], [], []
        now = datetime.now(timezone.utc)
        source_fresh: list[FeedItem] = []
        source_entries: list[tuple[str, str]] = []
        for item in items:
            source_key = _source_key(item)
            item_id = _item_id(item)
            seen = self._state.is_item_seen(
                source_key=source_key,
                item_id=item_id,
                ttl_hours=self._cfg.dedupe_seen_ttl_hours,
                now=now,
            )
            logger.info(
                "[proactive] item 去重检查 source=%s item_id=%s seen=%s title=%r",
                source_key,
                item_id[:16],
                seen,
                (item.title or "")[:60],
            )
            if seen:
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
                "text": _semantic_text(item, self._cfg.semantic_dedupe_text_max_chars),
            }
            for item, (source_key, item_id) in zip(source_fresh, source_entries)
        ]
        if not payload:
            return [], [], []
        docs = [h["text"] for h in history] + [p["text"] for p in payload]
        vectors = _build_tfidf_vectors(docs, self._cfg.semantic_dedupe_ngram)
        history_vectors = vectors[: len(history)]
        payload_vectors = vectors[len(history):]

        keep_items: list[FeedItem] = []
        keep_entries: list[tuple[str, str]] = []
        duplicate_entries: list[tuple[str, str]] = []
        accepted_vectors: list[dict[str, float]] = []
        accepted_meta: list[dict[str, str]] = []
        threshold = self._cfg.semantic_dedupe_threshold
        for idx, p in enumerate(payload):
            vec = payload_vectors[idx]
            best_sim = 0.0
            best_kind = ""
            best_source = ""
            best_item_id = ""

            for h_idx, h_vec in enumerate(history_vectors):
                sim = _cosine_sparse(vec, h_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_kind = "history"
                    best_source = history[h_idx].get("source_key", "")
                    best_item_id = history[h_idx].get("item_id", "")

            for a_idx, a_vec in enumerate(accepted_vectors):
                sim = _cosine_sparse(vec, a_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_kind = "batch"
                    best_source = accepted_meta[a_idx].get("source_key", "")
                    best_item_id = accepted_meta[a_idx].get("item_id", "")

            logger.info(
                "[proactive] 语义去重检查 source=%s item_id=%s best_sim=%.4f threshold=%.2f matched_kind=%s matched_source=%s matched_item=%s title=%r",
                p["source_key"],
                p["item_id"][:16],
                best_sim,
                threshold,
                best_kind or "-",
                best_source or "-",
                (best_item_id or "-")[:16],
                (p["item"].title or "")[:80],
            )

            if best_sim >= threshold:
                duplicate_entries.append((p["source_key"], p["item_id"]))
                logger.info(
                    "[proactive] 语义去重命中，过滤 item source=%s item_id=%s sim=%.4f",
                    p["source_key"],
                    p["item_id"][:16],
                    best_sim,
                )
                continue

            keep_items.append(p["item"])
            keep_entries.append((p["source_key"], p["item_id"]))
            accepted_vectors.append(vec)
            accepted_meta.append({"source_key": p["source_key"], "item_id": p["item_id"]})
        logger.info(
            "[proactive] 语义去重结果 keep=%d duplicate=%d history_candidates=%d",
            len(keep_items),
            len(duplicate_entries),
            len(history),
        )
        return keep_items, keep_entries, duplicate_entries

    def _collect_recent(self) -> list[dict]:
        """取目标会话最近 N 条消息（只取 user/assistant 文本）。"""
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        if not channel or not chat_id:
            logger.info("[proactive] collect_recent 跳过：目标 channel/chat_id 未配置")
            return []
        key = f"{channel}:{chat_id}"

        try:
            session = self._sessions.get_or_create(key)
            msgs = session.messages[-self._cfg.recent_chat_messages:]
            logger.info(
                "[proactive] collect_recent 成功 key=%s total=%d selected=%d",
                key,
                len(session.messages),
                len(msgs),
            )
            return [
                {"role": m["role"], "content": str(m.get("content", ""))[:200]}
                for m in msgs
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
        except Exception as e:
            logger.warning(f"[proactive] 加载 session {key!r} 失败: {e}")
            return []

    async def _reflect(
        self,
        items: list[FeedItem],
        recent: list[dict],
        energy: float = 0.0,
        urge: float = 0.0,
        is_crisis: bool = False,
    ) -> _Decision:
        now_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
        feed_text = _format_items(items) or "（暂无订阅内容）"
        chat_text = _format_recent(recent) or "（无近期对话记录）"

        # 始终注入全量记忆——可靠助手的基础语境
        memory_text = self._collect_global_memory()

        # 待了解的问题列表（作为话题素材）
        questions_text = ""
        if self._memory:
            try:
                questions_text = self._memory.read_questions().strip()
            except Exception:
                pass

        # 危机模式：额外随机抽取一条记忆话题作为开场建议
        crisis_hint = ""
        if is_crisis:
            topic_chunks = self._sample_random_memory(n=1)
            topic_hint = topic_chunks[0] if topic_chunks else ""
            session_key = self._target_session_key()
            last_at = self._presence.get_last_user_at(session_key) if self._presence else None
            elapsed = ""
            if last_at:
                hours = (datetime.now(timezone.utc) - last_at).total_seconds() / 3600
                elapsed = f"距离上次对话已超过 {hours:.0f} 小时。"
            topic_section = (
                f"\n\n## 随机话题建议（危机开场用）\n\n{topic_hint}"
                if topic_hint else ""
            )
            crisis_hint = (
                f"\n[危机模式] {elapsed}"
                "用户可能已忘记你的存在，需要主动找一个自然的切入点重新联系。"
                "可以从下方随机话题出发，或用关心/有趣内容开场。"
                f"{topic_section}"
            )

        system_msg = (
            "你是一个陪伴型 AI 助手，正在决定是否主动联系用户。"
            "你了解用户订阅的信息流和最近的对话内容。"
            "你的目标是在恰当的时机分享有价值的信息，而不是频繁打扰用户。"
        )

        user_msg = f"""当前时间：{now_str}

## 主动性上下文

当前电量（与用户的互动新鲜度）: {energy:.2f}  (0=完全冷却, 1=刚刚对话)
主动冲动指数: {urge:.2f}  (0=不需要说, 1=非常需要联系){crisis_hint}

## 订阅信息流（最新内容）

{feed_text}

## 长期记忆（用户画像/偏好）

{memory_text}
{f"## 待了解的话题（可作为开场素材）\n\n{questions_text}\n" if questions_text else ""}
## 近期对话

{chat_text}

## 任务

综合以上信息，判断是否值得主动联系用户。考虑：
- 信息流里有没有用户可能感兴趣的内容
- 现在说点什么是否自然、不唐突
- 与近期对话有无关联或延伸
- 电量越低越需要主动联系，危机模式时哪怕简单关心也有价值

只输出 JSON，不要其他内容：
{{
  "reasoning": "内心独白（不会显示给用户，说清楚你的判断依据）",
  "score": 0.0,
  "should_send": false,
  "message": "",
  "evidence_item_ids": []
}}

score 说明：0.0=完全没必要  0.5=有点想说  0.7=比较值得  1.0=非常值得立刻说
message 若 should_send=true，写要发给用户的话（口语化，不要像系统通知）
evidence_item_ids 从订阅信息流里挑选支持你判断的 item_id（可为空数组）"""

        try:
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                tools=[],
                model=self._model,
                max_tokens=self._max_tokens,
            )
            content = resp.content or ""
            logger.info("[proactive] LLM 原始输出预览: %r", content[:240])
            return _parse_decision(content)
        except Exception as e:
            logger.error(f"[proactive] LLM 反思失败: {e}")
            return _Decision(score=0.0, should_send=False, message="", reasoning=str(e))

    def _collect_global_memory(self) -> str:
        if not self._cfg.use_global_memory:
            logger.info("[proactive] 全局记忆已禁用（use_global_memory=false）")
            return "（全局记忆已禁用）"
        if not self._memory:
            logger.info("[proactive] 未注入 MemoryStore，跳过全局记忆")
            return "（无全局记忆）"
        try:
            raw = self._memory.get_memory_context().strip()
            if not raw:
                logger.info("[proactive] 全局记忆为空")
                return "（无全局记忆）"
            text = raw[: max(self._cfg.global_memory_max_chars, 256)]
            logger.info(
                "[proactive] 已注入全局记忆 chars=%d truncated=%s",
                len(text),
                len(raw) > len(text),
            )
            return text
        except Exception as e:
            logger.warning("[proactive] 读取全局记忆失败: %s", e)
            return "（读取全局记忆失败）"

    async def _send(self, message: str) -> bool:
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        if not channel or not chat_id:
            logger.warning("[proactive] default_channel/default_chat_id 未配置，跳过发送")
            return False
        logger.info(
            "[proactive] 准备发送主动消息 channel=%s chat_id=%s message_len=%d",
            channel,
            chat_id,
            len(message),
        )
        try:
            result = await self._push.execute(
                channel=channel,
                chat_id=chat_id,
                message=message,
            )
            logger.info("[proactive] message_push 返回: %r", result[:200])
            if "已发送" not in result:
                logger.warning(f"[proactive] 发送未成功: {result}")
                return False
            key = f"{channel}:{chat_id}"
            session = self._sessions.get_or_create(key)
            session.add_message(
                "assistant",
                message,
                proactive=True,
                tools_used=["message_push"],
            )
            self._sessions.save(session)
            if self._presence:
                self._presence.record_proactive_sent(key)
            logger.info(f"[proactive] 已发送主动消息并写入会话 → {channel}:{chat_id}")
            return True
        except Exception as e:
            logger.error(f"[proactive] 发送失败: {e}")
            return False


# ── helpers ──────────────────────────────────────────────────────

def _format_items(items: list[FeedItem]) -> str:
    if not items:
        return ""
    lines = []
    for item in items:
        pub = ""
        if item.published_at:
            try:
                pub = " (" + item.published_at.astimezone().strftime("%m-%d %H:%M") + ")"
            except Exception:
                pass
        title = item.title or "(无标题)"
        lines.append(f"[{item.source_name}|item_id={_item_id(item)}]{pub} {title}")
        if item.content:
            lines.append(f"  {item.content[:200]}")
        if item.url:
            lines.append(f"  {item.url}")
    return "\n".join(lines)


def _format_recent(msgs: list[dict]) -> str:
    if not msgs:
        return ""
    lines = []
    for m in msgs[-10:]:   # 最多展示最近 10 条
        role = "用户" if m["role"] == "user" else "助手"
        content = str(m.get("content", ""))[:150]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_decision(text: str) -> _Decision:
    """从 LLM 输出中提取 JSON 决策。"""
    # 先尝试提取 ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    raw = match.group(1) if match else text

    # 找第一个完整的 { ... }
    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if not brace_match:
        logger.warning(f"[proactive] 无法提取 JSON: {text[:200]!r}")
        return _Decision(score=0.0, should_send=False, message="", reasoning="parse error")

    try:
        d = json.loads(brace_match.group())
        evidence = d.get("evidence_item_ids", [])
        if not isinstance(evidence, list):
            evidence = []
        return _Decision(
            score=float(d.get("score", 0.0)),
            should_send=_strict_bool(d.get("should_send", False)),
            message=str(d.get("message", "")),
            reasoning=str(d.get("reasoning", "")),
            evidence_item_ids=[str(x).strip() for x in evidence if str(x).strip()],
        )
    except Exception as e:
        logger.warning(f"[proactive] JSON 解析失败: {e}  raw={raw[:200]!r}")
        return _Decision(score=0.0, should_send=False, message="", reasoning=str(e))


def _strict_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
    return False


def _source_key(item: FeedItem) -> str:
    return f"{(item.source_type or '').strip().lower()}:{(item.source_name or '').strip().lower()}"


def _normalize_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        p = urlsplit(url.strip())
        scheme = (p.scheme or "").lower()
        netloc = (p.netloc or "").lower()
        path = p.path.rstrip("/")
        return urlunsplit((scheme, netloc, path, p.query, ""))
    except Exception:
        return (url or "").strip()


def _item_id(item: FeedItem) -> str:
    url = _normalize_url(item.url)
    if url:
        return "u_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    raw = "|".join([
        (item.source_type or "").strip().lower(),
        (item.source_name or "").strip().lower(),
        (item.title or "").strip().lower(),
        (item.content or "").strip().lower()[:200],
        item.published_at.isoformat() if item.published_at else "",
    ])
    return "h_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _resolve_evidence_item_ids(decision: _Decision, items: list[FeedItem]) -> list[str]:
    valid = {_item_id(i) for i in items}
    selected = [x for x in decision.evidence_item_ids if x in valid]
    if selected:
        return sorted(set(selected))
    fallback = sorted(valid)
    return fallback[:5]


def _build_delivery_key(item_ids: list[str], message: str) -> str:
    canonical_ids = "|".join(sorted(set(item_ids)))
    canonical_msg = re.sub(r"\s+", " ", (message or "").strip().lower())
    raw = f"{canonical_ids}::{canonical_msg}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _semantic_text(item: FeedItem, max_chars: int) -> str:
    title = (item.title or "").strip().lower()
    content = (item.content or "").strip().lower()
    merged = f"{title} {content}".strip()
    merged = re.sub(r"\s+", " ", merged)
    if not merged:
        merged = _normalize_url(item.url)
    return merged[: max(max_chars, 32)]


def _semantic_entries(items: list[FeedItem], max_chars: int) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for item in items:
        entries.append(
            {
                "source_key": _source_key(item),
                "item_id": _item_id(item),
                "text": _semantic_text(item, max_chars),
            }
        )
    return entries


def _char_ngrams(text: str, n: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not cleaned:
        return []
    n = max(1, n)
    if len(cleaned) <= n:
        return [cleaned]
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def _build_tfidf_vectors(texts: list[str], n: int) -> list[dict[str, float]]:
    if not texts:
        return []
    tokenized: list[list[str]] = [_char_ngrams(t, n) for t in texts]
    doc_freq: Counter[str] = Counter()
    for toks in tokenized:
        doc_freq.update(set(toks))

    total_docs = len(tokenized)
    vectors: list[dict[str, float]] = []
    for toks in tokenized:
        if not toks:
            vectors.append({})
            continue
        tf = Counter(toks)
        total = float(sum(tf.values()))
        vec: dict[str, float] = {}
        for tok, cnt in tf.items():
            idf = math.log((1.0 + total_docs) / (1.0 + doc_freq[tok])) + 1.0
            vec[tok] = (cnt / total) * idf
        vectors.append(vec)
    return vectors


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = 0.0
    for k, v in small.items():
        dot += v * large.get(k, 0.0)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
