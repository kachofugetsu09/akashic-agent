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

from agent.provider import LLMProvider
from agent.memory import MemoryStore
from agent.tools.message_push import MessagePushTool
from feeds.base import FeedItem
from feeds.buffer import FeedBuffer
from feeds.registry import FeedRegistry
from proactive.energy import (
    compute_energy,
    d_energy,
    next_tick_from_score,
)
from proactive.components import (
    ProactiveFeatureScorer,
    ProactiveItemFilter,
    ProactiveMessageComposer,
    ProactiveMessageDeduper,
    ProactiveReflector,
    ProactiveSender,
    ReflectHooks,
)
from proactive.anyaction import AnyActionGate, QuotaStore
from proactive.engine import ProactiveEngine
from proactive.item_id import compute_item_id, compute_source_key, normalize_url
from proactive.memory_sampler import sample_memory_chunks
from proactive.ports import DefaultDecidePort, DefaultSensePort
from proactive.presence import PresenceStore
from proactive.schedule import ScheduleStore
from proactive.state import ProactiveStateStore
from proactive.interest import InterestFilterConfig
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
    interest_filter: InterestFilterConfig = field(default_factory=InterestFilterConfig)
    # ── 多维打分 ──
    score_weight_energy: float = 0.40    # D_energy 权重（互动饥渴度）
    score_weight_content: float = 0.40   # D_content 权重（信息流新鲜度）
    score_weight_recent: float = 0.20    # D_recent 权重（对话语境丰富度）
    score_content_halfsat: float = 3.0   # D_content 半饱和点（新条目数）
    score_recent_scale: float = 10.0     # D_recent 消息数归一化基数
    score_llm_threshold: float = 0.40    # draw_score 超过此值才调 LLM
    score_pre_threshold: float = 0.05    # pre_score 低于此值直接跳过（省 feed 拉取）
    decision_score_random_strength: float = 0.0  # 最终发送分数随机扰动幅度（0=关闭）
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
    quiet_hours_end: int = 10             # 静默结束（本地时间）
    quiet_hours_weight: float = 0.0      # 静默时段权重，0=完全不发，0.1=低概率仍可发
    # ── tick 间隔（由 base_score 驱动）──
    tick_interval_s0: int = 4800         # base_score ≤ 0.20 → ~80 min
    tick_interval_s1: int = 2400         # base_score > 0.20 → ~40 min
    tick_interval_s2: int = 1080         # base_score > 0.40 → ~18 min
    tick_interval_s3: int = 420          # base_score > 0.70 → ~7 min
    tick_jitter: float = 0.3            # 随机抖动幅度，0.3=±30%，0=关闭
    # ── AnyAction 通用层（后台动作门控）──
    anyaction_enabled: bool = False
    anyaction_daily_max_actions: int = 24
    anyaction_min_interval_seconds: int = 300
    anyaction_reset_hour_local: int = 12
    anyaction_timezone: str = "Asia/Shanghai"
    anyaction_probability_min: float = 0.03
    anyaction_probability_max: float = 0.45
    anyaction_idle_scale_minutes: float = 240.0
    # ── AI特征打分 + 算法决策（可选）──
    feature_scoring_enabled: bool = False
    feature_send_threshold: float = 0.52
    feature_weight_topic_continuity: float = 0.24
    feature_weight_interest_match: float = 0.24
    feature_weight_content_novelty: float = 0.20
    feature_weight_reconnect_value: float = 0.16
    feature_weight_message_readiness: float = 0.16
    feature_weight_disturb_risk: float = 0.70
    feature_weight_interrupt_penalty: float = 0.30
    feature_weight_d_recent_bonus: float = 0.10
    feature_weight_d_content_bonus: float = 0.10
    feature_weight_d_energy_bonus: float = 0.08
    # ── 发送前消息语义去重 ──
    message_dedupe_enabled: bool = True   # 发送前用 LLM 检测是否与近期 proactive 消息重复
    message_dedupe_recent_n: int = 5      # 取最近 N 条 proactive 消息做比对
    # ── LLM 拒绝冷却 ──
    llm_reject_cooldown_hours: int = 12  # LLM 拒绝后的软冷却时长（小时）；0 = 禁用
    # ── Feed 轮询器（后台解耦拉取）──
    feed_poller_enabled: bool = False           # 启用后台 FeedPoller（解耦拉取与决策）
    feed_poller_interval_seconds: int = 300     # 拉取间隔（秒）
    feed_poller_fetch_limit: int = 20           # 每源拉取条数上限
    feed_poller_buffer_ttl_hours: int = 48      # buffer 条目有效期（小时）
    feed_poller_buffer_max_per_source: int = 100  # 每源最多保留条数
    feed_poller_read_limit: int = 50            # 引擎从 buffer 读取的条数上限（0=全部）
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


def _decision_with_randomized_score(
    decision: _Decision,
    *,
    strength: float,
    rng: _random_module.Random | None = None,
) -> tuple[_Decision, float]:
    """对最终发送分数注入随机扰动，返回新 decision 与扰动值。"""
    s = max(0.0, min(1.0, strength))
    if s <= 0:
        return decision, 0.0
    delta = (rng or _random_module).uniform(-s, s)
    score = max(0.0, min(1.0, decision.score + delta))
    return _Decision(
        score=score,
        should_send=decision.should_send,
        message=decision.message,
        reasoning=decision.reasoning,
        evidence_item_ids=decision.evidence_item_ids,
    ), delta


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
        schedule: ScheduleStore | None = None,
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
        self._schedule = schedule
        self._rng = rng
        self._running = False
        # FeedBuffer：feed_poller_enabled=True 时由 FeedPoller 写入；False 则为 None（直接拉取）
        self.feed_buffer: FeedBuffer | None = (
            FeedBuffer(
                ttl_hours=config.feed_poller_buffer_ttl_hours,
                max_per_source=config.feed_poller_buffer_max_per_source,
            )
            if config.feed_poller_enabled
            else None
        )
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
        logger.info(
            "[proactive] interest_filter enabled=%s min_score=%.2f top_k=%d explore=%.2f",
            self._cfg.interest_filter.enabled,
            self._cfg.interest_filter.min_score,
            self._cfg.interest_filter.top_k,
            self._cfg.interest_filter.exploration_ratio,
        )
        self._item_filter = ProactiveItemFilter(
            cfg=self._cfg,
            state=self._state,
            source_key_fn=_source_key,
            item_id_fn=_item_id,
            semantic_text_fn=_semantic_text,
            build_tfidf_vectors_fn=_build_tfidf_vectors,
            cosine_fn=_cosine_sparse,
        )
        self._reflector = ProactiveReflector(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
            cfg=self._cfg,
            memory_store=self._memory,
            presence=self._presence,
            hooks=ReflectHooks(
                format_items=_format_items,
                format_recent=_format_recent,
                parse_decision=_parse_decision,
                collect_global_memory=self._collect_global_memory,
                sample_random_memory=self._sample_random_memory,
                target_session_key=self._target_session_key,
                on_reflect_error=lambda e: _Decision(
                    score=0.0,
                    should_send=False,
                    message="",
                    reasoning=str(e),
                ),
            ),
        )
        self._sender = ProactiveSender(
            cfg=self._cfg,
            push_tool=self._push,
            sessions=self._sessions,
            presence=self._presence,
        )
        self._feature_scorer = ProactiveFeatureScorer(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
            format_items=_format_items,
            format_recent=_format_recent,
            collect_global_memory=self._collect_global_memory,
        )
        self._message_composer = ProactiveMessageComposer(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
            format_items=_format_items,
            format_recent=_format_recent,
            collect_global_memory=self._collect_global_memory,
        )
        quota_path = (self._state.path.parent / "proactive_quota.json") if hasattr(self._state, "path") else Path("proactive_quota.json")
        self._anyaction = AnyActionGate(
            cfg=self._cfg,
            quota_store=QuotaStore(quota_path),
            rng=self._rng,
        )
        self._sense = DefaultSensePort(
            cfg=self._cfg,
            feeds=self._feeds,
            sessions=self._sessions,
            state=self._state,
            item_filter=self._item_filter,
            memory=self._memory,
            presence=self._presence,
            schedule=self._schedule,
            rng=self._rng,
            feed_buffer=self.feed_buffer,
        )
        self._decide = DefaultDecidePort(
            reflector=self._reflector,
            randomize_fn=lambda decision: _decision_with_randomized_score(
                decision,
                strength=self._cfg.decision_score_random_strength,
                rng=self._rng,
            ),
            source_key_fn=_source_key,
            item_id_fn=_item_id,
            semantic_text_fn=_semantic_text,
            semantic_text_max_chars=self._cfg.semantic_dedupe_text_max_chars,
            feature_scorer=self._feature_scorer,
            message_composer=self._message_composer,
        )
        self._message_deduper = (
            ProactiveMessageDeduper(
                provider=self._provider,
                model=self._model,
                max_tokens=self._max_tokens,
            )
            if self._cfg.message_dedupe_enabled
            else None
        )
        self._engine = ProactiveEngine(
            cfg=self._cfg,
            state=self._state,
            presence=self._presence,
            rng=self._rng,
            sense=self._sense,
            decide=self._decide,
            act=self._sender,
            anyaction=self._anyaction,
            message_deduper=self._message_deduper,
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
        return self._sense.target_session_key()

    def _quiet_hours(self) -> tuple[int, int, float]:
        """从 schedule.json 读取静默时段配置，缺失时回退 cfg 默认值。"""
        return self._sense.quiet_hours()

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
        return self._sense.has_global_memory()

    def _read_memory_text(self) -> str:
        return self._sense.read_memory_text()

    def _compute_energy(self) -> float:
        """计算目标 session 的当前电量（取目标与全局较高值）。"""
        return self._sense.compute_energy()

    def _compute_interruptibility(
        self,
        *,
        now_hour: int,
        now_utc: datetime,
        recent_msg_count: int,
    ) -> tuple[float, dict[str, float]]:
        """计算软打扰系数（0~1），并注入随机探索，避免长期锁死。"""
        return self._sense.compute_interruptibility(
            now_hour=now_hour,
            now_utc=now_utc,
            recent_msg_count=recent_msg_count,
        )

    # ── internal ──────────────────────────────────────────────────

    async def _tick(self) -> float | None:
        """执行一次主动判断循环。
        返回 base_score 供调度器调整间隔；None 表示 gate 按能量自算（不强制最长间隔）。
        """
        return await self._engine.tick()

    def _filter_new_items(
        self, items: list[FeedItem]
    ) -> tuple[list[FeedItem], list[tuple[str, str]], list[tuple[str, str]]]:
        return self._sense.filter_new_items(items)

    def _collect_recent(self) -> list[dict]:
        """取目标会话最近 N 条消息（只取 user/assistant 文本）。"""
        return self._sense.collect_recent()

    async def _reflect(
        self,
        items: list[FeedItem],
        recent: list[dict],
        energy: float = 0.0,
        urge: float = 0.0,
        is_crisis: bool = False,
        decision_signals: dict[str, object] | None = None,
    ) -> _Decision:
        return await self._reflector.reflect(
            items=items,
            recent=recent,
            energy=energy,
            urge=urge,
            is_crisis=is_crisis,
            decision_signals=decision_signals,
        )

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
        return await self._sender.send(message)


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
    return compute_source_key(item)


def _item_id(item: FeedItem) -> str:
    return compute_item_id(item)


def _normalize_url(url: str | None) -> str:
    return normalize_url(url)


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
