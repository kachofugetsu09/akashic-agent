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
from proactive.energy import compute_energy, content_weight, next_tick_interval, random_weight, time_weight, urge_base
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
    # ── 动态电量 / 昼夜节律 ──
    energy_cool_threshold: float = 0.20   # 低于此值才开始计算冲动
    energy_crisis_threshold: float = 0.05 # 低于此值进入危机模式
    energy_min_urge: float = 0.10         # 低于此冲动值跳过 LLM
    quiet_hours_start: int = 23           # 静默开始（本地时间）
    quiet_hours_end: int = 8              # 静默结束（本地时间）
    quiet_hours_weight: float = 0.0      # 静默时段权重，0=完全不发，0.1=低概率仍可发
    tick_interval_high: int = 7200        # 电量高时间隔（秒）
    tick_interval_normal: int = 1800      # 正常间隔（秒）
    tick_interval_low: int = 900          # 电量低时间隔（秒）
    tick_interval_crisis: int = 600       # 危机模式间隔（秒）


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
        while self._running:
            interval = self._next_interval()
            logger.info("[proactive] 下次 tick 间隔=%ds", interval)
            await asyncio.sleep(interval)
            try:
                await self._tick()
            except Exception:
                logger.exception("ProactiveLoop tick 异常")

    def _next_interval(self) -> int:
        """根据当前电量返回自适应等待秒数。无 presence 时回退固定间隔。"""
        if not self._presence:
            return self._cfg.interval_seconds
        session_key = self._target_session_key()
        last_user_at = self._presence.get_last_user_at(session_key)
        energy = compute_energy(last_user_at)
        return next_tick_interval(
            energy,
            cool_threshold=self._cfg.energy_cool_threshold,
            crisis_threshold=self._cfg.energy_crisis_threshold,
            tick_high=self._cfg.tick_interval_high,
            tick_normal=self._cfg.tick_interval_normal,
            tick_low=self._cfg.tick_interval_low,
            tick_crisis=self._cfg.tick_interval_crisis,
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

    def _compute_energy_urge(self) -> tuple[float, float]:
        """计算目标 session 的当前电量和基础冲动值。"""
        if not self._presence:
            return 0.0, 1.0  # 无 presence 时视作完全放电，冲动最大
        session_key = self._target_session_key()
        # 取目标 session 与全局最近活跃中的较高值，防误判危机
        last_target = self._presence.get_last_user_at(session_key)
        last_global = self._presence.most_recent_user_at()
        energy_target = compute_energy(last_target)
        energy_global = compute_energy(last_global) * 0.6
        energy = max(energy_target, energy_global)
        urge = urge_base(energy, self._cfg.energy_cool_threshold)
        return energy, urge

    # ── internal ──────────────────────────────────────────────────

    async def _tick(self) -> None:
        logger.info("[proactive] tick 开始")
        self._state.cleanup(
            seen_ttl_hours=self._cfg.dedupe_seen_ttl_hours,
            delivery_ttl_hours=self._cfg.delivery_dedupe_hours,
            semantic_ttl_hours=max(self._cfg.dedupe_seen_ttl_hours, self._cfg.semantic_dedupe_window_hours),
        )

        # 0. 能量前置门控：电量高 → 冲动为 0 → 直接跳过，省 LLM 调用
        energy, urge = self._compute_energy_urge()
        now_hour = datetime.now().hour
        w_time = time_weight(now_hour, self._cfg.quiet_hours_start, self._cfg.quiet_hours_end, self._cfg.quiet_hours_weight)

        # 快速剪枝：冲动或时间权重为零 → 跳过，不拉 feed
        if self._presence and urge * w_time == 0.0:
            if urge == 0.0:
                logger.info("[proactive] 电量充足（%.2f），无需主动联系，跳过本轮", energy)
            else:
                logger.info("[proactive] 静默时段（%02d:xx），W_time=%.1f，跳过本轮", now_hour, w_time)
            return

        # 1. 并发拉取信息流
        items = await self._feeds.fetch_all(self._cfg.items_per_source)
        logger.info(f"[proactive] 拉取到 {len(items)} 条信息")
        new_items, new_entries, semantic_duplicate_entries = self._filter_new_items(items)
        logger.info(
            "[proactive] 去重后剩余新信息 %d 条（过滤重复 %d 条）",
            len(new_items),
            len(items) - len(new_items),
        )
        if semantic_duplicate_entries:
            self._state.mark_items_seen(semantic_duplicate_entries)
            logger.info(
                "[proactive] 已标记语义重复条目为 seen count=%d（避免跨源重复重试）",
                len(semantic_duplicate_entries),
            )

        # 2. W_content + W_random → 最终冲动
        # 无 presence 时不触发危机模式——内容门控依赖真实电量数据
        is_crisis = self._presence is not None and energy < self._cfg.energy_crisis_threshold
        has_memory = bool(self._memory and self._memory.read_long_term().strip())
        w_content = content_weight(
            new_items=len(new_items),
            has_memory=has_memory,
            is_crisis=is_crisis,
        )
        w_random = random_weight(rng=self._rng)
        effective_urge = urge * w_time * w_content * w_random
        logger.info(
            "[proactive] 电量=%.3f 冲动=%.3f W_time=%.1f W_content=%.2f W_random=%.2f → 最终冲动=%.3f 阈值=%.2f",
            energy, urge, w_time, w_content, w_random, effective_urge, self._cfg.energy_min_urge,
        )
        if effective_urge < self._cfg.energy_min_urge:
            logger.info("[proactive] 最终冲动不足，跳过本轮反思")
            return

        # 3. 最近聊天上下文
        recent = self._collect_recent()
        logger.info("[proactive] 最近会话消息条数=%d", len(recent))

        # 4. LLM 反思
        decision = await self._reflect(
            new_items, recent,
            energy=energy, urge=effective_urge,
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
                return
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
