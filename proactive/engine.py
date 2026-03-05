from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from feeds.base import FeedItem
from proactive.energy import (
    composite_score,
    d_content,
    d_energy,
    d_recent,
    random_weight,
)
from proactive.interest import select_interesting_items
from proactive.presence import PresenceStore
from proactive.anyaction import AnyActionGate
from proactive.ports import ActPort, DecidePort, SensePort
from proactive.state import ProactiveStateStore
from proactive.skill_action import SkillActionRunner

logger = logging.getLogger(__name__)

# Sentinel returned by stage methods to signal "stop tick, return None to caller"
_STOP_NONE: object = object()


class DecisionLike(Protocol):
    score: float
    should_send: bool
    message: str
    reasoning: str
    evidence_item_ids: list[str]


class _PseudoDecision:
    def __init__(self, message: str) -> None:
        self.message = message
        self.evidence_item_ids: list[str] = []


def _sleep_policy_note(state: str, available: bool, prob: float | None = None) -> str:
    if not available:
        return "fitbit_unavailable: 不调整 chat/idle 概率"
    if state == "sleeping":
        return "sleeping_protect: chat 概率×0.20，idle 概率显著上升"
    if state == "uncertain":
        if prob is not None and prob >= 0.60:
            return "uncertain_high_prob_protect: chat 概率×0.20（按睡眠保护），idle 概率显著上升"
        return "cautious: chat 概率×0.50，idle 概率上升"
    if state == "awake":
        return "normal: 不降低 chat 概率"
    return "unknown: chat 概率轻微下调(×0.88)"


# ---------------------------------------------------------------------------
# DecisionContext — carries state across tick() stages (replaces scattered locals)
# ---------------------------------------------------------------------------


@dataclass
class DecisionContext:
    """每轮 tick 的决策上下文，阶段间传递，禁止裸 dict 穿越边界。"""

    now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Stage 1 — gate
    gate_passed: bool = True

    # Stage 2 — sense
    sleep_ctx: Any = None
    health_events: list[dict] = field(default_factory=list)
    energy: float = 0.0
    now_hour: int = 0
    recent: list[dict] = field(default_factory=list)
    interruptibility: float = 1.0
    interrupt_detail: dict[str, float] = field(default_factory=dict)
    interrupt_factor: float = 1.0
    sleep_mod: float = 1.0
    de: float = 0.0  # d_energy
    dr: float = 0.0  # d_recent

    # Stage 3 — pre-score
    pre_score: float = 0.0

    # Stage 4 — fetch & filter
    items: list[FeedItem] = field(default_factory=list)
    new_items: list[FeedItem] = field(default_factory=list)
    new_entries: list[tuple[str, str]] = field(default_factory=list)
    semantic_duplicate_entries: list[tuple[str, str]] = field(default_factory=list)
    has_memory: bool = False

    # Stage 5 — score
    dc: float = 0.0  # d_content
    base_score: float = 0.0
    draw_score: float = 0.0
    force_reflect: bool = False
    session_key: str = ""
    target_last_user: datetime | None = None
    last_proactive_at: datetime | None = None
    is_crisis: bool = False
    fresh_items_24h: int = 0
    sent_24h: int = 0

    # Stage 6 — decide
    decision_signals: dict[str, object] = field(default_factory=dict)
    feature_payload: dict[str, float | str] = field(default_factory=dict)
    feature_final_score: float | None = None
    decision: Any = None
    decision_message: str = ""
    should_send: bool = False

    # Stage 7 — act
    high_events: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------


class ProactiveEngine:
    """主动循环引擎：编排一次完整 tick，不直接依赖 ProactiveLoop。"""

    def __init__(
        self,
        *,
        cfg: Any,
        state: ProactiveStateStore,
        presence: PresenceStore | None,
        rng: Any,
        sense: SensePort,
        decide: DecidePort,
        act: ActPort,
        anyaction: AnyActionGate | None = None,
        message_deduper: Any | None = None,
        skill_action_runner: SkillActionRunner | None = None,
        light_provider: Any | None = None,
        light_model: str = "",
        passive_busy_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._presence = presence
        self._rng = rng
        self._sense = sense
        self._decide = decide
        self._act = act
        self._anyaction = anyaction
        self._message_deduper = message_deduper
        self._skill_action_runner = skill_action_runner
        self._light_provider = light_provider
        self._light_model = light_model
        # 可选：AgentLoop 注入的被动处理信号，用于跳过与被动回复并发的主动发送
        self._passive_busy_fn = passive_busy_fn

    async def tick(self) -> float | None:
        """执行一次主动判断循环。

        返回 base_score（float）供调度器调整下次 tick 间隔；
        返回 None 表示 gate 以 min_interval/probability 拒绝，由调度器按能量自算间隔。

        阶段：
          1. gate — 清理状态 + anyaction gate（早退）
          2. sense — 采集环境信号
          3. pre_score — 预评分，过低则尝试 skill action 后早退
          4. fetch_filter — 拉取并去重 feed 条目
          5. score — 计算 base_score / draw_score，未过门槛则早退
          6. decide — 构建 decision_signals，LLM 评分/生成
          7. act — 发送 + 状态标记
        """
        logger.info("[proactive] tick 开始")
        ctx = DecisionContext()

        # 1. gate
        gate_result = await self._stage_gate(ctx)
        if gate_result is _STOP_NONE:
            return None
        if gate_result is not None:
            return gate_result  # type: ignore[return-value]

        # 2. sense
        self._stage_sense(ctx)

        # 3. pre-score
        pre_result = await self._stage_pre_score(ctx)
        if pre_result is not None:
            return pre_result

        # 4. fetch & filter
        await self._stage_fetch_filter(ctx)

        # 5. score
        score_result = await self._stage_score(ctx)
        if score_result is not None:
            return score_result

        # 6. decide
        decide_result = await self._stage_decide(ctx)
        if decide_result is not None:
            return decide_result

        # 7. act
        await self._stage_act(ctx)
        return ctx.base_score

    # ------------------------------------------------------------------
    # Stage 1 — gate: state cleanup + anyaction gate
    # ------------------------------------------------------------------

    async def _stage_gate(self, ctx: DecisionContext) -> float | object | None:
        """清理过期状态；若 anyaction gate 拒绝则提前返回退出码。

        返回值：
          None         — gate 通过，继续后续阶段
          0.0          — quota_exhausted，tick 应返回 0.0
          _STOP_NONE   — min_interval/probability 拒绝，tick 应返回 None
        """
        # 1. 清理过期条目
        self._state.cleanup(
            seen_ttl_hours=self._cfg.dedupe_seen_ttl_hours,
            delivery_ttl_hours=self._cfg.delivery_dedupe_hours,
            semantic_ttl_hours=max(
                self._cfg.dedupe_seen_ttl_hours, self._cfg.semantic_dedupe_window_hours
            ),
            rejection_cooldown_ttl_hours=getattr(
                self._cfg, "llm_reject_cooldown_hours", 0
            ),
        )

        # 2. anyaction gate 判定
        if not (self._cfg.anyaction_enabled and self._anyaction):
            return None

        should_act, meta = self._anyaction.should_act(
            now_utc=ctx.now_utc,
            last_user_at=self._sense.last_user_at(),
        )
        if not should_act:
            logger.info(
                "[proactive] gate_result=reject selected_action=null meta=%s", meta
            )
            reason = meta.get("reason", "")
            if reason == "quota_exhausted":
                return 0.0  # 今日配额已满，用最长间隔（tick_s0）
            return _STOP_NONE  # pyright: ignore[reportReturnType]  # min_interval / probability：调度器按能量自算

        logger.info("[proactive] gate_result=pass meta=%s", meta)
        return None

    # ------------------------------------------------------------------
    # Stage 2 — sense: collect environment signals
    # ------------------------------------------------------------------

    def _stage_sense(self, ctx: DecisionContext) -> None:
        """采集睡眠上下文、能量、打扰度等环境信号，填充 ctx。"""
        # 1. 刷新睡眠上下文
        refreshed = bool(getattr(self._sense, "refresh_sleep_context", lambda: False)())
        if refreshed:
            logger.debug("[proactive] fitbit 上下文已在本轮决策前主动刷新")
        ctx.sleep_ctx = getattr(self._sense, "sleep_context", lambda: None)()

        # 2. 提取健康事件
        health_events: list[dict] = (
            getattr(ctx.sleep_ctx, "health_events", [])
            if ctx.sleep_ctx is not None
            else []
        )
        ctx.health_events = health_events if isinstance(health_events, list) else []

        # 3. 采集能量 & 打扰度
        ctx.energy = self._sense.compute_energy()
        ctx.now_hour = datetime.now().hour
        ctx.recent = self._sense.collect_recent()
        ctx.de = d_energy(ctx.energy)
        ctx.dr = d_recent(len(ctx.recent), self._cfg.score_recent_scale)
        ctx.interruptibility, ctx.interrupt_detail = (
            self._sense.compute_interruptibility(
                now_hour=ctx.now_hour,
                now_utc=ctx.now_utc,
                recent_msg_count=len(ctx.recent),
            )
        )
        ctx.interrupt_factor = 0.6 + 0.4 * ctx.interruptibility

        # 4. Fitbit 睡眠修正
        ctx.sleep_mod = (
            ctx.sleep_ctx.sleep_modifier if ctx.sleep_ctx is not None else 1.0
        )
        if ctx.sleep_mod != 1.0:
            ctx.interrupt_factor *= ctx.sleep_mod

        sleep_state = (
            ctx.sleep_ctx.state if ctx.sleep_ctx is not None else "unavailable"
        )
        sleep_available = bool(
            ctx.sleep_ctx is not None and getattr(ctx.sleep_ctx, "available", False)
        )
        logger.info(
            "[proactive][sleep-policy] state=%s available=%s prob=%s lag=%s sleep_mod=%.2f policy=%s",
            sleep_state,
            sleep_available,
            (ctx.sleep_ctx.prob if ctx.sleep_ctx is not None else None),
            (ctx.sleep_ctx.data_lag_min if ctx.sleep_ctx is not None else None),
            ctx.sleep_mod,
            _sleep_policy_note(
                sleep_state,
                sleep_available,
                (ctx.sleep_ctx.prob if ctx.sleep_ctx is not None else None),
            ),
        )

    # ------------------------------------------------------------------
    # Stage 3 — pre-score: quick score before fetching items
    # ------------------------------------------------------------------

    async def _stage_pre_score(self, ctx: DecisionContext) -> float | None:
        """计算 pre_score；过低时尝试 skill action 后提前返回。"""
        # 1. 计算加权 pre_score
        w_sum = self._cfg.score_weight_energy + self._cfg.score_weight_recent
        ctx.pre_score = (
            (
                self._cfg.score_weight_energy * ctx.de
                + self._cfg.score_weight_recent * ctx.dr
            )
            / w_sum
            if w_sum > 0
            else 0.0
        ) * ctx.interrupt_factor

        logger.info(
            "[proactive] pre_score=%.3f interrupt=%.3f factor=%.3f sleep_mod=%.2f"
            " (reply=%.2f activity=%.2f fatigue=%.2f rand=%+.2f)"
            " D_energy=%.3f D_recent=%.3f energy=%.3f msg_count=%d",
            ctx.pre_score,
            ctx.interruptibility,
            ctx.interrupt_factor,
            ctx.sleep_mod,
            ctx.interrupt_detail["f_reply"],
            ctx.interrupt_detail["f_activity"],
            ctx.interrupt_detail["f_fatigue"],
            ctx.interrupt_detail["random_delta"],
            ctx.de,
            ctx.dr,
            ctx.energy,
            len(ctx.recent),
        )

        high_events = [e for e in ctx.health_events if e.get("severity") == "high"]
        if high_events:
            ctx.force_reflect = True
            if ctx.pre_score < self._cfg.score_pre_threshold:
                logger.info(
                    "[proactive] health fast-path: pre_score=%.3f 低于阈值但存在 %d 个 high 事件，强制继续",
                    ctx.pre_score,
                    len(high_events),
                )
                return None

        # 2. 过低则跳过 chat，尝试 skill action
        if ctx.pre_score < self._cfg.score_pre_threshold:
            logger.info(
                "[proactive] pre_score 过低（%.3f < %.2f），跳过 chat，尝试 skill action",
                ctx.pre_score,
                self._cfg.score_pre_threshold,
            )
            await self._try_skill_action(now_utc=ctx.now_utc)
            return ctx.pre_score

        return None

    # ------------------------------------------------------------------
    # Stage 4 — fetch & filter: pull items, deduplicate, interest filter
    # ------------------------------------------------------------------

    async def _stage_fetch_filter(self, ctx: DecisionContext) -> None:
        """拉取 feed 条目，去重，应用兴趣筛选，填充 ctx.new_items 等。"""
        # 1. 拉取原始条目
        ctx.items = await self._sense.fetch_items(self._cfg.items_per_source)
        logger.info("[proactive] 拉取到 %d 条信息", len(ctx.items))

        # 2. 基础去重（seen / semantic）
        ctx.new_items, ctx.new_entries, ctx.semantic_duplicate_entries = (
            self._sense.filter_new_items(ctx.items)
        )
        logger.info(
            "[proactive] 去重后剩余新信息 %d 条（过滤重复 %d 条）",
            len(ctx.new_items),
            len(ctx.items) - len(ctx.new_items),
        )
        if ctx.semantic_duplicate_entries:
            logger.info(
                "[proactive] 语义重复条目 count=%d 不写入 seen_items，72h 窗口自然抑制",
                len(ctx.semantic_duplicate_entries),
            )

        # 3. memory 兴趣筛选
        if self._cfg.interest_filter.enabled and ctx.new_items:
            memory_text = self._sense.read_memory_text()
            if memory_text:
                filtered_items, ranked = select_interesting_items(
                    ctx.new_items, memory_text, self._cfg.interest_filter
                )
                keep_ids = {self._decide.item_id_for(item) for item in filtered_items}
                old_count = len(ctx.new_items)
                ctx.new_items = filtered_items
                ctx.new_entries = [
                    (source_key, item_id)
                    for source_key, item_id in ctx.new_entries
                    if item_id in keep_ids
                ]
                top_preview = ", ".join(
                    f"{(pair[0].title or '')[:28]}:{pair[1]:.2f}" for pair in ranked[:3]
                )
                logger.info(
                    "[proactive] memory 兴趣筛选 old=%d kept=%d min_score=%.2f top=%s",
                    old_count,
                    len(ctx.new_items),
                    self._cfg.interest_filter.min_score,
                    top_preview or "-",
                )
            else:
                logger.info("[proactive] memory 兴趣筛选跳过：memory 为空")

        # 4. 全局记忆检查（影响 force_reflect 判断）
        ctx.has_memory = self._sense.has_global_memory()

    # ------------------------------------------------------------------
    # Stage 5 — score: compute base_score / draw_score
    # ------------------------------------------------------------------

    async def _stage_score(self, ctx: DecisionContext) -> float | None:
        """计算 base_score / draw_score；未过门槛时提前返回。"""
        # 1. only_new_items_trigger 早退
        if (
            self._cfg.only_new_items_trigger
            and not ctx.new_items
            and not self._presence
            and not ctx.has_memory
        ):
            logger.info(
                "[proactive] 无新信息且 only_new_items_trigger=true（无 presence），跳过本轮反思"
            )
            logger.info("[proactive] selected_action=idle reason=no_new_items")
            return ctx.pre_score

        # 2. 计算 base_score / draw_score
        ctx.dc = d_content(len(ctx.new_items), self._cfg.score_content_halfsat)
        ctx.base_score = (
            composite_score(
                ctx.de,
                ctx.dc,
                ctx.dr,
                self._cfg.score_weight_energy,
                self._cfg.score_weight_content,
                self._cfg.score_weight_recent,
            )
            * ctx.interrupt_factor
        )
        w_random = random_weight(rng=self._rng)
        ctx.draw_score = ctx.base_score * w_random

        # 3. 采集 presence 信号
        ctx.session_key = self._sense.target_session_key()
        ctx.target_last_user = (
            self._presence.get_last_user_at(ctx.session_key)
            if self._presence and ctx.session_key
            else None
        )
        ctx.last_proactive_at = (
            self._presence.get_last_proactive_at(ctx.session_key)
            if self._presence and ctx.session_key
            else None
        )
        ctx.force_reflect = ctx.force_reflect or (
            ctx.energy < 0.05
            or (
                self._presence is not None
                and bool(ctx.session_key)
                and ctx.target_last_user is None
            )
            or (ctx.energy < 0.20 and ctx.has_memory)
        )
        ctx.is_crisis = ctx.energy < 0.05
        ctx.sent_24h = (
            self._state.count_deliveries_in_window(ctx.session_key, 24, now=ctx.now_utc)
            if ctx.session_key
            else 0
        )
        ctx.fresh_items_24h = sum(
            1
            for item in ctx.new_items
            if item.published_at
            and (ctx.now_utc - item.published_at).total_seconds() <= 24 * 3600
        )

        logger.info(
            "[proactive] base_score=%.3f  D_energy=%.3f D_content=%.3f D_recent=%.3f"
            "  interrupt=%.3f W_random=%.2f → draw_score=%.3f 阈值=%.2f force_reflect=%s",
            ctx.base_score,
            ctx.de,
            ctx.dc,
            ctx.dr,
            ctx.interruptibility,
            w_random,
            ctx.draw_score,
            self._cfg.score_llm_threshold,
            ctx.force_reflect,
        )

        # 4. 判断是否过 LLM 门槛
        if ctx.draw_score < self._cfg.score_llm_threshold and not ctx.force_reflect:
            logger.info("[proactive] draw_score 未过门槛，跳过本轮反思")
            logger.info("[proactive] selected_action=idle reason=draw_score")
            await self._try_skill_action(now_utc=ctx.now_utc)
            return ctx.base_score
        if ctx.draw_score < self._cfg.score_llm_threshold and ctx.force_reflect:
            logger.info("[proactive] draw_score 未过门槛，但命中兜底条件，继续反思")

        return None

    # ------------------------------------------------------------------
    # Stage 6 — decide: build decision_signals, run LLM scoring/compose
    # ------------------------------------------------------------------

    async def _stage_decide(self, ctx: DecisionContext) -> float | None:
        """构建决策信号，调用 LLM 评分或生成消息，填充 ctx.should_send / decision_message。"""
        # 1. 构建 decision_signals 结构
        mins_since_last_user = (
            int((ctx.now_utc - ctx.target_last_user).total_seconds() / 60)
            if ctx.target_last_user
            else None
        )
        mins_since_last_proactive = (
            int((ctx.now_utc - ctx.last_proactive_at).total_seconds() / 60)
            if ctx.last_proactive_at
            else None
        )
        replied_after_last_proactive = bool(
            ctx.target_last_user
            and ctx.last_proactive_at
            and ctx.target_last_user > ctx.last_proactive_at
        )
        ctx.decision_signals = {
            "minutes_since_last_user": mins_since_last_user,
            "minutes_since_last_proactive": mins_since_last_proactive,
            "user_replied_after_last_proactive": replied_after_last_proactive,
            "proactive_sent_24h": ctx.sent_24h,
            "interruptibility": round(ctx.interruptibility, 3),
            "interrupt_breakdown": {
                "reply": round(ctx.interrupt_detail["f_reply"], 3),
                "activity": round(ctx.interrupt_detail["f_activity"], 3),
                "fatigue": round(ctx.interrupt_detail["f_fatigue"], 3),
            },
            "scores": {
                "pre_score": round(ctx.pre_score, 3),
                "base_score": round(ctx.base_score, 3),
                "draw_score": round(ctx.draw_score, 3),
                "llm_threshold": round(self._cfg.score_llm_threshold, 3),
                "send_threshold": round(self._cfg.threshold, 3),
            },
            "candidate_items": len(ctx.new_items),
            "fresh_items_24h": ctx.fresh_items_24h,
        }
        # 只有 StatEngine 检测到事件时才向 LLM 暴露健康信息
        if ctx.health_events:
            ctx.decision_signals["health_events"] = ctx.health_events
        ctx.high_events = [
            e for e in ctx.health_events if str((e or {}).get("severity", "")) == "high"
        ]
        logger.info(
            "[proactive] fitbit_signal events=%d high=%d sleep_state=%s",
            len(ctx.health_events),
            len(ctx.high_events),
            ctx.sleep_ctx.state if ctx.sleep_ctx is not None else "unavailable",
        )

        # 2. feature scoring 路径
        if self._cfg.feature_scoring_enabled:
            features = await self._decide.score_features(
                items=ctx.new_items,
                recent=ctx.recent,
                decision_signals=ctx.decision_signals,
            )
            ctx.feature_payload = features or {}
            feature_final_base = _feature_final_score(
                cfg=self._cfg,
                features=ctx.feature_payload,
                de=ctx.de,
                dc=ctx.dc,
                dr=ctx.dr,
                interruptibility=ctx.interruptibility,
            )
            health_bonus = _health_priority_bonus(
                alert_count=len(ctx.health_events),
                notify_count=len(ctx.high_events),
            )
            ctx.feature_final_score = min(1.0, feature_final_base + health_bonus)
            logger.info(
                "[proactive] feature_score enabled base=%.3f health_bonus=%.3f final=%.3f threshold=%.3f features=%s",
                feature_final_base,
                health_bonus,
                ctx.feature_final_score,
                self._cfg.feature_send_threshold,
                ctx.feature_payload,
            )
            if ctx.feature_final_score < self._cfg.feature_send_threshold:
                logger.info("[proactive] selected_action=idle reason=feature_score")
                # Fix C：feature_score 拒绝 = LLM 已评估，写拒绝冷却防止短期重入
                self._state.mark_rejection_cooldown(
                    ctx.new_entries,
                    hours=getattr(self._cfg, "llm_reject_cooldown_hours", 0),
                )
                return ctx.base_score

        # 3. LLM 生成/反思
        if self._cfg.feature_scoring_enabled:
            ctx.decision_message = await self._decide.compose_message(
                items=ctx.new_items,
                recent=ctx.recent,
                decision_signals=ctx.decision_signals,
            )
            ctx.should_send = bool(ctx.decision_message.strip()) and (
                ctx.feature_final_score is not None
                and ctx.feature_final_score >= self._cfg.feature_send_threshold
            )
            logger.info(
                "[proactive] feature_mode compose_len=%d should_send=%s reasons={topic:%r,interest:%r,novel:%r,reconnect:%r,disturb:%r,readiness:%r,conf:%r}",
                len(ctx.decision_message),
                ctx.should_send,
                ctx.feature_payload.get("topic_continuity_reason", ""),
                ctx.feature_payload.get("interest_match_reason", ""),
                ctx.feature_payload.get("content_novelty_reason", ""),
                ctx.feature_payload.get("reconnect_value_reason", ""),
                ctx.feature_payload.get("disturb_risk_reason", ""),
                ctx.feature_payload.get("message_readiness_reason", ""),
                ctx.feature_payload.get("confidence_reason", ""),
            )
        else:
            ctx.decision = await self._decide.reflect(
                ctx.new_items,
                ctx.recent,
                energy=ctx.energy,
                urge=ctx.draw_score,
                is_crisis=ctx.is_crisis,
                decision_signals=ctx.decision_signals,
            )
            ctx.decision, decision_delta = self._decide.randomize_decision(ctx.decision)
            logger.info(
                f"[proactive] score={ctx.decision.score:.2f}  "
                f"score_delta={decision_delta:+.2f}  "
                f"send={ctx.decision.should_send}  "
                f"reasoning={ctx.decision.reasoning[:80]!r}"
            )
            ctx.should_send = (
                ctx.decision.should_send and ctx.decision.score >= self._cfg.threshold
            )
            ctx.decision_message = ctx.decision.message

        return None

    # ------------------------------------------------------------------
    # Stage 7 — act: dedup checks, send, mark state
    # ------------------------------------------------------------------

    async def _stage_act(self, ctx: DecisionContext) -> None:
        """去重检查、发送消息、标记已发送状态。"""
        if not ctx.should_send:
            logger.info("[proactive] 决定不主动发送")
            logger.info("[proactive] 本轮未发送，不标记 seen，后续可再次尝试")
            # Fix C：LLM 已评估并拒绝，写拒绝冷却防止短期内重复进 LLM
            self._state.mark_rejection_cooldown(
                ctx.new_entries,
                hours=getattr(self._cfg, "llm_reject_cooldown_hours", 0),
            )
            await self._try_skill_action(now_utc=ctx.now_utc)
            return

        # 1. 解析证据 & 构建 delivery_key
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        ctx.session_key = f"{channel}:{chat_id}" if channel and chat_id else ""
        evidence_ids = self._decide.resolve_evidence_item_ids(
            ctx.decision
            if ctx.decision is not None
            else _PseudoDecision(message=ctx.decision_message),
            ctx.new_items,
        )
        delivery_key = self._decide.build_delivery_key(
            evidence_ids, ctx.decision_message
        )
        logger.info(
            "[proactive] 发送前去重检查 session=%s evidence_count=%d delivery_key=%s",
            ctx.session_key or "（未配置）",
            len(evidence_ids),
            delivery_key[:16],
        )

        # 2. delivery 去重
        if ctx.session_key and self._state.is_delivery_duplicate(
            session_key=ctx.session_key,
            delivery_key=delivery_key,
            window_hours=self._cfg.delivery_dedupe_hours,
        ):
            logger.info("[proactive] 命中发送去重，跳过发送")
            self._state.mark_items_seen(ctx.new_entries)
            self._state.mark_semantic_items(
                self._decide.semantic_entries(ctx.new_items)
            )
            logger.info(
                "[proactive] 已按去重命中标记本轮条目为 seen（视为已送达过同等内容）"
            )
            logger.info("[proactive] selected_action=idle reason=delivery_dedupe")
            return

        # 3. message 语义去重
        if self._message_deduper is not None:
            recent_proactive = self._sense.collect_recent_proactive(
                getattr(self._cfg, "message_dedupe_recent_n", 5)
            )
            is_dup, dup_reason = await self._message_deduper.is_duplicate(
                ctx.decision_message, recent_proactive
            )
            if is_dup:
                logger.info(
                    "[proactive] 消息语义去重命中，跳过发送 reason=%r", dup_reason
                )
                logger.info("[proactive] selected_action=idle reason=message_dedupe")
                # Fix B：message_dedupe 误判率较高，只写 semantic_items（72h 软抑制）
                self._state.mark_semantic_items(
                    self._decide.semantic_entries(ctx.new_items)
                )
                await self._try_skill_action(now_utc=ctx.now_utc)
                return

        # 4. 被动处理并发检查
        if (
            ctx.session_key
            and self._passive_busy_fn
            and self._passive_busy_fn(ctx.session_key)
        ):
            logger.info(
                "[proactive] 目标会话 %s 正在处理被动回复，跳过本轮发送"
                " selected_action=idle reason=passive_busy",
                ctx.session_key,
            )
            return

        # 5. 发送
        sent = await self._act.send(ctx.decision_message)
        if sent:
            # 配额记录：动作成功即消耗，与 session_key 无关
            if self._cfg.anyaction_enabled and self._anyaction:
                self._anyaction.record_action(now_utc=ctx.now_utc)
            # item 标记：全局去重，不依赖 session
            self._state.mark_items_seen(ctx.new_entries)
            self._state.mark_semantic_items(
                self._decide.semantic_entries(ctx.new_items)
            )
            # delivery 去重：仅 chat 类 action 有 session_key
            if ctx.session_key:
                self._state.mark_delivery(ctx.session_key, delivery_key)
            # 健康事件 ACK：发送成功后消费本轮全部事件（high + medium）
            if ctx.health_events:
                acked_ids = [
                    e["id"]
                    for e in ctx.health_events
                    if isinstance(e, dict) and e.get("id")
                ]
                if acked_ids:
                    getattr(self._sense, "acknowledge_health_events", lambda _: None)(
                        acked_ids
                    )
                    logger.info(
                        "[proactive] acknowledged %d 健康事件 ids=%s",
                        len(acked_ids),
                        acked_ids,
                    )
            logger.debug("[proactive] 已发送成功并标记本轮条目为 seen")
            logger.debug("[proactive] selected_action=chat")
        else:
            logger.info("[proactive] 本轮发送未成功，不标记 seen，后续可再次尝试")
            logger.info("[proactive] selected_action=idle reason=send_failed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _try_skill_action(self, *, now_utc: datetime) -> None:
        """在 chat idle 时，尝试从注册的 skill actions 中随机抽取并执行一个。"""
        if not self._skill_action_runner:
            logger.info("[proactive] selected_action=idle reason=decision")
            return
        avail = self._skill_action_runner.available_count()
        logger.info("[proactive] skill_action 可用数量=%d", avail)
        action = self._skill_action_runner.pick()
        if action is None:
            logger.info(
                "[proactive] selected_action=idle reason=decision "
                "(无可用 skill action，配额已满或间隔未到)"
            )
            return
        logger.info(
            "[proactive] selected_action=skill_action id=%s name=%r",
            action.id,
            action.name,
        )
        # 消耗 anyaction 配额（与 chat 共享）
        if self._cfg.anyaction_enabled and self._anyaction:
            self._anyaction.record_action(now_utc=now_utc)
        success, stdout_str = await self._skill_action_runner.run(action)
        logger.info(
            "[proactive] skill_action 完成 id=%s success=%s",
            action.id,
            success,
        )
        # 成功后从 stdout JSON 里读 proactive_text 并直接发送
        if success and stdout_str:
            await self._try_send_proactive_text(action.id, stdout_str)

    async def _try_send_proactive_text(self, action_id: str, stdout_str: str) -> None:
        """解析 skill action stdout，若有 proactive_text 字段则直接发送。"""
        try:
            data = json.loads(stdout_str)
        except Exception:
            logger.debug(
                "[proactive] skill_reaction: stdout 非 JSON，跳过 id=%s", action_id
            )
            return
        proactive_text = (data.get("proactive_text") or "").strip()
        if not proactive_text:
            logger.debug(
                "[proactive] skill_reaction: 无 proactive_text 字段，跳过 id=%s",
                action_id,
            )
            return
        logger.info(
            "[proactive] skill_reaction 发送 proactive_text id=%s chars=%d",
            action_id,
            len(proactive_text),
        )
        try:
            await self._act.send(proactive_text)
        except Exception as e:
            logger.warning(
                "[proactive] skill_reaction: 发送失败 id=%s error=%s", action_id, e
            )


def _feature_final_score(
    *,
    cfg: Any,
    features: dict[str, float | str],
    de: float,
    dc: float,
    dr: float,
    interruptibility: float,
) -> float:
    def f(name: str, default: float = 0.5) -> float:
        try:
            val = float(features.get(name, default))
        except Exception:
            val = default
        return max(0.0, min(1.0, val))

    topic_continuity = f("topic_continuity")
    interest_match = f("interest_match")
    content_novelty = f("content_novelty")
    reconnect_value = f("reconnect_value")
    disturb_risk = f("disturb_risk")
    message_readiness = f("message_readiness")
    confidence = f("confidence")

    utility = (
        cfg.feature_weight_topic_continuity * topic_continuity
        + cfg.feature_weight_interest_match * interest_match
        + cfg.feature_weight_content_novelty * content_novelty
        + cfg.feature_weight_reconnect_value * reconnect_value
        + cfg.feature_weight_message_readiness * message_readiness
    )
    risk = (
        cfg.feature_weight_disturb_risk * disturb_risk
        + cfg.feature_weight_interrupt_penalty
        * (1.0 - max(0.0, min(1.0, interruptibility)))
    )
    system_bonus = (
        cfg.feature_weight_d_recent_bonus * dr
        + cfg.feature_weight_d_content_bonus * dc
        + cfg.feature_weight_d_energy_bonus * de
    )
    raw = utility - risk + system_bonus
    conf_adjusted = raw * (0.7 + 0.3 * confidence)
    return max(0.0, min(1.0, conf_adjusted))


def _health_priority_bonus(*, alert_count: int, notify_count: int) -> float:
    """健康信号优先级加分：仅高优先级 notify 才加分，普通 alert 不影响决策。"""
    if notify_count > 0:
        return min(0.08, 0.04 + 0.02 * min(notify_count, 2))
    return 0.0
