from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from feeds.base import FeedItem
from proactive.item_id import compute_item_id, compute_source_key
from proactive.energy import (
    composite_score,
    d_content,
    d_energy,
    d_recent,
    random_weight,
)
from proactive.interest import select_interesting_items
from proactive.json_utils import extract_json_object
from proactive.presence import PresenceStore
from proactive.anyaction import AnyActionGate
from proactive.ports import (
    ActPort,
    DecidePort,
    MemoryRetrievalPort,
    ProactiveRetrievedMemory,
    ProactiveSendMeta,
    ProactiveSourceRef,
    RecentProactiveMessage,
    SensePort,
)
from proactive.state import ProactiveStateStore
from proactive.skill_action import SkillActionRunner
from core.common.strategy_trace import build_strategy_trace_envelope

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


def _json_safe(value: Any) -> Any:
    if value is _STOP_NONE:
        return "STOP_NONE"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


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
# DecisionContext — carries state across tick() stages via engine state + snapshots
# ---------------------------------------------------------------------------


@dataclass
class EngineState:
    now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_key: str = ""
    target_last_user: datetime | None = None
    last_proactive_at: datetime | None = None


@dataclass
class SenseSnapshot:
    sleep_ctx: Any = None
    health_events: list[dict] = field(default_factory=list)
    energy: float = 0.0
    now_hour: int = 0
    recent: list[dict] = field(default_factory=list)
    interruptibility: float = 1.0
    interrupt_detail: dict[str, float] = field(default_factory=dict)
    interrupt_factor: float = 1.0
    sleep_mod: float = 1.0
    de: float = 0.0
    dr: float = 0.0


@dataclass
class FetchSnapshot:
    items: list[FeedItem] = field(default_factory=list)
    new_items: list[FeedItem] = field(default_factory=list)
    new_entries: list[tuple[str, str]] = field(default_factory=list)
    semantic_duplicate_entries: list[tuple[str, str]] = field(default_factory=list)
    has_memory: bool = False


@dataclass
class ScoreSnapshot:
    pre_score: float = 0.0
    dc: float = 0.0
    base_score: float = 0.0
    draw_score: float = 0.0
    force_reflect: bool = False
    is_crisis: bool = False
    fresh_items_24h: int = 0
    sent_24h: int = 0


@dataclass
class DecideSnapshot:
    decision_signals: dict[str, object] = field(default_factory=dict)
    feature_payload: dict[str, float | str] = field(default_factory=dict)
    feature_final_score: float | None = None
    decision: Any = None
    decision_message: str = ""
    should_send: bool = False
    memory_query: str = ""
    retrieved_memory_block: str = ""
    retrieved_memory_item_ids: list[str] = field(default_factory=list)
    history_channel_open: bool = False
    history_gate_reason: str = "disabled"
    history_scope_mode: str = "disabled"
    memory_fallback_reason: str = ""


@dataclass
class ActSnapshot:
    compose_items: list[FeedItem] = field(default_factory=list)
    compose_entries: list[tuple[str, str]] = field(default_factory=list)
    state_summary_tag: str = "none"
    source_refs: list[ProactiveSourceRef] = field(default_factory=list)
    high_events: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceBundle:
    source_items: list[FeedItem]
    source_entries: list[tuple[str, str]]
    evidence_items: list[FeedItem]
    evidence_entries: list[tuple[str, str]]
    evidence_item_ids: list[str]
    source_refs: list[ProactiveSourceRef]


@dataclass
class DecisionContext:
    """每轮 tick 的决策上下文，只通过 state 和各 stage snapshot 显式访问。"""

    state: EngineState = field(default_factory=EngineState)
    sense: SenseSnapshot | None = None
    fetch: FetchSnapshot | None = None
    score: ScoreSnapshot | None = None
    decide: DecideSnapshot | None = None
    act: ActSnapshot | None = None

    def ensure_sense(self) -> SenseSnapshot:
        if self.sense is None:
            self.sense = SenseSnapshot()
        return self.sense

    def ensure_fetch(self) -> FetchSnapshot:
        if self.fetch is None:
            self.fetch = FetchSnapshot()
        return self.fetch

    def ensure_score(self) -> ScoreSnapshot:
        if self.score is None:
            self.score = ScoreSnapshot()
        return self.score

    def ensure_decide(self) -> DecideSnapshot:
        if self.decide is None:
            self.decide = DecideSnapshot()
        return self.decide

    def ensure_act(self) -> ActSnapshot:
        if self.act is None:
            self.act = ActSnapshot()
        return self.act


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    proceed: bool
    stop_result: float | object | None
    reason_code: Literal["pass", "quota_exhausted", "scheduler_reject"]


@dataclass(frozen=True)
class SenseResult:
    sleep_state: str
    sleep_available: bool
    health_event_count: int
    energy: float
    recent_count: int
    interruptibility: float
    interrupt_factor: float
    sleep_mod: float


@dataclass(frozen=True)
class PreScoreResult:
    proceed: bool
    return_score: float | None
    reason_code: Literal["continue", "health_fast_path", "below_threshold"]


@dataclass(frozen=True)
class ScoreResult:
    proceed: bool
    return_score: float | None
    reason_code: Literal[
        "continue",
        "no_candidates",
        "draw_score_below_threshold",
        "draw_score_force_reflect",
    ]
    base_score: float
    draw_score: float
    force_reflect: bool


@dataclass(frozen=True)
class FetchFilterResult:
    total_items: int
    discovered_count: int
    selected_count: int
    semantic_duplicate_count: int
    pending_enabled: bool
    has_memory: bool


@dataclass(frozen=True)
class DecideResult:
    proceed: bool
    return_score: float | None
    reason_code: Literal["continue", "feature_score_reject"]
    should_send: bool
    decision_message: str
    decision_mode: Literal["feature", "reflect"]
    feature_final_score: float | None
    history_gate_reason: str
    history_scope_mode: str


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
        memory_retrieval: MemoryRetrievalPort | None = None,
        anyaction: AnyActionGate | None = None,
        message_deduper: Any | None = None,
        skill_action_runner: SkillActionRunner | None = None,
        light_provider: Any | None = None,
        light_model: str = "",
        passive_busy_fn: Callable[[str], bool] | None = None,
        stage_trace_writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._presence = presence
        self._rng = rng
        self._sense = sense
        self._decide = decide
        self._act = act
        self._memory_retrieval = memory_retrieval
        self._anyaction = anyaction
        self._message_deduper = message_deduper
        self._skill_action_runner = skill_action_runner
        self._light_provider = light_provider
        self._light_model = light_model
        # 可选：AgentLoop 注入的被动处理信号，用于跳过与被动回复并发的主动发送
        self._passive_busy_fn = passive_busy_fn
        self._stage_trace_writer = stage_trace_writer

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

        # 1. 先做 gate，决定这一轮是否连主动判断资格都没有。
        gate_result = await self._stage_gate(ctx)
        self._trace_stage_result(ctx, stage="gate", result=gate_result)
        if not gate_result.proceed:
            if gate_result.stop_result is _STOP_NONE:
                return None
            return gate_result.stop_result  # type: ignore[return-value]

        # 2. 采集环境信号，把睡眠/能量/近期上下文写入 ctx.sense。
        sense_result = self._stage_sense(ctx)
        self._trace_stage_result(ctx, stage="sense", result=sense_result)

        # 3. 做一轮轻量预评分，过低时直接早退，避免后续昂贵路径。
        pre_result = await self._stage_pre_score(ctx)
        self._trace_stage_result(ctx, stage="pre_score", result=pre_result)
        if not pre_result.proceed:
            return pre_result.return_score

        # 4. 拉 feed 并做去重/兴趣筛选，得到本轮候选条目。
        fetch_result = await self._stage_fetch_filter(ctx)
        self._trace_stage_result(ctx, stage="fetch_filter", result=fetch_result)

        # 5. 计算 base_score / draw_score，判断是否值得进入 LLM 决策。
        score_result = await self._stage_score(ctx)
        self._trace_stage_result(ctx, stage="score", result=score_result)
        if not score_result.proceed:
            return score_result.return_score

        # 6. 进入 decide，生成 decision_signals，并走 feature 或 reflect 模式。
        decide_result = await self._stage_decide(ctx)
        self._trace_stage_result(ctx, stage="decide", result=decide_result)
        if not decide_result.proceed:
            return decide_result.return_score

        # 7. 真正执行发送、副作用落地和状态标记。
        await self._stage_act(ctx)
        return ctx.ensure_score().base_score

    # ------------------------------------------------------------------
    # Stage 1 — gate: state cleanup + anyaction gate
    # ------------------------------------------------------------------

    async def _stage_gate(self, ctx: DecisionContext) -> GateResult:
        """清理过期状态；若 anyaction gate 拒绝则提前返回退出码。

        返回值：
          proceed=True               — gate 通过，继续后续阶段
          stop_result=0.0            — quota_exhausted，tick 应返回 0.0
          stop_result=_STOP_NONE     — min_interval/probability 拒绝，tick 应返回 None
        """
        # 1. 先清理 state store 里的过期痕迹，避免本轮判断用到脏状态。
        self._state.cleanup(
            seen_ttl_hours=self._cfg.dedupe_seen_ttl_hours,
            delivery_ttl_hours=self._cfg.delivery_dedupe_hours,
            semantic_ttl_hours=max(
                self._cfg.dedupe_seen_ttl_hours, self._cfg.semantic_dedupe_window_hours
            ),
            rejection_cooldown_ttl_hours=getattr(
                self._cfg, "llm_reject_cooldown_hours", 0
            ),
            pending_ttl_hours=self._cfg_int("pending_item_ttl_hours", 24),
        )

        # 2. 再跑 anyaction gate，决定今天额度/最小间隔/概率是否允许继续。
        if not (self._cfg.anyaction_enabled and self._anyaction):
            return GateResult(proceed=True, stop_result=None, reason_code="pass")

        should_act, meta = self._anyaction.should_act(
            now_utc=ctx.state.now_utc,
            last_user_at=self._sense.last_user_at(),
        )
        if not should_act:
            logger.info(
                "[proactive] gate_result=reject selected_action=null meta=%s", meta
            )
            reason = meta.get("reason", "")
            if reason == "quota_exhausted":
                return GateResult(
                    proceed=False,
                    stop_result=0.0,
                    reason_code="quota_exhausted",
                )
            return GateResult(
                proceed=False,
                stop_result=_STOP_NONE,
                reason_code="scheduler_reject",
            )

        logger.info("[proactive] gate_result=pass meta=%s", meta)
        return GateResult(proceed=True, stop_result=None, reason_code="pass")

    # ------------------------------------------------------------------
    # Stage 2 — sense: collect environment signals
    # ------------------------------------------------------------------

    def _stage_sense(self, ctx: DecisionContext) -> SenseResult:
        """采集睡眠上下文、能量、打扰度等环境信号，填充 ctx。"""
        state = ctx.state
        sense = ctx.ensure_sense()
        # 1. 先刷新睡眠上下文，确保 Fitbit 相关信号是本轮最新值。
        refreshed = bool(getattr(self._sense, "refresh_sleep_context", lambda: False)())
        if refreshed:
            logger.debug("[proactive] fitbit 上下文已在本轮决策前主动刷新")
        sense.sleep_ctx = getattr(self._sense, "sleep_context", lambda: None)()

        # 2. 从睡眠上下文中抽取健康事件，后面 pre-score / act 都会用到。
        health_events: list[dict] = (
            getattr(sense.sleep_ctx, "health_events", [])
            if sense.sleep_ctx is not None
            else []
        )
        sense.health_events = health_events if isinstance(health_events, list) else []

        # 3. 采集能量、近期消息和 interruptibility，形成 score 的基础输入。
        sense.energy = self._sense.compute_energy()
        sense.now_hour = datetime.now().hour
        sense.recent = self._sense.collect_recent()
        sense.de = d_energy(sense.energy)
        sense.dr = d_recent(len(sense.recent), self._cfg.score_recent_scale)
        sense.interruptibility, sense.interrupt_detail = (
            self._sense.compute_interruptibility(
                now_hour=sense.now_hour,
                now_utc=state.now_utc,
                recent_msg_count=len(sense.recent),
            )
        )
        sense.interrupt_factor = 0.6 + 0.4 * sense.interruptibility

        # 4. 用睡眠修正项微调 interrupt_factor，避免深睡时仍然太激进。
        sense.sleep_mod = (
            sense.sleep_ctx.sleep_modifier if sense.sleep_ctx is not None else 1.0
        )
        if sense.sleep_mod != 1.0:
            sense.interrupt_factor *= sense.sleep_mod

        sleep_state = (
            sense.sleep_ctx.state if sense.sleep_ctx is not None else "unavailable"
        )
        sleep_available = bool(
            sense.sleep_ctx is not None and getattr(sense.sleep_ctx, "available", False)
        )
        logger.info(
            "[proactive][sleep-policy] state=%s available=%s prob=%s lag=%s sleep_mod=%.2f policy=%s",
            sleep_state,
            sleep_available,
            (sense.sleep_ctx.prob if sense.sleep_ctx is not None else None),
            (sense.sleep_ctx.data_lag_min if sense.sleep_ctx is not None else None),
            sense.sleep_mod,
            _sleep_policy_note(
                sleep_state,
                sleep_available,
                (sense.sleep_ctx.prob if sense.sleep_ctx is not None else None),
            ),
        )
        result = SenseResult(
            sleep_state=sleep_state,
            sleep_available=sleep_available,
            health_event_count=len(sense.health_events),
            energy=sense.energy,
            recent_count=len(sense.recent),
            interruptibility=sense.interruptibility,
            interrupt_factor=sense.interrupt_factor,
            sleep_mod=sense.sleep_mod,
        )
        return result

    # ------------------------------------------------------------------
    # Stage 3 — pre-score: quick score before fetching items
    # ------------------------------------------------------------------

    async def _stage_pre_score(self, ctx: DecisionContext) -> PreScoreResult:
        """计算 pre_score；过低时尝试 skill action 后提前返回。"""
        state = ctx.state
        sense = ctx.ensure_sense()
        score = ctx.ensure_score()
        # 1. 用 energy/recent 两个轻量信号先算 pre_score，尽量早决定要不要继续。
        w_sum = self._cfg.score_weight_energy + self._cfg.score_weight_recent
        score.pre_score = (
            (
                self._cfg.score_weight_energy * sense.de
                + self._cfg.score_weight_recent * sense.dr
            )
            / w_sum
            if w_sum > 0
            else 0.0
        ) * sense.interrupt_factor

        logger.info(
            "[proactive] pre_score=%.3f interrupt=%.3f factor=%.3f sleep_mod=%.2f"
            " (reply=%.2f activity=%.2f fatigue=%.2f rand=%+.2f)"
            " D_energy=%.3f D_recent=%.3f energy=%.3f msg_count=%d",
            score.pre_score,
            sense.interruptibility,
            sense.interrupt_factor,
            sense.sleep_mod,
            sense.interrupt_detail["f_reply"],
            sense.interrupt_detail["f_activity"],
            sense.interrupt_detail["f_fatigue"],
            sense.interrupt_detail["random_delta"],
            sense.de,
            sense.dr,
            sense.energy,
            len(sense.recent),
        )

        high_events = [e for e in sense.health_events if e.get("severity") == "high"]
        if high_events:
            score.force_reflect = True
            if score.pre_score < self._cfg.score_pre_threshold:
                logger.info(
                    "[proactive] health fast-path: pre_score=%.3f 低于阈值但存在 %d 个 high 事件，强制继续",
                    score.pre_score,
                    len(high_events),
                )
                return PreScoreResult(
                    proceed=True,
                    return_score=None,
                    reason_code="health_fast_path",
                )

        # 2. 如果 pre_score 太低，就不再进入 LLM 路径，直接尝试 skill action 兜底。
        if score.pre_score < self._cfg.score_pre_threshold:
            logger.info(
                "[proactive] pre_score 过低（%.3f < %.2f），跳过 chat，尝试 skill action",
                score.pre_score,
                self._cfg.score_pre_threshold,
            )
            await self._try_skill_action(now_utc=state.now_utc)
            return PreScoreResult(
                proceed=False,
                return_score=score.pre_score,
                reason_code="below_threshold",
            )

        return PreScoreResult(
            proceed=True,
            return_score=None,
            reason_code="continue",
        )

    # ------------------------------------------------------------------
    # Stage 4 — fetch & filter: pull items, deduplicate, interest filter
    # ------------------------------------------------------------------

    async def _stage_fetch_filter(self, ctx: DecisionContext) -> FetchFilterResult:
        """拉取 feed 条目，去重，应用兴趣筛选，填充 ctx.new_items 等。"""
        state = ctx.state
        fetch = ctx.ensure_fetch()
        # 1. 先从 feed 拉原始条目，后续所有筛选都基于这批候选。
        fetch.items = await self._sense.fetch_items(self._cfg.items_per_source)
        logger.info("[proactive] 拉取到 %d 条信息", len(fetch.items))

        # 2. 先做 seen / semantic 去重，避免重复条目进入后续决策。
        discovered_items, discovered_entries, fetch.semantic_duplicate_entries = (
            self._sense.filter_new_items(fetch.items)
        )
        logger.info(
            "[proactive] 去重后剩余新信息 %d 条（过滤重复 %d 条）",
            len(discovered_items),
            len(fetch.items) - len(discovered_items),
        )
        if fetch.semantic_duplicate_entries:
            logger.info(
                "[proactive] 语义重复条目 count=%d 不写入 seen_items，72h 窗口自然抑制",
                len(fetch.semantic_duplicate_entries),
            )

        # 3. 如果开启兴趣筛选，再用 memory 对候选做一次收缩。
        if self._cfg.interest_filter.enabled and discovered_items:
            memory_text = self._sense.read_memory_text()
            if memory_text:
                filtered_items, ranked = select_interesting_items(
                    discovered_items, memory_text, self._cfg.interest_filter
                )
                keep_ids = {self._item_id_for(item) for item in filtered_items}
                old_count = len(discovered_items)
                discovered_items = filtered_items
                discovered_entries = [
                    (source_key, item_id)
                    for source_key, item_id in discovered_entries
                    if item_id in keep_ids
                ]
                top_preview = ", ".join(
                    f"{(pair[0].title or '')[:28]}:{pair[1]:.2f}" for pair in ranked[:3]
                )
                logger.info(
                    "[proactive] memory 兴趣筛选 old=%d kept=%d min_score=%.2f top=%s",
                    old_count,
                    len(filtered_items),
                    self._cfg.interest_filter.min_score,
                    top_preview or "-",
                )
            else:
                logger.info("[proactive] memory 兴趣筛选跳过：memory 为空")

        pending_enabled = bool(getattr(self._cfg, "pending_queue_enabled", True))
        if pending_enabled:
            self._state.upsert_pending_items(
                discovered_items,
                max_per_source=self._cfg_int("pending_max_per_source", 20),
                max_total=self._cfg_int("pending_max_total", 200),
                now=state.now_utc,
            )
            fetch.new_items, fetch.new_entries = self._select_pending_candidates(
                now=state.now_utc,
                limit=self._cfg_int("pending_candidate_limit", 3),
            )
            logger.info(
                "[proactive] pending 选出候选=%d pending_stats=%s",
                len(fetch.new_items),
                self._state.pending_stats(),
            )
        else:
            fetch.new_items = discovered_items
            fetch.new_entries = discovered_entries

        # 4. 最后补充全局记忆命中状态，供后面的 force_reflect 判断使用。
        fetch.has_memory = self._sense.has_global_memory()
        result = FetchFilterResult(
            total_items=len(fetch.items),
            discovered_count=len(discovered_items),
            selected_count=len(fetch.new_items),
            semantic_duplicate_count=len(fetch.semantic_duplicate_entries),
            pending_enabled=pending_enabled,
            has_memory=fetch.has_memory,
        )
        return result

    # ------------------------------------------------------------------
    # Stage 5 — score: compute base_score / draw_score
    # ------------------------------------------------------------------

    async def _stage_score(self, ctx: DecisionContext) -> ScoreResult:
        """计算 base_score / draw_score；未过门槛时提前返回。"""
        state = ctx.state
        # 1. 先把 score snapshot 算完整，保证 base/draw_score 都落在 ctx.score。
        w_random = self._compute_score_snapshot(ctx)
        # 2. 再刷新 presence 相关状态，把 target 会话和近 24h 统计补齐。
        self._refresh_presence_state(ctx)
        # 3. 基于完整 snapshot 生成 ScoreResult，决定本轮是否继续。
        result = self._build_score_result(ctx, w_random=w_random)
        # 4. draw_score 不够但又值得跑 skill action 时，在这里统一做兜底。
        if result.reason_code == "draw_score_below_threshold":
            logger.info("[proactive] draw_score 未过门槛，跳过本轮反思")
            logger.info("[proactive] selected_action=idle reason=draw_score")
            await self._try_skill_action(now_utc=state.now_utc)
        return result

    # ------------------------------------------------------------------
    # Stage 6 — decide: build decision_signals, run LLM scoring/compose
    # ------------------------------------------------------------------

    async def _stage_decide(self, ctx: DecisionContext) -> DecideResult:
        """构建决策信号，调用 LLM 评分或生成消息，填充 ctx.should_send / decision_message。"""
        # 1. 先把决策所需的显式 signals 组好，后续 feature/reflect 共用。
        self._populate_decision_signals(ctx)
        # 2. 再补记忆检索结果，把 memory block 和 route 元信息挂到 decide snapshot。
        await self._retrieve_decision_memory(ctx)
        # 3. feature 模式走“评分 + compose”的闭环；否则走 reflect 模式。
        if self._cfg.feature_scoring_enabled:
            self._prepare_feature_compose_candidates(ctx)
            return await self._run_feature_decision(ctx)
        await self._run_reflect_decision(ctx)
        # 4. 两条分支最后都统一落成 DecideResult，交给 act 阶段消费。
        return self._build_continue_decide_result(ctx, decision_mode="reflect")

    # ------------------------------------------------------------------
    # Stage 7 — act: dedup checks, send, mark state
    # ------------------------------------------------------------------

    async def _stage_act(self, ctx: DecisionContext) -> None:
        """去重检查、发送消息、标记已发送状态。"""
        state = ctx.state
        decide = ctx.ensure_decide()
        act = ctx.ensure_act()
        evidence = self._build_evidence_bundle(ctx)
        act.source_refs = evidence.source_refs

        # 1. 如果 decide 已经判定不发，就只做 rejection / fallback，不进入发送链。
        if not decide.should_send:
            logger.info("[proactive] 决定不主动发送")
            logger.info("[proactive] 本轮未发送，不标记 seen，后续可再次尝试")
            await self._reject_and_try_skill_action(ctx, evidence)
            return

        # 2. 准备本轮 delivery key 和证据集，后面的所有去重都围绕这组输入。
        delivery_key = self._prepare_delivery_attempt(ctx, evidence)
        logger.info(
            "[proactive] 发送前去重检查 session=%s evidence_count=%d delivery_key=%s",
            state.session_key or "（未配置）",
            len(evidence.evidence_item_ids),
            delivery_key[:16],
        )

        # 3. 先做 delivery 去重，避免同一组证据重复发到同一会话。
        if state.session_key and self._state.is_delivery_duplicate(
            session_key=state.session_key,
            delivery_key=delivery_key,
            window_hours=self._cfg.delivery_dedupe_hours,
        ):
            logger.info("[proactive] 命中发送去重，跳过发送")
            self._consume_evidence_entries(evidence)
            logger.info(
                "[proactive] 已按去重命中消费证据条目 count=%d",
                len(evidence.evidence_entries),
            )
            logger.info("[proactive] selected_action=idle reason=delivery_dedupe")
            return

        # 4. 再做 state summary 防复读，必要时重写或拒绝这次发送。
        recent_proactive = self._sense.collect_recent_proactive(
            getattr(self._cfg, "message_dedupe_recent_n", 5)
        )
        if not await self._rewrite_or_reject_repeated_state_summary(
            ctx,
            evidence,
            recent_proactive,
        ):
            return

        # 5. 再做 message 语义去重，挡住“话术不同但本质重复”的消息。
        if not await self._passes_message_deduper(ctx, evidence, recent_proactive):
            return

        # 6. 最后检查目标会话是否正忙于被动回复，避免双写并发。
        if (
            state.session_key
            and self._passive_busy_fn
            and self._passive_busy_fn(state.session_key)
        ):
            logger.info(
                "[proactive] 目标会话 %s 正在处理被动回复，跳过本轮发送"
                " selected_action=idle reason=passive_busy",
                state.session_key,
            )
            return

        # 7. 真正发送；成功后落 delivery/seen/ack，失败则保持可重试状态。
        sent = await self._act.send(
            decide.decision_message,
            ProactiveSendMeta(
                evidence_item_ids=evidence.evidence_item_ids,
                source_refs=act.source_refs,
                state_summary_tag=act.state_summary_tag,
            ),
        )
        if sent:
            self._finalize_successful_send(ctx, evidence, delivery_key)
            logger.debug("[proactive] 已发送成功并标记本轮条目为 seen")
            logger.debug("[proactive] selected_action=chat")
        else:
            logger.info("[proactive] 本轮发送未成功，不标记 seen，后续可再次尝试")
            logger.info("[proactive] selected_action=idle reason=send_failed")

    def _compute_score_snapshot(self, ctx: DecisionContext) -> float:
        """把 score 阶段的纯计算部分集中到一个 helper，避免和副作用混在一起。"""
        sense = ctx.ensure_sense()
        fetch = ctx.ensure_fetch()
        score = ctx.ensure_score()
        # 1. 先把内容新鲜度算出来，补全三维 score 输入。
        score.dc = d_content(len(fetch.new_items), self._cfg.score_content_halfsat)
        # 2. 再合成 base_score，并乘上 interrupt_factor 做当前轮的打扰修正。
        score.base_score = (
            composite_score(
                sense.de,
                score.dc,
                sense.dr,
                self._cfg.score_weight_energy,
                self._cfg.score_weight_content,
                self._cfg.score_weight_recent,
            )
            * sense.interrupt_factor
        )
        # 3. 最后叠加随机扰动，得到 draw_score，供阈值判断使用。
        w_random = random_weight(rng=self._rng)
        score.draw_score = score.base_score * w_random
        return w_random

    def _refresh_presence_state(self, ctx: DecisionContext) -> None:
        """把 score 阶段依赖的 presence / 会话状态拉平到 ctx.state / ctx.score。"""
        sense = ctx.ensure_sense()
        fetch = ctx.ensure_fetch()
        score = ctx.ensure_score()
        # 1. 先解析目标 session，并读取最近用户消息/主动触达时间。
        ctx.state.session_key = self._sense.target_session_key()
        ctx.state.target_last_user = (
            self._presence.get_last_user_at(ctx.state.session_key)
            if self._presence and ctx.state.session_key
            else None
        )
        ctx.state.last_proactive_at = (
            self._presence.get_last_proactive_at(ctx.state.session_key)
            if self._presence and ctx.state.session_key
            else None
        )
        # 2. 再计算 presence 兜底条件，决定是否要 force_reflect。
        presence_force_reflect = self._presence is not None and (
            sense.energy < 0.05
            or (bool(ctx.state.session_key) and ctx.state.target_last_user is None)
            or (sense.energy < 0.20 and fetch.has_memory)
        )
        # 3. 最后补 24h 维度的统计量，供 decide 阶段构建 signals。
        score.force_reflect = score.force_reflect or presence_force_reflect
        score.is_crisis = sense.energy < 0.05
        score.sent_24h = (
            self._state.count_deliveries_in_window(
                ctx.state.session_key,
                24,
                now=ctx.state.now_utc,
            )
            if ctx.state.session_key
            else 0
        )
        score.fresh_items_24h = sum(
            1
            for item in fetch.new_items
            if item.published_at
            and (ctx.state.now_utc - item.published_at).total_seconds() <= 24 * 3600
        )

    def _build_score_result(self, ctx: DecisionContext, *, w_random: float) -> ScoreResult:
        """把 score 阶段的分支判断统一映射成 ScoreResult。"""
        sense = ctx.ensure_sense()
        fetch = ctx.ensure_fetch()
        score = ctx.ensure_score()
        # 1. 没有候选、没有健康事件、也没有强制兜底时，直接 no_candidates 早退。
        if not fetch.new_items and not sense.health_events and not score.force_reflect:
            logger.info("[proactive] 无候选信息且无健康事件，跳过本轮反思")
            logger.info("[proactive] selected_action=idle reason=no_candidates")
            return ScoreResult(
                proceed=False,
                return_score=score.base_score,
                reason_code="no_candidates",
                base_score=score.base_score,
                draw_score=score.draw_score,
                force_reflect=score.force_reflect,
            )
        # 2. 正常情况下先记录完整 score 诊断信息，便于理解本轮为什么继续/停止。
        logger.info(
            "[proactive] base_score=%.3f  D_energy=%.3f D_content=%.3f D_recent=%.3f"
            "  interrupt=%.3f W_random=%.2f → draw_score=%.3f 阈值=%.2f force_reflect=%s",
            score.base_score,
            sense.de,
            score.dc,
            sense.dr,
            sense.interruptibility,
            w_random,
            score.draw_score,
            self._cfg.score_llm_threshold,
            score.force_reflect,
        )
        # 3. draw_score 低于阈值时，区分普通早退和 force_reflect 两条路径。
        if score.draw_score < self._cfg.score_llm_threshold and not score.force_reflect:
            return ScoreResult(
                proceed=False,
                return_score=score.base_score,
                reason_code="draw_score_below_threshold",
                base_score=score.base_score,
                draw_score=score.draw_score,
                force_reflect=score.force_reflect,
            )
        if score.draw_score < self._cfg.score_llm_threshold and score.force_reflect:
            logger.info("[proactive] draw_score 未过门槛，但命中兜底条件，继续反思")
            return ScoreResult(
                proceed=True,
                return_score=None,
                reason_code="draw_score_force_reflect",
                base_score=score.base_score,
                draw_score=score.draw_score,
                force_reflect=score.force_reflect,
            )
        return ScoreResult(
            proceed=True,
            return_score=None,
            reason_code="continue",
            base_score=score.base_score,
            draw_score=score.draw_score,
            force_reflect=score.force_reflect,
        )

    def _populate_decision_signals(self, ctx: DecisionContext) -> None:
        """把 decide 阶段真正依赖的显式信号集中写入 decide snapshot。"""
        state = ctx.state
        sense = ctx.ensure_sense()
        fetch = ctx.ensure_fetch()
        score = ctx.ensure_score()
        decide = ctx.ensure_decide()
        act = ctx.ensure_act()
        # 1. 先把与“最近是否被打扰过”有关的时间差算出来。
        mins_since_last_user = (
            int((state.now_utc - state.target_last_user).total_seconds() / 60)
            if state.target_last_user
            else None
        )
        mins_since_last_proactive = (
            int((state.now_utc - state.last_proactive_at).total_seconds() / 60)
            if state.last_proactive_at
            else None
        )
        replied_after_last_proactive = bool(
            state.target_last_user
            and state.last_proactive_at
            and state.target_last_user > state.last_proactive_at
        )
        # 2. 再组织成给 feature / reflect 共用的 decision_signals。
        decide.decision_signals = {
            "minutes_since_last_user": mins_since_last_user,
            "minutes_since_last_proactive": mins_since_last_proactive,
            "user_replied_after_last_proactive": replied_after_last_proactive,
            "proactive_sent_24h": score.sent_24h,
            "interruptibility": round(sense.interruptibility, 3),
            "interrupt_breakdown": {
                "reply": round(sense.interrupt_detail["f_reply"], 3),
                "activity": round(sense.interrupt_detail["f_activity"], 3),
                "fatigue": round(sense.interrupt_detail["f_fatigue"], 3),
            },
            "scores": {
                "pre_score": round(score.pre_score, 3),
                "base_score": round(score.base_score, 3),
                "draw_score": round(score.draw_score, 3),
                "llm_threshold": round(self._cfg.score_llm_threshold, 3),
                "send_threshold": round(self._cfg.threshold, 3),
            },
            "candidate_items": len(fetch.new_items),
            "fresh_items_24h": score.fresh_items_24h,
        }
        # 3. 最后抽取高优先级健康事件，给后面的发送和 ack 路径使用。
        if sense.health_events:
            decide.decision_signals["health_events"] = sense.health_events
        act.high_events = [
            e for e in sense.health_events if str((e or {}).get("severity", "")) == "high"
        ]
        logger.info(
            "[proactive] fitbit_signal events=%d high=%d sleep_state=%s",
            len(sense.health_events),
            len(act.high_events),
            sense.sleep_ctx.state if sense.sleep_ctx is not None else "unavailable",
        )

    async def _retrieve_decision_memory(self, ctx: DecisionContext) -> None:
        """按当前会话和候选条目补 memory block，不在这里做任何发送决策。"""
        state = ctx.state
        sense = ctx.ensure_sense()
        fetch = ctx.ensure_fetch()
        score = ctx.ensure_score()
        decide = ctx.ensure_decide()
        channel = ""
        chat_id = ""
        # 1. 先从 session_key 中拆出 channel/chat_id，供 proactive memory 检索使用。
        if state.session_key and ":" in state.session_key:
            channel, chat_id = state.session_key.split(":", 1)
        # 2. 然后跑 retrieval port，把 block 和 route 元信息一次性带回。
        if self._memory_retrieval is not None:
            retrieved = await self._memory_retrieval.retrieve_proactive_context(
                session_key=state.session_key,
                channel=channel,
                chat_id=chat_id,
                items=fetch.new_items,
                recent=sense.recent,
                decision_signals=decide.decision_signals,
                is_crisis=score.is_crisis,
            )
        else:
            retrieved = ProactiveRetrievedMemory.empty("retrieval_disabled")
        # 3. 最后把检索结果全部落进 decide snapshot，后面不再重复解析。
        decide.memory_query = retrieved.query
        decide.retrieved_memory_block = retrieved.block
        decide.retrieved_memory_item_ids = retrieved.item_ids
        decide.history_channel_open = retrieved.history_channel_open
        decide.history_gate_reason = retrieved.history_gate_reason
        decide.history_scope_mode = retrieved.history_scope_mode
        decide.memory_fallback_reason = retrieved.fallback_reason

    def _prepare_feature_compose_candidates(self, ctx: DecisionContext) -> None:
        fetch = ctx.ensure_fetch()
        act = ctx.ensure_act()
        act.compose_items = fetch.new_items[:1]
        act.compose_entries = self._entries_for_items(act.compose_items, fetch.new_entries)

    async def _run_feature_decision(self, ctx: DecisionContext) -> DecideResult:
        """feature 模式：先打分，再在通过阈值时 compose_message。"""
        sense = ctx.ensure_sense()
        fetch = ctx.ensure_fetch()
        score = ctx.ensure_score()
        decide = ctx.ensure_decide()
        act = ctx.ensure_act()
        # 1. 先对当前候选打 feature 分数，并缓存原始 feature payload。
        features = await self._decide.score_features(
            items=fetch.new_items,
            recent=sense.recent,
            decision_signals=decide.decision_signals,
            retrieved_memory_block=decide.retrieved_memory_block,
        )
        # 2. 再合成最终 feature_final_score，并决定是否在这里直接拒绝。
        decide.feature_payload = features or {}
        feature_final_base = _feature_final_score(
            cfg=self._cfg,
            features=decide.feature_payload,
            de=sense.de,
            dc=score.dc,
            dr=sense.dr,
            interruptibility=sense.interruptibility,
        )
        health_bonus = _health_priority_bonus(
            alert_count=len(sense.health_events),
            notify_count=len(act.high_events),
        )
        decide.feature_final_score = min(1.0, feature_final_base + health_bonus)
        logger.info(
            "[proactive] feature_score enabled base=%.3f health_bonus=%.3f final=%.3f threshold=%.3f features=%s",
            feature_final_base,
            health_bonus,
            decide.feature_final_score,
            self._cfg.feature_send_threshold,
            decide.feature_payload,
        )
        if decide.feature_final_score < self._cfg.feature_send_threshold:
            logger.info("[proactive] selected_action=idle reason=feature_score")
            self._state.mark_rejection_cooldown(
                self._primary_candidate_entries(fetch.new_entries),
                hours=getattr(self._cfg, "llm_reject_cooldown_hours", 0),
            )
            return DecideResult(
                proceed=False,
                return_score=score.base_score,
                reason_code="feature_score_reject",
                should_send=False,
                decision_message="",
                decision_mode="feature",
                feature_final_score=decide.feature_final_score,
                history_gate_reason=decide.history_gate_reason,
                history_scope_mode=decide.history_scope_mode,
            )
        # 3. 只有通过 feature gate，才会继续 compose_message 并产出 should_send。
        decide.decision_message = await self._decide.compose_message(
            items=act.compose_items,
            recent=sense.recent,
            decision_signals=decide.decision_signals,
            retrieved_memory_block=decide.retrieved_memory_block,
        )
        decide.should_send = bool(decide.decision_message.strip()) and (
            decide.feature_final_score is not None
            and decide.feature_final_score >= self._cfg.feature_send_threshold
        )
        logger.info(
            "[proactive] feature_mode compose_len=%d should_send=%s reasons={topic:%r,interest:%r,novel:%r,reconnect:%r,disturb:%r,readiness:%r,conf:%r}",
            len(decide.decision_message),
            decide.should_send,
            decide.feature_payload.get("topic_continuity_reason", ""),
            decide.feature_payload.get("interest_match_reason", ""),
            decide.feature_payload.get("content_novelty_reason", ""),
            decide.feature_payload.get("reconnect_value_reason", ""),
            decide.feature_payload.get("disturb_risk_reason", ""),
            decide.feature_payload.get("message_readiness_reason", ""),
            decide.feature_payload.get("confidence_reason", ""),
        )
        return self._build_continue_decide_result(ctx, decision_mode="feature")

    async def _run_reflect_decision(self, ctx: DecisionContext) -> None:
        """reflect 模式：直接让 LLM 做反思与消息草拟。"""
        sense = ctx.ensure_sense()
        fetch = ctx.ensure_fetch()
        score = ctx.ensure_score()
        decide = ctx.ensure_decide()
        # 1. 先请求 reflect 结果，得到 score / should_send / message / reasoning。
        decide.decision = await self._decide.reflect(
            fetch.new_items,
            sense.recent,
            energy=sense.energy,
            urge=score.draw_score,
            is_crisis=score.is_crisis,
            decision_signals=decide.decision_signals,
            retrieved_memory_block=decide.retrieved_memory_block,
        )
        # 2. 再叠加随机化决策，并用最终阈值决定 should_send。
        decide.decision, decision_delta = self._decide.randomize_decision(decide.decision)
        logger.info(
            f"[proactive] score={decide.decision.score:.2f}  "
            f"score_delta={decision_delta:+.2f}  "
            f"send={decide.decision.should_send}  "
            f"reasoning={decide.decision.reasoning[:80]!r}"
        )
        decide.should_send = (
            decide.decision.should_send and decide.decision.score >= self._cfg.threshold
        )
        decide.decision_message = decide.decision.message

    def _build_continue_decide_result(
        self,
        ctx: DecisionContext,
        *,
        decision_mode: Literal["feature", "reflect"],
    ) -> DecideResult:
        decide = ctx.ensure_decide()
        return DecideResult(
            proceed=True,
            return_score=None,
            reason_code="continue",
            should_send=decide.should_send,
            decision_message=decide.decision_message,
            decision_mode=decision_mode,
            feature_final_score=decide.feature_final_score,
            history_gate_reason=decide.history_gate_reason,
            history_scope_mode=decide.history_scope_mode,
        )

    def _build_evidence_bundle(self, ctx: DecisionContext) -> EvidenceBundle:
        """统一构建 act 阶段要消费的证据视图，避免 source/evidence 概念混用。"""
        fetch = ctx.ensure_fetch()
        decide = ctx.ensure_decide()
        act = ctx.ensure_act()
        # 1. 先确定 source_items/source_entries，它们代表本轮发送候选来源。
        source_items = act.compose_items or fetch.new_items or fetch.items
        source_entries = act.compose_entries or self._entries_for_items(
            source_items,
            fetch.new_entries,
        )
        # 2. 再从 decision 里解析 evidence ids，缩成真正支撑这次发送的证据集。
        evidence_item_ids = self._decide.resolve_evidence_item_ids(
            decide.decision
            if decide.decision is not None
            else _PseudoDecision(message=decide.decision_message),
            source_items,
        )
        # 3. 最后把 source/evidence/source_refs 一起打包给 act 阶段使用。
        evidence_items, evidence_entries = self._resolve_evidence_entries(
            source_items,
            source_entries,
            evidence_item_ids,
        )
        return EvidenceBundle(
            source_items=source_items,
            source_entries=source_entries,
            evidence_items=evidence_items,
            evidence_entries=evidence_entries,
            evidence_item_ids=evidence_item_ids,
            source_refs=self._build_source_refs(evidence_items),
        )

    def _candidate_entries_for_rejection(
        self,
        ctx: DecisionContext,
        evidence: EvidenceBundle,
    ) -> list[tuple[str, str]]:
        fetch = ctx.ensure_fetch()
        return evidence.evidence_entries or self._primary_candidate_entries(
            evidence.source_entries or fetch.new_entries
        )

    def _reject_current_candidates(self, entries: list[tuple[str, str]]) -> None:
        self._state.mark_rejection_cooldown(
            entries,
            hours=getattr(self._cfg, "llm_reject_cooldown_hours", 0),
        )

    async def _reject_and_try_skill_action(
        self,
        ctx: DecisionContext,
        evidence: EvidenceBundle,
    ) -> None:
        state = ctx.state
        self._reject_current_candidates(
            self._candidate_entries_for_rejection(ctx, evidence)
        )
        await self._try_skill_action(now_utc=state.now_utc)

    def _consume_evidence_entries(self, evidence: EvidenceBundle) -> None:
        self._state.mark_items_seen(evidence.evidence_entries)
        self._state.remove_pending_items(evidence.evidence_entries)
        self._state.mark_semantic_items(
            self._decide.semantic_entries(evidence.evidence_items)
        )

    def _prepare_delivery_attempt(
        self,
        ctx: DecisionContext,
        evidence: EvidenceBundle,
    ) -> str:
        state = ctx.state
        decide = ctx.ensure_decide()
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        state.session_key = f"{channel}:{chat_id}" if channel and chat_id else ""
        return self._decide.build_delivery_key(
            evidence.evidence_item_ids,
            decide.decision_message,
        )

    async def _rewrite_or_reject_repeated_state_summary(
        self,
        ctx: DecisionContext,
        evidence: EvidenceBundle,
        recent_proactive: list[RecentProactiveMessage],
    ) -> bool:
        state = ctx.state
        decide = ctx.ensure_decide()
        act = ctx.ensure_act()
        original_tag = await self._classify_state_summary_tag(decide.decision_message)
        act.state_summary_tag = original_tag
        if original_tag == "none" or not self._seen_state_summary_in_current_silence(
            original_tag,
            recent_proactive,
            state.target_last_user,
        ):
            return True

        rewritten = await self._rewrite_without_repeated_state(
            decide.decision_message,
            original_tag,
            act.source_refs,
        )
        if not rewritten:
            logger.info(
                "[proactive] 当前沉默周期内 state_summary_tag=%s 已出现，且重写失败，跳过发送",
                original_tag,
            )
            await self._reject_and_try_skill_action(ctx, evidence)
            return False

        rewritten_tag = await self._classify_state_summary_tag(rewritten)
        if rewritten_tag == original_tag and rewritten_tag != "none":
            logger.info(
                "[proactive] state_summary_tag=%s 重写后仍重复，跳过发送",
                rewritten_tag,
            )
            await self._reject_and_try_skill_action(ctx, evidence)
            return False

        decide.decision_message = rewritten
        act.state_summary_tag = rewritten_tag
        return True

    async def _passes_message_deduper(
        self,
        ctx: DecisionContext,
        evidence: EvidenceBundle,
        recent_proactive: list[RecentProactiveMessage],
    ) -> bool:
        if self._message_deduper is None:
            return True
        decide = ctx.ensure_decide()
        act = ctx.ensure_act()
        is_dup, dup_reason = await self._message_deduper.is_duplicate(
            decide.decision_message,
            recent_proactive,
            act.state_summary_tag,
        )
        if not is_dup:
            return True
        logger.info("[proactive] 消息语义去重命中，跳过发送 reason=%r", dup_reason)
        logger.info("[proactive] selected_action=idle reason=message_dedupe")
        await self._reject_and_try_skill_action(ctx, evidence)
        return False

    def _finalize_successful_send(
        self,
        ctx: DecisionContext,
        evidence: EvidenceBundle,
        delivery_key: str,
    ) -> None:
        state = ctx.state
        sense = ctx.ensure_sense()
        if self._cfg.anyaction_enabled and self._anyaction:
            self._anyaction.record_action(now_utc=state.now_utc)
        self._consume_evidence_entries(evidence)
        if state.session_key:
            self._state.mark_delivery(state.session_key, delivery_key)
        if not sense.health_events:
            return
        acked_ids = [
            e["id"]
            for e in sense.health_events
            if isinstance(e, dict) and e.get("id")
        ]
        if acked_ids:
            getattr(self._sense, "acknowledge_health_events", lambda _: None)(acked_ids)
            logger.info(
                "[proactive] acknowledged %d 健康事件 ids=%s",
                len(acked_ids),
                acked_ids,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trace_stage_result(
        self,
        ctx: DecisionContext,
        *,
        stage: str,
        result: object,
    ) -> None:
        state = ctx.state
        if self._stage_trace_writer is None:
            return
        payload = {
            "stage": stage,
            "result": _json_safe(asdict(result)) if is_dataclass(result) else {},
            "session_key": state.session_key,
        }
        try:
            self._stage_trace_writer(
                build_strategy_trace_envelope(
                    trace_type="proactive_stage",
                    source="proactive.engine",
                    subject_kind="global",
                    subject_id=f"proactive-stage:{stage}",
                    payload=payload,
                )
            )
        except Exception:
            logger.exception("[proactive] stage trace write failed stage=%s", stage)

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

    def _select_pending_candidates(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> tuple[list[FeedItem], list[tuple[str, str]]]:
        items = self._state.list_pending_candidates(limit=0, now=now)
        selected_items: list[FeedItem] = []
        selected_entries: list[tuple[str, str]] = []
        stale_entries: list[tuple[str, str]] = []
        cooldown_hours = getattr(self._cfg, "llm_reject_cooldown_hours", 0)
        for item in items:
            source_key = compute_source_key(item)
            item_id = self._item_id_for(item)
            if self._state.is_item_seen(
                source_key=source_key,
                item_id=item_id,
                ttl_hours=self._cfg.dedupe_seen_ttl_hours,
                now=now,
            ):
                stale_entries.append((source_key, item_id))
                continue
            if cooldown_hours > 0 and self._state.is_rejection_cooled(
                source_key=source_key,
                item_id=item_id,
                ttl_hours=cooldown_hours,
                now=now,
            ):
                continue
            selected_items.append(item)
            selected_entries.append((source_key, item_id))
            if limit > 0 and len(selected_items) >= limit:
                break
        if stale_entries:
            self._state.remove_pending_items(stale_entries)
        return selected_items, selected_entries

    def _build_source_refs(self, items: list[FeedItem]) -> list[ProactiveSourceRef]:
        refs: list[ProactiveSourceRef] = []
        for item in items[:1]:
            published_at = None
            if item.published_at is not None:
                try:
                    published_at = item.published_at.isoformat()
                except Exception:
                    published_at = str(item.published_at)
            refs.append(
                ProactiveSourceRef(
                    item_id=self._item_id_for(item),
                    source_type=str(item.source_type or ""),
                    source_name=str(item.source_name or ""),
                    title=str(item.title or ""),
                    url=str(item.url).strip() if item.url else None,
                    published_at=published_at,
                )
            )
        return refs

    def _seen_state_summary_in_current_silence(
        self,
        tag: str,
        recent_proactive: list[RecentProactiveMessage],
        last_user_reply_at: datetime | None,
    ) -> bool:
        if not tag or tag == "none":
            return False
        for msg in reversed(recent_proactive):
            msg_tag = str(getattr(msg, "state_summary_tag", "none") or "none")
            if msg_tag != tag:
                continue
            ts = getattr(msg, "timestamp", None)
            if last_user_reply_at is None:
                return True
            if ts is None:
                continue
            try:
                if ts > last_user_reply_at:
                    return True
            except Exception:
                continue
        return False

    async def _classify_state_summary_tag(self, message: str) -> str:
        text = (message or "").strip()
        if not text:
            return "none"
        heuristic = _heuristic_state_summary_tag(text)
        if self._light_provider is None or not self._light_model:
            return heuristic
        prompt = (
            "你是主动消息分类器。判断消息是否包含“用户状态总结/安慰框架”。\n"
            "只允许输出 JSON："
            '{"state_summary_tag":"none"}\n'
            "可选标签只有：none, interview_anxiety_reassurance, health_nudge, sleep_concern, general_encouragement。\n"
            "如果消息主要是在概括用户最近的压力、焦虑、别太逼自己、底子还在、先歇歇等，优先标 interview_anxiety_reassurance 或 general_encouragement。\n"
            "如果消息主要是直接推送兴趣资讯，不概括用户状态，标 none。"
        )
        try:
            content = await self._request_light_text(
                system_content=prompt,
                user_content=text,
                max_tokens=64,
            )
            data = extract_json_object(content)
            tag = str(data.get("state_summary_tag", "") or "").strip()
            if tag in {
                "none",
                "interview_anxiety_reassurance",
                "health_nudge",
                "sleep_concern",
                "general_encouragement",
            }:
                return tag
        except Exception:
            logger.debug("[proactive] state_summary_tag 分类失败，回退 heuristic")
        return heuristic

    async def _rewrite_without_repeated_state(
        self,
        message: str,
        tag: str,
        source_refs: list[ProactiveSourceRef],
    ) -> str:
        text = (message or "").strip()
        if not text:
            return ""
        if self._light_provider is None or not self._light_model:
            return ""
        source_hint = ""
        if source_refs:
            ref = source_refs[0]
            parts = [p for p in [ref.source_name, ref.title, ref.url] if p]
            if parts:
                source_hint = "\n来源信息：" + " | ".join(parts)
        prompt = (
            "你要重写一条主动消息。要求：删除重复的用户状态总结/安慰前缀，"
            "保留真正的新资讯与来源行；如果原消息没有实质新内容，返回空字符串。\n"
            "只输出最终消息，不要解释。"
        )
        user_prompt = (
            f"重复的 state_summary_tag={tag}\n"
            "用户在最近这次主动消息后还没有回复，所以不能再重复同类安慰。\n"
            f"原消息：{text}{source_hint}"
        )
        try:
            return await self._request_light_text(
                system_content=prompt,
                user_content=user_prompt,
                max_tokens=min(192, max(64, len(text) + 64)),
            )
        except Exception:
            logger.debug("[proactive] 去重复写失败")
            return ""

    async def _request_light_text(
        self,
        *,
        system_content: str,
        user_content: str,
        max_tokens: int,
    ) -> str:
        resp = await self._light_provider.chat(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            tools=[],
            model=self._light_model,
            max_tokens=max_tokens,
        )
        return (resp.content or "").strip()

    def _resolve_evidence_entries(
        self,
        items: list[FeedItem],
        entries: list[tuple[str, str]],
        evidence_ids: list[str],
    ) -> tuple[list[FeedItem], list[tuple[str, str]]]:
        if not evidence_ids:
            return [], []
        entry_by_id = {item_id: source_key for source_key, item_id in entries}
        wanted = set(evidence_ids)
        selected_items: list[FeedItem] = []
        selected_entries: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in items:
            item_id = self._item_id_for(item)
            if item_id not in wanted or item_id in seen:
                continue
            selected_items.append(item)
            selected_entries.append(
                (entry_by_id.get(item_id, compute_source_key(item)), item_id)
            )
            seen.add(item_id)
        return selected_items, selected_entries

    def _entries_for_items(
        self,
        items: list[FeedItem],
        entries: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        entry_by_id: dict[str, str] = {}
        for source_key, item_id in entries:
            entry_by_id.setdefault(item_id, source_key)
        selected: list[tuple[str, str]] = []
        for item in items:
            item_id = self._item_id_for(item)
            selected.append((entry_by_id.get(item_id, compute_source_key(item)), item_id))
        return selected

    def _primary_candidate_entries(
        self, entries: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        return entries[:1]

    def _cfg_int(self, name: str, default: int) -> int:
        try:
            value = int(getattr(self._cfg, name, default))
        except Exception:
            value = default
        return max(1, value)

    def _item_id_for(self, item: FeedItem) -> str:
        try:
            item_id = self._decide.item_id_for(item)
            if isinstance(item_id, str) and item_id.strip():
                return item_id
        except Exception:
            pass
        return compute_item_id(item)


def _heuristic_state_summary_tag(message: str) -> str:
    text = (message or "").strip().lower()
    if not text:
        return "none"

    def has_any(words: tuple[str, ...]) -> bool:
        return any(word in text for word in words)

    if has_any(("早点睡", "睡眠", "熬夜", "先睡", "休息一下", "去睡")):
        return "sleep_concern"
    if has_any(("喝水", "站起来", "活动一下", "别久坐", "休息会", "身体")):
        return "health_nudge"
    if has_any(("面试", "八股", "力扣", "月底")) and has_any(
        ("焦虑", "压力", "别太逼", "不用慌", "底子在", "先歇", "放轻松", "烦")
    ):
        return "interview_anxiety_reassurance"
    if has_any(("别太逼", "不用慌", "底子在", "先歇", "放轻松", "撑住", "别慌")):
        return "general_encouragement"
    return "none"


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
