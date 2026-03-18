from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal
from uuid import uuid4

from proactive.event import AlertEvent, ContentEvent, GenericAlertEvent, GenericContentEvent
from proactive.item_id import compute_item_id, compute_source_key
from proactive.energy import (
    composite_score,
    d_content,
    d_energy,
    d_recent,
    random_weight,
)
from proactive.json_utils import extract_json_object
from proactive.presence import PresenceStore
from proactive.anyaction import AnyActionGate
from proactive.sender import ProactiveSendMeta, ProactiveSourceRef
from proactive.sensor import RecentProactiveMessage
from proactive.state import ProactiveStateStore
from proactive.skill_action import SkillActionRunner
from core.common.strategy_trace import build_strategy_trace_envelope
from core.observe.events import ProactiveDecisionTrace

logger = logging.getLogger(__name__)

_TOPIC_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2,8}")
_TOPIC_STOPWORDS = frozenset(
    {
        "更新",
        "上线",
        "发布",
        "视频",
        "新闻",
        "消息",
        "内容",
        "官方",
        "作者",
        "活动",
        "版本",
        "game",
        "games",
        "news",
        "update",
        "review",
    }
)
class _PseudoDecision:
    def __init__(self, message: str) -> None:
        self.message = message
        self.evidence_item_ids: list[str] = []


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
    preference_block: str = ""

    @classmethod
    def empty(cls, fallback_reason: str = "") -> "ProactiveRetrievedMemory":
        return cls(fallback_reason=fallback_reason)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _excerpt_text(text: str, limit: int = 80) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _build_recent_proactive_context_signal(
    recent_proactive: list[RecentProactiveMessage],
    *,
    last_user_at: datetime | None,
    candidate_items_count: int,
) -> dict[str, object]:
    # 1. 先筛出"自上次用户发言后"的主动消息，识别当前沉默期里 agent 已经主动续写了几次。
    active_since_user: list[RecentProactiveMessage] = []
    for msg in recent_proactive:
        ts = getattr(msg, "timestamp", None)
        if last_user_at is not None and ts is not None and ts <= last_user_at:
            continue
        active_since_user.append(msg)
    latest = active_since_user[-1] if active_since_user else None
    # 2. 再把"连续主动续写"的强弱信号压成结构化字段，交给后续决策判断是否该收住。
    return {
        "exists": latest is not None,
        "count_since_last_user": len(active_since_user),
        "already_followed_up": len(active_since_user) >= 1,
        "followup_fatigue": "high" if len(active_since_user) >= 2 else ("medium" if len(active_since_user) == 1 else "none"),
        "has_new_feed": candidate_items_count > 0,
        "latest_excerpt": (
            _excerpt_text(getattr(latest, "content", "")) if latest is not None else ""
        ),
    }


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
    tick_id: str = ""
    now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_key: str = ""
    target_last_user: datetime | None = None
    last_proactive_at: datetime | None = None


@dataclass
class SenseSnapshot:
    sleep_ctx: Any = None
    health_events: list[AlertEvent] = field(default_factory=list)
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
    items: list[ContentEvent] = field(default_factory=list)
    new_items: list[ContentEvent] = field(default_factory=list)
    new_entries: list[tuple[str, str]] = field(default_factory=list)
    semantic_duplicate_entries: list[tuple[str, str]] = field(default_factory=list)
    has_memory: bool = False
    background_context: list[dict] = field(default_factory=list)
    # 三源分离
    alert_items: list[AlertEvent] = field(default_factory=list)
    content_items: list[ContentEvent] = field(default_factory=list)
    selected_primary_source: str = "none"  # alert|content|context|none
    context_mode: str = "none"  # none|assist|context_only
    # Evidence-First Research
    research_result: object | None = None  # ResearchResult


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
    preference_block: str = ""  # 偏好专项 RAG 结果，独立字段传给 score_features
    prefetch_urls: list[str] = field(default_factory=list)
    compose_no_content: bool = False
    judge_dims: dict[str, object] = field(default_factory=dict)
    judge_final_score: float | None = None
    judge_vetoed_by: str | None = None


@dataclass
class ActSnapshot:
    compose_items: list[ContentEvent] = field(default_factory=list)
    compose_entries: list[tuple[str, str]] = field(default_factory=list)
    state_summary_tag: str = "none"
    source_refs: list[ProactiveSourceRef] = field(default_factory=list)
    high_events: list[AlertEvent] = field(default_factory=list)
    evidence_item_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceBundle:
    source_items: list[ContentEvent]
    source_entries: list[tuple[str, str]]
    evidence_items: list[ContentEvent]
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
class GateSenseResult:
    proceed: bool
    return_score: float | None
    reason_code: Literal[
        "continue",
        "quota_exhausted",
        "scheduler_reject",
        "health_fast_path",
        "below_threshold",
    ]
    sleep_state: str
    sleep_available: bool
    health_event_count: int
    energy: float
    recent_count: int
    interruptibility: float
    interrupt_factor: float
    sleep_mod: float


@dataclass(frozen=True)
class EvaluateResult:
    proceed: bool
    return_score: float | None
    reason_code: Literal[
        "continue",
        "draw_score_below_threshold",
        "draw_score_force_reflect",
        "no_valid_source",
    ]
    base_score: float
    draw_score: float
    force_reflect: bool
    total_items: int
    discovered_count: int
    selected_count: int
    semantic_duplicate_count: int
    has_memory: bool


@dataclass(frozen=True)
class _ScoreDecision:
    proceed: bool
    return_score: float | None
    reason_code: Literal[
        "continue",
        "draw_score_below_threshold",
        "draw_score_force_reflect",
    ]
    base_score: float
    draw_score: float
    force_reflect: bool


@dataclass(frozen=True)
class ComposeResult:
    proceed: bool
    return_score: float | None
    reason_code: Literal["continue", "compose_no_content", "judge_reject", "no_valid_source"]
    decision_message: str
    compose_no_content: bool
    history_gate_reason: str
    history_scope_mode: str


@dataclass(frozen=True)
class GuardResult:
    should_send: bool
    return_score: float | None
    reason_code: Literal[
        "sent_ready",
        "judge_reject",
        "delivery_dedupe",
        "state_summary_repeat",
        "message_dedupe",
        "passive_busy",
    ]
    delivery_key: str | None
    judge_dims: dict[str, object]
    judge_final_score: float | None
    judge_vetoed_by: str | None


class ProactiveTick:
    """主动循环引擎：编排一次完整 tick，不直接依赖 ProactiveLoop。"""

    def __init__(
        self,
        *,
        cfg: Any,
        state: ProactiveStateStore,
        presence: PresenceStore | None,
        rng: Any,
        sensor: Any | None = None,
        decide: Any | None = None,
        composer: Any | None = None,
        judge: Any | None = None,
        sender: Any | None = None,
        memory_retriever: Any | None = None,
        sense: Any | None = None,
        act: Any | None = None,
        memory_retrieval: Any | None = None,
        anyaction: AnyActionGate | None = None,
        message_deduper: Any | None = None,
        skill_action_runner: SkillActionRunner | None = None,
        provider: Any | None = None,
        model: str = "",
        light_provider: Any | None = None,
        light_model: str = "",
        passive_busy_fn: Callable[[str], bool] | None = None,
        stage_trace_writer: Callable[[dict[str, Any]], None] | None = None,
        observe_writer: Any | None = None,
        tool_registry: dict | None = None,
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._presence = presence
        self._rng = rng
        self._sense = sensor or sense
        self._decide = decide
        self._composer = composer or getattr(decide, "_composer", None)
        self._judge = judge or getattr(decide, "_judge", None)
        self._act = sender or act
        self._memory_retrieval = memory_retriever or memory_retrieval
        self._anyaction = anyaction
        self._message_deduper = message_deduper
        self._skill_action_runner = skill_action_runner
        self._provider = provider
        self._model = model
        self._light_provider = light_provider
        self._light_model = light_model
        self._tool_registry = tool_registry
        # 可选：AgentLoop 注入的被动处理信号，用于跳过与被动回复并发的主动发送
        self._passive_busy_fn = passive_busy_fn
        self._trace_writer = stage_trace_writer
        self._observe_writer = observe_writer

    async def tick(self) -> float | None:
        logger.debug("[proactive] tick 开始")
        ctx = DecisionContext()
        ctx.state.tick_id = uuid4().hex

        gate_result = await self._gate_and_sense(ctx)
        self._trace(ctx, stage="gate_and_sense", result=gate_result)
        if not gate_result.proceed:
            return gate_result.return_score

        evaluate_result = await self._evaluate(ctx)
        self._trace(ctx, stage="evaluate", result=evaluate_result)
        if not evaluate_result.proceed:
            return evaluate_result.return_score

        # Research 阶段：对 content/alert 主源进行事实检索
        research_result = await self._research(ctx)
        if research_result is not None:
            ctx.ensure_fetch().research_result = research_result
            self._trace(ctx, stage="research", result=research_result)

        compose_result = await self._compose(ctx)
        self._trace(ctx, stage="compose", result=compose_result)
        if not compose_result.proceed:
            return compose_result.return_score

        return await self._judge_and_send(ctx)

    def _load_alert_events(self) -> list[AlertEvent]:
        """从 MCP 配置的告警源拉取 alert 事件。"""
        try:
            from proactive import mcp_sources

            return [
                GenericAlertEvent.from_mcp_payload(p)
                for p in mcp_sources.fetch_alert_events()
            ]
        except Exception as _mcp_err:
            logger.warning("[proactive] MCP alert 拉取失败: %s", _mcp_err)
            return []

    async def _gate_and_sense(self, ctx: DecisionContext) -> GateSenseResult:
        """本地计算阶段：gate + sense + pre_score。"""
        state = ctx.state
        sense = ctx.ensure_sense()
        score = ctx.ensure_score()
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
        )

        # 2. 先探测 alert；无 alert 时才需要走 anyaction gate。
        sense.health_events = self._load_alert_events()
        if not sense.health_events and self._cfg.anyaction_enabled and self._anyaction:
            should_act, meta = self._anyaction.should_act(
                now_utc=ctx.state.now_utc,
                last_user_at=self._sense.last_user_at(),
            )
            if not should_act:
                logger.info(
                    "[proactive] gate_result=reject selected_action=null meta=%s", meta
                )
                reason = meta.get("reason", "")
                return GateSenseResult(
                    proceed=False,
                    return_score=0.0 if reason == "quota_exhausted" else None,
                    reason_code=(
                        "quota_exhausted"
                        if reason == "quota_exhausted"
                        else "scheduler_reject"
                    ),
                    sleep_state="unavailable",
                    sleep_available=False,
                    health_event_count=0,
                    energy=0.0,
                    recent_count=0,
                    interruptibility=1.0,
                    interrupt_factor=1.0,
                    sleep_mod=1.0,
                )
        elif sense.health_events:
            logger.info(
                "[proactive] gate_result=pass reason=alert_bypass alert_count=%d",
                len(sense.health_events),
            )

        # 3. 再刷新睡眠上下文，确保 Fitbit 相关信号是本轮最新值。
        refreshed = bool(getattr(self._sense, "refresh_sleep_context", lambda: False)())
        if refreshed:
            logger.debug("[proactive] fitbit 上下文已在本轮决策前主动刷新")
        sense.sleep_ctx = getattr(self._sense, "sleep_context", lambda: None)()

        # 4. 采集能量、近期消息和 interruptibility，形成 score 的基础输入。
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

        # 5. 用睡眠修正项微调 interrupt_factor，避免深睡时仍然太激进。
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
        # 6. 最后只用本地信号计算 pre_score，尽量在访问 MCP/LLM 之前早退。
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

        high_events = [e for e in sense.health_events if e.is_urgent()]
        if high_events:
            score.force_reflect = True
            if score.pre_score < self._cfg.score_pre_threshold:
                logger.info(
                    "[proactive] health fast-path: pre_score=%.3f 低于阈值但存在 %d 个 high 事件，强制继续",
                    score.pre_score,
                    len(high_events),
                )
                return GateSenseResult(
                    proceed=True,
                    return_score=None,
                    reason_code="health_fast_path",
                    sleep_state=sleep_state,
                    sleep_available=sleep_available,
                    health_event_count=len(sense.health_events),
                    energy=sense.energy,
                    recent_count=len(sense.recent),
                    interruptibility=sense.interruptibility,
                    interrupt_factor=sense.interrupt_factor,
                    sleep_mod=sense.sleep_mod,
                )

        # 7. pre_score 太低时不再进入 MCP / LLM 主链路，直接尝试 skill action。
        if score.pre_score < self._cfg.score_pre_threshold:
            logger.info(
                "[proactive] pre_score 过低（%.3f < %.2f），跳过 chat，尝试 skill action",
                score.pre_score,
                self._cfg.score_pre_threshold,
            )
            await self._try_skill_action(now_utc=state.now_utc)
            return GateSenseResult(
                proceed=False,
                return_score=score.pre_score,
                reason_code="below_threshold",
                sleep_state=sleep_state,
                sleep_available=sleep_available,
                health_event_count=len(sense.health_events),
                energy=sense.energy,
                recent_count=len(sense.recent),
                interruptibility=sense.interruptibility,
                interrupt_factor=sense.interrupt_factor,
                sleep_mod=sense.sleep_mod,
            )

        return GateSenseResult(
            proceed=True,
            return_score=None,
            reason_code="continue",
            sleep_state=sleep_state,
            sleep_available=sleep_available,
            health_event_count=len(sense.health_events),
            energy=sense.energy,
            recent_count=len(sense.recent),
            interruptibility=sense.interruptibility,
            interrupt_factor=sense.interrupt_factor,
            sleep_mod=sense.sleep_mod,
        )

    async def _load_content_snapshot(
        self,
        ctx: DecisionContext,
    ) -> tuple[list[ContentEvent], list[tuple[str, str]], list[dict]]:
        """拉取候选内容与背景上下文，供 evaluate 阶段统一消费。"""
        # 1. 先导入 MCP 模块；导入失败时直接退化为空快照。
        try:
            from proactive import mcp_sources as _mcp_sources
        except Exception as _import_err:
            logger.warning("[proactive] mcp_sources 导入失败: %s", _import_err)
            return [], [], []

        # 2. 再读取 content 源，并显式带上 ack_server 作为 source_key 前缀。
        try:
            items = [
                GenericContentEvent.from_mcp_payload(p)
                for p in _mcp_sources.fetch_content_events()
            ]
        except Exception as _mcp_err:
            logger.warning("[proactive] MCP content 拉取失败: %s", _mcp_err)
            items = []
        entries = [
            (
                f"mcp:{getattr(event, '_ack_server', None) or event.source_name}:{event.event_id}",
                self._item_id_for(event),
            )
            for event in items
        ]
        try:
            background_context = _mcp_sources.fetch_context_data()
        except Exception as _ctx_err:
            logger.warning("[proactive] MCP context 拉取失败: %s", _ctx_err)
            background_context = []
        return items, entries, background_context

    async def _evaluate(self, ctx: DecisionContext) -> EvaluateResult:
        """MCP I/O + score 阶段：拉候选、做 cooldown 过滤、计算分数。"""
        state = ctx.state
        fetch = ctx.ensure_fetch()
        # 1. 先拉取 MCP 候选和背景上下文，统一写入 fetch snapshot。
        fetch.items = []
        fetch.new_items = []
        fetch.new_entries = []
        fetch.semantic_duplicate_entries = []
        fetch.items, fetch.new_entries, fetch.background_context = (
            await self._load_content_snapshot(ctx)
        )
        fetch.new_items = list(fetch.items)

        # 2. 再做 rejection cooldown 过滤，避免刚被拒过的内容立刻重试。
        cooldown_hours = getattr(self._cfg, "llm_reject_cooldown_hours", 0)
        if cooldown_hours > 0:
            filtered_events: list[ContentEvent] = []
            filtered_entries: list[tuple[str, str]] = []
            for event, (source_key, item_id) in zip(fetch.new_items, fetch.new_entries):
                if self._state.is_rejection_cooled(
                    source_key=source_key,
                    item_id=item_id,
                    ttl_hours=cooldown_hours,
                    now=ctx.state.now_utc,
                ):
                    logger.debug(
                        "[proactive] evaluate rejection_cooldown 跳过 source=%s item_id=%s ttl_hours=%d",
                        source_key,
                        item_id[:16],
                        cooldown_hours,
                    )
                    continue
                filtered_events.append(event)
                filtered_entries.append((source_key, item_id))
            fetch.items = filtered_events
            fetch.new_items = filtered_events
            fetch.new_entries = filtered_entries
        logger.debug("[proactive] 从 MCP 拉取到 %d 条内容", len(fetch.items))

        # 3. 分离三类源并决定主源
        sense = ctx.ensure_sense()
        fetch.alert_items = sense.health_events  # alert 来自 sense 阶段
        fetch.content_items = fetch.new_items  # content 来自 MCP content
        # context 来自 background_context（已在 _load_content_snapshot 中加载）

        # 按优先级决定主源
        has_alert = len(fetch.alert_items) > 0
        has_content = len(fetch.content_items) > 0
        has_context = len(fetch.background_context) > 0

        context_as_assist_enabled = getattr(self._cfg, "context_as_assist_enabled", True)
        context_only_enabled = getattr(self._cfg, "context_only_enabled", True)

        if has_alert:
            fetch.selected_primary_source = "alert"
            fetch.context_mode = "assist" if (context_as_assist_enabled and has_context) else "none"
            logger.info("[proactive] 主源=alert context_mode=%s", fetch.context_mode)
        elif has_content:
            fetch.selected_primary_source = "content"
            fetch.context_mode = "assist" if (context_as_assist_enabled and has_context) else "none"
            logger.info("[proactive] 主源=content context_mode=%s", fetch.context_mode)
        elif has_context and context_only_enabled:
            # context-only 需要检查配额
            session_key = state.session_key or ""
            context_only_daily_max = getattr(self._cfg, "context_only_daily_max", 1)
            context_only_min_interval_hours = getattr(self._cfg, "context_only_min_interval_hours", 12)

            count_24h = self._state.count_context_only_in_window(session_key, 24, state.now_utc)
            if count_24h >= context_only_daily_max:
                fetch.selected_primary_source = "none"
                fetch.context_mode = "none"
                logger.info("[proactive] context-only 24h 配额已满 count=%d max=%d", count_24h, context_only_daily_max)
            else:
                last_context_only_at = self._state.get_last_context_only_at(session_key)
                if last_context_only_at is not None:
                    elapsed_hours = (state.now_utc - last_context_only_at).total_seconds() / 3600
                    if elapsed_hours < context_only_min_interval_hours:
                        fetch.selected_primary_source = "none"
                        fetch.context_mode = "none"
                        logger.info("[proactive] context-only 最小间隔未满足 elapsed=%.1fh min=%dh", elapsed_hours, context_only_min_interval_hours)
                    else:
                        fetch.selected_primary_source = "context"
                        fetch.context_mode = "context_only"
                        logger.info("[proactive] 主源=context context_mode=context_only")
                else:
                    fetch.selected_primary_source = "context"
                    fetch.context_mode = "context_only"
                    logger.info("[proactive] 主源=context context_mode=context_only (首次)")
        else:
            fetch.selected_primary_source = "none"
            fetch.context_mode = "none"
            logger.info("[proactive] 无可用主源")

        # 如果主源是 none，提前返回
        if fetch.selected_primary_source == "none":
            logger.info("[proactive] 无可用主源，跳过本轮")
            await self._try_skill_action(now_utc=state.now_utc)
            return EvaluateResult(
                proceed=False,
                return_score=None,
                reason_code="no_valid_source",
                base_score=0.0,
                draw_score=0.0,
                force_reflect=False,
                total_items=len(fetch.items),
                discovered_count=len(fetch.new_items),
                selected_count=0,
                semantic_duplicate_count=len(fetch.semantic_duplicate_entries),
                has_memory=fetch.has_memory,
            )

        # 4. 最后计算 base_score / draw_score，并决定是否进入 LLM 路径。
        fetch.has_memory = self._sense.has_global_memory()
        w_random = self._compute_score_snapshot(ctx)
        self._refresh_presence_state(ctx)
        result = self._build_score_result(ctx, w_random=w_random)
        if result.reason_code == "draw_score_below_threshold":
            logger.info("[proactive] draw_score 未过门槛，跳过本轮反思")
            logger.info("[proactive] selected_action=idle reason=draw_score")
            await self._try_skill_action(now_utc=state.now_utc)
        return EvaluateResult(
            proceed=result.proceed,
            return_score=result.return_score,
            reason_code=result.reason_code,
            base_score=result.base_score,
            draw_score=result.draw_score,
            force_reflect=result.force_reflect,
            total_items=len(fetch.items),
            discovered_count=len(fetch.new_items),
            selected_count=len(fetch.new_items),
            semantic_duplicate_count=len(fetch.semantic_duplicate_entries),
            has_memory=fetch.has_memory,
        )

    async def _research(self, ctx: DecisionContext) -> object | None:
        """Research 阶段：对候选主源进行事实检索。

        Returns:
            ResearchResult | None: 检索结果，None 表示跳过 research
        """
        fetch = ctx.ensure_fetch()

        # 检查是否启用 research
        if not getattr(self._cfg, "research_enabled", True):
            logger.debug("[proactive] research_enabled=False，跳过 research")
            return None

        # 检查主源类型
        primary_source = fetch.selected_primary_source
        if primary_source == "none":
            return None

        # alert 可配置跳过
        if primary_source == "alert" and getattr(self._cfg, "research_skip_alert", True):
            logger.info("[proactive] alert 主源跳过 research")
            return None

        # context-only 根据配置决定
        if primary_source == "context" and not getattr(self._cfg, "research_apply_on_context_only", False):
            logger.info("[proactive] context-only 跳过 research")
            return None

        # 确定候选 items
        if primary_source == "alert":
            items = fetch.alert_items
        elif primary_source == "content":
            items = fetch.content_items
        elif primary_source == "context":
            # context-only: 从 background_context 构造 research items
            items = self._build_context_research_items(fetch)
        else:
            items = []

        if not items:
            logger.info("[proactive] 无候选 items，跳过 research")
            return None

        # 调用 Researcher
        try:
            from proactive.researcher import Researcher

            # 获取 provider 和 model
            provider = getattr(self, "_provider", None)
            model = getattr(self, "_model", "") or getattr(self._cfg, "model", "")
            tool_registry = getattr(self, "_tool_registry", None)

            researcher = Researcher(
                max_iterations=getattr(self._cfg, "research_max_iterations", 10),
                allowed_tools=getattr(self._cfg, "research_tools", ["web_search", "web_fetch", "read_file"]),
                min_body_chars=getattr(self._cfg, "research_min_body_chars", 200),
                timeout_seconds=getattr(self._cfg, "research_timeout_seconds", 30),
                provider=provider,
                model=model,
                tool_registry=tool_registry,
                include_all_mcp_tools=getattr(self._cfg, "research_include_all_mcp_tools", False),
            )

            logger.info("[proactive] 开始 research primary_source=%s items_count=%d", primary_source, len(items))
            result = await researcher.research(items=items, primary_source=primary_source)
            logger.info(
                "[proactive] research 完成 status=%s rounds=%d evidence_count=%d",
                result.status,
                result.rounds_used,
                len(result.evidence),
            )
            return result
        except Exception as e:
            logger.warning("[proactive] research 失败: %s", e, exc_info=True)
            # 返回 error 状态的 ResearchResult
            from proactive.researcher import ResearchResult
            return ResearchResult(
                status="error",
                rounds_used=0,
                reason=f"research_exception: {e}",
            )

    def _build_context_research_items(self, fetch: FetchSnapshot) -> list:
        """从 background_context 构造 research items。

        Returns:
            list: 候选项列表，每个项包含 title/content/url
        """
        if not fetch.background_context:
            return []

        items = []
        # background_context 可能是 dict 或 list
        if isinstance(fetch.background_context, dict):
            # 格式: {source_name: {topic, summary, items: [{title, content, ...}]}}
            for source_name, context_data in fetch.background_context.items():
                topic = context_data.get("topic", "")
                summary = context_data.get("summary", "")
                context_items = context_data.get("items", [])

                # 如果有具体的 items，使用 items
                if context_items:
                    for item in context_items[:3]:  # 最多取前 3 个
                        items.append(type('obj', (object,), {
                            'source_name': source_name,
                            'source_type': 'context',
                            'title': item.get("title", topic),
                            'content': item.get("content", summary),
                            'url': item.get("url", ""),
                        })())
                # 否则使用 topic/summary 构造一个候选项
                elif topic or summary:
                    items.append(type('obj', (object,), {
                        'source_name': source_name,
                        'source_type': 'context',
                        'title': topic or "用户最近活动",
                        'content': summary or topic,
                        'url': "",
                    })())
        elif isinstance(fetch.background_context, list):
            # 格式: [{_source, topic, summary, ...}] 或 Steam 格式
            for context_data in fetch.background_context[:5]:
                source_name = context_data.get("_source", "context")

                # 处理 Steam 类型的 context（原始结构：games/realtime）
                if source_name == "steam" and context_data.get("available") is not False:
                    realtime = context_data.get("realtime", {})
                    games = context_data.get("games", [])
                    currently_playing = realtime.get("currently_playing")

                    if currently_playing:
                        items.append(type('obj', (object,), {
                            'source_name': 'steam',
                            'source_type': 'context',
                            'title': f"Steam: 正在游玩 {currently_playing}",
                            'content': f"用户当前正在 Steam 上游玩 {currently_playing}",
                            'url': "",
                        })())
                    elif games:
                        # 提取近期活跃游戏（recent_2w_hours >= 5）
                        active_games = []
                        for g in games[:5]:
                            hours = float(g.get("recent_2w_hours", 0) or 0)
                            if hours >= 5:
                                active_games.append(g.get("name", ""))
                        if active_games:
                            items.append(type('obj', (object,), {
                                'source_name': 'steam',
                                'source_type': 'context',
                                'title': f"Steam: 近期活跃游戏",
                                'content': f"用户近期在 Steam 上活跃游玩的游戏: {', '.join(active_games[:3])}",
                                'url': "",
                            })())
                # 处理通用 topic/summary 结构
                else:
                    topic = context_data.get("topic", "")
                    summary = context_data.get("summary", "")
                    if topic or summary:
                        items.append(type('obj', (object,), {
                            'source_name': source_name,
                            'source_type': 'context',
                            'title': topic or "用户最近活动",
                            'content': summary or topic,
                            'url': "",
                        })())

        logger.info("[proactive] context-only 构造 research items: %d", len(items))
        return items

    async def _compose(self, ctx: DecisionContext) -> ComposeResult:
        sense = ctx.ensure_sense()
        fetch = ctx.ensure_fetch()
        score = ctx.ensure_score()
        decide = ctx.ensure_decide()
        act = ctx.ensure_act()

        # 0. 根据主源选择 compose 候选
        if fetch.selected_primary_source == "alert":
            # alert 主源：将 alert 转换为 ContentEvent 格式供 compose 使用
            alert_as_content = []
            for alert in fetch.alert_items:
                # 将 AlertEvent 转换为 ContentEvent 格式
                content_event = type('obj', (object,), {
                    'source_name': getattr(alert, 'source_name', 'alert'),
                    'source_type': getattr(alert, 'source_type', 'alert'),
                    'title': getattr(alert, 'title', ''),
                    'content': getattr(alert, 'content', ''),
                    'url': getattr(alert, 'url', None),
                    'event_id': getattr(alert, 'event_id', ''),
                    'severity': getattr(alert, 'severity', 'high'),
                    'published_at': getattr(alert, 'published_at', None),
                })()
                alert_as_content.append(content_event)
            # 覆盖 fetch.new_items，确保 compose 使用 alert 作为主源
            fetch.new_items = alert_as_content
            fetch.new_entries = [(f"alert:{a.event_id}", a.event_id) for a in fetch.alert_items]
            logger.info("[proactive] compose 使用 alert 作为主源 count=%d", len(alert_as_content))
        elif fetch.selected_primary_source == "content":
            # content 主源：保持原有逻辑
            logger.info("[proactive] compose 使用 content 作为主源 count=%d", len(fetch.content_items))
        elif fetch.selected_primary_source == "context":
            # context-only：清空 items，让 compose 基于 background_context 生成
            fetch.new_items = []
            fetch.new_entries = []
            logger.info("[proactive] compose 使用 context-only 模式")
        else:
            # 不应该到这里，因为 evaluate 阶段已经过滤了
            logger.warning("[proactive] compose 阶段遇到无效主源: %s", fetch.selected_primary_source)
            return ComposeResult(
                proceed=False,
                return_score=score.base_score,
                reason_code="no_valid_source",
                decision_message="",
                compose_no_content=False,
                history_gate_reason="",
                history_scope_mode="",
            )

        # 1. 先补 decision signals 和 retrieval memory，后续 compose/judge 都只读 ctx。
        self._populate_decision_signals(ctx)
        await self._retrieve_decision_memory(ctx)
        # 2. 再准备 compose 候选，并做 pre-veto 判定。
        self._prepare_feature_compose_candidates(ctx)
        compose_entries = act.compose_entries or self._primary_candidate_entries(
            fetch.new_entries
        )
        age_hours = self._candidate_age_hours(fetch.new_items, now_utc=ctx.state.now_utc)
        judge_port = self._judge_port()
        pre_veto = getattr(judge_port, "pre_compose_veto", lambda **_: None)(
            age_hours=age_hours,
            sent_24h=score.sent_24h,
            interrupt_factor=sense.interrupt_factor,
        )
        if pre_veto:
            decide.should_send = False
            decide.judge_vetoed_by = pre_veto
            decide.judge_final_score = 0.0
            decide.judge_dims = {}
            if pre_veto != "balance":
                self._state.mark_rejection_cooldown(
                    compose_entries,
                    hours=getattr(self._cfg, "llm_reject_cooldown_hours", 0),
                )
            return ComposeResult(
                proceed=False,
                return_score=score.base_score,
                reason_code="judge_reject",
                decision_message="",
                compose_no_content=False,
                history_gate_reason=decide.history_gate_reason,
                history_scope_mode=decide.history_scope_mode,
            )
        no_content_token = str(
            getattr(self._cfg, "compose_no_content_token", "<no_content/>")
        )
        compose_recent = self._compose_recent_messages(
            recent=sense.recent,
            has_compose_candidates=bool(act.compose_items),
        )
        compose_port = self._compose_port()
        decide.decision_message = await compose_port.compose_for_judge(
            items=act.compose_items,
            recent=compose_recent,
            preference_block=decide.preference_block,
            no_content_token=no_content_token,
            background_context=fetch.background_context if fetch.context_mode == "context_only" else None,
            research_result=fetch.research_result,
            fail_policy=getattr(self._cfg, "research_fail_policy", "drop"),
            transparent_message=getattr(self._cfg, "research_transparent_message", ""),
        )
        message = (decide.decision_message or "").strip()
        if message and message != no_content_token:
            decide.compose_no_content = False
            return ComposeResult(
                proceed=True,
                return_score=None,
                reason_code="continue",
                decision_message=message,
                compose_no_content=False,
                history_gate_reason=decide.history_gate_reason,
                history_scope_mode=decide.history_scope_mode,
            )
        decide.compose_no_content = True
        decide.should_send = False
        decide.decision_message = ""
        self._state.mark_rejection_cooldown(compose_entries, hours=8)
        return ComposeResult(
            proceed=False,
            return_score=score.base_score,
            reason_code="compose_no_content",
            decision_message="",
            compose_no_content=True,
            history_gate_reason=decide.history_gate_reason,
            history_scope_mode=decide.history_scope_mode,
        )

    async def _judge_and_send(self, ctx: DecisionContext) -> float | None:
        state = ctx.state
        sense = ctx.ensure_sense()
        fetch = ctx.ensure_fetch()
        score = ctx.ensure_score()
        decide = ctx.ensure_decide()
        act = ctx.ensure_act()
        # 1. 先调用 judge，并把判定结果和证据集落进 ctx。
        evidence = self._build_evidence_bundle(ctx)
        act.source_refs = evidence.source_refs
        act.evidence_item_ids = evidence.evidence_item_ids
        compose_entries = act.compose_entries or self._primary_candidate_entries(
            fetch.new_entries
        )

        age_hours = self._candidate_age_hours(fetch.new_items, now_utc=ctx.state.now_utc)
        recent_proactive_text = self._recent_proactive_text()
        judge_port = self._judge_port()
        judge_result = (
            await judge_port.judge_message(
                message=decide.decision_message,
                recent=sense.recent,
                recent_proactive_text=recent_proactive_text,
                preference_block=decide.preference_block,
                age_hours=age_hours,
                sent_24h=score.sent_24h,
                interrupt_factor=sense.interrupt_factor,
            )
            if judge_port is not None and hasattr(judge_port, "judge_message")
            else None
        )
        if judge_result is None:
            decide.should_send = True
            decide.judge_final_score = 1.0
            decide.judge_dims = {}
            decide.judge_vetoed_by = None
        else:
            decide.judge_dims = {
                "deterministic": dict(
                    getattr(judge_result, "dims_deterministic", {}) or {}
                ),
                "llm": dict(getattr(judge_result, "dims_llm", {}) or {}),
                "llm_raw": dict(getattr(judge_result, "dims_llm_raw", {}) or {}),
            }
            decide.judge_final_score = float(
                getattr(judge_result, "final_score", 0.0) or 0.0
            )
            decide.judge_vetoed_by = getattr(judge_result, "vetoed_by", None)
            decide.should_send = bool(getattr(judge_result, "should_send", False))

        # context-only 应用更高阈值（但有证据时降低阈值）
        if fetch.context_mode == "context_only":
            # 检查是否有 research 证据
            has_evidence = (
                fetch.research_result is not None
                and getattr(fetch.research_result, "status", "") == "success"
                and len(getattr(fetch.research_result, "evidence", [])) > 0
            )

            if has_evidence:
                # 有证据时使用较低阈值
                context_only_threshold = getattr(self._cfg, "context_only_judge_threshold_with_evidence", 0.68)
                logger.info(
                    "[proactive] context-only 有证据，使用较低阈值 threshold=%.3f",
                    context_only_threshold,
                )
            else:
                # 无证据时使用默认阈值
                context_only_threshold = getattr(self._cfg, "context_only_judge_threshold", 0.72)

            if decide.judge_final_score < context_only_threshold:
                logger.info(
                    "[proactive] context-only judge 分数不足 score=%.3f threshold=%.3f has_evidence=%s",
                    decide.judge_final_score,
                    context_only_threshold,
                    has_evidence,
                )
                decide.should_send = False
                decide.judge_vetoed_by = "context_only_threshold"

        guard = GuardResult(
            should_send=decide.should_send,
            return_score=None if decide.should_send else score.base_score,
            reason_code="sent_ready" if decide.should_send else "judge_reject",
            delivery_key=None,
            judge_dims=decide.judge_dims,
            judge_final_score=decide.judge_final_score,
            judge_vetoed_by=decide.judge_vetoed_by,
        )

        # 2. judge 不通过时直接返回 base_score，并保留 rejection cooldown 行为。
        if not decide.should_send:
            if decide.judge_vetoed_by != "balance":
                self._state.mark_rejection_cooldown(
                    compose_entries,
                    hours=self._judge_rejection_cooldown_hours(decide.judge_vetoed_by),
                )
            self._trace(ctx, stage="judge_and_send", result=guard)
            return score.base_score

        # 3. 再做 dedupe / passive_busy 守卫；只要任一失败就中止发送。
        delivery_key = self._prepare_delivery_attempt(ctx, evidence)
        guard = GuardResult(
            should_send=True,
            return_score=None,
            reason_code="sent_ready",
            delivery_key=delivery_key,
            judge_dims=decide.judge_dims,
            judge_final_score=decide.judge_final_score,
            judge_vetoed_by=decide.judge_vetoed_by,
        )
        if state.session_key and self._state.is_delivery_duplicate(
            session_key=state.session_key,
            delivery_key=delivery_key,
            window_hours=self._cfg.delivery_dedupe_hours,
        ):
            self._consume_evidence_entries(evidence)
            guard = GuardResult(
                should_send=False,
                return_score=score.base_score,
                reason_code="delivery_dedupe",
                delivery_key=delivery_key,
                judge_dims=decide.judge_dims,
                judge_final_score=decide.judge_final_score,
                judge_vetoed_by=decide.judge_vetoed_by,
            )
            self._trace(ctx, stage="judge_and_send", result=guard)
            return score.base_score
        sense_port = getattr(self, "_sense", None)
        if sense_port is not None:
            recent_proactive = sense_port.collect_recent_proactive(
                getattr(self._cfg, "message_dedupe_recent_n", 5)
            )
        else:
            recent_proactive = []
        if not await self._rewrite_or_reject_repeated_state_summary(
            ctx,
            evidence,
            recent_proactive,
        ):
            guard = GuardResult(
                should_send=False,
                return_score=score.base_score,
                reason_code="state_summary_repeat",
                delivery_key=delivery_key,
                judge_dims=decide.judge_dims,
                judge_final_score=decide.judge_final_score,
                judge_vetoed_by=decide.judge_vetoed_by,
            )
            self._trace(ctx, stage="judge_and_send", result=guard)
            return score.base_score
        if not await self._passes_message_deduper(ctx, evidence, recent_proactive):
            guard = GuardResult(
                should_send=False,
                return_score=score.base_score,
                reason_code="message_dedupe",
                delivery_key=delivery_key,
                judge_dims=decide.judge_dims,
                judge_final_score=decide.judge_final_score,
                judge_vetoed_by=decide.judge_vetoed_by,
            )
            self._trace(ctx, stage="judge_and_send", result=guard)
            return score.base_score
        if (
            state.session_key
            and getattr(self, "_passive_busy_fn", None)
            and self._passive_busy_fn(state.session_key)
        ):
            guard = GuardResult(
                should_send=False,
                return_score=score.base_score,
                reason_code="passive_busy",
                delivery_key=delivery_key,
                judge_dims=decide.judge_dims,
                judge_final_score=decide.judge_final_score,
                judge_vetoed_by=decide.judge_vetoed_by,
            )
            self._trace(ctx, stage="judge_and_send", result=guard)
            return score.base_score

        # 4. 所有守卫通过后实际发送，并在成功时落地状态。
        self._trace(ctx, stage="judge_and_send", result=guard)
        decide = ctx.ensure_decide()
        act = ctx.ensure_act()
        sent = await self._act.send(
            decide.decision_message,
            ProactiveSendMeta(
                evidence_item_ids=evidence.evidence_item_ids,
                source_refs=act.source_refs,
                state_summary_tag=act.state_summary_tag,
            ),
        )
        if sent:
            self._finalize_successful_send(ctx, evidence, guard.delivery_key or "")
            self._emit_observe_decision(
                ctx,
                stage="send",
                reason_code="sent",
                should_send=True,
                action="chat",
                delivery_key=guard.delivery_key,
                delivery_attempted=True,
                delivery_result="sent",
            )
            return score.base_score
        self._emit_observe_decision(
            ctx,
            stage="send",
            reason_code="send_failed",
            should_send=True,
            action="idle",
            delivery_key=guard.delivery_key,
            delivery_attempted=True,
            delivery_result="send_failed",
        )
        return score.base_score

    def _judge_rejection_cooldown_hours(self, vetoed_by: str | None) -> int:
        if vetoed_by == "llm_dim":
            return 12
        return 8

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

    def _build_score_result(
        self, ctx: DecisionContext, *, w_random: float
    ) -> _ScoreDecision:
        """把 evaluate 阶段的分支判断统一映射成内部 score 决策。"""
        sense = ctx.ensure_sense()
        fetch = ctx.ensure_fetch()
        score = ctx.ensure_score()
        # 1. 无候选内容时不再硬退出，仍走 draw_score 门槛判断。
        # 能量和时间信号足够时（D_energy + D_recent），即使没有 feed 内容也可进入反思，
        # 允许 LLM 基于持久上下文（如 steam/github 等背景知识）主动发起对话。
        if not fetch.new_items and not sense.health_events and not score.force_reflect:
            logger.info("[proactive] 无候选内容，继续走 draw_score 门槛判断（关心模式）")
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
            return _ScoreDecision(
                proceed=False,
                return_score=score.base_score,
                reason_code="draw_score_below_threshold",
                base_score=score.base_score,
                draw_score=score.draw_score,
                force_reflect=score.force_reflect,
            )
        if score.draw_score < self._cfg.score_llm_threshold and score.force_reflect:
            logger.info("[proactive] draw_score 未过门槛，但命中兜底条件，继续反思")
            return _ScoreDecision(
                proceed=True,
                return_score=None,
                reason_code="draw_score_force_reflect",
                base_score=score.base_score,
                draw_score=score.draw_score,
                force_reflect=score.force_reflect,
            )
        return _ScoreDecision(
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
        # 1. 先把与"最近是否被打扰过"有关的时间差算出来。
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
        sense_port = getattr(self, "_sense", None)
        if sense_port is not None:
            recent_proactive = sense_port.collect_recent_proactive(
                getattr(self._cfg, "message_dedupe_recent_n", 5)
            )
        else:
            recent_proactive = []
        # 2. 再组织成决策共用的 decision_signals。
        sleep_signal: dict[str, object] = {"state": "unavailable"}
        if sense.sleep_ctx is not None:
            sleep_signal = {
                "state": sense.sleep_ctx.state,
                "prob": sense.sleep_ctx.prob,
                "data_lag_min": sense.sleep_ctx.data_lag_min,
                "available": sense.sleep_ctx.available,
            }
        decide.decision_signals = {
            "minutes_since_last_user": mins_since_last_user,
            "minutes_since_last_proactive": mins_since_last_proactive,
            "user_replied_after_last_proactive": replied_after_last_proactive,
            "proactive_sent_24h": score.sent_24h,
            "sleep": sleep_signal,
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
            "recent_proactive_context": _build_recent_proactive_context_signal(
                recent_proactive,
                last_user_at=state.target_last_user,
                candidate_items_count=len(fetch.new_items),
            ),
        }
        # background_context：持久背景感知数据（如 Steam 游戏活动），每次反思均可读取。
        # 注入条件：
        # 1. context_mode != "none" (assist 或 context_only) 时必须注入
        # 2. context_mode == "none" 时，仅当 context_as_assist_enabled=True 才注入
        should_inject_context = fetch.background_context and (
            fetch.context_mode != "none"
            or getattr(self._cfg, "context_as_assist_enabled", True)
        )
        if should_inject_context:
            processed_bg = _process_bg_context_sources(fetch.background_context)
            decide.decision_signals["background_context"] = {
                "_description": (
                    "用户自身近期行为数据，非外部资讯。每轮反思均可读取，无信息流内容时也可据此主动搭话。"
                    "recent_activity 字段为活动强度（heavy/moderate/light），不含精确时长。"
                    "currently_playing 非 null 时表示用户此刻正在游戏中（唯一实时信号）。"
                ),
                "sources": processed_bg,
            }
            # bg_context_quota：主 topic 冷却信号。
            last_main = self._state.get_bg_context_last_main_at()
            min_interval_hours = self._cfg.bg_context_main_topic_min_interval_hours
            if last_main is not None:
                elapsed_min = int((state.now_utc - last_main).total_seconds() / 60)
                min_interval_min = min_interval_hours * 60
                available = elapsed_min >= min_interval_min
                cooldown_remaining = max(0, min_interval_min - elapsed_min)
            else:
                available = True
                cooldown_remaining = 0
            decide.decision_signals["bg_context_quota"] = {
                "available": available,
                "cooldown_remaining_min": cooldown_remaining,
                "min_interval_hours": min_interval_hours,
            }
        # 3. 最后抽取高优先级告警事件，给后面的发送和 ack 路径使用。
        # alert_events：所有 AlertEvent，供 history gate / memory query 等通用路径使用。
        # health_events：source_type=="health_event" 的子集，保留给健康相关提示词兼容层。
        if sense.health_events:
            alert_signals = [e.to_signal_dict() for e in sense.health_events]
            decide.decision_signals["alert_events"] = alert_signals
            health_signals = [
                s for s in alert_signals if s.get("source_type") == "health_event"
            ]
            if health_signals:
                decide.decision_signals["health_events"] = health_signals
        act.high_events = [e for e in sense.health_events if e.is_urgent()]
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
        memory_retrieval = getattr(self, "_memory_retrieval", None)
        if memory_retrieval is not None:
            retrieved = await memory_retrieval.retrieve_proactive_context(
                session_key=state.session_key,
                channel=channel,
                chat_id=chat_id,
                items=fetch.new_items,
                recent=sense.recent,
                decision_signals=decide.decision_signals,
                is_crisis=score.is_crisis,
                tick_id=state.tick_id,
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
        decide.preference_block = retrieved.preference_block

    def _prepare_feature_compose_candidates(self, ctx: DecisionContext) -> None:
        fetch = ctx.ensure_fetch()
        act = ctx.ensure_act()
        decide = ctx.ensure_decide()
        ranked_items = self._rank_items_by_interest(
            fetch.new_items,
            decide.preference_block,
        )
        act.compose_items, act.compose_entries = self._select_compose_items(
            ranked_items,
            fetch.new_entries,
        )

    def _compose_recent_messages(
        self,
        *,
        recent: list[dict],
        has_compose_candidates: bool,
    ) -> list[dict]:
        # 1. 有内容候选时保留完整 recent，让模型正常结合上下文组织资讯消息。
        if has_compose_candidates:
            return recent
        # 2. 无内容候选时只保留用户消息，避免模型复述最近 assistant 主动发过的旧资讯。
        filtered = [msg for msg in recent if str(msg.get("role", "")) == "user"]
        # 3. 若近期没有用户消息，则显式返回空列表，交给 compose prompt 自行判断 no_content。
        return filtered

    def _candidate_age_hours(
        self,
        items: list[ContentEvent],
        *,
        now_utc: datetime,
    ) -> float:
        ages: list[float] = []
        for item in items:
            published = getattr(item, "published_at", None)
            if published is None:
                continue
            ages.append(max(0.0, (now_utc - published).total_seconds() / 3600.0))
        if not ages:
            return 24.0
        return min(ages)

    def _recent_proactive_text(self, limit: int = 5) -> str:
        sense_port = getattr(self, "_sense", None)
        if sense_port is None:
            return ""
        rows = sense_port.collect_recent_proactive(limit)
        lines = [str(getattr(row, "content", "") or "").strip() for row in rows]
        lines = [line for line in lines if line]
        return "\n---\n".join(lines)

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
            (
                decide.decision
                if decide.decision is not None
                else _PseudoDecision(message=decide.decision_message)
            ),
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

    async def _reject_and_try_skill_action(
        self,
        ctx: DecisionContext,
        evidence: EvidenceBundle,
    ) -> None:
        state = ctx.state
        await self._try_skill_action(now_utc=state.now_utc)

    def _consume_evidence_entries(self, evidence: EvidenceBundle) -> None:
        self._state.mark_items_seen(evidence.evidence_entries)
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

    def _compose_port(self) -> Any:
        port = getattr(self, "_composer", None)
        if port is not None and hasattr(port, "compose_for_judge"):
            return port
        port = getattr(self, "_decide", None)
        if port is not None and hasattr(port, "compose_for_judge"):
            return port
        port = getattr(self, "_judge", None)
        if port is not None and hasattr(port, "compose_for_judge"):
            return port
        raise AttributeError("compose_for_judge port is not configured")

    def _judge_port(self) -> Any | None:
        port = getattr(self, "_judge", None)
        if port is not None and (
            hasattr(port, "judge_message") or hasattr(port, "pre_compose_veto")
        ):
            return port
        port = getattr(self, "_decide", None)
        if port is not None and (
            hasattr(port, "judge_message") or hasattr(port, "pre_compose_veto")
        ):
            return port
        return None

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
            self._emit_observe_decision(
                ctx,
                stage="act",
                reason_code="state_summary_repeat",
                should_send=True,
                action="idle",
                delivery_attempted=False,
                delivery_result="state_summary_repeat",
            )
            await self._reject_and_try_skill_action(ctx, evidence)
            return False

        rewritten_tag = await self._classify_state_summary_tag(rewritten)
        if rewritten_tag == original_tag and rewritten_tag != "none":
            logger.info(
                "[proactive] state_summary_tag=%s 重写后仍重复，跳过发送",
                rewritten_tag,
            )
            self._emit_observe_decision(
                ctx,
                stage="act",
                reason_code="state_summary_repeat",
                should_send=True,
                action="idle",
                delivery_attempted=False,
                delivery_result="state_summary_repeat",
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
        self._emit_observe_decision(
            ctx,
            stage="act",
            reason_code="message_dedupe",
            should_send=True,
            action="idle",
            is_message_duplicate=True,
            delivery_attempted=False,
            delivery_result="message_dedupe",
        )
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
        fetch = ctx.ensure_fetch()
        if self._cfg.anyaction_enabled and getattr(self, "_anyaction", None):
            self._anyaction.record_action(now_utc=state.now_utc)
        self._consume_evidence_entries(evidence)
        # 若本次发送无 feed 证据（纯 background_context 驱动），更新主 topic 冷却时间。
        if not evidence.evidence_item_ids:
            self._state.mark_bg_context_main_send(state.now_utc)
        # 若本次是 context-only 发送，记录 context-only 状态
        if fetch.context_mode == "context_only" and state.session_key:
            self._state.mark_context_only_send(state.session_key, state.now_utc)
            logger.info(
                "[proactive] context-only 发送已记录 session=%s",
                state.session_key,
            )
        if state.session_key:
            self._state.mark_delivery(state.session_key, delivery_key)
        try:
            from proactive import mcp_sources
            # 只 ack 本次作为证据发出的条目（7天不再返回）。
            # 未用到的候选保持 eligible，下次 tick 继续参与评分，避免错过感兴趣的内容。
            mcp_sources.acknowledge_content_entries(evidence.evidence_entries)
        except Exception as _ack_err:
            logger.warning("[proactive] MCP content ack 失败: %s", _ack_err)
        if not sense.health_events:
            return
        try:
            from proactive import mcp_sources
            mcp_sources.acknowledge_events(sense.health_events)
        except Exception as _ack_err:
            logger.warning("[proactive] MCP ack 失败: %s", _ack_err)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trace(
        self,
        ctx: DecisionContext,
        *,
        stage: str,
        result: object,
    ) -> None:
        state = ctx.state
        payload = {
            "stage": stage,
            "result": _json_safe(asdict(result)) if is_dataclass(result) else {},
            "session_key": state.session_key,
        }
        trace_writer = getattr(self, "_trace_writer", None)
        if trace_writer is not None:
            try:
                trace_writer(
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
        self._emit_observe_decision(ctx, stage=stage, result=result)

    def _emit_observe_decision(
        self,
        ctx: DecisionContext,
        *,
        stage: str,
        result: object | None = None,
        reason_code: str | None = None,
        should_send: bool | None = None,
        action: str | None = None,
        delivery_key: str | None = None,
        is_delivery_duplicate: bool | None = None,
        is_message_duplicate: bool | None = None,
        delivery_attempted: bool | None = None,
        delivery_result: str | None = None,
        error: str | None = None,
    ) -> None:
        writer = getattr(self, "_observe_writer", None)
        if writer is None:
            return
        state = ctx.state
        fetch = ctx.fetch
        score = ctx.score
        decide = ctx.decide
        sense = ctx.sense
        candidate_item_ids = (
            [self._item_id_for(item) for item in fetch.new_items[:5]]
            if fetch is not None
            else []
        )
        decision = decide.decision if decide is not None else None
        decision_signals = decide.decision_signals if decide is not None else {}
        scores = (
            decision_signals.get("scores", {})
            if isinstance(decision_signals.get("scores", {}), dict)
            else {}
        )
        sleep_ctx = sense.sleep_ctx if sense is not None else None
        stage_result_json = (
            json.dumps(_json_safe(asdict(result)), ensure_ascii=False)
            if result is not None and is_dataclass(result)
            else None
        )
        decision_signals_json = (
            json.dumps(_json_safe(decision_signals), ensure_ascii=False)
            if decision_signals
            else None
        )
        # 候选内容：content events + alert events，完整原始字段塞进去
        candidates_json: str | None = None
        if fetch is not None or sense is not None:
            _candidates = []
            for item in ((fetch.new_items if fetch else []) or []):
                _candidates.append({
                    "kind": getattr(item, "kind", "content"),
                    "source_type": getattr(item, "source_type", ""),
                    "source_name": getattr(item, "source_name", ""),
                    "title": getattr(item, "title", None),
                    "content": str(getattr(item, "content", "") or "")[:600],
                    "url": getattr(item, "url", None),
                    "published_at": (
                        item.published_at.isoformat()
                        if getattr(item, "published_at", None) is not None
                        else None
                    ),
                })
            for evt in (sense.health_events if sense else []):
                _candidates.append({
                    "kind": getattr(evt, "kind", "alert"),
                    "source_type": getattr(evt, "source_type", ""),
                    "source_name": getattr(evt, "source_name", ""),
                    "title": getattr(evt, "title", None),
                    "content": str(getattr(evt, "content", "") or "")[:600],
                    "url": getattr(evt, "url", None),
                    "severity": getattr(evt, "severity", None),
                    "published_at": (
                        evt.published_at.isoformat()
                        if getattr(evt, "published_at", None) is not None
                        else None
                    ),
                })
            if _candidates:
                candidates_json = json.dumps(_candidates, ensure_ascii=False)
        # 实际发送的消息正文：拒发类 act 打点和成功发送的 send 打点都应保留正文。
        sent_message: str | None = None
        if (
            stage in {"act", "send"}
            and decide is not None
            and decide.decision_message
        ):
            sent_message = decide.decision_message
        trace = ProactiveDecisionTrace(
            tick_id=state.tick_id,
            session_key=state.session_key or "",
            stage=stage,
            reason_code=reason_code
            or getattr(result, "reason_code", None)
            or delivery_result,
            should_send=(
                should_send
                if should_send is not None
                else (decide.should_send if decide is not None else None)
            ),
            action=action,
            gate_reason=(
                decide.history_gate_reason
                if decide is not None and decide.history_gate_reason != "disabled"
                else None
            ),
            pre_score=score.pre_score if score is not None else None,
            base_score=score.base_score if score is not None else None,
            draw_score=score.draw_score if score is not None else None,
            decision_score=getattr(decision, "score", None),
            send_threshold=(
                float(scores.get("send_threshold"))
                if "send_threshold" in scores
                else self._cfg.threshold
            ),
            interruptibility=(
                float(decision_signals.get("interruptibility"))
                if "interruptibility" in decision_signals
                else (ctx.sense.interruptibility if ctx.sense is not None else None)
            ),
            candidate_count=(len(fetch.new_items) if fetch is not None else None),
            candidate_item_ids=candidate_item_ids,
            sleep_state=(getattr(sleep_ctx, "state", None) if sleep_ctx is not None else None),
            sleep_prob=(
                float(sleep_ctx.prob)
                if sleep_ctx is not None and getattr(sleep_ctx, "prob", None) is not None
                else None
            ),
            sleep_available=(
                bool(getattr(sleep_ctx, "available", False))
                if sleep_ctx is not None
                else None
            ),
            sleep_data_lag_min=(
                int(sleep_ctx.data_lag_min)
                if sleep_ctx is not None
                and getattr(sleep_ctx, "data_lag_min", None) is not None
                else None
            ),
            user_replied_after_last_proactive=(
                bool(decision_signals["user_replied_after_last_proactive"])
                if "user_replied_after_last_proactive" in decision_signals
                else None
            ),
            proactive_sent_24h=(
                int(decision_signals["proactive_sent_24h"])
                if "proactive_sent_24h" in decision_signals
                else (score.sent_24h if score is not None else None)
            ),
            fresh_items_24h=(
                int(decision_signals["fresh_items_24h"])
                if "fresh_items_24h" in decision_signals
                else (score.fresh_items_24h if score is not None else None)
            ),
            delivery_key=delivery_key,
            is_delivery_duplicate=is_delivery_duplicate,
            is_message_duplicate=is_message_duplicate,
            delivery_attempted=delivery_attempted,
            delivery_result=delivery_result,
            reasoning_preview=(
                str(getattr(decision, "reasoning", ""))[:500] or None
                if decision is not None
                else None
            ),
            reasoning=(
                str(getattr(decision, "reasoning", "")) or None
                if decision is not None
                else None
            ),
            evidence_item_ids=(
                list(getattr(decision, "evidence_item_ids", None) or [])
                if decision is not None
                else (
                    list(ctx.act.evidence_item_ids)
                    if ctx.act is not None
                    else []
                )
            ),
            source_refs_json=(
                json.dumps(
                    [
                        {
                            "source_name": getattr(r, "source_name", ""),
                            "title": getattr(r, "title", None),
                            "url": getattr(r, "url", None),
                        }
                        for r in ctx.act.source_refs
                    ],
                    ensure_ascii=False,
                )
                if ctx.act is not None and ctx.act.source_refs
                else None
            ),
            fetched_urls=list(getattr(decision, "fetched_urls", None) or []),
            stage_result_json=stage_result_json,
            decision_signals_json=decision_signals_json,
            error=error,
            sent_message=sent_message,
            candidates_json=candidates_json,
            # Evidence-First Research 字段
            research_status=(
                getattr(fetch.research_result, "status", None)
                if fetch is not None and fetch.research_result is not None
                else None
            ),
            research_rounds_used=(
                getattr(fetch.research_result, "rounds_used", None)
                if fetch is not None and fetch.research_result is not None
                else None
            ),
            research_tools_called=(
                list(getattr(fetch.research_result, "tools_called", []))
                if fetch is not None and fetch.research_result is not None
                else []
            ),
            research_evidence_count=(
                len(getattr(fetch.research_result, "evidence", []))
                if fetch is not None and fetch.research_result is not None
                else None
            ),
            research_reason=(
                getattr(fetch.research_result, "reason", None)
                if fetch is not None and fetch.research_result is not None
                else None
            ),
            fact_claims_count=(
                len(getattr(fetch.research_result, "fact_claims", []))
                if fetch is not None and fetch.research_result is not None
                else None
            ),
        )
        try:
            writer.emit(trace)
        except Exception:
            logger.exception("[proactive] observe emit failed stage=%s", stage)

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

    def _build_source_refs(self, items: list[ContentEvent]) -> list[ProactiveSourceRef]:
        refs: list[ProactiveSourceRef] = []
        for item in items[:3]:
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

    def _select_compose_items(
        self,
        items: list[ContentEvent],
        entries: list[tuple[str, str]],
    ) -> tuple[list[ContentEvent], list[tuple[str, str]]]:
        if not items:
            return [], []
        max_items = min(3, len(items))
        seed = items[0]
        group = [seed]
        # 1. 先围绕最高兴趣 seed 聚合最多 2 条上下文。
        # 2. 聚合条件只做 MVP：同来源近时间，或跨来源同话题。
        # 3. 最后按发布时间正序，保证输出叙事稳定。
        for candidate in items[1:]:
            if self._should_aggregate(seed, candidate):
                group.append(candidate)
            if len(group) >= max_items:
                break
        group = self._sort_items_by_published_at(group)
        return group, self._entries_for_items(group, entries)

    def _sort_items_by_published_at(self, items: list[ContentEvent]) -> list[ContentEvent]:
        if len(items) <= 1:
            return items

        def _ts(item: ContentEvent) -> tuple[int, datetime]:
            ts = getattr(item, "published_at", None)
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    return (0, ts.replace(tzinfo=timezone.utc))
                return (0, ts)
            return (1, datetime.max.replace(tzinfo=timezone.utc))
        return sorted(items, key=_ts)

    def _rank_items_by_interest(
        self,
        items: list[ContentEvent],
        preference_block: str,
    ) -> list[ContentEvent]:
        # 1. 兜底检查：无候选 / 未开启 / 无偏好文本时保持原顺序。
        if not items:
            return []
        cfg = getattr(self._cfg, "interest_filter", None)
        if not cfg or not getattr(cfg, "enabled", False):
            return items
        if not (preference_block or "").strip():
            return items
        # 2. 构造兴趣配置并打分（不做硬过滤，只排序）。
        try:
            from proactive.interest import (
                InterestFilterConfig,
                score_items_by_memory,
            )
        except Exception:
            return items
        interest_cfg = InterestFilterConfig(
            enabled=True,
            memory_max_chars=getattr(cfg, "memory_max_chars", 4000),
            keyword_max_count=getattr(cfg, "keyword_max_count", 80),
            min_token_len=getattr(cfg, "min_token_len", 2),
            min_score=getattr(cfg, "min_score", 0.14),
            top_k=getattr(cfg, "top_k", 10),
            exploration_ratio=getattr(cfg, "exploration_ratio", 0.2),
        )
        ranked = score_items_by_memory(items, preference_block, interest_cfg)
        # 3. 按兴趣分数降序返回。
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return [item for item, _ in ranked]

    def _should_aggregate(self, left: ContentEvent, right: ContentEvent) -> bool:
        if self._is_same_source_window(left, right):
            return True
        return self._shares_topic(left, right)

    def _is_same_source_window(self, left: ContentEvent, right: ContentEvent) -> bool:
        if (left.source_name or "").strip().lower() != (
            right.source_name or ""
        ).strip().lower():
            return False
        left_ts = getattr(left, "published_at", None)
        right_ts = getattr(right, "published_at", None)
        if not isinstance(left_ts, datetime) or not isinstance(right_ts, datetime):
            return True
        if left_ts.tzinfo is None:
            left_ts = left_ts.replace(tzinfo=timezone.utc)
        if right_ts.tzinfo is None:
            right_ts = right_ts.replace(tzinfo=timezone.utc)
        return abs((left_ts - right_ts).total_seconds()) <= 12 * 3600

    def _shares_topic(self, left: ContentEvent, right: ContentEvent) -> bool:
        left_title = self._topic_text(left)
        right_title = self._topic_text(right)
        if not left_title or not right_title:
            return False
        if left_title[:12] == right_title[:12]:
            return True
        left_tokens = set(self._topic_tokens(left_title))
        right_tokens = set(self._topic_tokens(right_title))
        return bool(
            left_tokens and right_tokens and left_tokens.intersection(right_tokens)
        )

    @staticmethod
    def _topic_text(item: ContentEvent) -> str:
        title = (item.title or "").strip()
        if title:
            return title.lower()
        return (item.content or "").strip().lower()

    @staticmethod
    def _topic_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for token in _TOPIC_TOKEN_PATTERN.findall((text or "").lower()):
            if token in _TOPIC_STOPWORDS or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens

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
            "你是主动消息分类器。判断消息是否包含「用户状态总结/安慰框架」。\n"
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
        items: list[ContentEvent],
        entries: list[tuple[str, str]],
        evidence_ids: list[str],
    ) -> tuple[list[ContentEvent], list[tuple[str, str]]]:
        if not evidence_ids:
            return [], []
        entry_by_id = {item_id: source_key for source_key, item_id in entries}
        wanted = set(evidence_ids)
        selected_items: list[ContentEvent] = []
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
        items: list[ContentEvent],
        entries: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        entry_by_id: dict[str, str] = {}
        for source_key, item_id in entries:
            entry_by_id.setdefault(item_id, source_key)
        selected: list[tuple[str, str]] = []
        for item in items:
            item_id = self._item_id_for(item)
            selected.append(
                (entry_by_id.get(item_id, compute_source_key(item)), item_id)
            )
        return selected

    def _primary_candidate_entries(
        self, entries: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        return entries[:1]

    def _item_id_for(self, item: ContentEvent) -> str:
        try:
            item_id = self._decide.item_id_for(item)
            if isinstance(item_id, str) and item_id.strip():
                return item_id
        except Exception:
            pass
        return compute_item_id(item)


def _process_bg_context_sources(sources: list[dict]) -> list[dict]:
    """处理 background_context 原始数据：隐藏精确数字，只暴露活动强度级别。

    防止 LLM 把统计时长数字直接引述进消息或用于实时状态推断。
    唯一保留实时精确信息的字段是 currently_playing（来自 Steam API 实时查询）。
    """
    result = []
    for src in sources:
        if src.get("_source") == "steam" and src.get("available"):
            games = src.get("games", [])
            realtime = src.get("realtime", {})
            processed_games = []
            for g in games:
                h = float(g.get("recent_2w_hours", 0) or 0)
                level = (
                    "heavy" if h >= 20
                    else "moderate" if h >= 5
                    else "light" if h > 0
                    else "none"
                )
                processed_games.append({
                    "name": g["name"],
                    "recent_activity": level,
                    "all_time_familiar": float(g.get("all_time_hours", 0) or 0) >= 50,
                })
            result.append({
                "_source": "steam",
                "currently_playing": realtime.get("currently_playing"),
                "online_status": realtime.get("online_status"),
                "recent_games": processed_games,
                "data_freshness_hours": src.get("data_freshness_hours"),
            })
        else:
            result.append(src)
    return result


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


ProactiveEngine = ProactiveTick


def _health_priority_bonus(*, alert_count: int, notify_count: int) -> float:
    """健康信号优先级加分：仅高优先级 notify 才加分，普通 alert 不影响决策。"""
    if notify_count > 0:
        return min(0.08, 0.04 + 0.02 * min(notify_count, 2))
    return 0.0
