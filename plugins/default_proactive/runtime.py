"""
ProactiveFlowRuntime — 主动回复链路业务执行服务。

设计对齐被动链路的 PassiveTurnPipeline.run()：
通过 run() 一个方法可见全链路。

┌─ tick trigger
│  └─ Lifecycle Modules
│     ├─ 1. Gate      准入检查（busy / cooldown / anyaction / fallback）
│     ├─ 2. Fetch     拉取数据（alerts / content / context → messages）
│     ├─ 3. Judge     LLM 评估（多轮工具调用：分类 → 草稿 → 收尾）
│     ├─ 4. Resolve   决策去重（skip/reply + delivery_dedupe + message_dedupe）
│     └─ 5. Deliver   执行发送（dispatch + ACK + persist + tick_log）
└─ done

段之间通过 AgentTickContext 传递状态，每段各司其职，不跨段直接访问对方内部实现。
后续可按需将任一段升级为 Phase 模块链，对外接口不变。
"""

from __future__ import annotations

import logging
import random as _random_module
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from agent.tool_hooks import ToolExecutor, ToolHook
from agent.turns.orchestrator import TurnOrchestrator
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from bus.event_bus import EventBus
from bus.events_lifecycle import ProactiveFinished
from core.common.diagnostic_log import diagnostic_context, diagnostic_line
from proactive_v2.config import ProactiveConfig
from plugins.default_proactive.context import AgentTickContext
from proactive_v2.frame import ProactiveFrame, ProactiveTickResult
from plugins.drift_flow.runtime import DriftTurnPipeline
from plugins.default_proactive.gateway import DataGateway, GatewayDeps, GatewayResult
from plugins.default_proactive.deliver import ProactiveDeliverer
from plugins.default_proactive.gate import GateResult, ProactiveGateChain
from plugins.proactive_flow.judge import ProactiveJudge
from plugins.proactive_flow.prompt import ProactivePromptBuilder
from plugins.default_proactive.resolve import (
    ProactiveResolver,
    ResolveResult,
    ack_discarded,
    ack_on_success,
    ack_post_guard_fail,
    build_delivery_key,
)
from plugins.proactive_flow.tools import ToolDeps

logger = logging.getLogger(__name__)

__all__ = [
    "ProactiveFlowRuntime",
    "ProactiveFlowDeps",
    "ResolveResult",
    "ack_discarded",
    "ack_on_success",
    "ack_post_guard_fail",
    "build_delivery_key",
]

# ── Fetch 步骤的输出 ──────────────────────────────────────────────────────

@dataclass
class FeedResult:
    """数据拉取结果。drift_entered=True 时跳过 Judge/Resolve，直接收尾。"""
    drift_entered: bool
    base_score: float | None
    messages: list[dict] = field(default_factory=list)


@dataclass
class ProactiveRunState:
    ctx: AgentTickContext
    started: float
    gateway: GatewayResult | None = None
    route: str = ""
    feed: FeedResult | None = None
    decision: ResolveResult | None = None
    base_score: float | None = None
    finished: bool = False


def _log_content_candidates(gw: GatewayResult) -> None:
    if not gw.content_meta:
        logger.info("[proactive_v2] content candidates: 0")
        return
    lines: list[str] = []
    for index, item in enumerate(gw.content_meta, 1):
        title = str(item.get("title") or "").strip() or "(no title)"
        source = str(item.get("source") or "").strip()
        line = f"[{index}] {title}"
        if source:
            line += f" | source={source}"
        lines.append(line)
    logger.info(
        "[proactive_v2] content candidates: %d\n%s",
        len(gw.content_meta),
        "\n".join(lines),
    )


# ── Pipeline 依赖容器 ─────────────────────────────────────────────────────

@dataclass
class ProactiveFlowDeps:
    cfg: ProactiveConfig
    session_key: str
    state_store: Any
    any_action_gate: Any | None
    last_user_at_fn: Callable[[], datetime | None]
    passive_busy_fn: Callable[[str], bool] | None
    turn_orchestrator: TurnOrchestrator | None
    deduper: Any | None
    tool_deps: ToolDeps
    gateway_deps: GatewayDeps | None
    workspace_context_fn: Callable[[], str] | None
    llm_fn: Any | None
    rng: Any | None
    recent_proactive_fn: Callable[[], list] | None
    drift_pipeline: DriftTurnPipeline | None
    schedule_fn: Callable[[float | None], int] | None = None
    event_bus: EventBus | None = None
    tool_hooks: list[ToolHook] | None = None


# ── 主 Pipeline ─────────────────────────────────────────────────────────

# 主动链路核心入口，串起 Gate → Fetch → Judge → Resolve → Deliver 五段。
#
# ┌─ tick 触发
# │  └─ ProactiveFlowRuntime.run
# │     ├─ 1. Gate（准入）── _gate_check
# │     │  └─ no_target / busy / cooldown / anyaction / context_fallback
# │     ├─ 2. Fetch（拉取）── _fetch_pull
# │     │  └─ DataGateway 并行拉取 → drift 分支 → 构建 system prompt + messages
# │     ├─ 3. Judge（评估）── _judge_evaluate
# │     │  └─ _run_tool_step 循环 → completeness_check → reflection_pass
# │     ├─ 4. Resolve（决策）── _resolve_decide
# │     │  └─ skip 判定 / delivery_dedupe / message_dedupe → TurnResult
# │     └─ 5. Deliver（发送）── _deliver_execute
# │        └─ _record_tick_log_finish → TurnOrchestrator.handle_proactive_turn
# └─ 完成

# 主动业务执行服务，由 Lifecycle Module 分段调用。
class ProactiveFlowRuntime:
    def __init__(self, deps: ProactiveFlowDeps) -> None:
        self._cfg = deps.cfg
        self._session_key = deps.session_key
        self._state_store = deps.state_store
        self._any_action_gate = deps.any_action_gate
        self._last_user_at_fn = deps.last_user_at_fn
        self._passive_busy_fn = deps.passive_busy_fn
        self._turn_orchestrator = deps.turn_orchestrator
        self._deduper = deps.deduper
        self._tool_deps = deps.tool_deps
        self._gateway_deps = deps.gateway_deps
        self._workspace_context_fn = deps.workspace_context_fn
        self._llm_fn = deps.llm_fn
        self._rng = deps.rng if deps.rng is not None else _random_module.Random()
        self._recent_proactive_fn = deps.recent_proactive_fn
        self._drift_pipeline = deps.drift_pipeline
        self._schedule_fn = deps.schedule_fn
        self._event_bus = deps.event_bus
        self._tool_executor = ToolExecutor(deps.tool_hooks or [])
        self._proactive_slots: dict[str, Any] = {}
        self._proactive_prompt_sections: list[str] = []
        self._proactive_effect_logs: list[dict[str, Any]] = []
        self._gate_chain = ProactiveGateChain(
            cfg=self._cfg,
            session_key=self._session_key,
            state_store=self._state_store,
            any_action_gate=self._any_action_gate,
            last_user_at_fn=self._last_user_at_fn,
            passive_busy_fn=self._passive_busy_fn,
            rng=self._rng,
        )
        self._prompt_builder = ProactivePromptBuilder(
            cfg=self._cfg,
            memory=self._tool_deps.memory,
            workspace_context_fn=self._workspace_context_fn,
        )
        self._resolver = ProactiveResolver(
            cfg=self._cfg,
            session_key=self._session_key,
            state_store=self._state_store,
            deduper=self._deduper,
            recent_proactive_fn=self._recent_proactive_fn,
            ack_fn=self._tool_deps.ack_fn,
            alert_ack_fn=self._tool_deps.alert_ack_fn,
        )
        self._judge = ProactiveJudge(
            cfg=self._cfg,
            session_key=self._session_key,
            llm_fn=self._llm_fn,
            tool_deps=self._tool_deps,
            tool_executor=self._tool_executor,
            record_step_fn=self._record_tick_step,
        )
        self._deliverer = ProactiveDeliverer(
            cfg=self._cfg,
            session_key=self._session_key,
            turn_orchestrator=self._turn_orchestrator,
            record_finish_fn=self._record_tick_log_finish,
        )

        # 1. drift_pipeline 的 step_recorder 指向本 pipeline 的记录方法。
        if self._drift_pipeline is not None and self._drift_pipeline.step_recorder is None:
            self._drift_pipeline.step_recorder = (
                lambda ctx, phase, tool_name, tool_call_id, tool_args, tool_result_text: (
                    self._record_tick_step(
                        ctx,
                        phase=phase,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        tool_args=tool_args,
                        tool_result_text=tool_result_text,
                    )
                )
            )

        self.last_ctx: AgentTickContext | None = None
        self._last_gateway_result: GatewayResult | None = None

    def begin(self, frame: ProactiveFrame) -> ProactiveRunState:
        self._proactive_slots = frame.slots
        ctx = AgentTickContext(
            session_key=frame.input.session_key,
            now_utc=frame.input.started_at,
        )
        return ProactiveRunState(ctx=ctx, started=time.perf_counter())

    async def gate(self, state: ProactiveRunState) -> None:
        ctx = state.ctx
        with diagnostic_context(session=self._session_key, flow="proactive", tick=ctx.tick_id):
            logger.info(
                diagnostic_line(
                    "ProactiveFlowRuntime.run",
                    event="start",
                    flow="proactive",
                    phase="pregate",
                    session=self._session_key,
                    tick=ctx.tick_id,
                    action="run",
                )
            )
            gate = self._gate_check(ctx)
            if gate.blocked:
                logger.info(
                    diagnostic_line(
                        "ProactiveFlowRuntime.run",
                        event="gate_exit",
                        flow="proactive",
                        phase="pregate",
                        session=self._session_key,
                        tick=ctx.tick_id,
                        action="skip",
                        reason=gate.reason,
                        duration_ms=int((time.perf_counter() - state.started) * 1000),
                    )
                )
                self._record_tick_log_finish(ctx, gate_exit=gate.reason)
                state.base_score = gate.base_score
                state.finished = True
                return

            ctx.context_as_fallback_open = gate.context_as_fallback_open
            self.last_ctx = ctx
            self._record_tick_log_start(ctx)
            logger.info(
                diagnostic_line(
                    "ProactiveFlowRuntime.run",
                    event="end",
                    flow="proactive",
                    phase="pregate",
                    session=self._session_key,
                    tick=ctx.tick_id,
                    action="continue",
                    duration_ms=int((time.perf_counter() - state.started) * 1000),
                )
            )

    def collect_plugin_state(self, state: ProactiveRunState) -> None:
        if not state.finished:
            self._collect_proactive_plugin_state()

    async def source(self, state: ProactiveRunState) -> None:
        if state.finished:
            return
        with diagnostic_context(phase="gateway"):
            state.gateway = await self._fetch_gateway(state.ctx)

    def select_route(self, state: ProactiveRunState) -> None:
        if state.finished:
            return
        if state.gateway is None:
            raise RuntimeError("主动链路缺少 GatewayResult")
        gw = state.gateway
        ctx = state.ctx
        if gw.alerts or gw.content_meta or ctx.context_as_fallback_open:
            state.route = "proactive"
            return
        if self._drift_pipeline is not None and self._cfg.drift_enabled:
            last_drift_at = self._state_store.get_last_drift_at(self._session_key)
            min_hours = max(0, int(self._cfg.drift_min_interval_hours or 0))
            if last_drift_at is None or min_hours == 0:
                state.route = "drift"
                return
            if (ctx.now_utc - last_drift_at).total_seconds() >= min_hours * 3600:
                state.route = "drift"
                return
            logger.info(
                "[proactive_v2] fetch: drift blocked by interval "
                "last_drift_at=%s min_interval_hours=%d",
                last_drift_at.isoformat(),
                min_hours,
            )
        state.route = "skip"

    async def drift(self, state: ProactiveRunState) -> None:
        if state.finished or state.route != "drift":
            return
        ctx = state.ctx
        if self._drift_pipeline is None:
            raise RuntimeError("Drift route 缺少 DriftFlow")
        logger.info("[proactive_v2] fetch: empty gateway, attempting drift")
        entered = await self._drift_pipeline.run(ctx, self._llm_fn)
        if entered:
            self._state_store.mark_drift_run(self._session_key, ctx.now_utc)
            if self._any_action_gate is not None:
                self._any_action_gate.record_action(now_utc=ctx.now_utc)
            logger.info(
                "[proactive_v2] fetch: drift entered, message_staged=%s",
                ctx.drift_message_staged,
            )
            self.last_ctx = ctx
            state.feed = FeedResult(drift_entered=True, base_score=0.0)
            state.base_score = 0.0
            return
        logger.info("[proactive_v2] fetch: drift not entered")
        state.route = "skip"

    def prepare_proactive(self, state: ProactiveRunState) -> None:
        if state.finished or state.route == "drift":
            return
        if state.gateway is None:
            raise RuntimeError("主动链路缺少 GatewayResult")
        state.feed = self._prepare_feed(state.ctx, state.gateway, state.route)

    async def judge(self, state: ProactiveRunState) -> None:
        if state.finished or state.feed is None:
            return
        ctx = state.ctx
        if state.feed.messages and ctx.terminal_action is None:
            with diagnostic_context(phase="agent_loop"):
                await self._judge_evaluate(ctx, state.feed.messages)

        if ctx.terminal_action == "reply" and self._any_action_gate is not None:
            self._any_action_gate.record_action(now_utc=ctx.now_utc)

    async def resolve(self, state: ProactiveRunState) -> None:
        if state.finished:
            return
        if state.route == "drift":
            state.decision = self._resolve_drift(state.ctx)
            return
        with diagnostic_context(phase="resolve"):
            state.decision = await self._resolve_decide(state.ctx)

    async def deliver(self, state: ProactiveRunState) -> None:
        if state.finished:
            return
        if state.decision is None:
            raise RuntimeError("主动链路缺少 ResolveResult")
        ctx = state.ctx
        try:
            state.base_score = await self._deliver_execute(ctx, state.decision)
        finally:
            if ctx.drift_entered and (ctx.draft_message or ctx.draft_media):
                sent = bool(self._deliverer.last_sent)
                if self._drift_pipeline is not None:
                    self._drift_pipeline.record_commit_result(ctx, sent)
        logger.info(
            diagnostic_line(
                "ProactiveFlowRuntime.run",
                event="end",
                flow="proactive",
                phase="resolve",
                session=self._session_key,
                tick=ctx.tick_id,
                action=state.decision.action,
                reason=ctx.skip_reason or "-",
                duration_ms=int((time.perf_counter() - state.started) * 1000),
                counts=f"steps:{ctx.steps_taken},interesting:{len(ctx.interesting_item_ids)},discarded:{len(ctx.discarded_item_ids)}",
            )
        )
        ctx.content_store.clear()
        state.finished = True

    def _resolve_drift(self, ctx: AgentTickContext) -> ResolveResult:
        if not ctx.draft_message and not ctx.draft_media:
            ctx.terminal_action = "skip"
            ctx.skip_reason = "no_content"
            return ResolveResult(
                action="skip",
                result=TurnResult(
                    decision="skip",
                    outbound=None,
                    trace=TurnTrace(
                        source="proactive",
                        extra={"source_mode": "drift", "skip_reason": "no_content"},
                    ),
                ),
            )
        ctx.terminal_action = "reply"
        ctx.final_message = ctx.draft_message
        return ResolveResult(
            action="reply",
            result=TurnResult(
                decision="reply",
                outbound=TurnOutbound(
                    session_key=self._session_key,
                    content=ctx.draft_message,
                    media=list(ctx.draft_media),
                ),
                trace=TurnTrace(source="proactive", extra={"source_mode": "drift"}),
            ),
        )

    def schedule(self, state: ProactiveRunState) -> int | None:
        if self._schedule_fn is None:
            return None
        return self._schedule_fn(state.base_score)

    # ── 1. Gate（准入）───────────────────────────────────────────────

    def _gate_check(self, ctx: AgentTickContext) -> GateResult:
        gate = self._gate_chain.check(ctx)
        if gate.blocked:
            return gate

        pass_probability = self._proactive_slots.get("proactive:gate:pass_probability")
        if pass_probability is None:
            return gate
        if self._rng.random() < float(pass_probability):
            return gate
        reason = str(self._proactive_slots.get("proactive:gate:reason") or "plugin_gate")
        return GateResult(blocked=True, reason=reason, base_score=None)

    def _collect_proactive_plugin_state(self) -> None:
        self._proactive_prompt_sections = [
            str(self._proactive_slots[key])
            for key in sorted(self._proactive_slots)
            if key.startswith("proactive:prompt:system_bottom:")
            and str(self._proactive_slots[key]).strip()
        ]
        self._proactive_effect_logs = [
            dict(value)
            for key, value in sorted(self._proactive_slots.items())
            if key.startswith("proactive:effect:")
            and isinstance(value, dict)
        ]

    # ── 2. Fetch（拉取）─────────────────────────────────────────────

    async def _fetch_gateway(self, ctx: AgentTickContext) -> GatewayResult:

        # 2.1 通过 DataGateway 并行拉取 alerts / content / context。
        gateway_deps = self._gateway_deps or GatewayDeps(
            alert_fn=None,
            feed_fn=None,
            context_fn=None,
            web_fetch_tool=self._tool_deps.web_fetch_tool,
            max_chars=self._tool_deps.max_chars,
            content_limit=self._cfg.agent_tick_content_limit,
        )
        gw = DataGateway(
            begin_fn=gateway_deps.begin_fn,
            alert_fn=gateway_deps.alert_fn,
            feed_fn=gateway_deps.feed_fn,
            context_fn=gateway_deps.context_fn,
            web_fetch_tool=gateway_deps.web_fetch_tool,
            max_chars=gateway_deps.max_chars,
            content_limit=gateway_deps.content_limit,
        )
        gw_result = await gw.run()
        self._last_gateway_result = gw_result
        _log_content_candidates(gw_result)
        logger.info(
            diagnostic_line(
                "ProactiveSourceModule.run",
                event="end",
                flow="proactive",
                phase="gateway",
                session=self._session_key,
                tick=ctx.tick_id,
                action="fetched",
                counts=f"alerts:{len(gw_result.alerts)},content:{len(gw_result.content_meta)},context:{len(gw_result.context)}",
            )
        )

        # 2.2 把拉取结果灌入 ctx。
        ctx.mark_alerts_prefetched(gw_result.alerts)
        fetched_contents: list[dict[str, Any]] = []
        for meta in gw_result.content_meta:
            source_id, separator, event_id = str(meta["id"]).rpartition(":")
            if not separator:
                source_id = ""
                event_id = str(meta["id"])
            fetched_contents.append(
                {
                    "id": event_id,
                    "event_id": event_id,
                    "ack_server": source_id,
                    "title": meta.get("title") or "",
                    "source": meta.get("source") or "",
                    "url": meta.get("url") or "",
                    "published_at": meta.get("published_at") or "",
                }
            )
        ctx.mark_contents_prefetched(fetched_contents, gw_result.content_store)
        ctx.mark_context_prefetched(gw_result.context)

        return gw_result

    def _prepare_feed(
        self,
        ctx: AgentTickContext,
        gw_result: GatewayResult,
        route: str,
    ) -> FeedResult:
        if route == "skip":
            logger.info("[proactive_v2] fetch: no data and fallback off → skip")
            logger.info(
                diagnostic_line(
                    "ProactivePrepareModule.run",
                    event="skip",
                    flow="proactive",
                    phase="gateway",
                    session=self._session_key,
                    tick=ctx.tick_id,
                    action="skip",
                    reason="no_content",
                    counts="alerts:0,content:0,context:0",
                )
            )
            ctx.terminal_action = "skip"
            ctx.skip_reason = "no_content"
            self.last_ctx = ctx
            return FeedResult(drift_entered=False, base_score=None)
        if self._llm_fn is None:
            self.last_ctx = ctx
            return FeedResult(drift_entered=False, base_score=None)
        system_msg = {
            "role": "system",
            "content": self._prompt_builder.build_system_prompt(
                self._proactive_prompt_sections
            ),
        }
        runtime_context_msg = self._prompt_builder.build_runtime_context_message(
            ctx,
            gw_result,
        )
        kickoff_msg = {
            "role": "user",
            "content": (
                "开始本轮 proactive 处理。"
                "请基于上面的候选内容和规则，必须通过工具逐步完成分类，"
                "最后通过 message_push + finish_turn(decision=reply)，或 finish_turn(decision=skip, reason=...) 收尾。"
            ),
        }
        messages: list[dict] = [system_msg, runtime_context_msg, kickoff_msg]
        return FeedResult(drift_entered=False, base_score=None, messages=messages)

    # ── 3. Judge（评估）─────────────────────────────────────────────

    async def _judge_evaluate(self, ctx: AgentTickContext, messages: list[dict]) -> None:
        await self._judge.evaluate(ctx, messages, self._last_gateway_result)
        self.last_ctx = ctx

    # ── 4. Resolve（决策）───────────────────────────────────────────

    async def _resolve_decide(self, ctx: AgentTickContext) -> ResolveResult:
        return await self._resolver.resolve(ctx)

    # ── 5. Deliver（发送）───────────────────────────────────────────

    async def _deliver_execute(self, ctx: AgentTickContext, decision: ResolveResult) -> float | None:
        return await self._deliverer.deliver(ctx, decision)

    # ── Tick 日志记录 ──────────────────────────────────────────────────

    def _record_tick_log_start(self, ctx: AgentTickContext) -> None:
        self._state_store.record_tick_log_start(
            tick_id=ctx.tick_id,
            session_key=self._session_key,
            started_at=ctx.now_utc.isoformat(),
            gate_exit=None,
        )

    def _record_tick_log_finish(
        self,
        ctx: AgentTickContext,
        *,
        gate_exit: str | None = None,
        result: TurnResult | None = None,
    ) -> None:
        decision = result.decision if result is not None else ctx.terminal_action
        if ctx.drift_entered and result is None and decision is None:
            decision = "reply" if ctx.drift_message_staged else "skip"
        trace_extra = result.trace.extra if result is not None and result.trace is not None else {}
        skip_reason = str(trace_extra.get("skip_reason") or ctx.skip_reason or "")
        final_message = ""
        if result is not None and result.outbound is not None:
            final_message = str(result.outbound.content or "")
        elif ctx.final_message:
            final_message = ctx.final_message
        self._state_store.record_tick_log_finish(
            tick_id=ctx.tick_id,
            session_key=self._session_key,
            started_at=ctx.now_utc.isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            gate_exit=gate_exit,
            terminal_action=decision,
            skip_reason=skip_reason,
            steps_taken=ctx.steps_taken,
            alert_count=len(ctx.fetched_alerts),
            content_count=len(ctx.fetched_contents),
            context_count=len(ctx.fetched_context),
            interesting_ids=sorted(ctx.interesting_item_ids),
            discarded_ids=sorted(ctx.discarded_item_ids),
            cited_ids=list(ctx.cited_item_ids),
            drift_entered=ctx.drift_entered,
            final_message=final_message,
            proactive_effects=[
                dict(effect)
                for effect in self._proactive_effect_logs
            ],
        )
        self._emit_proactive_finished(
            ctx,
            gate_exit=gate_exit,
            terminal_action=decision,
            skip_reason=skip_reason,
            final_message=final_message,
        )
        self._last_log_result = result

    def _emit_proactive_finished(
        self,
        ctx: AgentTickContext,
        *,
        gate_exit: str | None,
        terminal_action: str | None,
        skip_reason: str,
        final_message: str,
    ) -> None:
        if self._event_bus is None:
            return
        self._event_bus.enqueue(
            ProactiveFinished(
                session_key=self._session_key,
                tick_id=ctx.tick_id,
                mode="drift" if ctx.drift_entered else "proactive",
                terminal_action=terminal_action,
                gate_exit=gate_exit,
                skip_reason=skip_reason,
                steps_taken=ctx.steps_taken,
                alert_count=len(ctx.fetched_alerts),
                content_count=len(ctx.fetched_contents),
                context_count=len(ctx.fetched_context),
                final_message=final_message,
                llm_call_count=ctx.llm_call_count,
                cache_prompt_tokens=(
                    ctx.cache_prompt_tokens if ctx.cache_seen else None
                ),
                cache_hit_tokens=ctx.cache_hit_tokens if ctx.cache_seen else None,
                timestamp=ctx.now_utc,
            )
        )

    def _record_tick_step(
        self,
        ctx: AgentTickContext,
        *,
        phase: str,
        tool_name: str,
        tool_call_id: str,
        tool_args: dict[str, Any],
        tool_result_text: str,
    ) -> None:
        self._state_store.record_tick_step_log(
            tick_id=ctx.tick_id,
            step_index=ctx.steps_taken,
            phase=phase,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_args=tool_args,
            tool_result_text=tool_result_text,
            terminal_action_after=ctx.terminal_action,
            skip_reason_after=ctx.skip_reason,
            interesting_ids_after=sorted(ctx.interesting_item_ids),
            discarded_ids_after=sorted(ctx.discarded_item_ids),
            cited_ids_after=list(ctx.cited_item_ids),
            final_message_after=ctx.final_message,
        )


_RUN_STATE_SLOT = "run:state"


def get_run_state(frame: ProactiveFrame) -> ProactiveRunState:
    state = frame.slots.get(_RUN_STATE_SLOT)
    if not isinstance(state, ProactiveRunState):
        raise RuntimeError("主动 Lifecycle 缺少 run:state")
    return state


class ProactiveStartModule:
    slot = "proactive.run.start"
    produces = (_RUN_STATE_SLOT,)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        frame.slots[_RUN_STATE_SLOT] = self._runtime.begin(frame)
        return frame


class ProactiveAdmissionModule:
    slot = "proactive.admission.collect"
    requires = (_RUN_STATE_SLOT,)
    collects = ("proactive:gate:*",)
    produces = ("admission:result",)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = get_run_state(frame)
        await self._runtime.gate(state)
        frame.slots["admission:result"] = not state.finished
        return frame


class ProactivePluginStateModule:
    slot = "proactive.prompt.collect"
    requires = ("admission:result",)
    collects = ("proactive:prompt:system_bottom:*", "proactive:effect:*")
    produces = ("prompt:sections:collected",)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        self._runtime.collect_plugin_state(get_run_state(frame))
        frame.slots["prompt:sections:collected"] = True
        return frame


class ProactiveSourceModule:
    slot = "proactive.source.collect"
    requires = ("admission:result",)
    produces = ("source:gateway",)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = get_run_state(frame)
        await self._runtime.source(state)
        frame.slots["source:gateway"] = state.gateway
        return frame


class ProactiveRouteModule:
    slot = "proactive.route"
    requires = ("source:gateway",)
    produces = ("route:selected",)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = get_run_state(frame)
        self._runtime.select_route(state)
        frame.slots["route:selected"] = state.route
        return frame


class ProactiveResolveModule:
    slot = "proactive.proposal.resolve"
    requires = ("proposal:proactive",)
    produces = ("run:proposal",)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = get_run_state(frame)
        await self._runtime.resolve(state)
        frame.slots["run:proposal"] = state.decision
        return frame


class ProactiveCommitModule:
    slot = "proactive.commit"
    requires = ("run:proposal",)
    produces = ("run:result",)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = get_run_state(frame)
        await self._runtime.deliver(state)
        frame.output = ProactiveTickResult(base_score=state.base_score)
        frame.slots["run:result"] = frame.output
        return frame


class ProactiveScheduleModule:
    slot = "proactive.schedule"
    requires = ("run:result",)
    produces = ("run:next_wakeup",)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = get_run_state(frame)
        interval = self._runtime.schedule(state)
        if frame.output is None:
            frame.output = ProactiveTickResult(base_score=state.base_score)
        frame.output.next_interval_seconds = interval
        frame.slots["run:next_wakeup"] = interval
        return frame


def build_default_proactive_modules(
    runtime: ProactiveFlowRuntime,
) -> list[object]:
    modules: list[object] = [
        ProactiveStartModule(runtime),
        ProactiveAdmissionModule(runtime),
        ProactiveSourceModule(runtime),
        ProactivePluginStateModule(runtime),
        ProactiveRouteModule(runtime),
        ProactiveResolveModule(runtime),
        ProactiveCommitModule(runtime),
        ProactiveScheduleModule(runtime),
    ]
    return modules
