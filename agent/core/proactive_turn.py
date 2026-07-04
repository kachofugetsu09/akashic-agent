"""
ProactiveTurnPipeline — 主动回复链路顶层抽象。

设计对齐被动链路的 PassiveTurnPipeline.run()：
通过 run() 一个方法可见全链路。

┌─ tick trigger
│  └─ ProactiveTurnPipeline.run()
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
from agent.turns.result import TurnResult
from bus.event_bus import EventBus
from bus.events_lifecycle import ProactiveFinished
from core.common.diagnostic_log import diagnostic_context, diagnostic_line
from proactive_v2.config import ProactiveConfig
from proactive_v2.context import AgentTickContext
from proactive_v2.frame import ProactiveFrame, ProactiveTickResult
from agent.core.drift_turn import DriftTurnPipeline
from proactive_v2.gateway import DataGateway, GatewayDeps, GatewayResult
from proactive_v2.modules_deliver import ProactiveDeliverer
from proactive_v2.modules_gate import GateResult, ProactiveGateChain
from proactive_v2.modules_judge import ProactiveJudge
from proactive_v2.modules_prompt import ProactivePromptBuilder
from proactive_v2.modules_resolve import (
    ProactiveResolver,
    ResolveResult,
    ack_discarded,
    ack_on_success,
    ack_post_guard_fail,
    build_delivery_key,
)
from proactive_v2.tools import ToolDeps

logger = logging.getLogger(__name__)

__all__ = [
    "ProactiveTurnPipeline",
    "ProactiveTurnPipelineDeps",
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
class ProactiveTurnPipelineDeps:
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
    event_bus: EventBus | None = None
    tool_hooks: list[ToolHook] | None = None


# ── 主 Pipeline ─────────────────────────────────────────────────────────

# 主动链路核心入口，串起 Gate → Fetch → Judge → Resolve → Deliver 五段。
#
# ┌─ tick trigger
# │  └─ ProactiveTurnPipeline.run
# │     ├─ 1. Gate ── _gate_check
# │     │  └─ no_target / busy / cooldown / anyaction / context_fallback
# │     ├─ 2. Fetch ── _fetch_pull
# │     │  └─ DataGateway 并行拉取 → drift 分支 → 构建 system prompt + messages
# │     ├─ 3. Judge ── _judge_evaluate
# │     │  └─ _run_tool_step 循环 → completeness_check → reflection_pass
# │     ├─ 4. Resolve ── _resolve_decide
# │     │  └─ skip 判定 / delivery_dedupe / message_dedupe → TurnResult
# │     └─ 5. Deliver ── _deliver_execute
# │        └─ _record_tick_log_finish → TurnOrchestrator.handle_proactive_turn
# └─ done

class ProactiveTurnPipeline:
    slot = "proactive.tick.pipeline"
    phase = "proactive.deliver"

    def __init__(self, deps: ProactiveTurnPipelineDeps) -> None:
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

    # ── 入口 ──────────────────────────────────────────────────────────

    # 核心方法：处理一次主动 tick，串起 Gate → Fetch → Judge → Resolve → Deliver 五段链路。
    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        started = time.perf_counter()
        self._proactive_slots = frame.slots
        # 1. Gate — 该不该动？
        ctx = AgentTickContext(
            session_key=frame.input.session_key,
            now_utc=frame.input.started_at,
        )
        with diagnostic_context(session=self._session_key, flow="proactive", tick=ctx.tick_id):
            logger.info(
                diagnostic_line(
                    "ProactiveTurnPipeline.run",
                    event="start",
                    flow="proactive",
                    phase="pregate",
                    session=self._session_key,
                    tick=ctx.tick_id,
                    action="run",
                )
            )
            frame.output = ProactiveTickResult(
                base_score=await self._run_with_context(ctx, started)
            )
            return frame

    async def _run_with_context(self, ctx: AgentTickContext, started: float) -> float | None:
        gate = self._gate_check(ctx)
        if gate.blocked:
            logger.info(
                diagnostic_line(
                    "ProactiveTurnPipeline.run",
                    event="gate_exit",
                    flow="proactive",
                    phase="pregate",
                    session=self._session_key,
                    tick=ctx.tick_id,
                    action="skip",
                    reason=gate.reason,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            )
            self._record_tick_log_finish(ctx, gate_exit=gate.reason)
            return gate.base_score

        ctx.context_as_fallback_open = gate.context_as_fallback_open
        self.last_ctx = ctx
        self._record_tick_log_start(ctx)
        self._collect_proactive_plugin_state()
        logger.info(
            diagnostic_line(
                "ProactiveTurnPipeline.run",
                event="end",
                flow="proactive",
                phase="pregate",
                session=self._session_key,
                tick=ctx.tick_id,
                action="continue",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        )

        # 2. Fetch — 外面有什么新鲜事？
        with diagnostic_context(phase="gateway"):
            feed = await self._fetch_pull(ctx)
        if feed.drift_entered:
            self._finalize_after_drift(ctx)
            return feed.base_score

        # 3. Judge — LLM 评估哪些值得说
        if feed.messages and ctx.terminal_action is None:
            with diagnostic_context(phase="agent_loop"):
                await self._judge_evaluate(ctx, feed.messages)

        # 3.5 LLM 判定 reply 时记录 anyaction（drift 路径在 _finalize_after_drift 中处理）。
        if ctx.terminal_action == "reply" and self._any_action_gate is not None:
            self._any_action_gate.record_action(now_utc=ctx.now_utc)

        # 4. Resolve — 发还是不发？
        with diagnostic_context(phase="resolve"):
            decision = await self._resolve_decide(ctx)

        # 5. Deliver — 执行发送
        score = await self._deliver_execute(ctx, decision)
        logger.info(
            diagnostic_line(
                "ProactiveTurnPipeline.run",
                event="end",
                flow="proactive",
                phase="resolve",
                session=self._session_key,
                tick=ctx.tick_id,
                action=decision.action,
                reason=ctx.skip_reason or "-",
                duration_ms=int((time.perf_counter() - started) * 1000),
                counts=f"steps:{ctx.steps_taken},interesting:{len(ctx.interesting_item_ids)},discarded:{len(ctx.discarded_item_ids)}",
            )
        )
        ctx.content_store.clear()
        return score

    # ── 1. Gate ───────────────────────────────────────────────────────

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

    # ── 2. Fetch ──────────────────────────────────────────────────────

    async def _fetch_pull(self, ctx: AgentTickContext) -> FeedResult:
        """拉取本轮数据源，构建 LLM 输入 messages。"""

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
                "ProactiveTurnPipeline._fetch_pull",
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
        fetched_contents = [
            {
                "id": m["id"].split(":", 1)[1] if ":" in m["id"] else m["id"],
                "event_id": m["id"].split(":", 1)[1] if ":" in m["id"] else m["id"],
                "ack_server": m["id"].split(":", 1)[0],
                "title": m.get("title") or "",
                "source": m.get("source") or "",
                "url": m.get("url") or "",
                "published_at": m.get("published_at") or "",
            }
            for m in gw_result.content_meta
        ]
        ctx.mark_contents_prefetched(fetched_contents, gw_result.content_store)
        ctx.mark_context_prefetched(gw_result.context)

        # 2.3 快速 skip：无 alert、无 content、且 fallback 未开启时尝试 drift。
        if not gw_result.alerts and not gw_result.content_meta and not ctx.context_as_fallback_open:
            if self._drift_pipeline is not None and self._cfg.drift_enabled:
                last_drift_at = self._state_store.get_last_drift_at(self._session_key)
                min_interval_hours = max(0, int(self._cfg.drift_min_interval_hours or 0))
                if (
                    last_drift_at is not None
                    and min_interval_hours > 0
                    and (ctx.now_utc - last_drift_at).total_seconds() < min_interval_hours * 3600
                ):
                    logger.info(
                        diagnostic_line(
                            "ProactiveTurnPipeline._fetch_pull",
                            event="skip",
                            flow="proactive",
                            phase="gateway",
                            session=self._session_key,
                            tick=ctx.tick_id,
                            action="skip",
                            reason="cooldown",
                            counts="alerts:0,content:0,context:0",
                        )
                    )
                    logger.info(
                        "[proactive_v2] fetch: drift blocked by interval last_drift_at=%s min_interval_hours=%d",
                        last_drift_at.isoformat(),
                        min_interval_hours,
                    )
                    ctx.terminal_action = "skip"
                    ctx.skip_reason = "no_content"
                    self.last_ctx = ctx
                    return FeedResult(drift_entered=False, base_score=None)
                logger.info("[proactive_v2] fetch: empty gateway, attempting drift")
                entered_drift = await self._drift_pipeline.run(ctx, self._llm_fn)
                if entered_drift:
                    self._state_store.mark_drift_run(self._session_key, ctx.now_utc)
                    logger.info("[proactive_v2] fetch: drift entered, message_sent=%s", ctx.drift_message_sent)
                    self.last_ctx = ctx
                    return FeedResult(drift_entered=True, base_score=0.0)
                logger.info("[proactive_v2] fetch: drift not entered")
            logger.info("[proactive_v2] fetch: no data and fallback off → skip")
            logger.info(
                diagnostic_line(
                    "ProactiveTurnPipeline._fetch_pull",
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

        # 2.4 llm_fn 为空 → 无法进入 Judge，直接退出。
        if self._llm_fn is None:
            self.last_ctx = ctx
            return FeedResult(drift_entered=False, base_score=None)

        # 2.5 构造本轮 proactive 输入 messages。
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

    # ── 3. Judge ──────────────────────────────────────────────────────

    async def _judge_evaluate(self, ctx: AgentTickContext, messages: list[dict]) -> None:
        await self._judge.evaluate(ctx, messages, self._last_gateway_result)
        self.last_ctx = ctx

    # ── 4. Resolve ────────────────────────────────────────────────────

    async def _resolve_decide(self, ctx: AgentTickContext) -> ResolveResult:
        return await self._resolver.resolve(ctx)

    # ── 5. Deliver ────────────────────────────────────────────────────

    async def _deliver_execute(self, ctx: AgentTickContext, decision: ResolveResult) -> float | None:
        return await self._deliverer.deliver(ctx, decision)

    # ── drift 收尾 ────────────────────────────────────────────────────

    def _finalize_after_drift(self, ctx: AgentTickContext) -> None:
        """drift 进入后跳过正常 post_loop，直接收尾。"""
        if self._any_action_gate is not None:
            self._any_action_gate.record_action(now_utc=ctx.now_utc)
        logger.info(
            "[proactive_v2] drift entered, skipping normal post_loop message_sent=%s finished=%s",
            ctx.drift_message_sent,
            ctx.drift_finished,
        )
        self._record_tick_log_finish(ctx)
        ctx.content_store.clear()

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
            decision = "reply" if ctx.drift_message_sent else "skip"
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
