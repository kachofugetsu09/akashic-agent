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

import json
import logging
import random as _random_module
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Awaitable, Callable

from agent.plugins.proactive_effects import (
    ProactiveEffect,
    ProactiveEffectContext,
    ProactiveEffectProvider,
)
from agent.tool_hooks import ToolExecutionRequest, ToolExecutor, ToolHook
from agent.turns.orchestrator import TurnOrchestrator
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from core.common.diagnostic_log import diagnostic_context, diagnostic_line
from proactive_v2.config import ProactiveConfig
from proactive_v2.context import AgentTickContext
from agent.core.drift_turn import DriftTurnPipeline
from proactive_v2.gateway import DataGateway, GatewayDeps, GatewayResult
from proactive_v2.modules_gate import GateResult, ProactiveGateChain
from proactive_v2.modules_prompt import ProactivePromptBuilder
from proactive_v2.tools import TOOL_SCHEMAS, ToolDeps, dispatch

logger = logging.getLogger(__name__)

# ── ACK TTL 常量 ──────────────────────────────────────────────────────────
_CITED_ACK_TTL = 168
_UNCITED_ACK_TTL = 24
_POST_GUARD_ACK_TTL = 24
_DISCARDED_ACK_TTL = 720


# ── Fetch 步骤的输出 ──────────────────────────────────────────────────────

@dataclass
class FeedResult:
    """数据拉取结果。drift_entered=True 时跳过 Judge/Resolve，直接收尾。"""
    drift_entered: bool
    base_score: float | None
    messages: list[dict] = field(default_factory=list)


# ── Resolve 步骤的输出 ─────────────────────────────────────────────────────

@dataclass
class ResolveResult:
    """最终裁定结果。action="send" 时 outbound 非空。"""
    action: str  # "send" | "skip"
    result: TurnResult


# ── 副作用回调（复刻原 AgentTick._CallbackSideEffect）─────────────────────

@dataclass
class _CallbackSideEffect:
    callback: Callable[[], Awaitable[None]]
    name: str = "callback"

    async def run(self) -> None:
        await self.callback()


# ── delivery key 构建 ─────────────────────────────────────────────────────

def _normalize_delivery_url(raw: str) -> str:
    from urllib.parse import urlsplit, urlunsplit
    text = str(raw or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _build_delivery_refs(ctx: AgentTickContext) -> list[str]:
    if not ctx.cited_item_ids:
        return []
    content_map = {
        f"{e.get('ack_server', '')}:{e.get('event_id') or e.get('id', '')}": e
        for e in ctx.fetched_contents
        if e.get("ack_server") and (e.get("event_id") or e.get("id"))
    }
    refs: list[str] = []
    for key in sorted(set(ctx.cited_item_ids)):
        meta = content_map.get(key)
        if meta is None:
            refs.append(f"id:{key}")
            continue
        url = _normalize_delivery_url(str(meta.get("url") or ""))
        if url:
            refs.append(f"url:{url}")
            continue
        source = str(meta.get("source") or meta.get("source_name") or "").strip().lower()
        title = str(meta.get("title") or "").strip().lower()
        if title:
            refs.append(f"title:{source}|{title}")
            continue
        refs.append(f"id:{key}")
    return sorted(set(refs))


def build_delivery_key(ctx: AgentTickContext) -> str:
    refs = _build_delivery_refs(ctx)
    if refs and any(not ref.startswith("id:") for ref in refs):
        key_src = json.dumps(refs)
    elif ctx.cited_item_ids:
        key_src = json.dumps(sorted(ctx.cited_item_ids))
    else:
        key_src = ctx.final_message[:500]
    return sha1(key_src.encode()).hexdigest()[:16]


# ── ACK 副作用函数 ──────────────────────────────────────────────────────

async def ack_discarded(ctx: AgentTickContext, ack_fn) -> None:
    if ack_fn is None:
        return
    for key in ctx.discarded_item_ids:
        await ack_fn(key, _DISCARDED_ACK_TTL)


async def ack_post_guard_fail(ctx: AgentTickContext, ack_fn, *, alert_ack_fn=None) -> None:
    if ack_fn is None:
        return
    fetched_alert_keys = {
        f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}"
        for e in ctx.fetched_alerts
    }
    cited_set = set(ctx.cited_item_ids)

    async def _ack_alert(key: str) -> None:
        if alert_ack_fn is not None:
            await alert_ack_fn(key)
        else:
            await ack_fn(key, _POST_GUARD_ACK_TTL)

    for key in cited_set - fetched_alert_keys:
        await ack_fn(key, _POST_GUARD_ACK_TTL)
    for key in cited_set & fetched_alert_keys:
        await _ack_alert(key)
    for key in fetched_alert_keys - cited_set:
        await _ack_alert(key)
    for key in (ctx.interesting_item_ids - cited_set) - fetched_alert_keys:
        await ack_fn(key, _POST_GUARD_ACK_TTL)
    for key in ctx.discarded_item_ids:
        await ack_fn(key, _DISCARDED_ACK_TTL)


async def ack_on_success(ctx: AgentTickContext, ack_fn, *, alert_ack_fn=None) -> None:
    if ack_fn is None:
        return
    fetched_alert_keys = {
        f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}"
        for e in ctx.fetched_alerts
    }
    fetched_content_keys = {
        f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}"
        for e in ctx.fetched_contents
    }
    cited_set = set(ctx.cited_item_ids)
    for key in cited_set & fetched_content_keys:
        await ack_fn(key, _CITED_ACK_TTL)
    for key in cited_set & fetched_alert_keys:
        if alert_ack_fn is not None:
            await alert_ack_fn(key)
        else:
            await ack_fn(key, _CITED_ACK_TTL)
    for key in (ctx.interesting_item_ids - cited_set) - fetched_alert_keys:
        await ack_fn(key, _UNCITED_ACK_TTL)
    for key in ctx.discarded_item_ids:
        await ack_fn(key, _DISCARDED_ACK_TTL)


async def _mark_delivery(*, state_store: Any, session_key: str, delivery_key: str) -> None:
    state_store.mark_delivery(session_key, delivery_key)


async def _mark_context_only_send(
    *,
    state_store: Any,
    session_key: str,
    context_as_fallback_open: bool,
    has_cited: bool,
) -> None:
    if context_as_fallback_open and not has_cited:
        state_store.mark_context_only_send(session_key)


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
    tool_hooks: list[ToolHook] | None = None
    proactive_effect_providers: list[ProactiveEffectProvider] | None = None


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
        self._tool_executor = ToolExecutor(deps.tool_hooks or [])
        self._proactive_effect_providers = deps.proactive_effect_providers or []
        self._proactive_effects: list[ProactiveEffect] = []
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

        # 1. drift_pipeline 的 step_recorder 指向本 pipeline 的记录方法。
        if self._drift_pipeline is not None and getattr(self._drift_pipeline, "step_recorder", None) is None:
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
    async def run(self) -> float | None:
        started = time.perf_counter()
        # 1. Gate — 该不该动？
        ctx = AgentTickContext(
            session_key=self._session_key,
            now_utc=datetime.now(timezone.utc),
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
            try:
                return await self._run_with_context(ctx, started)
            except Exception as exc:
                logger.exception(
                    diagnostic_line(
                        "ProactiveTurnPipeline.run",
                        event="phase_error",
                        flow="proactive",
                        phase="tick",
                        session=self._session_key,
                        tick=ctx.tick_id,
                        action="fail",
                        reason="proactive_tick_error",
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        error_type=type(exc).__name__,
                        note=str(exc)[:160],
                    )
                )
                raise

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
        self._apply_proactive_effects(ctx)
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
        return self._gate_chain.check(ctx)

    def _apply_proactive_effects(self, ctx: AgentTickContext) -> None:
        last_user_at = self._last_user_at_fn()
        effect_ctx = ProactiveEffectContext(
            session_key=self._session_key,
            tick_id=ctx.tick_id,
            now_utc=ctx.now_utc,
            base_judge_send_threshold=float(
                getattr(self._cfg, "judge_send_threshold", 0.60)
            ),
            last_user_at=last_user_at,
        )
        effects: list[ProactiveEffect] = []
        for provider in self._proactive_effect_providers:
            try:
                effect = provider.build_proactive_effect(effect_ctx)
            except Exception:
                logger.exception(
                    "[proactive_v2] proactive effect provider failed: %s",
                    type(provider).__name__,
                )
                continue
            if effect is not None:
                effects.append(effect)
        self._proactive_effects = effects

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
                min_interval_hours = max(0, int(getattr(self._cfg, "drift_min_interval_hours", 0) or 0))
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
                self._proactive_effects
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
        """LLM 多轮工具调用：逐条内容分类 → 草稿 → 收尾。"""

        if self._llm_fn is None:
            return

        # 3.1 主 loop：模型自行决定调用工具，直到 finish_turn 或达到步数上限。
        while ctx.steps_taken < self._cfg.agent_tick_max_steps:
            ok = await self._run_tool_step(messages, ctx, loop_tag="loop", tool_choice="auto")
            if not ok:
                break
            if ctx.terminal_action is not None:
                break

        # 3.2 完整性检查：如果 finish_skip 了但还有未分类条目，补全。
        gw_result = self._last_gateway_result
        if ctx.terminal_action == "skip" and gw_result is not None and gw_result.content_meta:
            all_content_ids = {m["id"] for m in gw_result.content_meta}
            classified_ids = ctx.interesting_item_ids | ctx.discarded_item_ids
            unclassified_ids = all_content_ids - classified_ids
            if unclassified_ids:
                ctx.terminal_action = None
                ctx.skip_reason = ""
                ctx.skip_note = ""
                titles_hint = "; ".join(
                    f"{m['id']}（{m['title'][:40]}）"
                    for m in gw_result.content_meta
                    if m["id"] in unclassified_ids
                )
                completeness_msg = (
                    f"【系统提示】以下 {len(unclassified_ids)} 个条目尚未完成分类：\n"
                    f"{titles_hint}\n"
                    "请对每条调用 mark_interesting 或 mark_not_interesting，"
                    "全部分类完毕后再调用 message_push + finish_turn(decision=reply)，或 finish_turn(decision=skip, reason=...)。"
                )
                logger.info(
                    "[proactive_v2] judge completeness: %d unclassified, resetting → %s",
                    len(unclassified_ids),
                    sorted(unclassified_ids),
                )
                messages.append({"role": "user", "content": completeness_msg})
                for _ in range(5):
                    if ctx.terminal_action is not None or ctx.steps_taken >= self._cfg.agent_tick_max_steps:
                        break
                    ok = await self._run_tool_step(messages, ctx, loop_tag="complete")
                    if not ok:
                        break

        # 3.3 反思阶段：如果 interesting 已标好但还没 finish_turn，逼它收尾。
        if ctx.terminal_action is None and ctx.interesting_item_ids and ctx.steps_taken < self._cfg.agent_tick_max_steps:
            ids_str = ", ".join(sorted(ctx.interesting_item_ids))
            reflection = (
                f"【系统提示】你已将以下条目标记为 interesting：{ids_str}。\n"
                "所有条目均已分类完毕。你必须现在调用 message_push 撰写推送，然后调用 finish_turn(decision=reply)；"
                "或直接调用 finish_turn(decision=skip, reason=...)。不允许直接结束。"
            )
            logger.info("[proactive_v2] judge reflection: interesting=%d, injecting prompt", len(ctx.interesting_item_ids))
            messages.append({"role": "user", "content": reflection})
            for _ in range(3):
                if ctx.terminal_action is not None or ctx.steps_taken >= self._cfg.agent_tick_max_steps:
                    break
                ok = await self._run_tool_step(messages, ctx, loop_tag="reflect", tool_choice="auto")
                if not ok:
                    break

        self.last_ctx = ctx

    # ── 4. Resolve ────────────────────────────────────────────────────

    async def _resolve_decide(self, ctx: AgentTickContext) -> ResolveResult:
        """最终裁定：skip/reply + delivery 去重 + message 去重。"""

        ack_fn = self._tool_deps.ack_fn

        # 4.1 LLM 判定为 skip → 直接构建 skip 结果。
        if ctx.terminal_action != "reply":
            logger.info(
                diagnostic_line(
                    "ProactiveTurnPipeline._resolve_decide",
                    event="resolve",
                    flow="proactive",
                    phase="resolve",
                    session=self._session_key,
                    tick=ctx.tick_id,
                    action="skip",
                    reason=ctx.skip_reason or "no_content",
                    counts=f"steps:{ctx.steps_taken},interesting:{len(ctx.interesting_item_ids)},discarded:{len(ctx.discarded_item_ids)}",
                    note=ctx.skip_note or "-",
                )
            )
            logger.info(
                "[proactive_v2] resolve: action=%s steps=%d discarded=%d interesting=%d skip_reason=%s note=%s",
                ctx.terminal_action or "none",
                ctx.steps_taken,
                len(ctx.discarded_item_ids),
                len(ctx.interesting_item_ids),
                ctx.skip_reason,
                ctx.skip_note,
            )
            skip_result = TurnResult(
                decision="skip",
                outbound=None,
                trace=TurnTrace(
                    source="proactive",
                    extra={
                        "steps_taken": ctx.steps_taken,
                        "skip_reason": ctx.skip_reason,
                        "skip_note": ctx.skip_note,
                    },
                ),
                side_effects=[
                    _CallbackSideEffect(
                        callback=lambda: ack_discarded(ctx, ack_fn),
                        name="ack_discarded_skip",
                    )
                ],
            )
            return ResolveResult(action="skip", result=skip_result)

        # 4.2 delivery 去重：同一批来源内容短时间内不重复发。
        delivery_key = build_delivery_key(ctx)
        if self._state_store.is_delivery_duplicate(
            self._session_key, delivery_key, self._cfg.delivery_dedupe_hours
        ):
            logger.info(
                diagnostic_line(
                    "ProactiveTurnPipeline._resolve_decide",
                    event="resolve",
                    flow="proactive",
                    phase="resolve",
                    session=self._session_key,
                    tick=ctx.tick_id,
                    action="skip",
                    reason="already_sent_similar",
                    counts=f"steps:{ctx.steps_taken},interesting:{len(ctx.interesting_item_ids)},discarded:{len(ctx.discarded_item_ids)}",
                    note="delivery_dedupe",
                )
            )
            logger.info("[proactive_v2] resolve: delivery_dedupe hit")
            return ResolveResult(
                action="skip",
                result=TurnResult(
                    decision="skip",
                    outbound=None,
                    evidence=list(ctx.cited_item_ids),
                    trace=TurnTrace(
                        source="proactive",
                        extra={
                            "steps_taken": ctx.steps_taken,
                            "skip_reason": "already_sent_similar",
                            "dedupe": "delivery",
                        },
                    ),
                    side_effects=[
                        _CallbackSideEffect(
                            callback=lambda: ack_post_guard_fail(
                                ctx, ack_fn, alert_ack_fn=self._tool_deps.alert_ack_fn
                            ),
                            name="ack_post_guard_delivery",
                        )
                    ],
                ),
            )

        # 4.3 message 语义去重：和最近主动消息实质重复也跳过。
        if self._cfg.message_dedupe_enabled and self._deduper is not None:
            recent_proactive = (
                self._recent_proactive_fn()
                if self._recent_proactive_fn is not None
                else []
            )
            is_dup, reason = await self._deduper.is_duplicate(
                new_message=ctx.final_message,
                recent_proactive=recent_proactive,
                new_state_summary_tag="none",
            )
            if is_dup:
                logger.info(
                    diagnostic_line(
                        "ProactiveTurnPipeline._resolve_decide",
                        event="resolve",
                        flow="proactive",
                        phase="resolve",
                        session=self._session_key,
                        tick=ctx.tick_id,
                        action="skip",
                        reason="already_sent_similar",
                        counts=f"steps:{ctx.steps_taken},interesting:{len(ctx.interesting_item_ids)},discarded:{len(ctx.discarded_item_ids)}",
                        note=str(reason or "message_dedupe")[:160],
                    )
                )
                logger.info("[proactive_v2] resolve: message_dedupe hit: %s", reason)
                return ResolveResult(
                    action="skip",
                    result=TurnResult(
                        decision="skip",
                        outbound=None,
                        evidence=list(ctx.cited_item_ids),
                        trace=TurnTrace(
                            source="proactive",
                            extra={
                                "steps_taken": ctx.steps_taken,
                                "skip_reason": "already_sent_similar",
                                "dedupe": "message",
                                "dedupe_note": str(reason or ""),
                            },
                        ),
                        side_effects=[
                            _CallbackSideEffect(
                                callback=lambda: ack_post_guard_fail(
                                    ctx, ack_fn, alert_ack_fn=self._tool_deps.alert_ack_fn
                                ),
                                name="ack_post_guard_message",
                            )
                        ],
                    ),
                )

        # 4.4 两层 guard 都通过 → 构建 send 结果。
        logger.info(
            diagnostic_line(
                "ProactiveTurnPipeline._resolve_decide",
                event="resolve",
                flow="proactive",
                phase="resolve",
                session=self._session_key,
                tick=ctx.tick_id,
                action="send",
                reason="-",
                counts=f"steps:{ctx.steps_taken},interesting:{len(ctx.interesting_item_ids)},discarded:{len(ctx.discarded_item_ids)},cited:{len(ctx.cited_item_ids)}",
            )
        )
        send_result = TurnResult(
            decision="reply",
            outbound=TurnOutbound(session_key=self._session_key, content=ctx.final_message),
            evidence=list(ctx.cited_item_ids),
            trace=TurnTrace(
                source="proactive",
                extra={
                    "steps_taken": ctx.steps_taken,
                    "skip_reason": "",
                    "state_summary_tag": "none",
                },
            ),
            success_side_effects=[
                _CallbackSideEffect(
                    callback=lambda: _mark_delivery(
                        state_store=self._state_store,
                        session_key=self._session_key,
                        delivery_key=delivery_key,
                    ),
                    name="mark_delivery",
                ),
                _CallbackSideEffect(
                    callback=lambda: _mark_context_only_send(
                        state_store=self._state_store,
                        session_key=self._session_key,
                        context_as_fallback_open=ctx.context_as_fallback_open,
                        has_cited=bool(ctx.cited_item_ids),
                    ),
                    name="mark_context_only_send",
                ),
                _CallbackSideEffect(
                    callback=lambda: ack_on_success(
                        ctx,
                        ack_fn,
                        alert_ack_fn=self._tool_deps.alert_ack_fn,
                    ),
                    name="ack_on_success",
                ),
            ],
            failure_side_effects=[
                _CallbackSideEffect(
                    callback=lambda: ack_discarded(ctx, ack_fn),
                    name="ack_discarded_send_fail",
                )
            ],
        )
        return ResolveResult(action="send", result=send_result)

    # ── 5. Deliver ────────────────────────────────────────────────────

    async def _deliver_execute(self, ctx: AgentTickContext, decision: ResolveResult) -> float | None:
        """执行发送：记日志 → 通过 TurnOrchestrator 落会话、发消息、执行副作用。"""
        # 5.1 先记 tick 日志。
        self._record_tick_log_finish(ctx, result=decision.result)
        if self._turn_orchestrator is None:
            raise RuntimeError("proactive turn_orchestrator is required")
        # 5.2 再统一交给 TurnOrchestrator。
        await self._turn_orchestrator.handle_proactive_turn(
            result=decision.result,
            session_key=self._session_key,
            channel=str(self._cfg.default_channel or "").strip(),
            chat_id=str(self._cfg.default_chat_id or "").strip(),
        )
        return 0.0

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

    # ── LLM 工具单步 ──────────────────────────────────────────────────

    async def _run_tool_step(
        self,
        messages: list[dict],
        ctx: AgentTickContext,
        *,
        loop_tag: str,
        tool_choice: str | dict = "auto",
        schemas: list[dict] | None = None,
    ) -> bool:
        """用当前 messages 调一次模型，拿到本轮工具调用并执行。"""
        active_schemas = schemas or TOOL_SCHEMAS
        llm_fn = self._llm_fn
        if llm_fn is None:
            return False
        tool_call = await llm_fn(messages, active_schemas, tool_choice)
        if tool_call is None:
            logger.warning(
                "[proactive_v2] %s: llm_fn returned None at step %d, stopping",
                loop_tag,
                ctx.steps_taken,
            )
            return False
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("input", {})
        arg_summary = json.dumps(tool_args, ensure_ascii=False)[:200]
        logger.info(
            diagnostic_line(
                "ProactiveTurnPipeline._run_tool_step",
                event="tool_call",
                flow="proactive",
                phase="agent_loop",
                session=self._session_key,
                tick=ctx.tick_id,
                action=str(tool_name or "-"),
                counts=f"step:{ctx.steps_taken}",
            )
        )
        logger.info(
            "[proactive_v2] %s step %d: %s  args=%s",
            loop_tag,
            ctx.steps_taken,
            tool_name,
            arg_summary,
        )
        ctx.steps_taken += 1
        exec_result = await self._tool_executor.execute(
            ToolExecutionRequest(
                call_id=str(tool_call.get("id") or f"call_{ctx.steps_taken}"),
                tool_name=tool_name,
                arguments=tool_args,
                source="proactive",
                session_key=self._session_key,
            ),
            lambda name, args: dispatch(name, args, ctx, self._tool_deps),
        )
        if exec_result.status == "error":
            logger.warning(
                diagnostic_line(
                    "ProactiveTurnPipeline._run_tool_step",
                    event="tool_result",
                    flow="proactive",
                    phase="agent_loop",
                    session=self._session_key,
                    tick=ctx.tick_id,
                    action=str(tool_name or "-"),
                    reason="tool_error",
                    counts=f"step:{ctx.steps_taken}",
                    note=str(exec_result.output)[:160],
                )
            )
            logger.warning("[proactive_v2] %s: tool error: %s", loop_tag, exec_result.output)
            result = str(exec_result.output)
            call_id = tool_call.get("id") or f"call_{ctx.steps_taken}"
            self._record_tick_step(
                ctx,
                phase=f"{loop_tag}:error",
                tool_name=tool_name,
                tool_call_id=str(call_id),
                tool_args=tool_args,
                tool_result_text=result,
            )
            self._append_tool_messages(
                messages,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=call_id,
                result=result,
            )
            return False
        result = str(exec_result.output)
        logger.info(
            diagnostic_line(
                "ProactiveTurnPipeline._run_tool_step",
                event="tool_result",
                flow="proactive",
                phase="agent_loop",
                session=self._session_key,
                tick=ctx.tick_id,
                action=str(tool_name or "-"),
                reason="-",
                counts=f"step:{ctx.steps_taken}",
            )
        )
        call_id = tool_call.get("id") or f"call_{ctx.steps_taken}"
        self._record_tick_step(
            ctx,
            phase=loop_tag,
            tool_name=tool_name,
            tool_call_id=str(call_id),
            tool_args=tool_args,
            tool_result_text=result,
        )
        self._append_tool_messages(
            messages,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=call_id,
            result=result,
        )
        return True

    @staticmethod
    def _append_tool_messages(
        messages: list[dict],
        *,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        result: str,
    ) -> None:
        messages.append({
            "role": "assistant",
            "content": f"调用工具 {tool_name}",
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })

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
                {
                    "provider_name": effect.provider_name,
                    "threshold_delta": effect.threshold_delta,
                    "metadata": effect.metadata,
                }
                for effect in self._proactive_effects
            ],
        )
        self._last_log_result = result

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
