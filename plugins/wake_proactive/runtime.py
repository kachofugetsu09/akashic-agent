from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
from json import JSONDecodeError
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, cast

from agent.turns.result import TurnOutbound, TurnResult, TurnSideEffect, TurnTrace
from core.clock import Clock, ReplayClock, clock_from_env
from plugins.default_proactive.context import AgentTickContext
from plugins.drift_flow.factory import (
    build_drift_llm_fn,
    build_drift_pipeline,
    build_drift_recent_chat_fn,
)
from plugins.wake_proactive.context import WakeContext
from plugins.wake_proactive.context_drive import ContextDriveResult, NormalizedContext
from plugins.wake_proactive.drift_drive import (
    DriftDriveResult,
    advance_drift_drive,
    sample_drift_delay_hours,
)
from plugins.wake_proactive.event_tools import (
    EVENT_TOOL_SCHEMAS,
    EventToolResult,
    execute_event_tool,
)
from plugins.wake_proactive.hazard import (
    WAKE_ADMISSION_FLOOR,
    HazardResult,
    advance_hazard,
    rank_events,
)
from plugins.wake_proactive.prompt import build_messages
from plugins.wake_proactive.state import WakeStateStore
from plugins.wake_proactive.tools import TOOL_SCHEMAS, ToolDeps, execute
from proactive_v2 import mcp_sources
from proactive_v2.frame import ProactiveFrame
from proactive_v2.runtime_scope import ProactiveRuntimeScope
from session.embedding_store import MessageEmbeddingStore


logger = logging.getLogger(__name__)
_MAX_TITLES_PER_WAKE = 120
_SEMANTIC_CALIBRATION_POWER = 4
_CONTENT_MIN_RESIDENCE = timedelta(hours=24)
_CONTENT_MAX_AGE = timedelta(days=14)
_SCHEMA_BY_NAME = {
    schema["function"]["name"]: schema
    for schema in [*TOOL_SCHEMAS, *EVENT_TOOL_SCHEMAS]
}


def select_content_page(
    events: list[dict[str, Any]],
    *,
    now: datetime,
    limit: int = _MAX_TITLES_PER_WAKE,
) -> list[dict[str, Any]]:
    return rank_events(events, now=now)[: max(0, limit)]


@dataclass(slots=True)
class WakeRunState:
    ctx: WakeContext
    alerts: list[dict[str, Any]]
    contents: list[dict[str, Any]]
    base_score: float = 0.0
    next_interval_seconds: int = 300
    hazard_result: HazardResult | None = None
    context_results: list[ContextDriveResult] | None = None
    context_reevaluate: bool = False
    context_event: dict[str, Any] | None = None
    drift_result: DriftDriveResult | None = None
    drift_ctx: AgentTickContext | None = None
    content_completed: bool = False
    new_alert_count: int = 0
    new_content_count: int = 0
    new_content_ids: set[str] | None = None


@dataclass(slots=True)
class AsyncEffect:
    callback: Callable[[], Awaitable[None]]

    async def run(self) -> None:
        await self.callback()


class WakeRuntime:
    def __init__(
        self,
        scope: ProactiveRuntimeScope,
        *,
        state_store: WakeStateStore | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._scope = scope
        self._clock = clock or clock_from_env()
        self._rng = scope.rng or random.Random(
            0 if isinstance(self._clock, ReplayClock) else None
        )
        self._tick_interval_seconds = 1 if isinstance(self._clock, ReplayClock) else 300
        workspace = Path(getattr(scope.state_store, "workspace_dir", "."))
        self._session_db_path = workspace / "sessions.db"
        self._message_embeddings = (
            MessageEmbeddingStore(self._session_db_path)
            if self._session_db_path.exists()
            else None
        )
        self._state = state_store or WakeStateStore(workspace / "wake_proactive.db")
        web_fetch_tool = (
            scope.shared_tools.get_tool("web_fetch")
            if scope.shared_tools is not None
            else None
        )
        self._tool_deps = ToolDeps(
            web_fetch_tool=web_fetch_tool,
            memory=scope.memory,
            state_store=self._state,
            max_chars=int(getattr(scope.cfg, "agent_tick_web_fetch_max_chars", 8_000)),
        )
        self._drift_llm_fn = build_drift_llm_fn(scope)
        self._drift_pipeline = build_drift_pipeline(
            scope,
            build_drift_recent_chat_fn(scope),
        )

    def build_modules(self) -> list[object]:
        from plugins.wake_proactive.modules import (
            build_wake_content_modules,
            build_wake_drift_modules,
            build_wake_runtime_modules,
        )

        return [
            *build_wake_runtime_modules(self),
            *build_wake_content_modules(self),
            *build_wake_drift_modules(self),
        ]

    def begin(self, frame: ProactiveFrame) -> WakeRunState:
        return WakeRunState(
            ctx=WakeContext(
                session_key=frame.input.session_key,
                now_utc=self._clock.now(),
            ),
            alerts=[],
            contents=[],
            next_interval_seconds=self._tick_interval_seconds,
        )

    async def ingest(self, state: WakeRunState) -> None:
        """拉取三类 source，持久化新事件并输出一条可读摘要。"""

        # 1. 拉取所有 source，并刷新本轮运行状态
        await self._flush_pending_acknowledgements()
        channels = await self._fetch_source_channels()
        self._ingest_source_channels(state, channels)
        await self._flush_pending_acknowledgements()
        self._log_source_summary(state, channels)

        # 2. Alert 不依赖内容向量；普通轮次再刷新内容兴趣
        if state.alerts:
            return
        await self._cache_event_embeddings()
        state.contents = self._state.unread("content")
        self._apply_semantic_interest(state.contents, state.ctx.now_utc)

    async def _fetch_source_channels(self) -> dict[str, list[dict[str, Any]]]:
        """拉取所有 source；全量失败时保留原始异常。"""

        try:
            return await mcp_sources.fetch_sources_async(
                self._scope.mcp_gateway,
                self._scope.proactive_sources,
            )
        except Exception:
            logger.exception(
                "[wake.source] poll failed sources=%d",
                len(self._scope.proactive_sources),
            )
            raise

    def _ingest_source_channels(
        self,
        state: WakeRunState,
        channels: dict[str, list[dict[str, Any]]],
    ) -> None:
        """持久化三类 source，并填充本轮单条事件状态。"""

        state.new_alert_count = self._state.ingest(
            "alert", channels["alert"], state.ctx.now_utc
        )
        new_content_ids = self._state.ingest_with_ids(
            "content", channels["content"], state.ctx.now_utc
        )
        state.new_content_ids = set(new_content_ids)
        state.new_content_count = len(new_content_ids)
        self._state.queue_acknowledgements(
            _group_acknowledgements(channels["content"]),
            state.ctx.now_utc,
        )
        state.context_results = self._state.ingest_context(
            channels["context"], state.ctx.now_utc
        )
        state.context_event = next(
            (
                snapshot
                for snapshot, result in zip(
                    channels["context"], state.context_results, strict=False
                )
                if result.signal == "reevaluate"
            ),
            None,
        )
        state.context_reevaluate = (
            self._state.claim_context_reevaluation(state.ctx.now_utc)
            if state.context_event is not None
            else False
        )
        state.alerts = self._state.unread("alert")
        state.contents = self._state.unread("content")

    def _log_source_summary(
        self,
        state: WakeRunState,
        channels: dict[str, list[dict[str, Any]]],
    ) -> None:
        logger.info(
            "[wake.source] poll ok received=alerts:%d,content:%d,context:%d "
            "new=alerts:%d,content:%d unread=alerts:%d,content:%d "
            "context_reevaluate=%s samples=%s",
            len(channels["alert"]),
            len(channels["content"]),
            len(channels["context"]),
            state.new_alert_count,
            state.new_content_count,
            len(state.alerts),
            len(state.contents),
            state.context_reevaluate,
            _source_samples(channels),
        )

    async def decide_content(self, state: WakeRunState) -> bool:
        if state.alerts:
            await self._decide_event(state, "alert", state.alerts[0])
            state.next_interval_seconds = (
                1 if len(state.alerts) > 1 else self._tick_interval_seconds
            )
            return True
        if state.context_reevaluate and state.context_event is not None:
            await self._decide_event(state, "context", state.context_event)
            state.next_interval_seconds = self._tick_interval_seconds
            return True
        expired_ids: set[str] = set()
        if state.contents:
            ranked = rank_events(state.contents, now=state.ctx.now_utc)
            expired_ids = {
                str(event["id"])
                for event in ranked
                if _content_expired(event, state.ctx.now_utc)
            }
            if expired_ids:
                self._state.expire(sorted(expired_ids), state.ctx.now_utc)
                state.contents = [
                    event
                    for event in state.contents
                    if str(event["id"]) not in expired_ids
                ]
        new_content_ids = (state.new_content_ids or set()) - expired_ids
        should_evaluate_content = bool(state.contents and new_content_ids)
        if should_evaluate_content:
            hazard_state = self._state.load_hazard(state.ctx.session_key)
            last_wake_at = _parse_optional_time(
                hazard_state.get("last_wake_at") if hazard_state is not None else None
            )
            result = advance_hazard(
                state.contents,
                now=state.ctx.now_utc,
                new_item_ids=new_content_ids,
                random_draw=self._content_draw(
                    state.ctx.session_key,
                    state.ctx.now_utc,
                ),
                last_wake_at=last_wake_at,
            )
            state.hazard_result = result
            state.base_score = result.rate
            self._state.save_hazard_monitor(
                session_key=state.ctx.session_key,
                hazard=result,
                candidate_count=len(state.contents),
                evaluated_at=state.ctx.now_utc,
            )
            if result.should_wake:
                state.ctx.content_events = select_content_page(
                    state.contents,
                    now=state.ctx.now_utc,
                )
                state.ctx.content_backlog_count = (
                    len(state.contents) - len(state.ctx.content_events)
                )
                self._record_content_observation(state.ctx, result)
                await self._run_content_tools(state.ctx)
                completed = await self._commit_content_decision(state)
                self._state.save_hazard(
                    session_key=state.ctx.session_key,
                    hazard=result.hazard_after,
                    threshold=result.threshold,
                    updated_at=state.ctx.now_utc,
                    last_wake_at=state.ctx.now_utc if completed else last_wake_at,
                )
                state.next_interval_seconds = self._tick_interval_seconds
                return True
            self._state.save_hazard(
                session_key=state.ctx.session_key,
                hazard=result.hazard_after,
                threshold=result.threshold,
                updated_at=state.ctx.now_utc,
                last_wake_at=last_wake_at,
            )

        return False

    async def decide_drift(self, state: WakeRunState) -> None:
        await self._decide_drift(state)
        state.next_interval_seconds = self._tick_interval_seconds

    async def decide(self, state: WakeRunState) -> None:
        if not await self.decide_content(state):
            await self.decide_drift(state)

    def _record_content_observation(
        self,
        ctx: WakeContext,
        hazard: HazardResult,
    ) -> None:
        messages = build_messages(
            ctx=ctx,
            memory_text=self._read_memory(),
            proactive_context=str(self._scope.workspace_context_fn() or ""),
            recent_session=self._read_recent_session(ctx.session_key, ctx.now_utc),
            current_context=self._current_context_text(ctx.now_utc),
        )
        candidates = [
            {
                key: event.get(key)
                for key in (
                    "id",
                    "source_id",
                    "source_name",
                    "title",
                    "url",
                    "published_at",
                    "first_seen_at",
                    "preprocess_score",
                    "_wake_interest_score",
                    "_wake_semantic_interest",
                    "_wake_rank_score",
                    "_wake_rank_features",
                )
            }
            for event in ctx.content_events
        ]
        self._state.record_observation(
            wake_id=ctx.wake_id,
            session_key=ctx.session_key,
            kind="content",
            now=ctx.now_utc,
            trigger=_hazard_trace(hazard),
            candidates=candidates,
            llm_input=messages,
        )

    async def _run_content_tools(self, ctx: WakeContext) -> None:
        base_messages = build_messages(
            ctx=ctx,
            memory_text=self._read_memory(),
            proactive_context=str(self._scope.workspace_context_fn() or ""),
            recent_session=self._read_recent_session(ctx.session_key, ctx.now_utc),
            current_context=self._current_context_text(ctx.now_utc),
        )
        await self._run_phase(base_messages, ctx, {"scratchpad"}, "scratchpad")
        investigation = await execute(
            "investigate_candidates",
            {},
            ctx,
            self._tool_deps,
        )
        final_messages = [
            base_messages[0],
            {
                "role": "user",
                "content": (
                    f"{base_messages[1]['content']}\n\n"
                    "【已执行的初筛与并发调查结果】\n"
                    f"{investigation}\n\n"
                    "【本轮最终任务】\n标题初筛和并发调查已经完成。现在只做最终判断：调用 "
                    "share_content 分享有正文证据且此刻值得告诉用户的内容，或调用 "
                    "skip_content 保持安静。通常分享一到三条；只有同时出现多个彼此独立、"
                    "都高度相关的重要变化时才可扩展到五条。不要重复标题。"
                    "share_content 优先使用 message 写成完整自然的一段主动消息，items 只负责"
                    "声明引用证据。你知道自己是在主动找用户说话，可以自然地说刚看到、碰到或"
                    "发现了什么，但不要每次套同一句开场，也不要假装亲历未发生的事情。"
                    "语气像真正熟悉用户的协作者：可以自然接住稳定偏好和期待，例如对方特别"
                    "喜欢某类事物时可以带一点会心的判断，也可以偶尔使用双方已经稳定使用的"
                    "简称、昵称或梗；只有自然贴合当前内容时才用，不要每条都刻意套亲密称呼。"
                    "不要说‘根据记忆’或复述个人档案。"
                    "涉及敏感经历时允许共情，但必须与当前事实直接相关、轻柔且有帮助，不能"
                    "替用户定义感受或把焦虑当作推送理由。不要制造紧迫感，不强行提问。"
                    "只有当前 ContextEvent 明确支持时，才能描述用户正在睡眠、忙碌、"
                    "离线或游戏；unknown 时保持中性。唤醒只代表允许判断，不代表必须分享；"
                    "缺少新事实、用户已经知道、只有营销或泛泛观点时应调用 skip_content。"
                ),
            },
        ]
        await self._run_phase(
            final_messages,
            ctx,
            {"share_content", "skip_content"},
            None,
        )
        if ctx.terminal_action is None:
            raise RuntimeError("wake proactive LLM did not finish content decision")

    async def _run_phase(
        self,
        messages: list[dict[str, Any]],
        ctx: WakeContext,
        allowed: set[str],
        forced_name: str | None,
    ) -> None:
        call = await self._call_tool(messages, allowed, forced_name)
        _ = await execute(call.name, call.arguments, ctx, self._tool_deps)

    async def _call_tool(
        self,
        messages: list[dict[str, Any]],
        allowed: set[str],
        forced_name: str | None,
    ) -> Any:
        """调用一次带工具约束的 LLM，并返回经过校验的工具调用。"""

        # 1. 为当前 mode 选择最小工具集合
        schemas = [_SCHEMA_BY_NAME[name] for name in sorted(allowed)]
        tool_choice: str | dict[str, Any] = "required"
        if forced_name is not None:
            tool_choice = {"type": "function", "function": {"name": forced_name}}
        for attempt in range(2):
            try:
                response = await self._scope.provider.chat(
                    messages=messages,
                    tools=schemas,
                    model=str(
                        getattr(self._scope.cfg, "agent_tick_model", "")
                        or self._scope.model
                    ),
                    max_tokens=self._scope.max_tokens,
                    tool_choice=tool_choice,
                    disable_thinking=True,
                )
                break
            except JSONDecodeError:
                if attempt == 1:
                    raise

        # 2. 拒绝缺失或越权的工具调用
        if not response.tool_calls:
            raise RuntimeError("wake proactive phase requires one tool call")
        call = response.tool_calls[0]
        if call.name not in allowed:
            raise RuntimeError(f"wake proactive unexpected tool in phase: {call.name}")
        return call

    async def _commit_content_decision(self, state: WakeRunState) -> bool:
        if state.ctx.terminal_action == "skip":
            result = TurnResult(
                decision="skip",
                outbound=None,
                evidence=[],
                trace=TurnTrace(source="proactive"),
            )
            await self._require_orchestrator().handle_proactive_turn(
                result=result,
                session_key=state.ctx.session_key,
                channel=str(getattr(self._scope.cfg, "default_channel", "")),
                chat_id=str(getattr(self._scope.cfg, "default_chat_id", "")),
            )
            return True

        selected_ids = set(state.ctx.cited_item_ids)
        selected_events = [
            event
            for event in state.contents
            if str(event.get("id") or "") in selected_ids
        ]
        effect = AsyncEffect(
            lambda: self._consume_events(selected_events, state.ctx.now_utc)
        )
        result = TurnResult(
            decision="reply",
            outbound=TurnOutbound(
                session_key=state.ctx.session_key,
                content=state.ctx.final_message,
            ),
            evidence=list(state.ctx.cited_item_ids),
            trace=TurnTrace(
                source="proactive",
                extra={
                    "source_refs": list(state.ctx.source_refs),
                    "display_event_map": dict(state.ctx.display_event_map),
                },
            ),
            success_side_effects=[effect],
        )
        return bool(
            await self._require_orchestrator().handle_proactive_turn(
                result=result,
                session_key=state.ctx.session_key,
                channel=str(getattr(self._scope.cfg, "default_channel", "")),
                chat_id=str(getattr(self._scope.cfg, "default_chat_id", "")),
            )
        )

    def _content_draw(self, session_key: str, now: datetime) -> float:
        if isinstance(self._clock, ReplayClock):
            seed = f"wake-content:{session_key}:{now.isoformat()}"
            return random.Random(seed).random()
        return self._rng.random()

    def next_interval(self, state: WakeRunState) -> int:
        return state.next_interval_seconds

    def close(self) -> None:
        if self._message_embeddings is not None:
            self._message_embeddings.close()
        self._state.close()

    async def _cache_event_embeddings(self) -> None:
        embedding_api = getattr(self._scope.memory, "embedding_api", None)
        embed_batch = getattr(embedding_api, "embed_batch", None)
        if not callable(embed_batch):
            return
        embed = cast(
            Callable[[list[str]], Awaitable[list[list[float]]]],
            embed_batch,
        )
        pending = self._state.unembedded()
        if not pending:
            return
        embeddings = await embed([item["text"] for item in pending])
        self._state.save_event_embeddings(
            [item["item_id"] for item in pending],
            [list(vector) for vector in embeddings],
        )

    def _apply_semantic_interest(
        self, events: list[dict[str, Any]], now: datetime
    ) -> None:
        prototypes = self._load_turn_prototypes(now)
        for event in events:
            base = _preprocess_interest(event)
            raw_vector = event.get("_event_embedding")
            vector = (
                [float(value) for value in cast(list[object], raw_vector) if isinstance(value, (int, float))]
                if isinstance(raw_vector, list)
                else []
            )
            similarity = max(
                (_cosine(vector, prototype) for prototype in prototypes),
                default=0.0,
            )
            semantic_interest = min(
                0.999,
                max(0.0, similarity) ** _SEMANTIC_CALIBRATION_POWER,
            )
            event["_wake_semantic_interest"] = semantic_interest
            event["_wake_interest_score"] = 1 - (1 - base) * (1 - semantic_interest)

    def _load_turn_prototypes(self, now: datetime) -> list[list[float]]:
        embedding_api = getattr(self._scope.memory, "embedding_api", None)
        model = str(getattr(embedding_api, "model_id", "") or "")
        if self._message_embeddings is None or not model:
            return []
        visible = dict(
            self._message_embeddings.list_until(model=model, cutoff=now.isoformat())
        )
        if not visible:
            return []
        with closing(sqlite3.connect(str(self._session_db_path))) as db:
            rows = db.execute(
                """
                SELECT id, session_key, seq, role, extra, julianday(ts)
                FROM messages
                WHERE julianday(ts) <= julianday(?)
                ORDER BY session_key, seq
                """,
                (now.isoformat(),),
            ).fetchall()
        timestamped: list[tuple[float, str, int, list[float]]] = []
        pending_user: list[float] | None = None
        pending_session = ""
        for message_id, session_key, seq, role, extra_json, ts_julian in rows:
            vector = visible.get(str(message_id))
            if role == "user":
                pending_user = vector
                pending_session = str(session_key)
                continue
            if (
                role == "assistant"
                and vector is not None
                and pending_user is not None
                and pending_session == str(session_key)
                and not _is_proactive_message(extra_json)
            ):
                timestamped.append(
                    (
                        float(ts_julian),
                        str(session_key),
                        int(seq),
                        _normalize_weighted(pending_user, vector),
                    )
                )
                pending_user = None
        timestamped.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in timestamped[-256:]]

    async def _decide_event(
        self,
        state: WakeRunState,
        kind: Literal["alert", "context"],
        event: dict[str, Any],
    ) -> None:
        """让 LLM 独立处理一条 alert 或 context，并提交发送结果。"""

        # 1. 用统一 prompt 渲染单条事件并在调用前落审计
        messages = self._build_event_messages(state, kind, event)
        item_id = str(event.get("id") or event.get("event_id") or "")
        self._record_event_observation(state, kind, event, item_id, messages)
        logger.info(
            "[wake.event] llm start kind=%s event_id=%s title=%r",
            kind,
            item_id,
            str(event.get("title") or event.get("topic") or "")[:120],
        )

        # 2. Alert 必须自然化后发送；context 可以判断保持安静
        allowed = {"send_event"} if kind == "alert" else {"send_event", "skip_event"}
        call = await self._call_tool(
            messages,
            allowed,
            "send_event" if kind == "alert" else None,
        )
        decision = execute_event_tool(call.name, call.arguments)
        await self._commit_event_decision(state, kind, event, item_id, decision)
        logger.info(
            "[wake.event] llm done kind=%s event_id=%s decision=%s message=%r",
            kind,
            item_id,
            decision.decision,
            decision.message[:160],
        )

    def _build_event_messages(
        self,
        state: WakeRunState,
        kind: Literal["alert", "context"],
        event: dict[str, Any],
    ) -> list[dict[str, str]]:
        return build_messages(
            ctx=state.ctx,
            memory_text=self._read_memory(),
            proactive_context=str(self._scope.workspace_context_fn() or ""),
            recent_session=self._read_recent_session(
                state.ctx.session_key, state.ctx.now_utc
            ),
            current_context=self._current_context_text(state.ctx.now_utc),
            mode=kind,
            event=event,
        )

    def _record_event_observation(
        self,
        state: WakeRunState,
        kind: Literal["alert", "context"],
        event: dict[str, Any],
        item_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        self._state.record_observation(
            wake_id=state.ctx.wake_id,
            session_key=state.ctx.session_key,
            kind=kind,
            now=state.ctx.now_utc,
            trigger={"event_id": item_id, "source": _event_source(event)},
            candidates=[event],
            llm_input=messages,
        )

    async def _commit_event_decision(
        self,
        state: WakeRunState,
        kind: Literal["alert", "context"],
        event: dict[str, Any],
        item_id: str,
        decision: EventToolResult,
    ) -> None:
        """持久化单事件决策，并通过统一 orchestrator 提交副作用。"""

        # 1. 保存 LLM 决策，供 Dashboard 与审计读取
        state.ctx.terminal_action = decision.decision
        state.ctx.final_message = decision.message
        state.ctx.cited_item_ids = [item_id] if kind == "alert" and item_id else []
        self._state.save(state.ctx)

        # 2. 只在发送成功后消费 alert；context 不参与 reservoir ack
        effects: list[TurnSideEffect] = (
            [AsyncEffect(lambda: self._ack_and_consume([event], state.ctx.now_utc))]
            if kind == "alert"
            else []
        )
        result = TurnResult(
            decision=decision.decision,
            outbound=(
                TurnOutbound(
                    session_key=state.ctx.session_key,
                    content=decision.message,
                )
                if decision.decision == "reply"
                else None
            ),
            evidence=list(state.ctx.cited_item_ids),
            trace=TurnTrace(
                source="proactive",
                extra={"source_refs": [], "event_kind": kind},
            ),
            side_effects=effects if decision.decision == "skip" else [],
            success_side_effects=effects if decision.decision == "reply" else [],
        )
        await self._require_orchestrator().handle_proactive_turn(
            result=result,
            session_key=state.ctx.session_key,
            channel=str(getattr(self._scope.cfg, "default_channel", "")),
            chat_id=str(getattr(self._scope.cfg, "default_chat_id", "")),
        )

    async def _decide_drift(self, state: WakeRunState) -> None:
        if not bool(getattr(self._scope.cfg, "drift_enabled", True)):
            return
        stored = self._state.load_drift(state.ctx.session_key) or {}
        last_user_at = self._last_user_at(
            state.ctx.session_key,
            state.ctx.now_utc,
        )
        last_drift_at = _parse_optional_time(stored.get("last_drift_at"))
        repetition = min(1.0, float(stored.get("repeat_count") or 0) / 3.0)
        timer_anchor = _drift_timer_anchor(
            last_user_at=last_user_at,
            last_drift_at=last_drift_at,
            repetition=repetition,
        )
        next_attempt_at = _parse_optional_time(stored.get("next_attempt_at"))
        if stored.get("timer_anchor") != timer_anchor or next_attempt_at is None:
            next_attempt_at = self._schedule_drift_attempt(
                state,
                timer_anchor=timer_anchor,
                last_user_at=last_user_at,
                last_drift_at=last_drift_at,
                repetition=repetition,
            )
        if state.ctx.now_utc < next_attempt_at:
            return

        # 到期事件只负责开启一次 LLM 判别，不再依赖轮询 hazard 穿线
        result = advance_drift_drive(
            now=state.ctx.now_utc,
            hazard=0.0,
            threshold=0.0,
            updated_at=state.ctx.now_utc,
            last_user_at=last_user_at,
            last_drift_at=last_drift_at,
            content_evidence=0.0,
            repetition=repetition,
        )
        result = replace(result, decision="attempt")
        state.drift_result = result
        state.base_score = max(state.base_score, result.rate)
        if self._drift_pipeline is None:
            raise RuntimeError("Wake Drift 到期但缺少 DriftTurnPipeline")

        # 1. Wake 只决定到期时间；进入后改用 Default 的完整 Drift 上下文
        drift_ctx = AgentTickContext(
            now_utc=state.ctx.now_utc,
            session_key=state.ctx.session_key,
        )
        drift_ctx.mark_context_prefetched(
            [dict(context.raw) for context in self._active_contexts(state.ctx.now_utc)]
        )
        entered = await self._drift_pipeline.run(drift_ctx, self._drift_llm_fn)
        if not entered:
            return
        state.drift_ctx = drift_ctx

        # 2. 只有实际进入 pipeline 才消费本次到期事件
        self._state.record_observation(
            wake_id=state.ctx.wake_id,
            session_key=state.ctx.session_key,
            kind="drift",
            now=state.ctx.now_utc,
            trigger=_drift_trace(result),
            candidates=[],
            llm_input=[],
        )
        self._state.record_drift_observation(
            session_key=state.ctx.session_key,
            now=state.ctx.now_utc,
            threshold=result.threshold,
        )
        await self._commit_full_drift(state, drift_ctx)

    async def _commit_full_drift(
        self,
        state: WakeRunState,
        drift_ctx: AgentTickContext,
    ) -> None:
        """按 Default 的投递语义提交完整 Drift 结果。"""

        # 1. 将 pipeline 暂存的消息映射成统一主动 TurnResult
        has_outbound = bool(drift_ctx.draft_message or drift_ctx.draft_media)
        result = TurnResult(
            decision="reply" if has_outbound else "skip",
            outbound=(
                TurnOutbound(
                    session_key=state.ctx.session_key,
                    content=drift_ctx.draft_message,
                    media=list(drift_ctx.draft_media),
                )
                if has_outbound
                else None
            ),
            evidence=[],
            trace=TurnTrace(
                source="proactive",
                extra={"source_mode": "drift"},
            ),
        )
        delivered = await self._require_orchestrator().handle_proactive_turn(
            result=result,
            session_key=state.ctx.session_key,
            channel=str(getattr(self._scope.cfg, "default_channel", "")),
            chat_id=str(getattr(self._scope.cfg, "default_chat_id", "")),
        )

        # 2. 与 Default 一样，在真实投递后修正 Drift message_result
        if has_outbound:
            pipeline = self._drift_pipeline
            if pipeline is None:
                raise RuntimeError("完整 Drift 投递时 pipeline 不应为空")
            drift_ctx.drift_message_sent = bool(delivered)
            pipeline.record_commit_result(drift_ctx, bool(delivered))
        if has_outbound and delivered:
            self._state.record_drift_success(
                session_key=state.ctx.session_key,
                now=state.ctx.now_utc,
                fingerprint=drift_ctx.draft_message.strip().casefold(),
            )

    def _schedule_drift_attempt(
        self,
        state: WakeRunState,
        *,
        timer_anchor: str,
        last_user_at: datetime | None,
        last_drift_at: datetime | None,
        repetition: float,
    ) -> datetime:
        """根据当前活动状态派生并持久化一次 Drift 到期事件。"""

        # 1. 为回放提供确定性随机数，线上使用 Runtime RNG
        if isinstance(self._clock, ReplayClock):
            draw = random.Random(
                f"wake-drift:{state.ctx.session_key}:{timer_anchor}"
            ).random()
        else:
            draw = self._rng.random()
        idle_hours = (
            max(0.0, (state.ctx.now_utc - last_user_at).total_seconds() / 3600)
            if last_user_at is not None
            else 0.0
        )
        recent_drift = (
            math.exp(
                -max(0.0, (state.ctx.now_utc - last_drift_at).total_seconds())
                / (6 * 3600)
            )
            if last_drift_at is not None
            else 0.0
        )

        # 2. 一次性 timer 由状态事件重建，普通 tick 不会重新采样
        delay_hours = sample_drift_delay_hours(
            random_draw=draw,
            idle_hours=idle_hours,
            recent_drift_suppression=recent_drift,
            repetition_suppression=repetition,
        )
        next_attempt_at = state.ctx.now_utc + timedelta(hours=delay_hours)
        self._state.save_drift_timer(
            session_key=state.ctx.session_key,
            timer_anchor=timer_anchor,
            next_attempt_at=next_attempt_at,
            updated_at=state.ctx.now_utc,
        )
        return next_attempt_at

    def _last_user_at(self, session_key: str, now: datetime) -> datetime | None:
        if isinstance(self._clock, ReplayClock) and self._session_db_path.exists():
            with closing(sqlite3.connect(str(self._session_db_path))) as db:
                row = db.execute(
                    """
                    SELECT ts
                    FROM messages
                    WHERE session_key = ? AND role = 'user'
                      AND julianday(ts) <= julianday(?)
                    ORDER BY julianday(ts) DESC, seq DESC
                    LIMIT 1
                    """,
                    (session_key, now.isoformat()),
                ).fetchone()
            return _parse_optional_time(row[0]) if row is not None else None
        getter = getattr(getattr(self._scope, "presence", None), "get_last_user_at", None)
        value = getter(session_key) if callable(getter) else None
        return value if isinstance(value, datetime) else None

    def _active_contexts(self, now: datetime) -> list[NormalizedContext]:
        return [
            context
            for context in self._state.list_contexts()
            if (
                context.expires_at is not None
                and context.expires_at >= now
            )
            or (
                context.expires_at is None
                and context.observed_at is not None
                and 0 <= (now - context.observed_at).total_seconds() <= 30 * 60
            )
        ]

    def _current_context_text(self, now: datetime) -> str:
        """把所有仍有效的原始 ContextEvent 交给 Agent 判断。"""

        contexts = self._active_contexts(now)
        if not contexts:
            return "没有有效 ContextEvent"
        return "\n".join(
            json.dumps(context.raw, ensure_ascii=False, sort_keys=True)
            for context in contexts
        )

    async def _ack_and_consume(
        self, events: list[dict[str, Any]], now: datetime
    ) -> None:
        grouped: dict[str, list[str]] = {}
        for event in events:
            source_id = str(event.get("_reservoir_ack_source_id") or "")
            source_event_id = str(event.get("_reservoir_source_event_id") or "")
            if source_id and source_event_id:
                grouped.setdefault(source_id, []).append(source_event_id)
        self._state.consume_and_queue_ack(
            item_ids=[str(event["id"]) for event in events],
            acknowledgements=grouped,
            now=now,
        )
        await self._flush_pending_acknowledgements()

    async def _consume_events(
        self,
        events: list[dict[str, Any]],
        now: datetime,
    ) -> None:
        self._state.consume(
            [str(event["id"]) for event in events],
            now,
        )

    async def _flush_pending_acknowledgements(self) -> None:
        grouped = self._state.pending_acknowledgements()
        for source_id, event_ids in grouped.items():
            try:
                await mcp_sources.acknowledge_async(
                    self._scope.mcp_gateway,
                    self._scope.proactive_sources,
                    source_id,
                    event_ids,
                )
            except Exception as exc:
                logger.warning(
                    "wake proactive ack pending source=%s count=%d error=%s",
                    source_id,
                    len(event_ids),
                    exc,
                )
                continue
            self._state.mark_acknowledged(source_id, event_ids)

    def _read_memory(self) -> str:
        reader = getattr(self._scope.memory, "read_long_term", None)
        return str(reader() or "") if callable(reader) else ""

    def _read_recent_session(
        self,
        session_key: str,
        now: datetime,
        *,
        include_proactive: bool = False,
    ) -> str:
        if not self._session_db_path.exists():
            return ""
        with closing(sqlite3.connect(str(self._session_db_path))) as db:
            rows = db.execute(
                """
                SELECT role, content, extra
                FROM messages
                WHERE session_key = ? AND julianday(ts) <= julianday(?)
                ORDER BY seq DESC
                LIMIT 20
                """,
                (session_key, now.isoformat()),
            ).fetchall()
        lines: list[str] = []
        for role, content, extra_json in reversed(rows):
            proactive = role == "assistant" and _is_proactive_message(extra_json)
            if role != "user" and role != "assistant":
                continue
            if proactive and not include_proactive:
                continue
            label = "assistant(proactive)" if proactive else role
            lines.append(f"{label}: {str(content or '')[:300]}")
        return "\n".join(lines)[:3_000]

    def _require_orchestrator(self) -> Any:
        if self._scope.turn_orchestrator is None:
            raise RuntimeError("wake proactive requires turn_orchestrator")
        return self._scope.turn_orchestrator


def _parse_optional_time(value: object) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    return datetime.fromisoformat(str(value))


def _content_expired(event: dict[str, Any], now: datetime) -> bool:
    """只在内容超龄或度过驻留期后跌破衰减线时淘汰。"""

    # 1. 发布时间可靠时直接淘汰绝对陈旧内容
    published_at = _parse_optional_time(event.get("published_at"))
    if published_at is not None and now - published_at >= _CONTENT_MAX_AGE:
        return True

    # 2. 新内容先获得一次跟随后续事件进入判别的机会
    first_seen_at = _parse_optional_time(event.get("first_seen_at"))
    if first_seen_at is None:
        raise ValueError("wake content missing first_seen_at")
    if now - first_seen_at < _CONTENT_MIN_RESIDENCE:
        return False
    return (
        float(event["_wake_rank_features"]["admission_mass"])
        < WAKE_ADMISSION_FLOOR
    )


def _group_acknowledgements(
    events: list[dict[str, Any]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for event in events:
        source_id = str(
            event.get("ack_server") or event.get("_source") or ""
        ).strip()
        source_event_id = str(
            event.get("event_id") or event.get("id") or ""
        ).strip()
        if source_id and source_event_id:
            grouped.setdefault(source_id, []).append(source_event_id)
    return grouped


def _source_samples(channels: dict[str, list[dict[str, Any]]]) -> str:
    samples: list[str] = []
    for kind in ("alert", "content", "context"):
        for event in channels[kind]:
            label = str(
                event.get("title")
                or event.get("topic")
                or event.get("summary")
                or event.get("event_id")
                or ""
            ).strip()
            if label:
                samples.append(f"{kind}:{label[:80]}")
            if len(samples) == 3:
                return repr(samples)
    return repr(samples)


def _event_source(event: dict[str, Any]) -> str:
    return str(
        event.get("_reservoir_original_source_id")
        or event.get("source_id")
        or event.get("source_name")
        or event.get("_source")
        or ""
    )


def _preprocess_interest(event: dict[str, Any]) -> float:
    raw_features: dict[str, Any] | None = None
    candidate_features = event.get("preprocess_features")
    if isinstance(candidate_features, dict):
        raw_features = cast(dict[str, Any], candidate_features)
    if raw_features is None:
        payload = event.get("payload")
        if isinstance(payload, dict):
            payload_features = cast(dict[str, Any], payload).get("features")
            if isinstance(payload_features, dict):
                raw_features = cast(dict[str, Any], payload_features)
    raw_interest = (
        raw_features.get("interest")
        if isinstance(raw_features, dict)
        else event.get("preprocess_score")
    )
    try:
        return min(0.999, max(0.0, float(raw_interest or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_weighted(user: list[float], assistant: list[float]) -> list[float]:
    if len(user) != len(assistant) or not user:
        return []
    combined = [0.9 * left + 0.1 * right for left, right in zip(user, assistant, strict=True)]
    norm = math.sqrt(sum(value * value for value in combined))
    return [value / norm for value in combined] if norm > 0 else []


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _is_proactive_message(extra_json: object) -> bool:
    try:
        extra = json.loads(str(extra_json or "{}"))
    except json.JSONDecodeError:
        return False
    if not isinstance(extra, dict):
        return False
    payload = cast(dict[str, Any], extra)
    return bool(payload.get("proactive"))


def _drift_trace(result: DriftDriveResult) -> dict[str, Any]:
    return {
        "hazard_before": result.hazard_before,
        "hazard_after": result.hazard_after,
        "threshold": result.threshold,
        "rate": result.rate,
        "idle_hours": result.idle_hours,
        "content_suppression": result.content_suppression,
        "recent_drift_suppression": result.recent_drift_suppression,
        "repetition_suppression": result.repetition_suppression,
        "reasons": list(result.reasons),
    }


def _hazard_trace(result: HazardResult) -> dict[str, Any]:
    return {
        "hazard_before": result.hazard_before,
        "hazard_after": result.hazard_after,
        "threshold": result.threshold,
        "evidence": result.evidence,
        "refractory": result.refractory,
        "rate": result.rate,
        "preference_pressure": result.preference_pressure,
        "should_wake": result.should_wake,
        "driver_item_id": result.driver_item_id,
    }


def _drift_timer_anchor(
    *,
    last_user_at: datetime | None,
    last_drift_at: datetime | None,
    repetition: float,
) -> str:
    return "|".join(
        (
            last_user_at.isoformat() if last_user_at is not None else "none",
            last_drift_at.isoformat() if last_drift_at is not None else "none",
            f"{repetition:.6f}",
        )
    )
