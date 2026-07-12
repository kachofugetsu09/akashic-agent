from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
from hashlib import sha256
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from core.clock import Clock, ReplayClock, clock_from_env
from plugins.wake_proactive.context import WakeContext
from plugins.wake_proactive.context_drive import ContextDriveResult, NormalizedContext
from plugins.wake_proactive.drift_drive import DriftDriveResult, advance_drift_drive
from plugins.wake_proactive.drift_prompt import build_drift_messages
from plugins.wake_proactive.drift_tools import (
    DRIFT_TOOL_SCHEMAS,
    DriftToolResult,
    execute_drift_tool,
)
from plugins.wake_proactive.hazard import HazardResult, advance_hazard
from plugins.wake_proactive.modules import build_wake_modules
from plugins.wake_proactive.prompt import build_messages
from plugins.wake_proactive.state import WakeStateStore
from plugins.wake_proactive.tools import TOOL_SCHEMAS, ToolDeps, execute
from proactive_v2 import mcp_sources
from proactive_v2.frame import ProactiveFrame
from proactive_v2.runtime_scope import ProactiveRuntimeScope
from session.embedding_store import MessageEmbeddingStore


logger = logging.getLogger(__name__)
_SCHEMA_BY_NAME = {
    schema["function"]["name"]: schema
    for schema in TOOL_SCHEMAS
}


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
    drift_result: DriftDriveResult | None = None
    new_content_count: int = 0


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

    def build_modules(self) -> list[object]:
        return build_wake_modules(self)

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
        await self._flush_pending_acknowledgements()
        channels = await mcp_sources.fetch_sources_async(
            self._scope.mcp_gateway,
            self._scope.proactive_sources,
        )
        _ = self._state.ingest("alert", channels["alert"], state.ctx.now_utc)
        state.new_content_count = self._state.ingest(
            "content", channels["content"], state.ctx.now_utc
        )
        state.context_results = self._state.ingest_context(
            channels["context"], state.ctx.now_utc
        )
        state.context_reevaluate = any(
            result.signal == "reevaluate" for result in state.context_results
        )
        state.alerts = self._state.unread("alert")
        state.contents = self._state.unread("content")
        if state.alerts:
            return
        await self._cache_event_embeddings()
        state.contents = self._state.unread("content")
        self._apply_semantic_interest(state.contents, state.ctx.now_utc)

    async def decide(self, state: WakeRunState) -> None:
        if state.alerts:
            await self._deliver_alert(state, state.alerts[0])
            state.next_interval_seconds = (
                1 if len(state.alerts) > 1 else self._tick_interval_seconds
            )
            return
        should_evaluate_content = bool(state.contents)
        if should_evaluate_content:
            hazard_state = self._state.load_hazard(state.ctx.session_key)
            threshold = (
                float(hazard_state["threshold"])
                if hazard_state is not None
                else float(self._rng.gammavariate(3.0, 1 / 3))
            )
            updated_at = _parse_optional_time(
                hazard_state.get("updated_at") if hazard_state is not None else None
            )
            last_wake_at = _parse_optional_time(
                hazard_state.get("last_wake_at") if hazard_state is not None else None
            )
            current_hazard = (
                float(hazard_state["hazard"]) if hazard_state is not None else 0.0
            )
            result = advance_hazard(
                state.contents,
                now=state.ctx.now_utc,
                hazard=current_hazard,
                threshold=threshold,
                updated_at=updated_at,
                last_wake_at=last_wake_at,
            )
            state.hazard_result = result
            state.base_score = result.rate
            if result.should_wake:
                state.ctx.content_events = state.contents
                await self._run_content_tools(state.ctx)
                completed = await self._commit_content_decision(state)
                self._state.save_hazard(
                    session_key=state.ctx.session_key,
                    hazard=0.0 if completed else result.hazard_after,
                    threshold=(
                        float(self._rng.gammavariate(3.0, 1 / 3))
                        if completed
                        else result.threshold
                    ),
                    updated_at=state.ctx.now_utc,
                    last_wake_at=state.ctx.now_utc if completed else last_wake_at,
                )
                state.next_interval_seconds = self._tick_interval_seconds
                return
            self._state.save_hazard(
                session_key=state.ctx.session_key,
                hazard=result.hazard_after,
                threshold=result.threshold,
                updated_at=state.ctx.now_utc,
                last_wake_at=last_wake_at,
            )

        if state.context_reevaluate:
            state.next_interval_seconds = self._tick_interval_seconds
            return

        await self._decide_drift(state)
        state.next_interval_seconds = self._tick_interval_seconds

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
            base = min(0.999, max(0.0, float(event.get("preprocess_score") or 0.0)))
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
            semantic_interest = min(0.999, max(0.0, similarity))
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

    async def _deliver_alert(
        self, state: WakeRunState, alert: dict[str, Any]
    ) -> None:
        title = str(alert.get("title") or "提醒").strip()
        body = str(alert.get("content") or alert.get("body") or "").strip()
        message = title if not body else f"{title}\n\n{body}"
        item_id = str(alert["id"])
        result = TurnResult(
            decision="reply",
            outbound=TurnOutbound(session_key=state.ctx.session_key, content=message),
            evidence=[item_id],
            trace=TurnTrace(source="proactive", extra={"source_refs": []}),
            success_side_effects=[
                AsyncEffect(lambda: self._ack_and_consume([alert], state.ctx.now_utc))
            ],
        )
        orchestrator = self._require_orchestrator()
        await orchestrator.handle_proactive_turn(
            result=result,
            session_key=state.ctx.session_key,
            channel=str(getattr(self._scope.cfg, "default_channel", "")),
            chat_id=str(getattr(self._scope.cfg, "default_chat_id", "")),
        )

    async def _run_content_tools(self, ctx: WakeContext) -> None:
        base_messages = build_messages(
            ctx=ctx,
            memory_text=self._read_memory(),
            proactive_context=str(self._scope.workspace_context_fn() or ""),
            recent_session=self._read_recent_session(ctx.session_key, ctx.now_utc),
        )
        await self._run_phase(list(base_messages), ctx, {"scratchpad"}, "scratchpad")
        investigation = await self._run_investigation(ctx)
        final_messages = [
            {
                "role": "system",
                "content": (
                    "标题初筛和并发调查已经完成。现在只做最终判断："
                    "调用 share_content 分享有事实证据且值得现在说的内容，"
                    "或调用 skip_content 保持安静。不得调用其他工具，不要重新初筛。"
                    "最终最多产生一条自然消息，说明发生了什么以及为什么和用户有关。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{base_messages[1]['content']}\n\n"
                    "【已执行的初筛与并发调查结果】\n"
                    f"{investigation}\n\n"
                    "请做最终判断。"
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

    async def _run_investigation(
        self,
        ctx: WakeContext,
    ) -> str:
        return await execute(
            "investigate_candidates",
            {},
            ctx,
            self._tool_deps,
        )

    async def _run_phase(
        self,
        messages: list[dict[str, Any]],
        ctx: WakeContext,
        allowed: set[str],
        forced_name: str | None,
    ) -> None:
        schemas = [_SCHEMA_BY_NAME[name] for name in sorted(allowed)]
        tool_choice: str | dict[str, Any] = "required"
        if forced_name is not None:
            tool_choice = {"type": "function", "function": {"name": forced_name}}
        response = await self._scope.provider.chat(
            messages=messages,
            tools=schemas,
            model=str(getattr(self._scope.cfg, "agent_tick_model", "") or self._scope.model),
            max_tokens=self._scope.max_tokens,
            tool_choice=tool_choice,
            disable_thinking=True,
        )
        if not response.tool_calls:
            raise RuntimeError("wake proactive phase requires one tool call")
        call = response.tool_calls[0]
        if call.name not in allowed:
            raise RuntimeError(f"wake proactive unexpected tool in phase: {call.name}")
        output = await execute(call.name, call.arguments, ctx, self._tool_deps)
        messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    async def _commit_content_decision(self, state: WakeRunState) -> bool:
        events = list(state.contents)
        effect = AsyncEffect(lambda: self._ack_and_consume(events, state.ctx.now_utc))
        if state.ctx.terminal_action == "skip":
            result = TurnResult(
                decision="skip",
                outbound=None,
                evidence=[],
                trace=TurnTrace(source="proactive"),
                side_effects=[effect],
            )
            await self._require_orchestrator().handle_proactive_turn(
                result=result,
                session_key=state.ctx.session_key,
                channel=str(getattr(self._scope.cfg, "default_channel", "")),
                chat_id=str(getattr(self._scope.cfg, "default_chat_id", "")),
            )
            return True

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
        return await self._require_orchestrator().handle_proactive_turn(
            result=result,
            session_key=state.ctx.session_key,
            channel=str(getattr(self._scope.cfg, "default_channel", "")),
            chat_id=str(getattr(self._scope.cfg, "default_chat_id", "")),
        )

    async def _decide_drift(self, state: WakeRunState) -> None:
        if not bool(getattr(self._scope.cfg, "drift_enabled", True)):
            return
        stored = self._state.load_drift(state.ctx.session_key) or {}
        contexts = self._active_contexts(state.ctx.now_utc)
        last_user_at = self._last_user_at(state.ctx.session_key)
        content_evidence = max(
            (float(event.get("_wake_interest_score") or 0.0) for event in state.contents),
            default=0.0,
        )
        result = advance_drift_drive(
            now=state.ctx.now_utc,
            hazard=float(stored.get("hazard") or 0.0),
            threshold=float(stored.get("threshold") or 0.8),
            updated_at=_parse_optional_time(stored.get("updated_at")),
            last_user_at=last_user_at,
            last_drift_at=_parse_optional_time(stored.get("last_drift_at")),
            content_evidence=content_evidence,
            busy=any(
                context.presence == "offline"
                or (
                    context.presence not in {"sleeping", "in_game"}
                    and context.interruptibility <= 0.1
                )
                for context in contexts
            ),
            sleeping=any(context.presence == "sleeping" for context in contexts),
            in_game=any(context.presence == "in_game" for context in contexts),
            repetition=min(1.0, float(stored.get("repeat_count") or 0) / 3.0),
        )
        state.drift_result = result
        state.base_score = max(state.base_score, result.rate)
        self._state.save_drift_progress(
            session_key=state.ctx.session_key,
            hazard=result.hazard_after,
            threshold=result.threshold,
            updated_at=state.ctx.now_utc,
        )
        if result.decision != "attempt":
            return

        decision = await self._run_drift_tool(state.ctx, result)
        if decision.decision == "skip":
            self._state.save_drift_progress(
                session_key=state.ctx.session_key,
                hazard=0.0,
                threshold=result.threshold,
                updated_at=state.ctx.now_utc,
            )
            return

        fingerprint = _message_fingerprint(decision.message)
        effect = AsyncEffect(
            lambda: self._record_drift_success(
                state.ctx.session_key,
                state.ctx.now_utc,
                fingerprint,
            )
        )
        turn = TurnResult(
            decision="reply",
            outbound=TurnOutbound(
                session_key=state.ctx.session_key,
                content=decision.message,
            ),
            evidence=[],
            trace=TurnTrace(
                source="proactive",
                extra={"wake_flow": "drift", "drift_drive": _drift_trace(result)},
            ),
            success_side_effects=[effect],
        )
        _ = await self._require_orchestrator().handle_proactive_turn(
            result=turn,
            session_key=state.ctx.session_key,
            channel=str(getattr(self._scope.cfg, "default_channel", "")),
            chat_id=str(getattr(self._scope.cfg, "default_chat_id", "")),
        )

    async def _run_drift_tool(
        self,
        ctx: WakeContext,
        drive: DriftDriveResult,
    ) -> DriftToolResult:
        messages = build_drift_messages(
            memory_text=self._read_memory(),
            proactive_context=str(self._scope.workspace_context_fn() or ""),
            recent_session=self._read_recent_session(
                ctx.session_key,
                ctx.now_utc,
                include_proactive=True,
            ),
            drive=drive,
        )
        response = await self._scope.provider.chat(
            messages=messages,
            tools=DRIFT_TOOL_SCHEMAS,
            model=str(getattr(self._scope.cfg, "agent_tick_model", "") or self._scope.model),
            max_tokens=self._scope.max_tokens,
            tool_choice="required",
            disable_thinking=True,
        )
        if len(response.tool_calls) != 1:
            raise RuntimeError("wake proactive drift requires one tool call")
        call = response.tool_calls[0]
        return execute_drift_tool(call.name, call.arguments)

    async def _record_drift_success(
        self,
        session_key: str,
        now: datetime,
        fingerprint: str,
    ) -> None:
        self._state.record_drift_success(
            session_key=session_key,
            now=now,
            fingerprint=fingerprint,
        )

    def _last_user_at(self, session_key: str) -> datetime | None:
        getter = getattr(getattr(self._scope, "presence", None), "get_last_user_at", None)
        value = getter(session_key) if callable(getter) else None
        return value if isinstance(value, datetime) else None

    def _active_contexts(self, now: datetime) -> list[NormalizedContext]:
        return [
            context
            for context in self._state.list_contexts()
            if context.confidence >= 0.55
            and (context.expires_at is None or context.expires_at >= now)
        ]

    async def _ack_and_consume(
        self, events: list[dict[str, Any]], now: datetime
    ) -> None:
        grouped: dict[str, list[str]] = {}
        for event in events:
            source_id = str(event.get("_reservoir_source_id") or "")
            source_event_id = str(event.get("_reservoir_source_event_id") or "")
            if source_id and source_event_id:
                grouped.setdefault(source_id, []).append(source_event_id)
        self._state.consume_and_queue_ack(
            item_ids=[str(event["id"]) for event in events],
            acknowledgements=grouped,
            now=now,
        )
        await self._flush_pending_acknowledgements()

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


def _message_fingerprint(message: str) -> str:
    normalized = " ".join(message.lower().split())
    return sha256(normalized.encode("utf-8")).hexdigest()


def _drift_trace(result: DriftDriveResult) -> dict[str, Any]:
    return {
        "hazard_before": result.hazard_before,
        "hazard_after": result.hazard_after,
        "threshold": result.threshold,
        "rate": result.rate,
        "idle_hours": result.idle_hours,
        "content_suppression": result.content_suppression,
        "context_suppression": result.context_suppression,
        "recent_drift_suppression": result.recent_drift_suppression,
        "repetition_suppression": result.repetition_suppression,
        "reasons": list(result.reasons),
    }
