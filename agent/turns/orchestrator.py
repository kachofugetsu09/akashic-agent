from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agent.core.types import to_tool_call_groups
from agent.postturn.protocol import PostTurnEvent
from agent.turns.outbound import OutboundDispatch, OutboundPort
from agent.turns.result import TurnResult

if TYPE_CHECKING:
    from agent.core.runtime_support import SessionLike
    from agent.looping.ports import (
        SessionServices,
        ObservabilityServices,
    )
    from agent.postturn.protocol import PostTurnPipeline

logger = logging.getLogger("agent.turn_orchestrator")


@runtime_checkable
class _ObserveWriter(Protocol):
    def emit(self, event: object) -> None: ...


@dataclass
class TurnOrchestratorDeps:
    session: SessionServices
    trace: ObservabilityServices
    post_turn: PostTurnPipeline
    outbound: OutboundPort


class TurnOrchestrator:
    def __init__(self, deps: TurnOrchestratorDeps) -> None:
        self._session = deps.session
        self._trace = deps.trace
        self._post_turn = deps.post_turn
        self._outbound = deps.outbound

    async def handle_proactive_turn(
        self,
        *,
        result: TurnResult,
        session_key: str,
        channel: str,
        chat_id: str,
    ) -> bool:
        # 1. proactive 先处理 skip：不发消息，只跑 skip 路径副作用并记 trace。
        if result.decision == "skip":
            self._emit_proactive_observe(
                key=session_key,
                channel=channel,
                chat_id=chat_id,
                result=result,
                sent=False,
            )
            await self._run_side_effects(result)
            return False

        if result.outbound is None:
            raise ValueError("proactive reply result requires outbound")

        content = result.outbound.content
        session = self._session.session_manager.get_or_create(session_key)
        # 2. reply 路径先把主动消息写进 session，再安排 post_turn。
        self._persist_proactive_session(
            session=session,
            content=content,
            result=result,
        )
        await self._session.session_manager.append_messages(session, session.messages[-1:])

        self._schedule_proactive_post_turn(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            session=session,
            result=result,
        )

        sent = False
        try:
            # 3. 先执行发送前 side_effects，再真正 dispatch 到 outbound。
            await self._run_effects(result.side_effects)
            sent = await self._outbound.dispatch(
                OutboundDispatch(
                    channel=channel,
                    chat_id=chat_id,
                    content=content,
                    metadata={},
                )
            )
        except Exception as e:
            logger.warning("proactive outbound dispatch failed: %s", e)

        # 4. 根据是否真正发送成功，分别执行 success / failure side_effects。
        if sent:
            if self._session.presence:
                self._session.presence.record_proactive_sent(session_key)
            await self._run_effects(result.success_side_effects)
        else:
            await self._run_effects(result.failure_side_effects)

        self._emit_proactive_observe(
            key=session_key,
            channel=channel,
            chat_id=chat_id,
            result=result,
            sent=sent,
        )
        return sent

    async def _run_side_effects(self, result: TurnResult) -> None:
        await self._run_effects(result.side_effects)

    async def _run_effects(self, effects: list[Any]) -> None:
        for effect in effects:
            try:
                maybe = effect.run()
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception as e:
                logger.warning("turn side effect failed: %s", e)

    def _persist_proactive_session(
        self,
        *,
        session: SessionLike,
        content: str,
        result: TurnResult,
    ) -> None:
        source_refs = []
        state_summary_tag = "none"
        if result.trace is not None and isinstance(result.trace.extra, dict):
            raw_refs = result.trace.extra.get("source_refs", [])
            if isinstance(raw_refs, list):
                source_refs = [ref for ref in raw_refs if isinstance(ref, dict)]
            state_summary_tag = str(result.trace.extra.get("state_summary_tag", "none"))
        session.add_message(
            "assistant",
            content,
            proactive=True,
            tools_used=["message_push"],
            evidence_item_ids=[str(item_id) for item_id in result.evidence],
            source_refs=source_refs,
            state_summary_tag=state_summary_tag,
        )

    def _schedule_proactive_post_turn(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        session: SessionLike,
        result: TurnResult,
    ) -> None:
        tool_chain = _trace_tool_chain(result.trace)
        tools_used = _trace_tools_used(result.trace)
        self._post_turn.schedule(
            PostTurnEvent(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                user_message="",
                assistant_response=result.outbound.content if result.outbound else "",
                tools_used=tools_used,
                tool_chain=to_tool_call_groups(tool_chain),
                session=session,
            )
        )

    def _emit_proactive_observe(
        self,
        *,
        key: str,
        channel: str,
        chat_id: str,
        result: TurnResult,
        sent: bool,
    ) -> None:
        writer = self._trace.observe_writer
        if not isinstance(writer, _ObserveWriter):
            return
        from core.observe.events import TurnTrace as TurnTraceEvent

        trace = result.trace
        extra = trace.extra if trace is not None and isinstance(trace.extra, dict) else {}
        writer.emit(
            TurnTraceEvent(
                source="proactive",
                session_key=key,
                user_msg="",
                llm_output=result.outbound.content if result.outbound else "",
                tool_calls=[
                    {
                        "name": "proactive_turn",
                        "args": json.dumps(
                            {
                                "channel": channel,
                                "chat_id": chat_id,
                                "decision": result.decision,
                                "evidence": list(result.evidence),
                                "sent": sent,
                                "steps_taken": int(extra.get("steps_taken", 0) or 0),
                                "skip_reason": str(extra.get("skip_reason", "")),
                            },
                            ensure_ascii=False,
                        ),
                        "result": "",
                    }
                ],
            )
        )


def _trace_tools_used(trace: Any | None) -> list[str]:
    if trace is None:
        return []
    raw = trace.extra.get("tools_used", []) if isinstance(trace.extra, dict) else []
    if not isinstance(raw, list):
        return []
    return [str(name) for name in raw if isinstance(name, str)]


def _trace_tool_chain(trace: Any | None) -> list[dict]:
    if trace is None:
        return []
    raw = trace.extra.get("tool_chain", []) if isinstance(trace.extra, dict) else []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]
