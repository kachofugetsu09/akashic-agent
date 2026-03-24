from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from agent.looping.memory_gate import _update_session_runtime_metadata
from agent.looping.turn_types import ToolCall, ToolCallGroup
from agent.postturn.protocol import PostTurnEvent
from agent.turns.result import TurnResult
from bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from agent.looping.ports import (
        SessionLike,
        SessionServices,
        ObservabilityServices,
    )
    from agent.postturn.protocol import PostTurnPipeline

logger = logging.getLogger("agent.turn_orchestrator")


@dataclass
class TurnOrchestratorDeps:
    session: SessionServices
    trace: ObservabilityServices
    post_turn: PostTurnPipeline


class TurnOrchestrator:
    def __init__(self, deps: TurnOrchestratorDeps) -> None:
        self._session = deps.session
        self._trace = deps.trace
        self._post_turn = deps.post_turn

    async def handle_turn(
        self,
        *,
        msg: InboundMessage,
        result: TurnResult,
    ) -> OutboundMessage:
        if result.decision != "reply" or result.outbound is None:
            raise ValueError("passive turn result must be reply with outbound")
        key = result.outbound.session_key
        session = self._session.session_manager.get_or_create(key)
        final_content = result.outbound.content
        tools_used = _trace_tools_used(result.trace)
        tool_chain = _trace_tool_chain(result.trace)
        thinking = _trace_thinking(result.trace)
        retrieval_raw = _trace_retrieval_raw(result.trace)

        await self._persist_session(
            msg=msg,
            session=session,
            final_content=final_content,
            tools_used=tools_used,
            tool_chain=tool_chain,
        )
        self._emit_observe_traces(
            key=key,
            msg=msg,
            final_content=final_content,
            tool_chain=tool_chain,
            retrieval_raw=retrieval_raw,
        )
        self._post_turn.schedule(
            PostTurnEvent(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                user_message=msg.content,
                assistant_response=final_content,
                tools_used=tools_used,
                tool_chain=_to_tool_call_groups(tool_chain),
                session=session,
                timestamp=msg.timestamp,
            )
        )
        await self._run_side_effects(result)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            thinking=thinking,
            metadata={
                **(msg.metadata or {}),
                "tools_used": tools_used,
                "tool_chain": tool_chain,
            },
        )

    async def handle_proactive_turn(
        self,
        *,
        result: TurnResult,
        session_key: str,
        channel: str,
        chat_id: str,
        dispatch_outbound: Callable[[str], Awaitable[bool]],
    ) -> bool:
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
        self._persist_proactive_session(
            session=session,
            content=content,
            result=result,
        )
        await self._session.session_manager.append_messages(session, session.messages[-1:])

        self._emit_proactive_observe(
            key=session_key,
            channel=channel,
            chat_id=chat_id,
            result=result,
            sent=False,
        )
        self._schedule_proactive_post_turn(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            session=session,
            result=result,
        )

        sent = False
        try:
            sent = await dispatch_outbound(content)
        except Exception as e:
            logger.warning("proactive outbound dispatch failed: %s", e)

        if sent:
            if self._session.presence:
                self._session.presence.record_proactive_sent(session_key)
            await self._run_effects(result.success_side_effects)
            await self._run_effects(result.side_effects)
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

    async def _persist_session(
        self,
        *,
        msg: InboundMessage,
        session: SessionLike,
        final_content: str,
        tools_used: list[str],
        tool_chain: list[dict],
    ) -> None:
        if self._session.presence:
            self._session.presence.record_user_message(session.key)
        session.add_message("user", msg.content, media=msg.media if msg.media else None)
        session.add_message(
            "assistant",
            final_content,
            tools_used=tools_used if tools_used else None,
            tool_chain=tool_chain if tool_chain else None,
        )
        _update_session_runtime_metadata(
            session,
            tools_used=tools_used,
            tool_chain=tool_chain,
        )
        await self._session.session_manager.append_messages(session, session.messages[-2:])

    def _emit_observe_traces(
        self,
        *,
        key: str,
        msg: InboundMessage,
        final_content: str,
        tool_chain: list[dict],
        retrieval_raw: Any | None,
    ) -> None:
        writer = self._trace.observe_writer
        if writer is None:
            return
        from core.observe.events import TurnTrace

        tool_calls = [
            {
                "name": call.get("name", ""),
                "args": str(call.get("arguments", ""))[:300],
                "result": str(call.get("result", ""))[:500],
            }
            for group in tool_chain
            for call in (group.get("calls") or [])
        ]

        def _slim_chain(chain: list[dict]) -> list[dict]:
            out = []
            for group in chain:
                text = str(group.get("text") or "")
                calls = [
                    {
                        "name": c.get("name", ""),
                        "args": str(c.get("arguments", ""))[:800],
                        "result": str(c.get("result", ""))[:1200],
                    }
                    for c in (group.get("calls") or [])
                ]
                out.append({"text": text, "calls": calls})
            return out

        tool_chain_json = (
            json.dumps(_slim_chain(tool_chain), ensure_ascii=False) if tool_chain else None
        )

        writer.emit(
            TurnTrace(
                source="agent",
                session_key=key,
                user_msg=msg.content,
                llm_output=final_content,
                tool_calls=tool_calls,
                tool_chain_json=tool_chain_json,
            )
        )
        if retrieval_raw is not None:
            writer.emit(retrieval_raw)

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
                tool_chain=_to_tool_call_groups(tool_chain),
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
        if writer is None:
            return
        from core.observe.events import TurnTrace

        trace = result.trace
        extra = trace.extra if trace is not None and isinstance(trace.extra, dict) else {}
        writer.emit(
            TurnTrace(
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


def _to_tool_call_groups(raw_chain: list[dict]) -> list[ToolCallGroup]:
    groups: list[ToolCallGroup] = []
    for group in raw_chain:
        text = str(group.get("text", "") or "")
        calls: list[ToolCall] = []
        for call in (group.get("calls") or []):
            args = call.get("arguments")
            calls.append(
                ToolCall(
                    call_id=str(call.get("call_id", "") or ""),
                    name=str(call.get("name", "") or ""),
                    arguments=args if isinstance(args, dict) else {},
                    result=str(call.get("result", "") or ""),
                )
            )
        groups.append(ToolCallGroup(text=text, calls=calls))
    return groups


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


def _trace_thinking(trace: Any | None) -> str | None:
    if trace is None or not isinstance(trace.extra, dict):
        return None
    raw = trace.extra.get("thinking")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _trace_retrieval_raw(trace: Any | None) -> Any | None:
    if trace is None or not isinstance(trace.retrieval, dict):
        return None
    return trace.retrieval.get("raw")
