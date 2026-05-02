from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from contextlib import suppress
from typing import Protocol, cast, runtime_checkable

from agent.plugins import Plugin
from bus.events_lifecycle import TurnCommitted
from core.observe.retention import run_retention_if_needed
from core.observe.writer import TraceWriter

logger = logging.getLogger("plugin.observe")


@runtime_checkable
class _ObserveWriter(Protocol):
    def emit(self, event: object) -> None: ...


class ObservePlugin(Plugin):
    name = "observe"

    async def initialize(self) -> None:
        workspace = self.context.workspace
        if workspace is None:
            logger.warning("observe 插件缺少 workspace，跳过加载")
            return

        self._writer = TraceWriter(workspace / "observe" / "observe.db")
        self._writer_task = asyncio.create_task(
            self._writer.run(),
            name="observe_writer",
        )
        self._retention_task = asyncio.create_task(
            run_retention_if_needed(workspace / "observe" / "observe.db"),
            name="observe_retention",
        )
        self.context.event_bus.on(TurnCommitted, self._observe_turn_committed)

    async def terminate(self) -> None:
        for task in (
            getattr(self, "_retention_task", None),
            getattr(self, "_writer_task", None),
        ):
            if task is None:
                continue
            _ = task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def _observe_turn_committed(self, event: TurnCommitted) -> None:
        writer = getattr(self, "_writer", None)
        if not isinstance(writer, _ObserveWriter):
            return
        _emit_turn_trace(writer, event)


def _emit_turn_trace(writer: _ObserveWriter, event: TurnCommitted) -> None:
    from core.observe.events import TurnTrace as TurnTraceEvent

    post_reply_budget = event.post_reply_budget
    react_stats = event.react_stats
    tool_chain = event.tool_chain_raw
    tool_chain_json = (
        json.dumps(_slim_tool_chain(tool_chain), ensure_ascii=False)
        if tool_chain
        else None
    )
    tool_calls = _slim_tool_calls(tool_chain)
    writer.emit(
        TurnTraceEvent(
            source="agent",
            session_key=event.session_key,
            user_msg=event.persisted_user_message,
            llm_output=event.assistant_response,
            raw_llm_output=event.raw_reply,
            meme_tag=event.meme_tag,
            meme_media_count=event.meme_media_count,
            tool_calls=tool_calls,
            tool_chain_json=tool_chain_json,
            history_window=post_reply_budget.get("history_window"),
            history_messages=post_reply_budget.get("history_messages"),
            history_chars=post_reply_budget.get("history_chars"),
            history_tokens=post_reply_budget.get("history_tokens"),
            prompt_tokens=post_reply_budget.get("prompt_tokens"),
            next_turn_baseline_tokens=post_reply_budget.get(
                "next_turn_baseline_tokens"
            ),
            react_iteration_count=react_stats.get("iteration_count"),
            react_input_sum_tokens=react_stats.get("turn_input_sum_tokens"),
            react_input_peak_tokens=react_stats.get("turn_input_peak_tokens"),
            react_final_input_tokens=react_stats.get("final_call_input_tokens"),
            react_cache_prompt_tokens=react_stats.get("cache_prompt_tokens"),
            react_cache_hit_tokens=react_stats.get("cache_hit_tokens"),
        )
    )
    if event.retrieval_raw is not None:
        writer.emit(event.retrieval_raw)
    logger.info(
        "[observe] turn_trace 已入队 session=%s tool_calls=%d retrieval=%s",
        event.session_key,
        len(tool_calls),
        event.retrieval_raw is not None,
    )


def _slim_tool_calls(tool_chain: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {
            "name": str(call.get("name", "")),
            "args": str(call.get("arguments", ""))[:300],
            "result": str(call.get("result", ""))[:500],
        }
        for group in tool_chain
        for call in _group_calls(group)
    ]


def _slim_tool_chain(tool_chain: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "text": str(group.get("text") or ""),
            "calls": [
                {
                    "name": str(call.get("name", "")),
                    "args": str(call.get("arguments", ""))[:800],
                    "result": str(call.get("result", ""))[:1200],
                }
                for call in _group_calls(group)
            ],
        }
        for group in tool_chain
    ]


def _group_calls(group: dict[str, object]) -> list[dict[str, object]]:
    calls = group.get("calls")
    if not isinstance(calls, list):
        return []
    raw_calls = cast(list[object], calls)
    out: list[dict[str, object]] = []
    for call in raw_calls:
        if isinstance(call, Mapping):
            mapping = cast(Mapping[object, object], call)
            out.append(
                {
                    str(key): value
                    for key, value in mapping.items()
                    if isinstance(key, str)
                }
            )
    return out
