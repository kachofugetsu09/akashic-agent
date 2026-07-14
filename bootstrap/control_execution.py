from __future__ import annotations

from typing import Any, cast

import openai

from agent.control.errors import ControlExecutionError
from agent.control.ids import new_item_id
from agent.control.models import TurnItem, TurnItemKind, TurnRequest, TurnUsage
from agent.control.ports import ControlExecutionResult
from agent.looping.core import AgentLoop
from agent.model_runtime.errors import (
    AuthenticationError,
    ContextWindowError,
    QuotaError,
    RateLimitError,
    RetryableTransportError,
    TransportError,
)
from agent.provider import ContentSafetyError, ContextLengthError, LLMNetworkTimeoutError
from bus.event_bus import EventBus
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnCommitted,
)


async def execute_control_turn(
    loop: AgentLoop,
    event_bus: EventBus,
    request: TurnRequest,
) -> ControlExecutionResult:
    """执行正式被动 turn，并把工具与用量投影到控制面结果。"""

    turn_id = str(request.metadata["turnId"])
    completed_items: list[TurnItem] = []
    tool_item_ids: dict[str, str] = {}
    invalid_tool_events: list[str] = []
    deltas: list[str] = []
    committed: TurnCommitted | None = None

    def collect_tool(event: ToolCallCompleted) -> None:
        if event.turn_id == turn_id:
            item_id = tool_item_ids.pop(event.call_id, None)
            if item_id is None:
                invalid_tool_events.append(event.call_id)
                return
            item = _tool_item(event, item_id)
            completed_items.append(item)
            emit_item("item/completed", item)

    def collect_tool_started(event: ToolCallStarted) -> None:
        if event.turn_id != turn_id:
            return
        item_id = new_item_id()
        tool_item_ids[event.call_id] = item_id
        emit_item(
            "item/started",
            TurnItem(
                TurnItemKind.TOOL_CALL,
                item_id,
                {
                    "callId": event.call_id,
                    "name": event.tool_name,
                    "arguments": dict(event.arguments),
                    "status": "in_progress",
                },
            ),
        )

    def collect_committed(event: TurnCommitted) -> None:
        nonlocal committed
        if event.turn_id == turn_id:
            committed = event

    def collect_delta(event: StreamDeltaReady) -> None:
        if event.turn_id == turn_id and event.content_delta:
            deltas.append(event.content_delta)

    raw_emit_item = request.metadata.get("_controlItemEvent")
    if not callable(raw_emit_item):
        raise RuntimeError("control executor 缺少 item event sink")

    def emit_item(method: str, item: TurnItem) -> None:
        raw_emit_item(method, item)

    # 1. 仅在本 turn 生命周期内收集同 turn id 的领域事件。
    tool_subscription = event_bus.on(ToolCallCompleted, collect_tool)
    tool_started_subscription = event_bus.on(ToolCallStarted, collect_tool_started)
    committed_subscription = event_bus.on(TurnCommitted, collect_committed)
    delta_subscription = event_bus.on(StreamDeltaReady, collect_delta)
    try:
        try:
            outbound = await loop.process_direct_message(
                request.input,
                session_key=request.thread_id,
                channel=str(request.metadata.get("channel") or "programmatic"),
                chat_id=str(request.metadata.get("chatId") or request.thread_id),
                sender=str(request.metadata.get("sender") or "user"),
                media=_media_values(request.metadata.get("media")),
                turn_id=turn_id,
                stream_events=True,
            )
        except (openai.RateLimitError, RateLimitError) as exc:
            raise ControlExecutionError("provider_rate_limited", str(exc), retryable=True) from exc
        except (openai.APITimeoutError, LLMNetworkTimeoutError) as exc:
            raise ControlExecutionError("provider_timeout", str(exc), retryable=True) from exc
        except (openai.APIConnectionError, RetryableTransportError) as exc:
            raise ControlExecutionError("provider_connection_error", str(exc), retryable=True) from exc
        except openai.APIStatusError as exc:
            raise ControlExecutionError(
                "provider_error",
                str(exc),
                retryable=exc.status_code >= 500,
            ) from exc
        except (AuthenticationError, QuotaError) as exc:
            raise ControlExecutionError("provider_auth_error", str(exc), retryable=False) from exc
        except (ContextLengthError, ContextWindowError) as exc:
            raise ControlExecutionError("context_window_exceeded", str(exc), retryable=False) from exc
        except ContentSafetyError as exc:
            raise ControlExecutionError("content_safety", str(exc), retryable=False) from exc
        except TransportError as exc:
            raise ControlExecutionError("provider_transport_error", str(exc), retryable=False) from exc
    finally:
        delta_subscription.close()
        committed_subscription.close()
        tool_started_subscription.close()
        tool_subscription.close()

    # 2. 核心成功后 TurnCommitted 必须已同步 fanout，否则属于内部契约错误。
    if committed is None:
        raise RuntimeError(f"turn 缺少 TurnCommitted 事件: {turn_id}")
    if tool_item_ids or invalid_tool_events:
        raise RuntimeError(
            "tool call 生命周期不完整: "
            f"未完成={sorted(tool_item_ids)} 无开始={sorted(invalid_tool_events)}"
        )
    return ControlExecutionResult(
        response=outbound.content,
        assistant_data={
            "thinking": outbound.thinking,
            "replyTo": outbound.reply_to,
            "media": list(outbound.media),
            "metadata": dict(outbound.metadata),
        },
        items=completed_items,
        deltas=deltas,
        usage=_turn_usage(committed.model_usage),
    )


def _tool_item(event: ToolCallCompleted, item_id: str) -> TurnItem:
    return TurnItem(
        TurnItemKind.TOOL_CALL,
        item_id,
        {
            "callId": event.call_id,
            "name": event.tool_name,
            "arguments": dict(event.final_arguments),
            "status": event.status,
            "resultPreview": event.result_preview,
        },
    )


def _media_values(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("control metadata media 必须是字符串数组")
    return list(value)


def _turn_usage(value: dict[str, Any]) -> TurnUsage | None:
    if not value:
        return None
    return TurnUsage(
        input_tokens=cast(int | None, value.get("input_tokens")),
        cached_input_tokens=cast(int | None, value.get("cached_input_tokens")),
        output_tokens=cast(int | None, value.get("output_tokens")),
        reasoning_output_tokens=cast(int | None, value.get("reasoning_output_tokens")),
        request_count=cast(int, value.get("request_count", 0)),
        covered_request_count=cast(int, value.get("covered_request_count", 0)),
        coverage=cast(str, value.get("coverage", "unavailable")),
    )
