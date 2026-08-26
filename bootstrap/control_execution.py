from __future__ import annotations

from typing import Any, cast
from datetime import datetime

import openai

from agent.control.errors import ControlExecutionError
from agent.control.ids import new_item_id
from agent.control.models import TurnItem, TurnItemKind, TurnRequest, TurnUsage
from agent.control.ports import ControlExecutionResult
from agent.control.replay_format import (
    METADATA_ATTEMPT_REPLAY,
    METADATA_PRIOR_TOOL_CHAIN,
)
from agent.control.turn_scope import get_current_turn_scope
from agent.turn_effects import PostCommitEffect, TurnStorage, set_post_commit_effect
from agent.looping.core import AgentLoop
from agent.model_runtime.errors import (
    AuthenticationError,
    ContextWindowError,
    QuotaError,
    RateLimitError,
    RetryableTransportError,
    TransportError,
)
from agent.provider import (
    ContentSafetyError,
    ContextLengthError,
    LLMNetworkTimeoutError,
)
from agent.plugins.snapshot import RuntimeSelector
from bus.event_bus import EventBus
from bus.events import TurnDisposition
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
    interaction_id = str(request.metadata.get("interactionId") or turn_id)
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
            inbound_metadata = _inbound_metadata(
                request.metadata.get("inboundMetadata")
            )
            turn_scope = get_current_turn_scope()
            if turn_scope is not None:
                if turn_scope.storage is TurnStorage.IN_MEMORY:
                    inbound_metadata.update(
                        {
                            "omit_user_turn": True,
                            "omit_assistant_turn": True,
                        }
                    )
                    if not turn_scope.session_history_read:
                        inbound_metadata["skip_session_history"] = True
                if turn_scope.disabled_prompt_sections:
                    inbound_metadata["disabled_prompt_sections"] = sorted(
                        turn_scope.disabled_prompt_sections
                    )
                if turn_scope.post_commit_effect is PostCommitEffect.SUPPRESS:
                    set_post_commit_effect(inbound_metadata, PostCommitEffect.SUPPRESS)
            if request.metadata.get("_pluginRolloutGenerationId"):
                inbound_metadata["_pluginCandidateValidation"] = True
            input_source = request.metadata.get("_controlTurnInputSource")
            if input_source is None:
                raise RuntimeError("control executor 缺少 turn input source")
            outbound = await loop.process_direct_message(
                request.input,
                session_key=request.thread_id,
                busy_session_key=(
                    str(request.metadata["busySessionId"])
                    if request.metadata.get("busySessionId")
                    else None
                ),
                channel=str(request.metadata.get("channel") or "programmatic"),
                chat_id=str(request.metadata.get("chatId") or request.thread_id),
                sender=str(request.metadata.get("sender") or "user"),
                media=_media_values(request.metadata.get("media")),
                metadata=inbound_metadata,
                turn_input_source=input_source,
                timestamp=_input_timestamp(request.metadata.get("inputTimestamp")),
                turn_id=turn_id,
                interaction_id=interaction_id,
                attempt_replay=_attempt_replay(
                    request.metadata.get(METADATA_ATTEMPT_REPLAY)
                ),
                prior_tool_chain=_prior_tool_chain(
                    request.metadata.get(METADATA_PRIOR_TOOL_CHAIN)
                ),
                prior_input_count=_prior_input_count(
                    request.metadata.get("priorInputCount")
                ),
                stream_events=True,
                runtime_selector=cast(
                    RuntimeSelector,
                    request.metadata.get("runtime", "stable"),
                ),
            )
        except (openai.RateLimitError, RateLimitError) as exc:
            raise ControlExecutionError(
                "provider_rate_limited", str(exc), retryable=True
            ) from exc
        except (openai.APITimeoutError, LLMNetworkTimeoutError) as exc:
            raise ControlExecutionError(
                "provider_timeout", str(exc), retryable=True
            ) from exc
        except (openai.APIConnectionError, RetryableTransportError) as exc:
            raise ControlExecutionError(
                "provider_connection_error", str(exc), retryable=True
            ) from exc
        except openai.APIStatusError as exc:
            raise ControlExecutionError(
                "provider_error",
                str(exc),
                retryable=exc.status_code >= 500,
            ) from exc
        except (AuthenticationError, QuotaError) as exc:
            raise ControlExecutionError(
                "provider_auth_error", str(exc), retryable=False
            ) from exc
        except (ContextLengthError, ContextWindowError) as exc:
            raise ControlExecutionError(
                "context_window_exceeded", str(exc), retryable=False
            ) from exc
        except ContentSafetyError as exc:
            raise ControlExecutionError(
                "content_safety", str(exc), retryable=False
            ) from exc
        except TransportError as exc:
            raise ControlExecutionError(
                "provider_transport_error", str(exc), retryable=False
            ) from exc
    finally:
        delta_subscription.close()
        committed_subscription.close()
        tool_started_subscription.close()
        tool_subscription.close()

    # 2. 插件命令可在推理前合法短路；普通 turn 仍必须完成正式提交。
    short_circuited = outbound.turn_disposition is TurnDisposition.SHORT_CIRCUITED
    if committed is None and not short_circuited:
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
            "attachmentIds": [ref.artifact_id for ref in outbound.attachment_refs],
            "metadata": dict(outbound.metadata),
            "sessionMessageId": outbound.session_message_id,
        },
        items=completed_items,
        deltas=deltas,
        usage=_turn_usage(committed.model_usage) if committed is not None else None,
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
            "iteration": event.iteration,
            "runtimeProvenance": dict(event.runtime_provenance),
        },
    )


def _attempt_replay(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("control attempt replay 必须是对象数组")
    return [dict(cast(dict[str, Any], item)) for item in value]


def _prior_tool_chain(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("control prior tool chain 必须是对象数组")
    return [dict(cast(dict[str, Any], item)) for item in value]


def _prior_input_count(value: object) -> int:
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("control prior input count 必须是非负整数")
    return value


def _media_values(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("control metadata media 必须是字符串数组")
    return list(value)


def _inbound_metadata(value: object) -> dict[str, object]:
    """校验并复制渠道随入站消息提交的内部元数据。"""

    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("control inboundMetadata 必须是字符串键对象")
    metadata = dict(cast(dict[str, object], value))
    if "skip_post_memory" in metadata:
        raise ValueError(
            "control inboundMetadata.skip_post_memory 已移除；请声明 Turn effects"
        )
    return metadata


def _input_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("control inputTimestamp 必须是 RFC 3339 字符串")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("control inputTimestamp 必须包含时区")
    return parsed


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
