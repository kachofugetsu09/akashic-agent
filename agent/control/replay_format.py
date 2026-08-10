from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from agent.control.models import TurnItem, TurnItemKind, TurnRecord

ATTEMPT_INTERRUPTED_MARKER = "[execution attempt interrupted]"

METADATA_ATTEMPT_REPLAY = "_controlAttemptReplay"
METADATA_PRIOR_TOOL_CHAIN = "_controlPriorToolChain"


def replay_messages(
    attempts: list[TurnRecord],
    *,
    tool_group_from_item: Callable[[TurnItem], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """把中断 attempt 的 items 投影为模型历史，仅供下一 attempt 使用。"""

    messages: list[dict[str, Any]] = []
    for attempt in attempts:
        for item in attempt.items:
            if item.kind is TurnItemKind.USER_MESSAGE:
                content = item.data.get("content")
                media = item.data.get("media", [])
                if not isinstance(content, str):
                    raise ValueError(f"turn user content 必须是字符串: {item.id}")
                if not isinstance(media, list) or not all(
                    isinstance(value, str) for value in media
                ):
                    raise ValueError(f"turn user media 必须是字符串数组: {item.id}")
                if media:
                    content = "\n".join(
                        [
                            content,
                            "",
                            "[附加媒体]",
                            *(f"- {value}" for value in media),
                        ]
                    )
                messages.append({"role": "user", "content": content})
            elif item.kind is TurnItemKind.TOOL_CALL:
                group = tool_group_from_item(item)
                if group is None:
                    continue
                call = group["calls"][0]
                messages.append(
                    {
                        "role": "assistant",
                        "content": group["text"],
                        "tool_calls": [
                            {
                                "id": call["call_id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(
                                        call["arguments"], ensure_ascii=False
                                    ),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["call_id"],
                        "content": call["result"],
                    }
                )
        messages.append(
            {
                "role": "assistant",
                "content": ATTEMPT_INTERRUPTED_MARKER,
            }
        )
    return messages


def split_replay_batches(
    attempt_replay: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """把 replay 拆成已闭合的工具批次和未触碰的尾部。"""

    # 1. 每个 assistant tool call 之后必须有全部匹配结果。
    batches: list[list[dict[str, Any]]] = []
    batch_start = 0
    cursor = 0
    while cursor < len(attempt_replay):
        message = attempt_replay[cursor]
        raw_calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(raw_calls, list):
            cursor += 1
            continue
        call_ids = {
            str(call.get("id"))
            for call in raw_calls
            if isinstance(call, dict) and isinstance(call.get("id"), str)
        }
        if not call_ids or len(call_ids) != len(raw_calls):
            raise RuntimeError("control attempt replay tool call identity 无效")
        result_ids: set[str] = set()
        batch_end = cursor + 1
        while batch_end < len(attempt_replay):
            result = attempt_replay[batch_end]
            if result.get("role") != "tool":
                break
            result_id = result.get("tool_call_id")
            if not isinstance(result_id, str):
                raise RuntimeError("control attempt replay tool result identity 无效")
            result_ids.add(result_id)
            batch_end += 1
        if result_ids != call_ids:
            # 不完整的尾部保持 pending，永不提升为 active。
            if batches:
                return batches, list(attempt_replay[batch_start:])
            return [], list(attempt_replay)
        batches.append(list(attempt_replay[batch_start:batch_end]))
        batch_start = batch_end
        cursor = batch_end

    # 2. 中断标记和任何非工具后缀按顺序留在 pending。
    return batches, list(attempt_replay[batch_start:])
