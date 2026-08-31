from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agent.plugin_composition import ToolCall
from agent.tools.base import ToolResult, normalize_tool_result


def tool_call_batch_snapshot(
    tool_calls: Sequence[ToolCall],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": tool_call.name,
            "arguments": dict(tool_call.arguments),
        }
        for tool_call in tool_calls
    )


def append_assistant_tool_calls(
    messages: list[dict[str, Any]],
    *,
    content: str | None,
    tool_calls: Sequence[ToolCall],
    provider_fields: dict[str, Any] | None = None,
) -> None:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(
                        tool_call.arguments,
                        ensure_ascii=False,
                    ),
                },
            }
            for tool_call in tool_calls
        ],
    }
    if provider_fields:
        message.update(provider_fields)
    messages.append(message)


def append_tool_result(
    messages: list[dict[str, Any]],
    *,
    tool_call_id: str,
    content: str | ToolResult,
    tool_name: str | None = None,
    execution_status: str | None = None,
) -> None:
    result = normalize_tool_result(content)
    text = result.text or "工具执行完成。"
    if execution_status is not None:
        text = (
            f'<tool_execution transport_status="{execution_status}" />\n'
            f"{text}"
        )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": text,
        }
    )
    if result.content_blocks:
        prefix = f"以下是工具 {tool_name} 读取到的文件内容，请直接查看。" if tool_name else "以下是工具读取到的文件内容，请直接查看。"
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prefix}, *result.content_blocks],
            }
        )
