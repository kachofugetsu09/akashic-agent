from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from agent.tools.base import Tool, ToolResult, normalize_tool_result


def build_tool_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def build_tool_map(tools: list[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools}


def tool_call_batch_snapshot(tool_calls: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    batch: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        raw_arguments: object = getattr(tool_call, "arguments", {})
        snapshot_args: dict[str, Any] = {}
        if isinstance(raw_arguments, dict):
            for key, value in cast("dict[Any, Any]", raw_arguments).items():
                snapshot_args[str(key)] = value
        batch.append(
            {
                "name": str(getattr(tool_call, "name", "")),
                "arguments": snapshot_args,
            }
        )
    return tuple(batch)


def format_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    return [
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
    ]


def append_assistant_tool_calls(
    messages: list[dict[str, Any]],
    *,
    content: str | None,
    tool_calls: list[Any],
    provider_fields: dict[str, Any] | None = None,
) -> None:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "tool_calls": format_tool_calls(tool_calls),
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
) -> None:
    result = normalize_tool_result(content)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result.text or "工具执行完成。",
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
