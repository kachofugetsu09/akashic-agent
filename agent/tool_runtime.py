from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.tools.base import Tool, ToolResult, normalize_tool_result


@dataclass(frozen=True)
class PreparedToolset:
    tools: list[Tool]
    schemas: list[dict[str, Any]]
    tool_map: dict[str, Tool]


def prepare_toolset(tools: list[Tool]) -> PreparedToolset:
    return PreparedToolset(
        tools=list(tools),
        schemas=build_tool_schemas(tools),
        tool_map=build_tool_map(tools),
    )


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


def tool_call_signature(tool_calls: list[Any]) -> str:
    parts: list[str] = []
    for tool_call in tool_calls:
        args = json.dumps(tool_call.arguments, ensure_ascii=False, sort_keys=True)
        parts.append(f"{tool_call.name}:{args}")
    return "|".join(parts)


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
) -> None:
    messages.append(
        {
            "role": "assistant",
            "content": content,
            "tool_calls": format_tool_calls(tool_calls),
        }
    )


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
