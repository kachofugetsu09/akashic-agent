from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


DRIFT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "share_drift",
            "description": "发送一条自然、克制、与近期上下文有关的主动消息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "idle_drift",
            "description": "当前没有自然且值得打扰用户的话题，保持安静。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(frozen=True, slots=True)
class DriftToolResult:
    decision: Literal["reply", "skip"]
    message: str


def execute_drift_tool(name: str, arguments: dict[str, Any]) -> DriftToolResult:
    if name == "idle_drift":
        return DriftToolResult(decision="skip", message="")
    if name != "share_drift":
        raise ValueError(f"unknown drift tool: {name}")
    message = str(arguments.get("message") or "").strip()
    if not message:
        raise ValueError("share_drift message 不能为空")
    return DriftToolResult(decision="reply", message=message)
