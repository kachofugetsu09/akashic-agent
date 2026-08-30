from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""


@dataclass
class ToolCallGroup:
    text: str
    calls: list[ToolCall] = field(default_factory=list)


@dataclass
class HistoryMessage:
    role: str
    content: str
    tools_used: list[str] = field(default_factory=list)
    tool_chain: list[ToolCallGroup] = field(default_factory=list)


def to_tool_call_groups(raw_chain: list[dict[str, Any]]) -> list[ToolCallGroup]:
    groups: list[ToolCallGroup] = []
    for group_index, group in enumerate(raw_chain):
        text = str(group.get("text", "") or "")
        calls: list[ToolCall] = []
        for call_index, call in enumerate(group.get("calls") or []):
            args = call.get("arguments")
            if not isinstance(args, dict):
                raise TypeError(
                    "tool_chain arguments 必须是 dict: "
                    f"group={group_index} call={call_index} "
                    f"type={type(args).__name__}"
                )
            calls.append(
                ToolCall(
                    call_id=str(call.get("call_id", "") or ""),
                    name=str(call.get("name", "") or ""),
                    arguments=args,
                    result=str(call.get("result", "") or ""),
                )
            )
        groups.append(ToolCallGroup(text=text, calls=calls))
    return groups


@dataclass
class ContextBundle:
    skill_mentions: list[str] = field(default_factory=list)
    history_messages: list[Any] = field(default_factory=list)


@dataclass
class ContextRequest:
    history: list[dict[str, Any]]
    current_message: str
    multimodal: bool
    media: list[str] | None = None
    skill_names: list[str] | None = None
    channel: str | None = None
    chat_id: str | None = None
    message_timestamp: datetime | None = None
    disabled_sections: set[str] | None = None
    turn_injection_prompt: str | None = None


@dataclass
class ReasonerResult:
    reply: str
    thinking: str | None = None
    streamed: bool = False
    tools_used: list[str] = field(default_factory=list)
    tools_unlocked: list[str] = field(default_factory=list)
    tool_chain: list[dict[str, Any]] = field(default_factory=list)
    media: list[str] = field(default_factory=list)
    visible_names: set[str] | None = None
    react_stats: dict[str, Any] = field(default_factory=dict)
    model_state: dict[str, Any] | None = None
    mobile_attention: Literal["confirmation"] | None = None
