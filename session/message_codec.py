from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal, cast

from session.message import (
    Body,
    CallRef,
    ContentPart,
    Control,
    Input,
    Output,
    Part,
    ToolCall,
    ToolResult,
)


def json_value(value: object) -> object:
    """把已校验的不可变 JSON 转回可序列化容器。"""
    if isinstance(value, Mapping):
        return {
            key: json_value(item)
            for key, item in cast(Mapping[str, object], value).items()
        }
    if isinstance(value, tuple):
        return [json_value(item) for item in cast(tuple[object, ...], value)]
    return value


def encode_body(body: Body) -> str:
    """只编码一份消息事实，不创建 provider 或 UI 的平行持久模型。"""
    data: dict[str, object]
    if isinstance(body, Control):
        data = {
            "kind": "control",
            "action": body.action,
            "through_seq": body.through_seq,
            "reason": body.reason,
        }
    else:
        parts: list[dict[str, object]] = [
            (
                {
                    "kind": "tool_call",
                    "binding_id": part.binding_id,
                    "arguments": json_value(part.arguments),
                }
                if isinstance(part, ToolCall)
                else {"kind": part.kind, "value": json_value(part.value)}
            )
            for part in body.parts
        ]
        if isinstance(body, Input):
            data = {"kind": "input", "parts": parts}
        elif isinstance(body, Output):
            data = {
                "kind": "output",
                "parts": parts,
                "finish": body.finish,
            }
        else:
            data = {
                "kind": "tool_result",
                "parts": parts,
                "outcome": body.outcome,
                "call_ref": {
                    "message_id": body.call_ref.message_id,
                    "part_index": body.call_ref.part_index,
                },
            }
    return json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _object(value: object, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != fields:
        raise ValueError(f"消息对象字段必须为 {sorted(fields)}")
    return cast(dict[str, object], value)


def _part(value: object) -> Part:
    if not isinstance(value, dict):
        raise ValueError("消息 part 必须是对象")
    raw = cast(dict[str, object], value)
    if raw.get("kind") == "tool_call":
        data = _object(raw, {"kind", "binding_id", "arguments"})
        return ToolCall(
            cast(str, data["binding_id"]), cast(Mapping[str, object], data["arguments"])
        )
    data = _object(raw, {"kind", "value"})
    return ContentPart(cast(str, data["kind"]), data["value"])


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"消息 JSON 包含重复字段: {key}")
        result[key] = value
    return result


def decode_body(payload: str) -> Body:
    """在持久化边界拒绝未知结构；值域与不变量归消息构造函数。"""
    raw: object = json.loads(payload, object_pairs_hook=_unique_fields)
    if not isinstance(raw, dict):
        raise ValueError("消息 body 必须是对象")
    data = cast(dict[str, object], raw)
    kind = data.get("kind")
    if kind == "control":
        row = _object(data, {"kind", "action", "through_seq", "reason"})
        return Control(
            cast(Literal["pause", "resume", "abandon", "failure"], row["action"]),
            cast(int, row["through_seq"]),
            cast(str | None, row["reason"]),
        )
    fields = {
        "input": {"kind", "parts"},
        "output": {"kind", "parts", "finish"},
        "tool_result": {"kind", "parts", "call_ref", "outcome"},
    }
    if not isinstance(kind, str) or kind not in fields:
        raise ValueError("消息 body kind 无效")
    row = _object(data, fields[kind])
    raw_parts = row["parts"]
    if not isinstance(raw_parts, list):
        raise ValueError("消息 parts 必须是数组")
    parts = tuple(_part(part) for part in cast(list[object], raw_parts))
    if kind == "input":
        return Input(cast(tuple[ContentPart, ...], parts))
    if kind == "output":
        return Output(
            parts,
            cast(Literal["continue", "complete", "quiet"], row["finish"]),
        )
    call = _object(row["call_ref"], {"message_id", "part_index"})
    return ToolResult(
        CallRef(cast(str, call["message_id"]), cast(int, call["part_index"])),
        cast(Literal["success", "denied", "error", "unknown"], row["outcome"]),
        cast(tuple[ContentPart, ...], parts),
    )
