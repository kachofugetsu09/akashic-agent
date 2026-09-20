"""公开消息词汇表：Message、内容块、调用引用与终态值域。

这是插件可以依赖的结构合同模块。它只定义不可变值类型和 JSON 边界校验，
不导入任何 core 内部模块、服务实现或存储层；词汇表本身不拥有持久化、
控制流或生命周期。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Literal, cast


MAX_METADATA_BYTES = 64 * 1024


def freeze_metadata(value: Mapping[str, object]) -> Mapping[str, object]:
    """附加信息只接受有界 JSON 对象；插件内部结构不参与消息类型校验。"""
    if not isinstance(value, Mapping):
        raise TypeError("消息 metadata 必须是 JSON 对象")
    frozen = cast(Mapping[str, object], freeze_json(value))
    if any(not key for key in frozen):
        raise ValueError("消息 metadata 命名空间不能为空")
    # JSON 编码同时固定字节预算；不按 Python 对象大小或字符数计算。
    payload = json.dumps(frozen, default=_json_container, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(payload.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("消息 metadata 超过 64 KiB")
    return frozen


def _json_container(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, object], value))
    raise TypeError("消息 metadata 包含非 JSON 值")


def freeze_json(value: object) -> object:
    """在消息边界复制 JSON 值，阻止调用者随后改变已接纳内容。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("消息 JSON 不接受非有限浮点数")
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if any(not isinstance(key, str) for key in mapping):
            raise TypeError("消息 JSON 对象的 key 必须是字符串")
        return MappingProxyType(
            {cast(str, key): freeze_json(item) for key, item in mapping.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item) for item in cast(list[object] | tuple[object, ...], value)
        )
    raise TypeError(f"消息内容必须是 JSON 值，实际为 {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ContentPart:
    """内容类型及其不可变载荷；具体 schema 由声明该类型的能力校验。"""

    kind: str
    value: object

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind or self.kind == "tool_call":
            raise ValueError("内容类型不能为空")
        object.__setattr__(self, "value", freeze_json(self.value))


@dataclass(frozen=True, slots=True)
class ContentReferences:
    """内容检查声明的耐久链接；附件保留正文出现次序和重复项。"""

    binding_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for values in (self.binding_ids, self.artifact_ids):
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise TypeError("内容引用必须是非空 ID 的 tuple")


@dataclass(frozen=True, slots=True)
class ToolCall:
    binding_id: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.binding_id, str) or not self.binding_id:
            raise ValueError("工具调用必须固定 binding_id")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("工具参数必须是 JSON 对象")
        object.__setattr__(self, "arguments", freeze_json(self.arguments))


@dataclass(frozen=True, slots=True)
class CallRef:
    message_id: str
    part_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.message_id, str)
            or not self.message_id
            or type(self.part_index) is not int
            or self.part_index < 0
        ):
            raise ValueError("调用引用需要 message_id 和非负 part_index")


type Part = ContentPart | ToolCall


@dataclass(frozen=True, slots=True)
class Input:
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("Input 只能包含内容块")
        object.__setattr__(self, "parts", parts)


@dataclass(frozen=True, slots=True)
class Output:
    parts: tuple[Part, ...]
    finish: Literal["continue", "complete", "quiet"]

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if any(not isinstance(part, (ContentPart, ToolCall)) for part in parts):
            raise TypeError("Output 只能包含内容块或工具调用")
        object.__setattr__(self, "parts", parts)
        if self.finish not in {"continue", "complete", "quiet"}:
            raise ValueError("Output finish 无效")
        if self.finish != "continue" and any(
            isinstance(part, ToolCall) for part in self.parts
        ):
            raise ValueError("含工具调用的 Output 必须是 continue")


@dataclass(frozen=True, slots=True)
class ToolResult:
    # 仅持久化解码器记录旧表示；运行时 outcome 仍遵守当前值域。
    _legacy_unknown: bool = field(default=False, init=False, repr=False, compare=False)
    call_ref: CallRef
    outcome: Literal["success", "denied", "error", "interrupted"]
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.call_ref, CallRef):
            raise TypeError("工具结果需要有效的调用引用")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("ToolResult 只能包含内容块")
        object.__setattr__(self, "parts", parts)
        if self.outcome not in {"success", "denied", "error", "interrupted"}:
            raise ValueError("ToolResult outcome 无效")


@dataclass(frozen=True, slots=True)
class Control:
    action: Literal["pause", "resume", "abandon", "failure"]
    through_seq: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.action not in {"pause", "resume", "abandon", "failure"}:
            raise ValueError("Control action 无效")
        if type(self.through_seq) is not int or self.through_seq < 0:
            raise ValueError("Control through_seq 必须非负")
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("Control reason 必须是文本")


type Body = Input | Output | ToolResult | Control


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Message:
    """一条已接纳事实；作者、来源与消息用途分别表达独立信息。"""

    message_id: str
    session_id: str
    seq: int
    recorded_at: datetime
    author: str
    source: str
    body: Body
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"Session {self.session_id} Message {self.message_id} metadata 损坏: {error}"
            ) from error
        if not all(
            isinstance(value, str) and value
            for value in (self.message_id, self.session_id, self.author, self.source)
        ):
            raise ValueError("消息身份、Session、作者和来源不能为空")
        if type(self.seq) is not int or self.seq < 0:
            raise ValueError("消息 seq 必须非负")
        if (
            not isinstance(self.recorded_at, datetime)
            or self.recorded_at.utcoffset() is None
        ):
            raise ValueError("消息接纳时间必须包含时区")
        if not isinstance(self.body, (Input, Output, ToolResult, Control)):
            raise TypeError("消息 body 类型无效")
        if isinstance(self.body, Control) and self.body.through_seq >= self.seq:
            raise ValueError("Control 只能指向之前的已接纳前缀")


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


def body_to_dict(body: Body) -> dict[str, object]:
    """返回当前运行时消息字段，供展示和上下文使用。"""
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
    return data


def encode_body(body: Body, *, allow_legacy: bool = True) -> str:
    """保留已读消息的历史编码；新写入可显式拒绝旧表示。"""
    data = body_to_dict(body)
    if isinstance(body, ToolResult) and body._legacy_unknown:
        if not allow_legacy:
            raise ValueError("旧 unknown 工具结果只能重放已有消息，不能作为新消息写入")
        data["outcome"] = "unknown"
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
    result = ToolResult(
        CallRef(cast(str, call["message_id"]), cast(int, call["part_index"])),
        cast(Literal["success", "denied", "error", "interrupted"],
             "error" if row["outcome"] == "unknown" else row["outcome"]),
        cast(tuple[ContentPart, ...], parts),
    )

    if row["outcome"] == "unknown":
        object.__setattr__(result, "_legacy_unknown", True)
    return result
