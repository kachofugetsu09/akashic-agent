from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Protocol, cast

from agent.plugin_composition.models import BoundChatModel, ModelRequest
from agent.plugin_contracts import CallRef, ContentPart, ContentReferences, Control, Message, Output, ToolCall, ToolResult


MaterialData = Mapping[str, object]
SummaryData = Mapping[str, object]


def settled_prefixes(messages: tuple[Message, ...]) -> tuple[int, ...]:
    """返回工具已结算或被明确放弃的前缀长度；不伪造任何工具结果。"""
    pending: dict[CallRef, Message] = {}
    ends: list[int] = []
    for index, message in enumerate(messages):
        body = message.body
        if isinstance(body, Output):
            pending.update((CallRef(message.message_id, pos), message) for pos, part in enumerate(body.parts)
                           if isinstance(part, ToolCall))
        elif isinstance(body, ToolResult):
            _ = pending.pop(body.call_ref, None)
        elif isinstance(body, Control) and body.action == "abandon":
            pending = {ref: call for ref, call in pending.items()
                       if call.source != message.source or call.seq > body.through_seq}
        if not pending:
            ends.append(index + 1)
    return tuple(ends)


def check_summary(part: ContentPart) -> ContentReferences:
    """摘要使用记录只引用已固定的来源；内容 writer 不接受内联摘要。"""
    value = part.value
    if not isinstance(value, Mapping):
        raise ValueError("context.summary 必须是对象")
    value = cast(Mapping[str, object], value)
    if (set(value) != {"reference"} or not isinstance(value["reference"], str) or not value["reference"]):
        raise ValueError("context.summary 必须包含唯一的摘要 binding 引用")
    return ContentReferences(binding_ids=(value["reference"],))


def summary_range(snapshot: tuple[Message, ...], source_message_ids: tuple[str, ...]) -> range:
    """按真实身份定位摘要的连续区间；窗口外旧消息不冒充摘要来源。"""
    identities = tuple(message.message_id for message in snapshot)
    if not source_message_ids or source_message_ids[0] not in identities:
        raise ValueError("摘要来源缺少实际消息")
    start = identities.index(source_message_ids[0])
    end = start + len(source_message_ids)
    if identities[start:end] != source_message_ids:
        raise ValueError("摘要来源不等于实际连续消息范围")
    return range(start, end)


@dataclass(frozen=True, slots=True)
class Summary:
    """摘要 owner 已持久发布的内容与精确覆盖范围。"""

    reference: str
    source_message_ids: tuple[str, ...]
    content: str

    def __post_init__(self) -> None:
        ids = tuple(self.source_message_ids)
        if not self.reference or not isinstance(self.reference, str):
            raise ValueError("摘要必须有持久来源引用")
        if not ids or any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("摘要必须声明覆盖的消息")
        if len(set(ids)) != len(ids):
            raise ValueError("摘要消息引用不能重复")
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("摘要正文不能为空")
        object.__setattr__(self, "source_message_ids", ids)


@dataclass(frozen=True, slots=True)
class Reminder:
    """一次请求中的具名文本；priority 只决定显示顺序，不授予权限。"""

    name: str
    text: str
    priority: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("提醒必须有稳定的局部名称")
        if not isinstance(self.text, str):
            raise TypeError("提醒正文必须是字符串")
        if type(self.priority) is not int:
            raise TypeError("提醒 priority 必须是整数")


@dataclass(frozen=True, slots=True)
class Materials:
    """权限已由组合确定的 Prompt，以及保持低信任的检索材料。"""

    system_prompt: str
    reminders: tuple[Reminder, ...] = ()
    summary: Summary | None = None
    references: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str):
            raise TypeError("system Prompt 必须是字符串")
        parts = tuple(self.reminders)
        if any(not isinstance(part, Reminder) for part in parts):
            raise TypeError("提醒必须是已校验的文本块")
        if self.summary is not None and not isinstance(self.summary, Summary):
            raise TypeError("摘要必须来自已发布的 Summary")
        references = tuple(self.references)
        normalized_references: list[Mapping[str, object]] = []
        for reference in references:
            if not isinstance(reference, Mapping) or any(
                not isinstance(key, str) for key in reference
            ):
                raise TypeError("引用必须是结构化映射")
            normalized_references.append(MappingProxyType(dict(reference)))
        object.__setattr__(self, "reminders", parts)
        object.__setattr__(self, "references", tuple(normalized_references))


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是字符串键对象")
    data = cast(Mapping[str, object], value)
    if any(not isinstance(key, str) for key in data):
        raise TypeError(f"{label} 必须是字符串键对象")
    return data


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} 必须是数组")
    return tuple(cast(Sequence[object], value))


def _reference(value: object) -> Mapping[str, object]:
    """保留引用的结构数据；引用字段的领域校验由 Content owner 完成。"""
    data = _object(value, "reference")
    return MappingProxyType(dict(data))


def _summary(value: object) -> Summary | None:
    if value is None:
        return None
    data = _object(value, "summary")
    if set(data) != {"reference", "source_message_ids", "content"}:
        raise ValueError("summary 字段无效")
    reference = data["reference"]
    content = data["content"]
    if not isinstance(reference, str) or not isinstance(content, str):
        raise TypeError("summary 的 reference/content 必须是字符串")
    source_ids = _sequence(data["source_message_ids"], "summary.source_message_ids")
    if any(not isinstance(item, str) for item in source_ids):
        raise TypeError("summary.source_message_ids 必须是字符串数组")
    return Summary(reference, cast(tuple[str, ...], source_ids), content)


def _reminder(value: object) -> Reminder:
    data = _object(value, "reminder")
    if set(data) != {"name", "text", "priority"}:
        raise ValueError("reminder 字段无效")
    name, text, priority = data["name"], data["text"], data["priority"]
    if not isinstance(name, str) or not isinstance(text, str) or type(priority) is not int:
        raise TypeError("reminder 的 name/text/priority 类型无效")
    return Reminder(name, text, priority)


def decode_material(value: object) -> Materials:
    """在 Context 边界把 provider 的普通映射转换为内部材料值。"""
    data = _object(value, "materials")
    unknown = set(data) - {"system_prompt", "reminders", "summary", "references"}
    if unknown:
        raise ValueError(f"materials 包含未知字段: {sorted(unknown)}")
    prompt = data.get("system_prompt", "")
    if not isinstance(prompt, str):
        raise TypeError("materials.system_prompt 必须是字符串")
    reminders = tuple(_reminder(item) for item in _sequence(data.get("reminders", ()), "materials.reminders"))
    summary = _summary(data.get("summary"))
    references = tuple(_reference(item) for item in _sequence(data.get("references", ()), "materials.references"))
    return Materials(prompt, reminders, summary, references)


def material_data(materials: Materials) -> MaterialData:
    """把内部材料投影成 reducer 能消费的普通映射。"""
    summary: Mapping[str, object] | None = None if materials.summary is None else {
        "reference": materials.summary.reference,
        "source_message_ids": materials.summary.source_message_ids,
        "content": materials.summary.content,
    }
    return {
        "system_prompt": materials.system_prompt,
        "reminders": tuple({"name": item.name, "text": item.text, "priority": item.priority} for item in materials.reminders),
        "summary": summary,
        "references": tuple(dict(item) for item in materials.references),
    }


def decode_summary(value: object) -> Summary | None:
    """在摘要 owner 返回边界创建并校验内部 Summary。"""
    return _summary(value)


class ContextModel(Protocol):
    """Model 的只读请求投影；这里没有 complete 或工具执行权。"""

    @property
    def context_window(self) -> int | None: ...

    @property
    def max_tool_schemas(self) -> int | None: ...

    def render(self, messages: tuple[Message, ...], *, after_seq: int,
               summary_reference: str | None = None, fresh: bool = False) -> ModelRequest:
        """接收完整事实；after_seq 是摘要覆盖末尾，-1 表示没有覆盖。

        fresh 明确从选定近期窗口开始新请求，不接续旧 opaque 状态。
        summary_reference 明确要求从这份摘要开始新请求；只有同一摘要下的
        后续成功响应才接续 opaque state。只给 after_seq 不授权丢弃 replay。
        """
        ...

    def estimate(self, request: ModelRequest) -> int: ...


class SummaryReducer(Protocol):
    """摘要 owner 先持久发布再返回；None 表示保留已有摘要，不做缩减。"""

    async def __call__(
        self, snapshot: tuple[Message, ...], materials: MaterialData,
        request: ModelRequest, model: BoundChatModel, projection: ContextModel,
        *, source: str, force: bool,
    ) -> SummaryData | None: ...


class ContextOverflow(ValueError):
    def __init__(self, estimated_tokens: int, output_tokens: int, capacity: int,
                 *, request: ModelRequest):
        self.estimated_tokens = estimated_tokens
        self.output_tokens = output_tokens
        self.capacity = capacity
        self.request = request
        super().__init__(
            f"请求需要约 {estimated_tokens}+{output_tokens} tokens，容量 {capacity}"
        )
