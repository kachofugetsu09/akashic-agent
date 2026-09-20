from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import aclosing
from dataclasses import asdict, dataclass, field
from typing import Protocol, cast
from types import MappingProxyType

from session.log import MessagePage, MessageReader, SessionEntry
from session.message import ContentPart, Control, Input, Message, Output, ToolCall
from session.message_codec import json_value


PartDisplayProvider = Callable[[ContentPart], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class MessageDisplayProviders:
    """一次页面投影使用的只读回调；调用者负责同代 lease。"""

    tool_name: Callable[[str], str] | None = None
    part_display: Mapping[str, PartDisplayProvider] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "part_display", MappingProxyType(dict(self.part_display)))


class MessageDisplayReader(Protocol):
    """在自己的资源作用域内投影一页，不让客户端持有插件回调。"""

    async def __call__(self, page: MessagePage, *, display_only: bool) -> list[dict[str, object]]: ...


async def read_message_rows(
    page: MessagePage, *, display_only: bool = False,
    reader: MessageDisplayReader | None = None,
) -> list[dict[str, object]]:
    """展示缺少业务 provider 时保留明确的 unavailable 状态。"""
    if reader is None:
        return message_rows(page, display_only=display_only)
    return await reader(page, display_only=display_only)


def session_row(entry: SessionEntry) -> dict[str, object]:
    """列表只读首条消息；无文字的会话也保留入口。"""
    first = entry.first_message
    text = "" if first is None or isinstance(first.body, Control) else "\n".join(
        cast(str, part.value) for part in first.body.parts
        if isinstance(part, ContentPart) and part.kind == "text"
    )
    return {
        "key": entry.session_id,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "message_count": entry.message_count,
        "head_seq": entry.head_seq,
        "first_message_content": text,
    }


def message_rows(
    page: MessagePage,
    *,
    display_only: bool = False,
    providers: MessageDisplayProviders | None = None,
) -> list[dict[str, object]]:
    """把固定消息页转为两端共用的展示数据，不读取或修改运行状态。"""
    display = MessageDisplayProviders() if providers is None else providers
    return [
        _message_row(message, page, display_only=display_only, providers=display)
        for message in page.messages
    ]


async def follow_messages(
    reader: MessageReader,
    *,
    after_seq: int,
    display_only: bool = False,
    reader_display: MessageDisplayReader | None = None,
) -> AsyncGenerator[dict[str, object], None]:
    """从 seq 续读完整展示页；唤醒通知不携带第二份消息正文。"""
    # 1. 先注册日志通知，再按页补齐附件和 binding 展示字段。
    async with aclosing(reader.follow(after_seq=after_seq)) as follower:
        async for message in follower:
            if message.seq <= after_seq:
                continue
            page = reader.read_page(after_seq=after_seq, limit=50)
            while page.messages:
                next_seq = page.messages[-1].seq
                yield {"version": 2, "session_id": reader.session_id,
                       "items": await read_message_rows(page, display_only=display_only, reader=reader_display), "after_seq": after_seq,
                       "through_seq": page.through_seq, "next_after_seq": next_seq,
                       "has_more": page.has_more}
                after_seq = next_seq
                if not page.has_more:
                    break
                # 2. 当前批次固定 head；随后到达的事实由外层 follow 继续追赶。
                page = reader.read_page(after_seq=after_seq, through_seq=page.through_seq, limit=50)


def _message_row(
    message: Message,
    page: MessagePage,
    *,
    display_only: bool,
    providers: MessageDisplayProviders,
) -> dict[str, object]:
    """保留真实类型、顺序和引用，页面不推断执行结果或重新分配作者。"""
    # 1. 身份和消息用途分别呈现；Control 与晚到结果仍是独立行。
    body = message.body
    row: dict[str, object] = {
        "id": message.message_id,
        "session_id": message.session_id,
        "seq": message.seq,
        "timestamp": message.recorded_at.isoformat(),
        "author": message.author,
        "source": message.source,
        "metadata": json_value(message.metadata),
        "attachments": [asdict(ref) for ref in page.attachments[message.message_id]],
    }
    if isinstance(body, Control):
        row["body"] = {"kind": "control", "action": body.action,
                       "through_seq": body.through_seq, "reason": body.reason}
        return row
    parts = [
        _part(part, display_only=display_only, providers=providers)
        if isinstance(part, ContentPart)
        else _tool_call(part, providers=providers)
        for part in body.parts
    ]
    if isinstance(body, Input):
        row["body"] = {"kind": "input", "parts": parts}
    elif isinstance(body, Output):
        row["body"] = {"kind": "output", "parts": parts, "finish": body.finish}
    else:
        row["body"] = {"kind": "tool_result", "parts": parts,
                       "call_ref": asdict(body.call_ref), "outcome": body.outcome}
    return row


def _tool_call(part: ToolCall, *, providers: MessageDisplayProviders) -> dict[str, object]:
    """只通过工具 owner 的名称回调展示 binding，不解析或重开工具。"""
    name_reader = providers.tool_name
    if name_reader is None:
        return {
            "kind": "tool_call",
            "binding_id": part.binding_id,
            "display": "unavailable",
        }
    return {
        "kind": "tool_call",
        "binding_id": part.binding_id,
        "name": name_reader(part.binding_id),
        "arguments": json_value(part.arguments),
    }


def _part(
    part: ContentPart,
    *,
    display_only: bool,
    providers: MessageDisplayProviders,
) -> dict[str, object]:
    """只公开展示合同允许的字段，未知内容保留类型与不可展示的明确状态。"""
    # 1. 业务字段由插件 callback 选择；Core 不识别模型或工具的字段名。
    provider = providers.part_display.get(part.kind)
    if provider is not None:
        return {"kind": part.kind, "value": json_value(provider(part))}
    # 2. 展示端保留原 part 下标；不可展示的归档只传类型，权威正文不变。
    if display_only and part.kind in {"history.provenance", "history.record", "history.turn_input"}:
        return {"kind": part.kind, "display": "unavailable"}
    # 旧客户端和旧下载摘要仍使用原表示；可见 transcript 始终完整。
    if part.kind in {"history.provenance", "history.transcript", "history.record", "history.turn_input"}:
        return {"kind": part.kind, "archive": json_value(part.value)}
    if part.kind in {"text", "artifact_ref", "reply_ref"}:
        return {"kind": part.kind, "value": json_value(part.value)}
    return {"kind": part.kind, "display": "unavailable"}
