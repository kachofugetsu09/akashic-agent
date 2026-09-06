from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from dataclasses import asdict
from typing import cast

from plugins.models.projection import display_facts
from plugins.tools.api import display_name
from session.log import MessagePage, MessageReader, SessionEntry
from session.message import ContentPart, Control, Input, Message, Output, ToolCall
from session.message_codec import json_value


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


def message_rows(page: MessagePage) -> list[dict[str, object]]:
    """把固定消息页转为两端共用的展示数据，不读取或修改运行状态。"""
    return [_message_row(message, page) for message in page.messages]


async def follow_messages(reader: MessageReader, *, after_seq: int) -> AsyncGenerator[dict[str, object], None]:
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
                       "items": message_rows(page), "after_seq": after_seq,
                       "through_seq": page.through_seq, "next_after_seq": next_seq,
                       "has_more": page.has_more}
                after_seq = next_seq
                if not page.has_more:
                    break
                # 2. 当前批次固定 head；随后到达的事实由外层 follow 继续追赶。
                page = reader.read_page(after_seq=after_seq, through_seq=page.through_seq, limit=50)


def _message_row(message: Message, page: MessagePage) -> dict[str, object]:
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
        "attachments": [asdict(ref) for ref in page.attachments[message.message_id]],
    }
    if isinstance(body, Control):
        row["body"] = {"kind": "control", "action": body.action,
                       "through_seq": body.through_seq, "reason": body.reason}
        return row
    parts = [_part(part, page.bindings) for part in body.parts]
    if isinstance(body, Input):
        row["body"] = {"kind": "input", "parts": parts}
    elif isinstance(body, Output):
        row["body"] = {"kind": "output", "parts": parts, "finish": body.finish}
    else:
        row["body"] = {"kind": "tool_result", "parts": parts,
                       "call_ref": asdict(body.call_ref), "outcome": body.outcome}
    return row


def _part(part: ContentPart | ToolCall, bindings: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """只公开展示合同允许的字段，未知内容保留类型与不可展示的明确状态。"""
    # 1. 名称由原工具 binding 提供，模型事实由其 owner 限定字段。
    if isinstance(part, ToolCall):
        descriptor = bindings[part.binding_id]
        metadata = descriptor["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("工具 binding metadata 无效")
        return {"kind": "tool_call", "binding_id": part.binding_id,
                "name": display_name(cast(Mapping[str, object], metadata)),
                "arguments": json_value(part.arguments)}
    if part.kind == "model.facts":
        return {"kind": part.kind, "value": display_facts(part)}
    # 2. 旧归档原样留在当前 Message 内，不拆成新的工具调用或回复。
    if part.kind in {"history.provenance", "history.transcript", "history.record", "history.turn_input"}:
        return {"kind": part.kind, "archive": json_value(part.value)}
    if part.kind in {"text", "artifact_ref", "reply_ref"}:
        return {"kind": part.kind, "value": json_value(part.value)}
    return {"kind": part.kind, "display": "unavailable"}
