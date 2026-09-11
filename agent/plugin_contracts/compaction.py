"""压缩摘要的公开结构合同（纯函数）。

`source_text`/`window_starts`/`summary_groups` 是只读的摘要候选计算：它们只读取
已提交消息与投影，不发布摘要、不推进 head、不调用模型。Core 与插件共用，因此由
合同层拥有，实现留在 `plugins/compaction/message_summary.py`。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace

from agent.plugin_contracts.context import settled_prefixes
from agent.plugin_contracts.message import (
    CallRef,
    ContentPart,
    Control,
    Message,
    Output,
    ToolCall,
    ToolResult,
)
from agent.plugin_contracts.message_codec import body_to_dict
from agent.plugin_contracts.turn_projection import TurnProjectionPort as TurnProjection


def _protected_cuts(messages: tuple[Message, ...], projection: TurnProjection,
                    *, keep_open: bool) -> set[int]:
    """合并各来源的完整 Turn 区间，交错成员之间不能切开。"""
    positions = {message.message_id: index for index, message in enumerate(messages)}
    protected: set[int] = set()
    for source in sorted({message.source for message in messages}):
        for turn in projection.project(messages, source):
            if turn.status == "open" and not keep_open:
                continue
            ids = (*turn.message_ids, *(identity for _, identity in turn.observations))
            if turn.ending_message_id is not None:
                ids += (turn.ending_message_id,)
            if ids:
                indexes = [positions[identity] for identity in ids]
                protected.update(range(min(indexes) + 1, max(indexes) + 1))
    return protected

def source_text(messages: Sequence[Message]) -> str:
    """把精确原文作为低信任资料；不把模型私有 replay 当成待学习正文。"""
    rows: list[dict[str, object]] = []
    for message in messages:
        body = message.body
        if not isinstance(body, Control):
            body = replace(body, parts=tuple(
                part for part in body.parts
                if not isinstance(part, ContentPart) or part.kind not in {
                    "model.facts", "context.summary", "model.selection", "tool.selection",
                }
            ))
        rows.append({"message_id": message.message_id, "source": message.source, "author": message.author,
                     "seq": message.seq, "body": body_to_dict(body)})
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))

def window_starts(messages: tuple[Message, ...], projection: TurnProjection) -> tuple[int, ...]:
    """首次窗口只能从完整单元开始，当前未结束工作也作为整体保留。"""
    protected = _protected_cuts(messages, projection, keep_open=True)
    return tuple(index for index in (0, *settled_prefixes(messages))
                 if index < len(messages) and index not in protected)

def summary_groups(groups: tuple[tuple[Message, ...], ...],
                   snapshot: tuple[Message, ...]) -> tuple[tuple[Message, ...], ...]:
    """摘要不重新引入已放弃调用的迟到正文；调用归属从完整日志确定。"""
    # 1. abandon 可能早于上一份摘要，不能只检查本次增量。
    pending: dict[CallRef, Message] = {}
    abandoned: set[CallRef] = set()
    excluded: set[str] = set()
    for message in snapshot:
        body = message.body
        if isinstance(body, Output):
            pending.update((CallRef(message.message_id, index), message)
                           for index, part in enumerate(body.parts) if isinstance(part, ToolCall))
        elif isinstance(body, ToolResult):
            if body.call_ref in abandoned:
                excluded.add(message.message_id)
            _ = pending.pop(body.call_ref, None)
        elif isinstance(body, Control) and body.action == "abandon":
            abandoned.update(ref for ref, call in pending.items()
                             if call.source == message.source and call.seq <= body.through_seq)
            pending = {ref: call for ref, call in pending.items() if ref not in abandoned}
    # 2. 只改变本次摘要材料；原分组仍用于耐久覆盖范围和 raw tail 选择。
    selected = tuple(tuple(message for message in group if message.message_id not in excluded)
                     for group in groups)
    return tuple(group for group in selected if group)

