"""只摘要已结算的完整消息前缀；原始事实和 provider 调用账保持各自的 owner。"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from collections.abc import Sequence

from agent.plugin_composition.models import (
    BoundChatModel, ContextLengthError, ModelRequest, ModelTimeoutError,
    RateLimitError, TransportError,
)
from session.message import CallRef, ContentPart, Control, Message, Output, ToolCall, ToolResult
from plugins.context.api import settled_prefixes
from plugins.turn_projection.plugin import TurnProjection
from session.message_codec import encode_body

logger = logging.getLogger(__name__)

HEADINGS = (
    "## Goal", "## Constraints & Preferences", "## Progress", "### Done",
    "### In Progress", "### Blocked", "## Key Decisions", "## Next Steps", "## Critical Context",
)
PROMPT = """更新当前长任务的上下文压缩摘要。

只记录输入中已经出现的事实，不补充猜测，不把计划写成已完成。
摘要只替代已结算的旧消息；完整原文和工具结果仍保留。
必须严格使用以下标题，不得增加标题：
""" + "\n".join(HEADINGS) + """

保留路径、符号、命令、错误、数值、外部效果和验证结果。
仍在运行的执行必须保留 execution_id、命令和当前状态。
省略重复探索、无用日志和 provider 协议细节。只输出摘要正文。
"""


class SummaryError(ValueError):
    """摘要不能满足完整前缀、容量或输出合同。"""


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
        rows.append({"message_id": message.message_id, "source": message.source,
                     "seq": message.seq, "body": json.loads(encode_body(body))})
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


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


def window_starts(messages: tuple[Message, ...], projection: TurnProjection) -> tuple[int, ...]:
    """首次窗口只能从完整单元开始，当前未结束工作也作为整体保留。"""
    protected = _protected_cuts(messages, projection, keep_open=True)
    return tuple(index for index in (0, *settled_prefixes(messages))
                 if index < len(messages) and index not in protected)


def closed_groups(messages: tuple[Message, ...], projection: TurnProjection,
                  *, after: int = 0) -> tuple[tuple[Message, ...], ...]:
    """完整 Turn 不拆开；open 工作只在已结算批次后提供压缩边界。"""
    groups: list[tuple[Message, ...]] = []
    ends = set(settled_prefixes(messages)) - _protected_cuts(messages, projection, keep_open=False)
    start = after
    for index in range(after, len(messages)):
        message = messages[index]
        body = message.body
        closed = (isinstance(body, Output) and body.finish != "continue"
                  or isinstance(body, ToolResult)
                  or isinstance(body, Control) and body.action == "abandon")
        if closed and index + 1 in ends:
            groups.append(tuple(messages[start:index + 1]))
            start = index + 1
    return tuple(groups)


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


def _request(model: BoundChatModel, summary: str, groups: Sequence[tuple[Message, ...]]) -> ModelRequest:
    text = PROMPT + "\n[Previous summary]\n" + summary + "\n[Source messages]\n"
    text += source_text(tuple(message for group in groups for message in group))
    rows = ({"role": "user", "content": text},)
    window = model.descriptor.capabilities.context_window
    if window is None:
        raise SummaryError("摘要模型缺少已确认的 context_window")
    estimated = model.estimate_context_tokens(rows)
    if estimated >= int(window * 0.74):
        raise SummaryError("单个摘要请求超出模型软水位")
    # 复用 provider 默认输出长度，不以旧摘要上限截断长任务资料。
    return ModelRequest(rows, max_output_tokens=0, disable_reasoning=True)


async def _summarize(model: BoundChatModel, groups: tuple[tuple[Message, ...], ...],
                     previous: str) -> tuple[str, tuple[str, ...]]:
    """逐个有界请求更新摘要；provider 拒绝时只减小本批完整分组。"""
    remaining = groups
    summary = previous
    calls: list[str] = []
    while remaining:
        # 1. 二分选择可容纳的最大完整前缀，不拆单条消息或工具批次。
        low, high, size = 1, len(remaining), 0
        while low <= high:
            middle = (low + high) // 2
            try:
                _ = _request(model, summary, remaining[:middle])
            except SummaryError:
                high = middle - 1
            else:
                size = middle
                low = middle + 1
        if size == 0:
            raise SummaryError("一个完整消息组已超过摘要模型容量")
        while True:
            request = _request(model, summary, remaining[:size])
            try:
                response = await model.complete(request)
            except ContextLengthError:
                if size == 1:
                    raise
                size = max(1, size // 2)
                continue
            break
        # 2. 只接纳成功调用的真实正文；格式错误不会用空摘要掩盖。
        text = (response.content or "").strip()
        headings = tuple(line.strip() for line in text.splitlines() if line.lstrip().startswith("#"))
        if response.tool_calls or headings != HEADINGS:
            raise SummaryError("摘要响应没有遵守固定标题合同")
        if response.call_record_id is None:
            raise SummaryError("摘要响应缺少成功模型调用出处")
        summary = text
        calls.append(response.call_record_id)
        remaining = remaining[size:]
    return summary, tuple(calls)


async def summarize(groups: tuple[tuple[Message, ...], ...], *, previous: str,
                    model: BoundChatModel, fallback: BoundChatModel) -> tuple[str, tuple[str, ...]]:
    """主模型在本层可恢复的生成失败后，使用本次作用域已固定的 DEFAULT。"""
    try:
        return await _summarize(model, groups, previous)
    except (SummaryError, ContextLengthError, ModelTimeoutError, RateLimitError, TransportError) as failure:
        if fallback.descriptor.binding_id == model.descriptor.binding_id:
            raise
        logger.warning("摘要模型 %s 失败，改用已固定的 DEFAULT %s: %s",
                       model.descriptor.model_id, fallback.descriptor.model_id, failure)
        return await _summarize(fallback, groups, previous)
