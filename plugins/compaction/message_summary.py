"""只摘要已结算的完整消息前缀；原始事实和 provider 调用账保持各自的 owner。"""
from __future__ import annotations

from plugins.models.contract import ModelError
import json
import logging
from dataclasses import replace
from collections.abc import Awaitable, Callable, Sequence

from plugins.models.contract import ModelRequest
from plugins.models.contract import (
    ContextLengthError,
    ModelTimeoutError,
    RateLimitError,
    TransportError,
)
from plugins.models.contract import BoundChatModel
from plugins.ledger.contract import CallRef, ContentPart, Control, Message, Output, ToolCall, ToolResult
from plugins.ledger.contract import body_to_dict
from plugins.turn_projection.contract import TurnProjection

logger = logging.getLogger(__name__)

HEADINGS = (
    "## Goal", "## Constraints & Preferences", "## Progress", "### Done",
    "### In Progress", "### Blocked", "## Key Decisions", "## Next Steps", "## Critical Context",
)
PROMPT = """更新当前长任务的上下文压缩摘要。

只记录输入中已经出现的事实，不补充猜测，不把计划写成已完成。
摘要只替代已结算的旧消息；完整原文和工具结果仍保留。
用户消息是理解目标、要求、偏好和关系的核心证据。只有 author=user 的原话能够建立这些用户事实。
助手消息只记录助手的判断、计划和执行过程；不得用助手转述或猜测补全用户没有表达的上下文。
区分用户原话、助手判断及工具/后台结果，保留重要来源身份；转述不能升级成用户事实或偏好。
必须严格使用以下标题，不得增加标题：
""" + "\n".join(HEADINGS) + """

保留路径、符号、命令、错误、数值、外部效果和验证结果。
仍在运行的执行必须保留 execution_id、命令和当前状态。
省略重复探索、无用日志和 provider 协议细节。只输出摘要正文。
"""


class SummaryError(ValueError):
    """摘要不能满足完整前缀、容量或输出合同。"""


# 同 Pi 截断摘要资料中的工具结果，这里按整条消息限长，覆盖参数、多段结果与长回复；
# 放不下时逐级缩短。原文仍留在日志可回读，任何一条已结算消息都不会让摘要请求放不下。
EXCERPT_CHARS = (4000, 1000, 200)


def _excerpt(body: dict[str, object], chars: int) -> object:
    text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= chars:
        return body
    half = chars // 2
    return {"excerpt": f"{text[:half]}\n[…省略 {len(text) - 2 * half} 字符，原文仍在日志…]\n{text[-half:]}"}


def clip_summary(text: str, ratio: float) -> str:
    """按固定标题重建摘要，每节正文（含标题前的正文）只留前 ratio 比例；0 时只剩标题。"""
    bodies: list[list[str]] = [[] for _ in HEADINGS]
    index = 0
    for line in text.splitlines():
        if line.strip() in HEADINGS:
            index = HEADINGS.index(line.strip())
        else:
            bodies[index].append(line)
    sections: list[str] = []
    for heading, body in zip(HEADINGS, bodies):
        content = "\n".join(body).strip()
        keep = content[:int(len(content) * ratio)].rstrip()
        sections.append(heading + ("\n" + keep if keep else "") + ("…（篇幅所限已截断）" if keep != content else ""))
    return "\n".join(sections)


def recoverable(error: BaseException) -> bool:
    """摘要层可以降级处理的失败：格式/容量不合，或 provider 的可恢复错误。"""
    return isinstance(error, SummaryError) or ModelError.matches(
        error, ContextLengthError, ModelTimeoutError, RateLimitError, TransportError)


def source_text(messages: Sequence[Message], *, excerpt: int | None = None) -> str:
    """把精确原文作为低信任资料；不把模型私有 replay 当成待学习正文。"""
    rows: list[dict[str, object]] = []
    for message in messages:
        body = message.body
        if not isinstance(body, Control):
            body = replace(body, parts=tuple(
                part for part in body.parts
                if not isinstance(part, ContentPart) or part.kind not in {
                    "model.facts", "context.summary", "context.notice", "model.selection", "tool.selection",
                }
            ))
        data = body_to_dict(body)
        rows.append({"message_id": message.message_id, "source": message.source, "author": message.author,
                     "seq": message.seq, "body": data if excerpt is None else _excerpt(data, excerpt)})
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


def window_starts(
    messages: tuple[Message, ...], projection: TurnProjection, *,
    settled_prefixes: Callable[[tuple[Message, ...]], tuple[int, ...]],
) -> tuple[int, ...]:
    """首次窗口只能从完整单元开始，当前未结束工作也作为整体保留。"""
    protected = _protected_cuts(messages, projection, keep_open=True)
    return tuple(index for index in (0, *settled_prefixes(messages))
                 if index < len(messages) and index not in protected)


def closed_groups(
    messages: tuple[Message, ...], projection: TurnProjection, *,
    settled_prefixes: Callable[[tuple[Message, ...]], tuple[int, ...]],
    after: int = 0,
) -> tuple[tuple[Message, ...], ...]:
    """只在已结算批次后切分请求视图；持久 Turn 和学习边界保持完整。"""
    groups: list[tuple[Message, ...]] = []
    ends = set(settled_prefixes(messages))
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


def _request(model: BoundChatModel, summary: str, groups: Sequence[tuple[Message, ...]],
             limit: int, excerpt: int) -> ModelRequest:
    text = PROMPT + f"摘要正文不超过约 {limit:,} tokens。\n"
    text += "\n[Previous summary]\n" + summary + "\n[Source messages]\n"
    text += source_text(tuple(message for group in groups for message in group), excerpt=excerpt)
    rows = ({"role": "user", "content": text},)
    window = model.descriptor.capabilities.context_window
    if window is None:
        raise SummaryError("摘要模型缺少已确认的 context_window")
    estimated = model.estimate_context_tokens(rows)
    if estimated > int(window * 0.74):
        raise SummaryError("单个摘要请求超出模型软水位")
    # 复用 provider 默认输出长度，不以旧摘要上限截断长任务资料。
    return ModelRequest(rows, max_output_tokens=0, disable_reasoning=True)


def _fits(model: BoundChatModel, summary: str, groups: Sequence[tuple[Message, ...]],
          limit: int, excerpt: int) -> bool:
    try:
        _ = _request(model, summary, groups, limit, excerpt)
    except SummaryError:
        return False
    return True


# 发请求前把输入缩进摘要模型：优先保留已有摘要，其次缩短消息摘录，再取近期后缀。
def _fit_inputs(model: BoundChatModel, groups: tuple[tuple[Message, ...], ...], previous: str,
                limit: int) -> tuple[str, tuple[tuple[Message, ...], ...], int]:
    for ratio in (1.0, 0.5, 0.25, 0.0):
        summary = previous if ratio == 1.0 or not previous else clip_summary(previous, ratio)
        for excerpt in EXCERPT_CHARS:
            if not groups:
                if _fits(model, summary, (), limit, excerpt):
                    return summary, (), excerpt
                continue
            # 1. 二分选择最大近期后缀；更老的积压不再触发串行摘要。
            low, high, size = 1, len(groups), 0
            while low <= high:
                middle = (low + high) // 2
                if _fits(model, summary, groups[-middle:], limit, excerpt):
                    size, low = middle, middle + 1
                else:
                    high = middle - 1
            if size:
                return summary, groups[-size:], excerpt
            # 2. 最近一组仍放不下时只取它能放下的消息后缀，其余记为 omitted。
            last = groups[-1]
            count = next((count for count in range(len(last) - 1, 0, -1)
                          if _fits(model, summary, (last[-count:],), limit, excerpt)), 0)
            if count:
                return summary, (last[-count:],), excerpt
    raise SummaryError("摘要提示加一条最短消息摘录已超过摘要模型容量")


async def _summarize(model: BoundChatModel, groups: tuple[tuple[Message, ...], ...],
                     previous: str, limit: int) -> tuple[str, tuple[str, ...], tuple[tuple[Message, ...], ...]]:
    """输入先缩进摘要模型，再只发出一次摘要请求。"""
    summary, selected, excerpt = _fit_inputs(model, groups, previous, limit)
    response = await model.complete(_request(model, summary, selected, limit, excerpt))

    # 只接纳成功调用的真实正文；格式错误不会用空摘要掩盖。
    text = (response.content or "").strip()
    headings = tuple(line.strip() for line in text.splitlines() if line.lstrip().startswith("#"))
    if response.tool_calls or headings != HEADINGS:
        raise SummaryError("摘要响应没有遵守固定标题合同")
    if response.call_record_id is None:
        raise SummaryError("摘要响应缺少成功模型调用出处")
    return text, (response.call_record_id,), selected


async def summarize(groups: tuple[tuple[Message, ...], ...], *, previous: str, limit: int,
                    model: BoundChatModel, fallback: BoundChatModel,
                    on_fallback: Callable[[str], Awaitable[None]] | None = None,
                    ) -> tuple[str, tuple[str, ...], tuple[tuple[Message, ...], ...]]:
    """主模型在本层可恢复的生成失败后，使用本次作用域已固定的 DEFAULT。"""
    try:
        return await _summarize(model, groups, previous, limit)
    except (RuntimeError, TimeoutError, SummaryError) as failure:
        if not recoverable(failure):
            raise
        if fallback.descriptor.binding_id == model.descriptor.binding_id:
            raise
        logger.warning("摘要模型 %s 失败，改用已固定的 DEFAULT %s: %s",
                       model.descriptor.model_id, fallback.descriptor.model_id, failure)
        if on_fallback is not None:
            await on_fallback(f"摘要模型 {model.descriptor.model_id} 失败：{failure}；"
                              f"改用 {fallback.descriptor.model_id}")
        return await _summarize(fallback, groups, previous, limit)
