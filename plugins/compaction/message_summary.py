"""只摘要已结算的完整消息前缀；原始事实和 provider 调用账保持各自的 owner。"""
from __future__ import annotations

from plugins.models.contract import ModelError
import json
import logging
from dataclasses import dataclass, replace
from bisect import bisect_right
from collections.abc import Awaitable, Callable, Sequence
from uuid import uuid4

from plugins.models.contract import ModelRequest, StreamCallback
from plugins.models.contract import (
    ContextLengthError,
    ModelTimeoutError,
    RateLimitError,
    REJECTED_CALL_NOTICE,
    REJECTED_INPUT_NOTICE,
    REJECTED_KEEP_KINDS,
    REJECTED_OUTPUT_NOTICE,
    TransportError,
    rejected_result_notice,
    rejected_segments,
)
from plugins.models.contract import BoundChatModel, LLMResponse
from plugins.ledger.contract import CallRef, ContentPart, Control, Input, Message, Output, Part, ToolCall, ToolResult, origin
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
对照 [Previous summary] 与 [Source messages]：已在本次消息中完成、解决或改道的事项，必须从 ### In Progress、### Blocked、## Next Steps 移入 ### Done 或移除，不得把已结束的阻塞和旧待办留在活跃章节。
用户消息是理解目标、要求、偏好和关系的核心证据。只有 actor=user 的原话能够建立这些用户事实；actor=automation 是插件自动产生的输入，不是用户原话。
助手消息只记录助手的判断、计划和执行过程；不得用助手转述或猜测补全用户没有表达的上下文。
区分用户原话、助手判断及工具/后台结果，保留重要来源身份；转述不能升级成用户事实或偏好。
建议使用以下标题组织摘要，可按资料调整：
""" + "\n".join(HEADINGS) + """

保留路径、符号、命令、错误、数值、外部效果和验证结果。
仍在运行的执行必须保留 execution_id、命令和当前状态。
省略重复探索、无用日志和 provider 协议细节。只输出摘要正文。
"""


class SummaryError(ValueError):
    """摘要不能满足完整前缀、容量或输出合同。"""


class SummaryCapacityError(SummaryError):
    """摘要模型调用正常，但压缩已结算历史救不了当前请求；文案自带用户出口。"""


# 同 Pi 截断摘要资料中的工具结果，这里按整条消息限长，覆盖参数、多段结果与长回复；
# 放不下时逐级缩短。原文仍留在日志可回读，任何一条已结算消息都不会让摘要请求放不下。
EXCERPT_CHARS = (4000, 1000, 200)
# 本地估算放得下而 provider 仍拒绝时的裁剪次数。
MAX_SUMMARY_TRIMS = 4
# 摘要挡在回合前，上游要求的单次等待超过它就结束本次摘要，不把回合挂在退避上。
SUMMARY_MAX_RETRY_WAIT = 300.0
# 阻断回合的摘要：每个模型的裁剪和重试共用这份物理尝试额度，一次压缩还共用下面的总期限。
# 上界：主选 11 + 回退 11 + 缩写每模型 1 = 24 次物理尝试。
SUMMARY_MAX_ATTEMPTS = 11
SUMMARY_DEADLINE_SECONDS = 900.0


@dataclass(frozen=True)
class SummaryCall:
    """一次摘要在每个模型上的额度；trim 决定 provider 判超窗时能否丢弃更早的资料。"""
    max_attempts: int
    trim: bool


BLOCKING_SUMMARY = SummaryCall(SUMMARY_MAX_ATTEMPTS, True)
# 可选压缩有退路：摘要超窗就软失败保留原文，不为省空间丢历史；上界每模型 1 次。
OPTIONAL_SUMMARY = SummaryCall(1, False)


def _excerpt(body: dict[str, object], chars: int) -> object:
    text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= chars:
        return body
    half = chars // 2
    return {"excerpt": f"{text[:half]}\n[…省略 {len(text) - 2 * half} 字符，原文仍在日志…]\n{text[-half:]}"}


def clip_summary(text: str, ratio: float) -> str:
    """保留实际标题，每节正文只留前 ratio 比例；无标题正文也可缩短。"""
    blocks: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            blocks.append((line, []))
        else:
            blocks[-1][1].append(line)
    sections: list[str] = []
    for heading, body in blocks:
        content = "\n".join(body).strip()
        keep = content[:int(len(content) * ratio)].rstrip()
        sections.append("\n".join(part for part in (heading, keep) if part)
                        + ("…（篇幅所限已截断）" if keep != content else ""))
    return "\n".join(section for section in sections if section)


def recoverable(error: BaseException) -> bool:
    """摘要层可以降级处理的失败：格式/容量不合，或 provider 的可恢复错误。"""
    return isinstance(error, SummaryError) or ModelError.matches(
        error, ContextLengthError, ModelTimeoutError, RateLimitError, TransportError)


def source_text(messages: Sequence[Message], *, excerpt: int | None = None) -> str:
    """把精确原文作为低信任资料；不把模型私有 replay 当成待学习正文。actor 是 Ledger 判定的主体，模型不必再解析作者。"""
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
                     "actor": origin(message).actor.value, "seq": message.seq, "body": data if excerpt is None else _excerpt(data, excerpt)})
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
    """已结算批次或失败输入可供摘要；失败不改变持久 Turn 与学习资格。"""
    groups: list[tuple[Message, ...]] = []
    ends = set(settled_prefixes(messages))
    seqs = tuple(message.seq for message in messages)
    start = after
    for index in range(after, len(messages)):
        message = messages[index]
        body = message.body
        closed = (isinstance(body, Output) and body.finish != "continue"
                  or isinstance(body, ToolResult)
                  or isinstance(body, Control) and body.action in {"abandon", "failure"})
        if closed and index + 1 in ends:
            # A late failure only releases the prefix it actually observed.
            end = (bisect_right(seqs, body.through_seq, lo=start, hi=index + 1)
                   if isinstance(body, Control) and body.action == "failure" else index + 1)
            if end > start and end in ends:
                groups.append(tuple(messages[start:end]))
                start = end
    return tuple(groups)


# Summaries must not carry what the abandon took out of the model view; outcomes stay as facts.
def _without_rejected(message: Message, hidden: frozenset[str]) -> Message:
    body = message.body
    if isinstance(body, ToolResult) and body.call_ref.message_id in hidden:
        notice = ContentPart("text", rejected_result_notice(body.outcome))
        return replace(message, body=replace(body, parts=(notice,)))
    if message.message_id not in hidden:
        return message
    if isinstance(body, Input):
        return replace(message, body=Input((ContentPart("text", REJECTED_INPUT_NOTICE),)))
    assert isinstance(body, Output)
    parts: list[Part] = []
    for part in body.parts:
        if isinstance(part, ToolCall):
            parts.append(replace(part, arguments={}))
        elif part.kind == "model.tool_rejection":
            parts.append(ContentPart("text", REJECTED_CALL_NOTICE))
        elif part.kind in REJECTED_KEEP_KINDS:
            parts.append(part)
        elif not any(isinstance(item, ContentPart) and item.value == REJECTED_OUTPUT_NOTICE for item in parts):
            parts.append(ContentPart("text", REJECTED_OUTPUT_NOTICE))
    return replace(message, body=replace(body, parts=tuple(parts)))


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
    hidden = rejected_segments(snapshot)
    selected = tuple(tuple(_without_rejected(message, hidden) for message in group if message.message_id not in excluded)
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


# provider 拒绝说明本地估算偏低：已有摘要占比更大时先缩它，否则从最旧的组减半。
# 组是已结算批次，整组丢弃不切开工具配对；只剩一组时缩短摘录，再缩已有摘要，都到头返回 None。
def _trim_inputs(summary: str, groups: tuple[tuple[Message, ...], ...],
                 excerpt: int) -> tuple[str, tuple[tuple[Message, ...], ...], int] | None:
    clipped = clip_summary(summary, 0.5)
    material = source_text(tuple(message for group in groups for message in group), excerpt=excerpt)
    if len(summary) > len(material) and len(clipped) < len(summary):
        return clipped, groups, excerpt
    if len(groups) > 1:
        return summary, groups[len(groups) // 2:], excerpt
    shorter = next((chars for chars in EXCERPT_CHARS if chars < excerpt), None)
    if groups and shorter is not None:
        return summary, groups, shorter
    return (clipped, groups, excerpt) if len(clipped) < len(summary) else None


# 每个请求一个 key：同一请求的网络重试归 Models，裁剪后的新内容是新请求，不继承已终结的拒绝。
# 新 key 不刷新额度：Models 每次物理尝试开始时发一个 call_record_id，这里按它递减。
async def _complete_trimmed(model: BoundChatModel, inputs: tuple[str, tuple[tuple[Message, ...], ...], int],
                            limit: int, *, call: SummaryCall, on_delta: StreamCallback | None,
                            ) -> tuple[LLMResponse, tuple[tuple[Message, ...], ...]]:
    summary, selected, excerpt = inputs
    remaining, trims = call.max_attempts, 0
    attempts: set[str] = set()

    async def observe(value: dict[str, str]) -> None:
        if value.get("call_record_id"):
            attempts.add(value["call_record_id"])
        if on_delta is not None:
            await on_delta(value)

    while True:
        attempts.clear()
        request = replace(_request(model, summary, selected, limit, excerpt), request_key=uuid4().hex,
                          max_attempts=remaining, max_retry_wait=SUMMARY_MAX_RETRY_WAIT, on_delta=observe)
        try:
            return await model.complete(request), selected
        except (RuntimeError, TimeoutError) as failure:
            remaining -= max(1, len(attempts))
            if (not call.trim or remaining < 1 or trims >= MAX_SUMMARY_TRIMS
                    or not ModelError.matches(failure, ContextLengthError)):
                raise
            trimmed = _trim_inputs(summary, selected, excerpt)
            if trimmed is None:
                raise
            trims += 1
            logger.warning("摘要请求被 %s 判超窗，第 %d 次裁剪后重试", model.descriptor.model_id, trims)
            summary, selected, excerpt = trimmed


async def _summarize(model: BoundChatModel, groups: tuple[tuple[Message, ...], ...],
                     previous: str, limit: int, *, call: SummaryCall, on_delta: StreamCallback | None,
                     ) -> tuple[str, tuple[str, ...], tuple[tuple[Message, ...], ...]]:
    """输入先缩进摘要模型；阻断回合的摘要被 provider 判超窗时裁剪后重发，共用一份尝试额度。"""
    response, selected = await _complete_trimmed(
        model, _fit_inputs(model, groups, previous, limit), limit, call=call, on_delta=on_delta)

    # None is the shared response contract's unspecified normal completion.
    if response.finish_reason not in (None, "stop"):
        raise SummaryError(f"摘要生成未正常完成：{response.finish_reason}")
    # 只接纳成功调用的真实正文；标题差异不影响内容与调用出处。
    text = (response.content or "").strip()
    if response.tool_calls or not text:
        raise SummaryError("摘要响应必须有正文且不能包含工具调用")
    if response.call_record_id is None:
        raise SummaryError("摘要响应缺少成功模型调用出处")
    return text, (response.call_record_id,), selected


async def summarize(groups: tuple[tuple[Message, ...], ...], *, previous: str, limit: int,
                    model: BoundChatModel, fallback: BoundChatModel,
                    on_fallback: Callable[[str], Awaitable[None]] | None = None,
                    call: SummaryCall = OPTIONAL_SUMMARY, on_delta: StreamCallback | None = None,
                    ) -> tuple[str, tuple[str, ...], tuple[tuple[Message, ...], ...]]:
    """主模型在本层可恢复的生成失败后，使用本次作用域已固定的 DEFAULT。"""
    try:
        return await _summarize(model, groups, previous, limit, call=call, on_delta=on_delta)
    except (RuntimeError, TimeoutError, SummaryError) as failure:
        if not recoverable(failure):
            raise
        if fallback.descriptor.binding_id == model.descriptor.binding_id:
            raise
        logger.warning("[fallback] 压缩摘要: 摘要模型 %s 失败，改用已固定的 DEFAULT %s: %s",
                       model.descriptor.model_id, fallback.descriptor.model_id, failure)
        if on_fallback is not None:
            await on_fallback(f"摘要模型 {model.descriptor.model_id} 失败：{failure}；"
                              f"改用 {fallback.descriptor.model_id}")
        return await _summarize(fallback, groups, previous, limit, call=call, on_delta=on_delta)
