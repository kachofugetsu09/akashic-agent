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
from agent.plugin_contracts import CallRef, ContentPart, Control, Message, Output, ToolCall, ToolResult
from agent.plugin_contracts.context import settled_prefixes
from agent.plugin_contracts.turn_projection import TurnProjectionPort as TurnProjection
from agent.plugin_contracts import body_to_dict

logger = logging.getLogger(__name__)

HEADINGS = (
    "## Goal", "## Constraints & Preferences", "## Progress", "### Done",
    "### In Progress", "### Blocked", "## Key Decisions", "## Next Steps", "## Critical Context",
)
PROMPT = """更新当前长任务的上下文压缩摘要。

只记录输入中已经出现的事实，不补充猜测，不把计划写成已完成。
摘要只替代已结算的旧消息；完整原文和工具结果仍保留。
区分用户原话、助手判断及工具/后台结果，保留重要来源身份；转述不能升级成用户事实或偏好。
必须严格使用以下标题，不得增加标题：
""" + "\n".join(HEADINGS) + """

保留路径、符号、命令、错误、数值、外部效果和验证结果。
仍在运行的执行必须保留 execution_id、命令和当前状态。
省略重复探索、无用日志和 provider 协议细节。只输出摘要正文。
"""


class SummaryError(ValueError):
    """摘要不能满足完整前缀、容量或输出合同。"""








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

from agent.plugin_contracts.compaction import (  # noqa: E402,F401  (再导出)
    _protected_cuts,
    source_text,
    summary_groups,
    window_starts,
)
