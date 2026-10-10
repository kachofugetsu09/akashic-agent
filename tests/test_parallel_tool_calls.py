"""并行工具调用的顺序不变量：执行可重叠，ToolResult 与回执仍按模型顺序落盘。

守护概念：SES-002/C4 —— 同一来源的事实追加顺序是权威语义，重叠执行不得改变它。
两条路径即够：正常提交按模型顺序落盘；前驱失败时已重叠的后继也不得抢先落盘。
这些用例在没有 `parallel` 注册与提交门的旧实现上失败（register/react 无对应参数）。
"""

import asyncio
from typing import cast

import pytest

from plugins.models.contract import (
    LLMResponse,
    ToolCall as ModelToolCall,
)
from plugins.tools.execution import Result
from plugins.ledger.contract import ContentPart, Input, ToolResult
from tests.support.message_react import runtime


def _tool_results(log):
    return [message.body for message in log.reader("s").snapshot() if isinstance(message.body, ToolResult)]


@pytest.mark.asyncio
async def test_parallel_calls_overlap_but_results_commit_in_model_order(tmp_path):
    """执行可重叠，但 ToolResult 仍按模型顺序落盘。"""
    entered: list[int] = []
    finished: list[int] = []
    tail_done = asyncio.Event()
    release_first = asyncio.Event()
    requests = []

    async def complete(request):
        requests.append(request)
        if len(requests) == 1:
            return LLMResponse(None, [ModelToolCall("a", "example", {"call": 0}),
                                      ModelToolCall("b", "example", {"call": 1}),
                                      ModelToolCall("c", "example", {"call": 2})])
        rows = [row for row in request.messages if row["role"] == "tool"]
        assert [row["tool_call_id"] for row in rows] == ["a", "b", "c"]
        return LLMResponse("done")

    async def invoke(key, arguments):
        index = cast(int, arguments["call"])
        entered.append(index)
        if index == 0:
            await release_first.wait()
        finished.append(index)
        if len(finished) >= 2 and 0 not in finished:
            tail_done.set()
        return Result("success", (ContentPart("text", str(index)),))

    async with runtime(tmp_path, complete, invoke, parallel_ids=frozenset({"tool"}),
                       max_parallel_calls=4) as (conversation, log, _, run):
        await conversation.accept("u1", Input(()))
        task = await conversation.start(run)
        await asyncio.wait_for(tail_done.wait(), 5)
        assert _tool_results(log) == []
        release_first.set()
        await asyncio.wait_for(task.join(), 5)
        assert [item.call_ref.part_index for item in _tool_results(log)] == [0, 1, 2]
        assert sorted(entered) == sorted(finished) == [0, 1, 2]


@pytest.mark.asyncio
async def test_parallel_failure_does_not_commit_later_results_first(tmp_path):
    """前驱在提交前失败时，已重叠的后继不能先写 ToolResult。"""
    entered: list[int] = []
    overlapped = asyncio.Event()

    async def complete(request):
        if not any(row["role"] == "tool" for row in request.messages):
            return LLMResponse(None, [ModelToolCall("a", "example", {"call": 0}),
                                      ModelToolCall("b", "example", {"call": 1}),
                                      ModelToolCall("c", "example", {"call": 2})])
        rows = [row for row in request.messages if row["role"] == "tool"]
        assert [row["tool_call_id"] for row in rows] == ["a", "b", "c"]
        return LLMResponse("done")

    async def invoke(key, arguments):
        index = cast(int, arguments["call"])
        entered.append(index)
        if index == 0:
            await overlapped.wait()
            raise RuntimeError("boom")
        if {1, 2} <= set(entered):
            overlapped.set()
        return Result("success", (ContentPart("text", str(index)),))

    async with runtime(tmp_path, complete, invoke, parallel_ids=frozenset({"tool"}),
                       max_parallel_calls=4) as (conversation, log, _, run):
        await conversation.accept("u1", Input(()))
        task = await conversation.start(run)
        with pytest.raises(RuntimeError, match="boom"):
            await asyncio.wait_for(task.join(), 5)
        failed = _tool_results(log)
        assert [item.call_ref.part_index for item in failed] == [0, 1, 2]
        assert [item.outcome for item in failed] == ["error", "success", "success"]
        assert sorted(entered) == [0, 1, 2]
        assert await conversation.start(run) is None
