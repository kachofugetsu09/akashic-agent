from __future__ import annotations

from pathlib import Path

import pytest

from plugins.models.contract import (
    LLMResponse,
    ModelContinuation,
    ToolCall as ModelToolCall,
)
from plugins.ledger.contract import ContentPart, Control, Input, MessageConflict
from plugins.react.plugin import (
    STEP_LIMIT_ABANDON,
    STEP_LIMIT_FAILURE,
    StepLimit,
    step_limit_abandon_reason,
    step_limit_failure_reason,
)
from tests.support.message_react import runtime
from plugins.tools.execution import Result


def continue_response(step: int) -> LLMResponse:
    """一步工具调用输出：有 tool_calls 的响应会落 finish="continue"。"""
    return LLMResponse(
        None,
        [ModelToolCall(f"tool-{step}", "example", {})],
        continuation=ModelContinuation("model", {"step": step}),
    )


def controls(log) -> tuple[Control, ...]:
    return tuple(
        message.body
        for message in log.reader("s").snapshot()
        if isinstance(message.body, Control)
    )


async def invoke(_key, _arguments):
    return Result("success", (ContentPart("text", "synthetic tool result"),))


@pytest.mark.asyncio
async def test_step_limit_abandons_segment_when_no_call_is_open(tmp_path: Path) -> None:
    """纯空转（没有未回执调用）撞线时由引擎作废本段，新输入仍能真正发请求。"""
    requests: list[object] = []

    async def complete(_request):
        requests.append(_request)
        # 每一步都只产生工具调用、从不收尾：连续两步即撞上 max_steps=2。
        return continue_response(len(requests))

    async with runtime(tmp_path, complete, invoke, max_steps=2) as (conversation, log, _models, run):
        await conversation.accept("first-input", Input((ContentPart("text", "first"),)))
        first = await conversation.start(run)
        assert first is not None
        await first.join()
        # react 内部连跑两步后撞线：本轮就有 2 次请求，随后自动作废本段。
        assert len(requests) == 2
        written = controls(log)
        assert [body.action for body in written] == ["abandon"], written

        await conversation.accept("second-input", Input((ContentPart("text", "second"),)))
        second = await conversation.start(run)
        assert second is not None
        await second.join()

        written = controls(log)
        abandons = [body for body in written if body.action == "abandon"]
        # 第二轮又跑满 2 步后才作废：新输入不会被旧账卡住，也不会凭空清零。
        assert len(abandons) == 2, written
        for body in abandons:
            assert body.reason == step_limit_abandon_reason(2)
            assert body.reason is not None
            assert body.reason.startswith(STEP_LIMIT_ABANDON)
        # 自动放弃不需要用户再动手，因此不能写 failure 把来源留在暂停上。
        assert not [body for body in written if body.action == "failure"]
        # 两轮共 4 次请求：每段 2 步，第二轮没有因为旧账而一步都发不出。
        assert len(requests) == 4


@pytest.mark.asyncio
async def test_step_limit_without_abort_writer_keeps_failing(tmp_path: Path) -> None:
    """没有写控制能力的调用方仍得到明确失败，行为与修复前一致。"""
    async def complete(_request):
        return continue_response(0)

    async with runtime(
        tmp_path, complete, invoke, max_steps=1, without_abort=True,
    ) as (conversation, log, _models, run):
        await conversation.accept("first-input", Input((ContentPart("text", "first"),)))
        first = await conversation.start(run)
        assert first is not None
        with pytest.raises(RuntimeError) as failure:
            await first.join()
        assert isinstance(failure.value.args[0], StepLimit)
        assert not [body for body in controls(log) if body.action == "abandon"]


@pytest.mark.asyncio
async def test_step_limit_with_open_call_reports_unknown_effect(tmp_path: Path) -> None:
    """未回执调用可能已有外部效果：引擎不替用户作废，只停下并如实报告。"""
    requests: list[object] = []

    async def complete(_request):
        requests.append(_request)
        return continue_response(0)

    async with runtime(
        tmp_path, complete, invoke, max_steps=1,
        # 这个调用结算后不会留下回执：它就是「外部效果未知」的那类遗留工作。
        drop_receipt=lambda call: True,
    ) as (conversation, log, _models, run):
        await conversation.accept("first-input", Input((ContentPart("text", "first"),)))
        first = await conversation.start(run)
        assert first is not None
        # 跑完一步、留下未回执调用后在门槛处停下：直接报可识别的超限失败。
        with pytest.raises(RuntimeError) as failure:
            await first.join()

        limit = failure.value.args[0]
        assert isinstance(limit, StepLimit)
        assert str(limit) == step_limit_failure_reason(1, 1)
        # 界面按这个前缀认出「重试不会生效、只有放弃能解开」，所以它必须是稳定的事实。
        assert str(limit).startswith(STEP_LIMIT_FAILURE)
        # 引擎没有替用户作废：没有任何 abandon，计数也不会被清掉。
        assert not [body for body in controls(log) if body.action == "abandon"]


@pytest.mark.asyncio
async def test_user_abandon_refuses_already_closed_prefix(tmp_path: Path) -> None:
    """人工放弃沿用账本既有前提：已收尾的前缀不能被重新关闭，不会伪造清账。"""
    requests: list[object] = []

    async def complete(_request):
        requests.append(_request)
        return LLMResponse(f"answer-{len(requests)}")

    async with runtime(tmp_path, complete, invoke, max_steps=8) as (conversation, log, _models, run):
        await conversation.accept("first-input", Input((ContentPart("text", "first"),)))
        first = await conversation.start(run)
        assert first is not None
        await first.join()
        assert len(requests) == 1

        head = log.reader("s").head(source="conversation")
        with pytest.raises(MessageConflict):
            await conversation.control(
                "user-abandon", Control("abandon", head, "用户放弃本轮未完成工作"),
                expected_head=head, handle=None,
            )
        # 被拒绝的放弃不写入任何事实。
        assert not [body for body in controls(log) if body.action == "abandon"]
