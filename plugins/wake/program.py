from __future__ import annotations

from functools import partial

import asyncio
from collections.abc import Mapping

from agent.plugin_composition import Context
from plugins.ledger.contract import MESSAGE_WRITERS, MessageReader
from agent.plugin_composition.tasks import Task
from plugins.ledger.contract import Control, Message
from plugins.models.contract import (
    MODEL_CONTENT as MODEL_CONTENT,
    ModelContent as ModelContent,
)
from plugins.sources.contract import (
    SOURCE_CHECK_V2 as SOURCE_CHECK,
)
from plugins.reply_program.contract import (
    REPLY_EXECUTE_V4 as REPLY_EXECUTE,
)

from .messages import HINTS, render
from .request import TERMINAL_TOOLS, Request, Stage, WakeFailure, read_phase


# 每类职责最多运行的模型步数：技能读取、初筛、调查与提交决定共用一轮中心循环。
MAX_STEPS: dict[Stage, int] = {"content": 26, "alert": 3}


async def run(ctx: Context, task: Task, reader: MessageReader, request: Request) -> Message:
    """按归档工具运行一轮中心循环，已知失败也保存为普通 Control。"""
    request = Request.model_validate(request.model_dump())
    _, phase = read_phase(reader.snapshot(), request)
    fixed = dict(request.tools)

    async def authorize(binding: str, arguments: Mapping[str, object]) -> Mapping[str, object] | str:
        if binding not in fixed.values():
            return 'Wake 原请求未授予该工具'
        return {"source": "wake", "session_id": request.session_id}

    try:
        return await ctx.require(REPLY_EXECUTE)(
            ctx,
            task,
            reader,
            "wake",
            render_content=lambda part: render(
                part,
                fallback=lambda item: ctx.require(MODEL_CONTENT).render(item, artifacts={}),
            ),
            check_admission=partial(ctx.require(SOURCE_CHECK), task, reader, "wake", task.boundary_hint),
            authorize=authorize,
            tool_view=None,
            fixed_bindings=fixed,
            # 推理也占用输出预算，不另设会截断工具决定的小上限。
            max_output_tokens=0,
            max_steps=MAX_STEPS[phase.stage],
            terminal_tools=frozenset(fixed) & TERMINAL_TOOLS,
            # 告警只写通知，不需要记忆材料；其余职责初筛要结合已注入的记忆。
            exclude_material_kinds=frozenset({"recall", "profile"}) if phase.stage == "alert" else frozenset(),
            prompt_hints=(HINTS[phase.stage],),
        )
    except Exception as error:
        # Models owns bounded retries. A failed Wake round must not start that budget again.
        detail = f"{type(error).__name__}: {error}"
        reason = WakeFailure(message=detail, retryable=False).model_dump_json()
        _ = ctx.report_incident("wake_program_failed", f"flow={request.flow_id} {detail}")
    # Keep tool receipts intact; this control ends only the failed program.
    writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="wake", source="wake", body_types=(Control,), content={})(reader.session_id)
    task.on_close(writer.expire)
    try:
        if not task.active:
            raise asyncio.CancelledError
        return await writer.append_async(request.phase_id(phase.stage) + ":failure", Control("failure", reader.head(source="wake"), reason))
    finally:
        writer.expire()
