from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping

from typing import Protocol

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugin_composition.models import ModelError
from agent.plugin_composition.tasks import Task
from agent.plugin_composition.messages import MessageReader
from agent.plugin_contracts import ContentPart, Control, Message

from .messages import HINTS, render
from .request import Request, STAGE_TOOLS, WakeFailure, read_phase


REPLY_EXECUTE = ServiceKey[Callable[..., Awaitable[Message]]]("reply.execute.v1")


class ModelContent(Protocol):
    def render(self, part: ContentPart, *, artifacts: Mapping[str, tuple[Mapping[str, object], ...]]) -> tuple[Mapping[str, object], ...]: ...


MODEL_CONTENT = ServiceKey[ModelContent]("models.content.v1")

async def run(ctx: Context, task: Task, reader: MessageReader, request: Request) -> Message:
    """按归档程序和原工具运行一个真实阶段，已知失败也保存为普通 Control。"""
    request = Request.model_validate(request.model_dump())
    _, phase = read_phase(reader.snapshot(), request)
    names = STAGE_TOOLS[phase.stage]
    fixed = {name: request.tools[name] for name in names}

    async def authorize(binding: str, arguments: Mapping[str, object]) -> Mapping[str, object] | str:
        if binding not in fixed.values():
            return 'Wake 原阶段未授予该工具'
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
            authorize=authorize,
            tool_view=None,
            fixed_bindings=fixed,
            # 推理也占用输出预算，阶段不另设会截断工具决定的小上限。
            max_output_tokens=0,
            max_steps=(
                3
                if phase.stage == "screen"
                else 1
                if phase.stage == "alert"
                else 20
                if phase.stage == "investigate"
                else 40
            ),
            terminal_tools=frozenset(
                name for name in names if name not in {"recall_memory", "web_fetch"}
            ),
            exclude_materials=(
                frozenset({"akasha", "markdown_memory"})
                if phase.stage in {"investigate", "alert"}
                else frozenset()
            ),
            prompt_hints=(HINTS[phase.stage],),
        )
    except ModelError as error:
        reason = WakeFailure(message=str(error), retryable=error.retryable).model_dump_json()
    # 此处只结算本层已知的失败；未知工具效果和存储错误保持原事实并向上传播。
    writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="wake", source="wake", body_types=(Control,), content={})(reader.session_id)
    try:
        if not task.active:
            raise asyncio.CancelledError
        return writer.append(request.phase_id(phase.stage) + ":failure", Control("failure", reader.head(source="wake"), reason))
    finally:
        writer.expire()
