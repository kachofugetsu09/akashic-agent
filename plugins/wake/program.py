from __future__ import annotations

import asyncio
from collections.abc import Mapping

from agent.plugin_composition import CHAT_MODELS, Context
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugin_composition.models import ModelError
from agent.plugin_composition.tasks import Task
from plugins.content.plugin import CONTENT
from plugins.context.plugin import CONTEXT
from plugins.context.api import ContextOverflow
from plugins.context.materials import MATERIALS
from plugins.conversation.program import run_reply
from plugins.models.projection import MODEL_CALLS
from plugins.react.plugin import REACT, StepLimit
from plugins.tools.api import Denied
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.log import MessageReader
from session.message import Control, Message

from .messages import HINTS, render
from .request import Request, STAGE_TOOLS, WakeFailure, read_phase


async def run(ctx: Context, task: Task, reader: MessageReader, request: Request) -> Message:
    """按归档程序和原工具运行一个真实阶段，已知失败也保存为普通 Control。"""
    request = Request.model_validate(request.model_dump())
    _, phase = read_phase(reader.snapshot(), request)
    names = STAGE_TOOLS[phase.stage]
    fixed = {name: request.tools[name] for name in names}

    async def authorize(binding: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
        if binding not in fixed.values():
            raise Denied("Wake 原阶段未授予该工具")
        return {"source": "wake", "session_id": request.session_id}

    try:
        return await run_reply(
            ctx, task, reader, "wake", models=ctx.require(CHAT_MODELS), content=ctx.require(CONTENT),
            context=ctx.require(CONTEXT), tools=ctx.require(TOOLS), react=ctx.require(REACT),
            materials=ctx.require(MATERIALS), turn_projection=ctx.require(TURN_PROJECTION),
            read_call=ctx.require(MODEL_CALLS), render_content=render, authorize=authorize,
            tool_names=names, fixed_bindings=fixed, max_output_tokens=4096,
            max_steps=1 if phase.stage in {"screen", "alert"} else 20 if phase.stage == "investigate" else 40,
            terminal_tools=frozenset(name for name in names if name not in {"recall_memory", "web_fetch"}),
            exclude_materials=frozenset({"akasha", "markdown_memory"}) if phase.stage in {"investigate", "alert"} else frozenset(),
            prompt_hints=(HINTS[phase.stage],),
        )
    except ModelError as error:
        reason = WakeFailure(message=str(error), retryable=error.retryable).model_dump_json()
    except (StepLimit, ContextOverflow) as error:
        reason = str(error)
    # 此处只结算本层已知的失败；未知工具效果和存储错误保持原事实并向上传播。
    writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="wake", source="wake", body_types=(Control,), content={})(reader.session_id)
    try:
        if not task.active:
            raise asyncio.CancelledError
        return writer.append(request.phase_id(phase.stage) + ":failure", Control("failure", reader.head(source="wake"), reason))
    finally:
        writer.expire()
