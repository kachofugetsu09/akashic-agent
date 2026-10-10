from contextlib import nullcontext
from functools import partial
from typing import Any

from plugins.ledger.contract import Message
from plugins.models.contract import CONTENT_VIEWS
from plugins.context.contract import MaterialKind
from plugins.ledger.contract import MessageReader
from agent.plugin_composition.tasks import Task

from agent.plugin_composition import Context
from plugins.models.contract import CHAT_MODELS
from plugins.ledger.contract import ARTIFACT_READ
from plugins.ledger.contract import MESSAGE_WRITERS, OWNER_STATE
from plugins.reply_program.contract import (
    REPLY_EXECUTE_V4,
)
from plugins.sources.contract import (
    SourceGuard,
)

from .inputs import (
    CONTENT,
    CONTEXT,
    MATERIALS,
    MODEL_CALLS,
    MODEL_CHECKS,
    MODEL_CONTENT,
    MODEL_PROJECTION,
    MODEL_SELECTION,
    REACT,
    TOOL_CLEANUP,
    TOOL_PROGRAM,
    TOOLS,
    TURN_PROJECTION,
)
from .program import run_reply

api_version = 3
name = "reply_program"
version = "1.0.0"
desc = "一次回复的资源与执行组合；不拥有来源策略或后台监听"
inject = (CHAT_MODELS, CONTENT, CONTEXT, MATERIALS, MODEL_CALLS, MODEL_CHECKS,
          MODEL_CONTENT, MODEL_PROJECTION, MODEL_SELECTION, REACT, TOOL_CLEANUP,
          TOOL_PROGRAM, TOOLS, TURN_PROJECTION, OWNER_STATE, MESSAGE_WRITERS, ARTIFACT_READ)


async def apply(ctx: Context) -> None:
    """在同一代绑定程序依赖；每次调用仍自行持有实际执行租约。"""
    run = partial(
        run_reply, models=ctx.require(CHAT_MODELS), content=ctx.require(CONTENT),
        context=ctx.require(CONTEXT), tools=ctx.require(TOOLS), cleanup=ctx.require(TOOL_CLEANUP),
        react=ctx.require(REACT),
        turn_projection=ctx.require(TURN_PROJECTION), read_call=ctx.require(MODEL_CALLS),
        selection=ctx.require(MODEL_SELECTION),
        tool_program=ctx.require(TOOL_PROGRAM), model_checks=ctx.require(MODEL_CHECKS),
        model_content=ctx.require(MODEL_CONTENT), model_projection=ctx.require(MODEL_PROJECTION),
        writers=ctx.require(MESSAGE_WRITERS), owner_state=ctx.require(OWNER_STATE),
        artifact_reader=ctx.require(ARTIFACT_READ),
    )
    async def execute(caller: Context, task: Task, reader: MessageReader, source: str,
                      **options: Any) -> Message:
        """固定可选内容贡献者；无贡献服务的组合保留基础模型投影。"""
        with ctx.borrow(CONTENT_VIEWS) as views:
            async with (nullcontext(None) if views is None else views.bind()) as bound:
                prepare = None if bound is None else bound.prepare
                kinds = frozenset() if bound is None else bound.dynamic_kinds
                return await run(caller, task, reader, source, prepare_content=prepare,
                                 dynamic_content_kinds=kinds, **options)

    materials = ctx.require(MATERIALS)

    async def selected(ctx: Context, task: Task, reader: MessageReader, source: str, *,
                       check_admission: SourceGuard,
                       exclude_material_kinds: frozenset[MaterialKind] = frozenset(),
                       **options: Any) -> Message:
        """新合同只把用途选择交给当前材料 provider。"""
        return await execute(ctx, task, reader, source,
                             check_admission=check_admission,
                             materials=materials.bind(exclude_kinds=exclude_material_kinds), **options)

    _ = await ctx.provide(REPLY_EXECUTE_V4, ctx.entrypoint(selected))
