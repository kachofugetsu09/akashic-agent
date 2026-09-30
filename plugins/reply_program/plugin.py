from contextlib import nullcontext
from functools import partial
from typing import Any

from agent.plugin_contracts import Message
from agent.plugin_contracts.models import CONTENT_VIEWS
from agent.plugin_contracts.context import MATERIALS as LEGACY_MATERIALS, MaterialKind
from agent.plugin_composition.messages import MessageReader
from agent.plugin_composition.tasks import Task

from agent.plugin_composition import CHAT_MODELS, Context
from agent.plugin_composition.messages import MESSAGE_WRITERS, OWNER_STATE
from agent.plugin_contracts.reply import REPLY_EXECUTE, REPLY_EXECUTE_V2

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
    SOURCE_CHECK,
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
inject = (SOURCE_CHECK, CHAT_MODELS, CONTENT, CONTEXT, MATERIALS, MODEL_CALLS, MODEL_CHECKS,
          MODEL_CONTENT, MODEL_PROJECTION, MODEL_SELECTION, REACT, TOOL_CLEANUP,
          TOOL_PROGRAM, TOOLS, TURN_PROJECTION, OWNER_STATE, MESSAGE_WRITERS)


async def apply(ctx: Context) -> None:
    """在同一代绑定程序依赖；每次调用仍自行持有实际执行租约。"""
    run = partial(
        run_reply, models=ctx.require(CHAT_MODELS), content=ctx.require(CONTENT),
        context=ctx.require(CONTEXT), tools=ctx.require(TOOLS), cleanup=ctx.require(TOOL_CLEANUP),
        react=ctx.require(REACT),
        turn_projection=ctx.require(TURN_PROJECTION), read_call=ctx.require(MODEL_CALLS),
        check_source=ctx.require(SOURCE_CHECK), selection=ctx.require(MODEL_SELECTION),
        tool_program=ctx.require(TOOL_PROGRAM), model_checks=ctx.require(MODEL_CHECKS),
        model_content=ctx.require(MODEL_CONTENT), model_projection=ctx.require(MODEL_PROJECTION),
        writers=ctx.require(MESSAGE_WRITERS), owner_state=ctx.require(OWNER_STATE),
    )
    async def execute(caller: Context, task: Task, reader: MessageReader, source: str,
                      **options: Any) -> Message:
        """固定可选内容贡献者；无贡献服务的组合保留基础模型投影。"""
        with ctx.borrow(CONTENT_VIEWS) as views:
            async with (nullcontext(None) if views is None else views.bind()) as prepare:
                return await run(caller, task, reader, source, prepare_content=prepare, **options)

    materials = ctx.require(MATERIALS)

    async def selected(ctx: Context, task: Task, reader: MessageReader, source: str, *,
                       exclude_material_kinds: frozenset[MaterialKind] = frozenset(),
                       **options: Any) -> Message:
        """新合同只把用途选择交给当前材料 provider。"""
        return await execute(ctx, task, reader, source,
                             materials=materials.bind(exclude_kinds=exclude_material_kinds), **options)

    _ = await ctx.provide(REPLY_EXECUTE_V2, ctx.entrypoint(selected))

    async def legacy(child: Context) -> None:
        """旧调用恢复独立借用 v3；缺席不阻断新用途接口。"""
        old_materials = child.require(LEGACY_MATERIALS)

        async def named(ctx: Context, task: Task, reader: MessageReader, source: str, *,
                        exclude_materials: frozenset[str] = frozenset(), **options: Any) -> Message:
            """仅解释旧名称选择，回复程序与新入口共用。"""
            return await execute(ctx, task, reader, source,
                                 materials=old_materials.bind(exclude=exclude_materials), **options)

        _ = await child.provide(REPLY_EXECUTE, child.entrypoint(named))

    await ctx.inject((LEGACY_MATERIALS,), legacy, name="legacy-materials")
