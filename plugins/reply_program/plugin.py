from functools import partial

from agent.plugin_composition import CHAT_MODELS, Context
from agent.plugin_composition.artifacts import ARTIFACT_READ
from agent.plugin_composition.messages import MESSAGE_WRITERS, OWNER_STATE
from agent.plugin_contracts.reply import REPLY_EXECUTE as REPLY_EXECUTE

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
          TOOL_PROGRAM, TOOLS, TURN_PROJECTION, OWNER_STATE, MESSAGE_WRITERS, ARTIFACT_READ)


async def apply(ctx: Context) -> None:
    """在同一代绑定程序依赖；每次调用仍自行持有实际执行租约。"""
    _ = await ctx.provide(REPLY_EXECUTE, ctx.entrypoint(partial(
        run_reply, models=ctx.require(CHAT_MODELS), content=ctx.require(CONTENT),
        context=ctx.require(CONTEXT), tools=ctx.require(TOOLS), cleanup=ctx.require(TOOL_CLEANUP),
        react=ctx.require(REACT), materials=ctx.require(MATERIALS),
        turn_projection=ctx.require(TURN_PROJECTION), read_call=ctx.require(MODEL_CALLS),
        check_source=ctx.require(SOURCE_CHECK), selection=ctx.require(MODEL_SELECTION),
        tool_program=ctx.require(TOOL_PROGRAM), model_checks=ctx.require(MODEL_CHECKS),
        model_content=ctx.require(MODEL_CONTENT), model_projection=ctx.require(MODEL_PROJECTION),
        writers=ctx.require(MESSAGE_WRITERS), owner_state=ctx.require(OWNER_STATE),
        artifact_reader=ctx.require(ARTIFACT_READ),
    )))
