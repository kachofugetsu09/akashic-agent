from collections.abc import Awaitable, Callable
from functools import partial

from agent.plugin_composition import CHAT_MODELS, Context, ServiceKey
from agent.plugin_contracts import Message

from .inputs import (
    SOURCE_CHECK, CONTENT, CONTEXT, MATERIALS, MODEL_CALLS, MODEL_CHECKS, MODEL_CONTENT, MODEL_PROJECTION,
    MODEL_SELECTION, REACT, TOOL_CLEANUP, TOOL_PROGRAM, TOOLS, TURN_PROJECTION,
)
from .program import run_reply

api_version = 3
name = "reply_program"
version = "1.0.0"
desc = "一次回复的资源与执行组合；不拥有来源策略或后台监听"
inject = (SOURCE_CHECK, CHAT_MODELS, CONTENT, CONTEXT, MATERIALS, MODEL_CALLS, MODEL_CHECKS,
          MODEL_CONTENT, MODEL_PROJECTION, MODEL_SELECTION, REACT, TOOL_CLEANUP,
          TOOL_PROGRAM, TOOLS, TURN_PROJECTION)
REPLY_EXECUTE = ServiceKey[Callable[..., Awaitable[Message]]]("reply.execute.v1")


async def apply(ctx: Context, config: object) -> None:
    """在同一代绑定程序依赖；每次调用仍自行持有实际执行租约。"""
    _ = await ctx.provide(REPLY_EXECUTE, partial(
        run_reply, models=ctx.require(CHAT_MODELS), content=ctx.require(CONTENT),
        context=ctx.require(CONTEXT), tools=ctx.require(TOOLS), cleanup=ctx.require(TOOL_CLEANUP),
        react=ctx.require(REACT), materials=ctx.require(MATERIALS),
        turn_projection=ctx.require(TURN_PROJECTION), read_call=ctx.require(MODEL_CALLS),
    ))
