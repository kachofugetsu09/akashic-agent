from agent.plugin_composition import Context
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS
from plugins.content.plugin import check_text
from plugins.conversation.plugin import check_origin
from plugins.conversation.source import Conversation
from plugins.sources.plugin import SOURCES, SOURCE_CHANGED, Source
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.log import MessageReader
from session.message import Control, Input

from .control import PROGRAMMATIC, Programmatic, check_session

api_version = 3
name = "programmatic"
version = "1.0.0"
desc = "程序调用的输入、停止、恢复与结果；默认保存原文但排除学习"
inject = (SOURCES, MESSAGE_WRITERS, SESSION_ADMISSION, TURN_PROJECTION)


def open_source(ctx: Context, session_id: str) -> Conversation:
    """打开已明确创建的内部 Session，固定程序来源的写入身份。"""
    check_session(session_id)
    reader = ctx.require(MESSAGE_CATALOG).reader(session_id)
    if reader.attributes.visibility != "internal":
        raise ValueError("程序调用 Session 尚未通过内部来源准入")

    def changed(reader: MessageReader, source: str) -> None:
        listener = ctx.get(SOURCE_CHANGED)
        if listener is not None:
            listener(reader, source)

    writers = ctx.require(MESSAGE_WRITERS)
    return Conversation(reader=reader,
        inputs=writers.bind(ctx, author="user", source="programmatic", body_types=(Input,),
            content={"text": check_text, "channel.origin": check_origin})(session_id),
        controls=writers.bind(ctx, author="app", source="programmatic", body_types=(Control,),
            content={})(session_id),
        tasks=ctx.require(TASKS).open(ctx), changed=changed)


async def apply(ctx: Context, config: object) -> None:
    _ = await ctx.require(SOURCES).register(ctx, Source("programmatic", lambda session: open_source(ctx, session)))
    _ = await ctx.provide(PROGRAMMATIC, Programmatic(ctx))
