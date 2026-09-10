import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import aclosing
from typing import cast

from agent.plugin_composition import Context, RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS
from agent.plugin_composition.restart import RESTART_GATE
from agent.control.frame_book import CONTROL_FRAMES
from plugins.delivery.api import FINAL_OUTPUT_DELIVERY
from plugins.content.plugin import check_text
from plugins.conversation.plugin import check_origin
from plugins.conversation.source import Conversation
from plugins.sources.plugin import SOURCES, SOURCE_CHANGED, Source
from agent.plugin_contracts.turn_projection import TURN_PROJECTION
from session.log import MessageReader
from agent.plugin_contracts import Control, Input

from .control import PROGRAMMATIC, Programmatic, check_session

api_version = 3
name = "programmatic"
version = "1.0.0"
desc = "程序调用的输入、停止、恢复与结果；默认保存原文但排除学习"
inject = (SOURCES, MESSAGE_WRITERS, SESSION_ADMISSION, TURN_PROJECTION, RESTART_GATE, CONTROL_FRAMES)


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
        programmatic = ctx.get(PROGRAMMATIC)
        if programmatic is not None:
            programmatic.settle_changed(reader, source)

    writers = ctx.require(MESSAGE_WRITERS)
    return Conversation(reader=reader,
        inputs=writers.bind(ctx, author="user", source="programmatic", body_types=(Input,),
            content={"text": check_text, "channel.origin": check_origin})(session_id),
        controls=writers.bind(ctx, author="app", source="programmatic", body_types=(Control,),
            content={})(session_id),
        tasks=ctx.require(TASKS).open(ctx), changed=changed,
        restart_gate=ctx.require(RESTART_GATE))

async def apply(ctx: Context, config: object) -> None:
    _ = await ctx.require(SOURCES).register(ctx, Source("programmatic", lambda session: open_source(ctx, session)))
    programmatic = Programmatic(ctx)
    _ = await ctx.provide(PROGRAMMATIC, programmatic)
    catalog = ctx.require(MESSAGE_CATALOG)
    watcher: asyncio.Task[None] | None = None

    async def settle_committed_frames() -> None:
        """Follow committed heads so ordinary Output commits release idle routes."""
        stream = cast(AsyncGenerator[Mapping[str, int], None], catalog.follow())
        async with aclosing(stream):
            async for heads in stream:
                for session_id in heads:
                    programmatic.settle_changed(catalog.reader(session_id), "programmatic")

    async def start(_event: object) -> None:
        nonlocal watcher
        watcher = await ctx.spawn(settle_committed_frames(), name="programmatic-frame-settlement")

    async def stop(_event: object) -> None:
        nonlocal watcher
        if watcher is not None and not watcher.done():
            _ = watcher.cancel()
            _ = await asyncio.gather(watcher, return_exceptions=True)
        watcher = None

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)

    async def attach_delivery(child: Context) -> None:
        """在 Delivery 可用期间登记同一 programmatic provider，并随 child 释放。"""
        delivery = child.require(FINAL_OUTPUT_DELIVERY)

        def setup() -> Callable[[], None]:
            delivery.register("programmatic", programmatic)

            def cleanup() -> None:
                delivery.unregister("programmatic", programmatic)

            return cleanup

        _ = await child.effect(setup, label="programmatic-final-output")

    _ = await ctx.inject((FINAL_OUTPUT_DELIVERY,), attach_delivery,
                         name="programmatic-final-output-provider")
