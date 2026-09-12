import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import aclosing
from typing import Protocol, cast

from agent.plugin_composition import Context, Effect, ServiceKey, RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS, Task, TaskAdmission, RestartGate, RESTART_GATE
from agent.plugin_composition.control_frames import CONTROL_FRAMES
from agent.plugin_composition.rpc import rpc_method_key
from .result import TURN_PROJECTION
from agent.plugin_composition.channels import ChannelInboundMessage
from agent.plugin_composition.messages import MessageReader, MessageWriter
from agent.plugin_contracts import ContentPart, ContentReferences, Control, Input, Message

from .control import PROGRAMMATIC, Programmatic, FinalOutputTurn, check_session, rpc_methods



class ContentChecks(Protocol):
    def check_text(self, part: ContentPart) -> ContentReferences: ...


class FinalOutputWaiter(Protocol):
    async def wait(self, reader: MessageReader, turn: FinalOutputTurn) -> None: ...


class FinalOutputDelivery(Protocol):
    def register(self, source: str, provider: FinalOutputWaiter) -> None: ...
    def unregister(self, source: str, provider: FinalOutputWaiter) -> None: ...


CONTENT = ServiceKey[ContentChecks]("content.v2")
CHECK_ORIGIN = ServiceKey[Callable[[ContentPart], ContentReferences]]("conversation.check_origin.v1")
FINAL_OUTPUT_DELIVERY = ServiceKey[FinalOutputDelivery]("delivery.final_output.v1")


class SourceSession(Protocol):
    async def accept(self, message_id: str, body: Input) -> Message: ...
    async def pause(self, message_id: str) -> Message: ...
    async def resume(self, message_id: str, input_id: str) -> Message: ...
    async def start(self, program: Callable[[Task, MessageReader, str], Awaitable[object]]) -> Task | None: ...


class SessionFactory(Protocol):
    def __call__(
        self, *, reader: MessageReader, inputs: MessageWriter, controls: MessageWriter,
        tasks: TaskAdmission, changed: Callable[[MessageReader, str], None] | None = None,
        restart_gate: RestartGate | None = None,
    ) -> SourceSession: ...

    def needs_reply(self, reader: MessageReader, source: str) -> bool: ...


class SourceRegistry(Protocol):
    async def register(
        self, ctx: Context, *, name: str, open: Callable[[str], SourceSession],
        needs_reply: Callable[[MessageReader], bool],
        accept: Callable[[str, str, ChannelInboundMessage], Awaitable[Message]] | None = None,
        channels: tuple[str, ...] | None = (),
    ) -> Effect: ...


SOURCES = ServiceKey[SourceRegistry]("sources.v2")
SOURCE_SESSION = ServiceKey[SessionFactory]("source.session.v1")
SOURCE_CHANGED = ServiceKey[Callable[[MessageReader, str], None]]("source.changed.v1")


api_version = 3
name = "programmatic"
version = "1.0.0"
desc = "程序调用的输入、停止、恢复与结果；默认保存原文但排除学习"
inject = (CONTENT, CHECK_ORIGIN, SOURCES, SOURCE_SESSION, MESSAGE_WRITERS, SESSION_ADMISSION, TURN_PROJECTION, RESTART_GATE, CONTROL_FRAMES)


def open_source(ctx: Context, session_id: str) -> SourceSession:
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
    return ctx.require(SOURCE_SESSION)(reader=reader,
        inputs=writers.bind(ctx, author="user", source="programmatic", body_types=(Input,),
            content={"text": ctx.require(CONTENT).check_text, "channel.origin": ctx.require(CHECK_ORIGIN)})(session_id),
        controls=writers.bind(ctx, author="app", source="programmatic", body_types=(Control,),
            content={})(session_id),
        tasks=ctx.require(TASKS).open(ctx), changed=changed,
        restart_gate=ctx.require(RESTART_GATE))

async def apply(ctx: Context, config: object) -> None:
    _ = await ctx.require(SOURCES).register(ctx, name="programmatic", open=lambda session: open_source(ctx, session),
        needs_reply=lambda reader: ctx.require(SOURCE_SESSION).needs_reply(reader, "programmatic"))
    programmatic = Programmatic(ctx)
    _ = await ctx.provide(PROGRAMMATIC, programmatic)
    for method, operation in rpc_methods(programmatic).items():
        _ = await ctx.provide(rpc_method_key(method), operation)
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
