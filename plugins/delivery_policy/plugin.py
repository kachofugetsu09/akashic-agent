from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext
from functools import partial
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import Context, RUNTIME_STARTING, RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG
from plugins.conversation.plugin import check_origin
from plugins.delivery.api import Sink
from plugins.delivery.plugin import DELIVERY
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.reply.completion import REPLY_COMPLETION
from session.log import MessageReader
from session.message import ContentPart, Input, Message, Output, ToolCall

from .follow import follow

logger = logging.getLogger(__name__)

api_version = 3
name = "delivery_policy"
version = "1.0.0"
desc = "默认只发送完整可见回复；显式通知沿来源自己的固定发送选择"
inject = (DELIVERY, DELIVERY_SENDERS, BINDINGS, MESSAGE_CATALOG)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: tuple[str, ...] = Field(default=("conversation",), min_length=1)


def origin(reader: MessageReader, message: Message, sources: tuple[str, ...]) -> tuple[str, str] | None:
    """默认回复发往其消息前缀中最后一次同来源输入；后来消息不改变旧目的地。"""
    if reader.attributes.visibility == "internal":
        return None
    body = message.body
    if (message.source not in sources or not isinstance(body, Output) or body.finish != "complete"
            or any(isinstance(part, ToolCall) for part in body.parts)):
        return None
    visible = any(isinstance(part, ContentPart) and (
        part.kind == "artifact_ref" or part.kind == "text" and isinstance(part.value, str) and part.value.strip()
    ) for part in body.parts)
    if not visible:
        return None
    return input_origin(reader, message.source, through_seq=message.seq)


def input_origin(reader: MessageReader, source: str, *, through_seq: int) -> tuple[str, str] | None:
    """只从原输入的已验证渠道事实读取目的地。"""
    for previous in reversed(reader.snapshot(through_seq=through_seq)):
        if previous.source != source or not isinstance(previous.body, Input):
            continue
        parts = [part for part in previous.body.parts if part.kind == "channel.origin"]
        if not parts:
            return None
        if len(parts) != 1:
            raise ValueError("输入必须只有一个渠道来源")
        _ = check_origin(parts[0])
        value = cast(Mapping[str, str], parts[0].value)
        return value["channel"], value["chat_id"]
    return None


async def apply(ctx: Context, config: Config) -> None:
    """正式启动后跟随日志；不把策略、学习或来源 ACK 放进发送原子能力。"""
    watcher: asyncio.Task[None] | None = None
    recovery: dict[tuple[str, str], AbstractContextManager[None]] = {}

    def settled(message_id: str, sink: str) -> None:
        hold = recovery.pop((message_id, sink), None)
        if hold is not None:
            _ = hold.__exit__(None, None, None)

    def close_recovery() -> None:
        for hold in recovery.values():
            _ = hold.__exit__(None, None, None)
        recovery.clear()

    def prepare(_event: object) -> None:
        """正式接纳开放前占住原被动效果，直到 follower 实际恢复一次发送或查询。"""
        delivery = ctx.require(DELIVERY).open(ctx)
        for message_id, sink in delivery.pending():
            selection = delivery.selection(message_id)
            assert selection is not None
            if selection.passive and (message_id, sink) not in recovery:
                target = delivery.destination(message_id, sink)
                hold = delivery.activity(target.name, target.address)
                _ = hold.__enter__()
                recovery[message_id, sink] = hold

    _ = await ctx.effect(lambda: close_recovery, label="pending-delivery")

    def select(reader: MessageReader, message: Message) -> tuple[Sink, ...] | None:
        if message.source not in config.sources or reader.attributes.visibility == "internal":
            return None
        route = origin(reader, message, config.sources)
        if route is None:
            return ()
        name, address = route
        binding = ctx.require(DELIVERY_SENDERS).bind(name, ctx.require(BINDINGS))
        return (Sink(name=name, binding_id=binding, address=address),)

    class ReplyCompletion:
        def activity(self, reader: MessageReader, source: str) -> AbstractContextManager[None]:
            route = (input_origin(reader, source, through_seq=reader.head())
                     if source in config.sources and reader.attributes.visibility != "internal" else None)
            return nullcontext() if route is None else ctx.require(DELIVERY).open(ctx).activity(*route)

        @asynccontextmanager
        async def __call__(self, reader: MessageReader, source: str) -> AsyncGenerator[None]:
            """完成后固定选路并独立启动发送，新 Input 不取得旧发送的取消权。"""
            head = reader.head()
            delivery = ctx.require(DELIVERY).open(ctx)
            with self.activity(reader, source):
                try:
                    yield
                finally:
                    for message in reader.snapshot():
                        if message.seq <= head:
                            continue
                        if message.source != source or origin(reader, message, config.sources) is None:
                            continue
                        try:
                            sinks = select(reader, message) if delivery.selection(message.message_id) is None else ()
                            assert sinks is not None
                            selected = delivery.prepare(reader, message, sinks, passive=True)
                            for sink in selected.sinks:
                                _ = await delivery.start(message.message_id, sink)
                        except Exception:
                            # 回复事实已提交；发送失败不能反向改写来源为推理失败。
                            logger.exception("回复发送未结算，保留原效果 message=%s", message.message_id)

    _ = await ctx.provide(REPLY_COMPLETION, ReplyCompletion())

    async def start(_event: object) -> None:
        nonlocal watcher
        watcher = await ctx.spawn(follow(
            ctx, ctx.require(MESSAGE_CATALOG), partial(ctx.require(DELIVERY).open, ctx), select, settled=settled,
        ), name="delivery")

    async def stop(_event: object) -> None:
        try:
            if watcher is not None:
                _ = watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
        finally:
            close_recovery()

    _ = await ctx.on(RUNTIME_STARTING, prepare)
    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
