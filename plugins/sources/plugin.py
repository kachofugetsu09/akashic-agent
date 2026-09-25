from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass

from agent.plugin_composition import Context, Effect
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition.messages import MessageReader
from agent.plugin_contracts import Message
from agent.plugin_contracts.sources import (
    SOURCE_CHECK as SOURCE_CHECK,
    SOURCE_SESSION as SOURCE_SESSION,
    SOURCES as SOURCES,
    Source as Source,
    SourceSession as SourceSession,
)

api_version = 3
name = "sources"
version = "1.0.0"
desc = "按来源注册接纳与控制，供渠道和默认回复组合使用"
inject = ()

Accept = Callable[[str, str, ChannelInboundMessage], Awaitable[Message]]
Changed = Callable[[MessageReader, str], None]


from .session import SourceSession as _SourceSession, check_source


@dataclass(slots=True)
class _Registration:
    source: Source
    visible: bool = False


class Sources:
    """每个来源和输入渠道只绑定一个 owner；不拥有 Message 或活动 Task。"""

    def __init__(self, context: Context):
        self._context = context
        self._items: dict[str, _Registration] = {}
        self._changes: set[asyncio.Event] = set()

    def _notify_changes(self) -> None:
        """唤醒只读订阅者；订阅者随后重读唯一注册表事实。"""
        for event in tuple(self._changes):
            event.set()

    async def register(
        self, ctx: Context, *, name: str, open: Callable[[str], SourceSession],
        needs_reply: Callable[[MessageReader], bool], accept: Accept | None = None,
        channels: tuple[str, ...] | None = (),
    ) -> Effect:
        """来源注册随插件 effect 生灭；None channels 表示唯一默认输入来源。"""
        source = Source(
            ctx, name, open, needs_reply, accept,
            None if channels is None else tuple(channels),
        )
        if ctx.require(SOURCES) is not self:
            raise PermissionError("来源注册不属于当前组合")
        if not source.name or source.accept is None and source.channels != () or (
            source.channels is not None and any(not channel for channel in source.channels)
        ):
            raise ValueError("来源名称和渠道必须明确")

        registration: _Registration | None = None

        def setup():
            nonlocal registration
            if source.name in self._items:
                raise ValueError("来源已有 owner: " + source.name)
            for existing in self._items.values():
                if source.channels is None and existing.source.channels is None or (
                    source.channels is not None and existing.source.channels is not None
                    and set(source.channels).intersection(existing.source.channels)
                ):
                    raise ValueError("输入渠道已有来源 owner")
            registration = _Registration(source)
            self._items[source.name] = registration

            def close() -> None:
                assert registration is not None
                if self._items.get(source.name) is not registration:
                    return
                del self._items[source.name]
                if registration.visible:
                    self._notify_changes()

            return close

        effect = await ctx.effect(setup, label="source:" + source.name)
        assert registration is not None

        async def publish_when_ready() -> None:
            """首个 activation-ready 后发布 exact registration。"""
            if self._items.get(source.name) is not registration:
                return
            registration.visible = True
            self._notify_changes()

        try:
            _ = await ctx.spawn(
                publish_when_ready(), name="source-ready:" + source.name,
            )
        except BaseException as error:
            try:
                await effect.aclose()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "来源登记和通知任务清理失败", [error, cleanup_error]
                ) from None
            raise
        return effect

    def needs_reply(self, reader: MessageReader, source: str) -> bool:
        """来源自己解释是否待回复；注册表不猜来源的完成规则。"""
        return self._items[source].source.needs_reply(reader)

    def entries(self) -> tuple[Source, ...]:
        return tuple(
            registration.source
            for registration in self._items.values()
            if registration.visible
        )

    async def changes(self) -> AsyncGenerator[tuple[Source, ...], None]:
        """订阅登记的 active 快照，不保存 cursor，也不执行业务回调。"""
        event = asyncio.Event()
        self._changes.add(event)
        previous: tuple[Source, ...] | None = None
        try:
            while True:
                event.clear()
                current = self.entries()
                if previous is None or len(current) != len(previous) or any(
                    left is not right for left, right in zip(current, previous)
                ):
                    previous = current
                    yield current
                    continue
                await event.wait()
        finally:
            self._changes.discard(event)

    async def accept(self, session_id: str, message_id: str, message: ChannelInboundMessage) -> Message:
        """专属渠道先匹配；没有专属注册时交给来源声明的默认输入。"""
        async with self._context.runtime_scope():
            selected: Source | None = None
            default: Source | None = None
            for registration in self._items.values():
                source = registration.source
                if source.channels is None:
                    default = source
                elif message.channel in source.channels:
                    selected = source
                    break
            if selected is None:
                selected = default
            if selected is None:
                raise ValueError("输入渠道没有来源: " + message.channel)
            assert selected.accept is not None
            async with selected.context.runtime_scope():
                return await selected.accept(session_id, message_id, message)


async def apply(ctx: Context) -> None:
    sources = Sources(ctx)
    _ = await ctx.provide(SOURCE_CHECK, check_source)
    _ = await ctx.provide(SOURCE_SESSION, _SourceSession)
    _ = await ctx.provide(SOURCES, sources)
    _ = await ctx.provide(CHANNEL_INPUT, sources.accept)
