from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition.effect import Effect
from plugins.conversation.source import Changed, Conversation
from session.message import Message

api_version = 3
name = "sources"
version = "1.0.0"
desc = "按来源注册接纳与控制，供渠道和默认回复组合使用"
inject = ()

Accept = Callable[[str, str, ChannelInboundMessage], Awaitable[Message]]


@dataclass(frozen=True)
class Source:
    name: str
    open: Callable[[str], Conversation]
    accept: Accept | None = None
    channels: tuple[str, ...] | None = ()


class Sources:
    """每个来源和输入渠道只绑定一个 owner；不拥有 Message 或活动 Task。"""

    def __init__(self):
        self._items: dict[str, Source] = {}

    async def register(self, ctx: Context, source: Source) -> Effect:
        """来源注册随插件 effect 生灭；None channels 表示唯一默认输入来源。"""
        if ctx.require(SOURCES) is not self:
            raise PermissionError("来源注册不属于当前组合")
        if not source.name or source.accept is None and source.channels != () or (
            source.channels is not None and any(not channel for channel in source.channels)
        ):
            raise ValueError("来源名称和渠道必须明确")

        def setup():
            if source.name in self._items:
                raise ValueError("来源已有 owner: " + source.name)
            for existing in self._items.values():
                if source.channels is None and existing.channels is None or (
                    source.channels is not None and existing.channels is not None
                    and set(source.channels).intersection(existing.channels)
                ):
                    raise ValueError("输入渠道已有来源 owner")
            self._items[source.name] = source

            def close() -> None:
                del self._items[source.name]

            return close

        return await ctx.effect(setup, label="source:" + source.name)

    def entries(self) -> tuple[Source, ...]:
        return tuple(self._items.values())

    async def accept(self, session_id: str, message_id: str, message: ChannelInboundMessage) -> Message:
        """专属渠道先匹配；没有专属注册时交给来源声明的默认输入。"""
        default: Source | None = None
        for source in self._items.values():
            if source.channels is None:
                default = source
            elif message.channel in source.channels:
                assert source.accept is not None
                return await source.accept(session_id, message_id, message)
        if default is None:
            raise ValueError("输入渠道没有来源: " + message.channel)
        assert default.accept is not None
        return await default.accept(session_id, message_id, message)


SOURCES = ServiceKey[Sources]("sources.v1")
SOURCE_CHANGED = ServiceKey[Changed]("source.changed.v1")


async def apply(ctx: Context, config: object) -> None:
    sources = Sources()
    _ = await ctx.provide(SOURCES, sources)
    _ = await ctx.provide(CHANNEL_INPUT, sources.accept)
