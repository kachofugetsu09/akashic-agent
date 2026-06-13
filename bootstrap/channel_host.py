from __future__ import annotations

import logging
from collections.abc import Callable

from infra.channels.contract import Channel, ChannelContext

logger = logging.getLogger(__name__)


class ChannelHost:
    def __init__(
        self,
        ctx_factory: Callable[[Channel], ChannelContext],
    ) -> None:
        self._ctx_factory = ctx_factory
        self._channels: list[Channel] = []

    def add(self, channel: Channel) -> None:
        self._channels.append(channel)

    async def start_all(self) -> None:
        for channel in self._channels:
            try:
                await channel.start(self._ctx_factory(channel))
                print(f"渠道已启动: {channel.name}")
            except Exception as e:
                logger.error("渠道启动失败 %s: %s", channel.name, e)

    async def stop_all(self) -> None:
        for channel in reversed(self._channels):
            try:
                await channel.stop()
            except Exception as e:
                logger.warning("渠道停止失败 %s: %s", channel.name, e)

    @property
    def channels(self) -> list[Channel]:
        return list(self._channels)
