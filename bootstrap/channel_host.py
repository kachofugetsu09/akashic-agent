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
        self._plugin_channels: dict[str, tuple[Channel, ...]] = {}

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

    def bind_plugin_channels(
        self,
        channels: dict[str, tuple[Channel, ...]],
    ) -> None:
        self._plugin_channels = dict(channels)

    async def swap_plugin_channels(
        self,
        plugin_id: str,
        old_channels: tuple[Channel, ...],
        new_channels: tuple[Channel, ...],
    ) -> None:
        current = self._plugin_channels.get(plugin_id, ())
        if current != old_channels:
            raise RuntimeError(f"插件 Channel 代际不一致: {plugin_id}")
        old_positions = [
            self._channels.index(channel)
            for channel in old_channels
            if channel in self._channels
        ]
        stopped_old: list[Channel] = []
        try:
            for channel in reversed(old_channels):
                stopped_old.append(channel)
                await channel.stop()
        except BaseException:
            for channel in reversed(stopped_old):
                await channel.start(self._ctx_factory(channel))
            raise
        started: list[Channel] = []
        try:
            for channel in new_channels:
                started.append(channel)
                await channel.start(self._ctx_factory(channel))
        except BaseException:
            for channel in reversed(started):
                await channel.stop()
            for channel in old_channels:
                await channel.start(self._ctx_factory(channel))
            raise
        for channel in old_channels:
            if channel in self._channels:
                self._channels.remove(channel)
        insert_at = min(old_positions, default=len(self._channels))
        for offset, channel in enumerate(new_channels):
            self._channels.insert(insert_at + offset, channel)
        self._plugin_channels[plugin_id] = new_channels

    @property
    def channels(self) -> list[Channel]:
        return list(self._channels)
