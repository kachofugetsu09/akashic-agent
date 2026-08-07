from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable

from infra.channels.contract import Channel, ChannelContext

logger = logging.getLogger(__name__)


@dataclass
class ChannelSwap:
    plugin_id: str
    old_channels: tuple[Channel, ...]
    new_channels: tuple[Channel, ...]
    old_positions: tuple[int, ...]
    stopped_old: list[Channel] = field(default_factory=list)
    started_new: list[Channel] = field(default_factory=list)


class ChannelHost:
    def __init__(
        self,
        ctx_factory: Callable[[Channel], ChannelContext],
    ) -> None:
        self._ctx_factory = ctx_factory
        self._channels: list[Channel] = []
        self._plugin_channels: dict[str, tuple[Channel, ...]] = {}
        self._resources: dict[int, _ChannelResources] = {}
        self._started: set[int] = set()

    def add(self, channel: Channel) -> None:
        self._channels.append(channel)

    async def start_all(self) -> None:
        failures: list[str] = []
        try:
            for channel in self._channels:
                try:
                    await self._start_channel(channel)
                    print(f"渠道已启动: {channel.name}")
                except Exception as e:
                    logger.error("渠道启动失败 %s: %s", channel.name, e)
                    failures.append(f"{channel.name}: {e}")
        except asyncio.CancelledError:
            await self.stop_all()
            raise
        if failures:
            raise RuntimeError("渠道启动失败: " + "; ".join(failures))

    async def stop_all(self) -> None:
        cancellation: asyncio.CancelledError | None = None
        for channel in reversed(self._channels):
            try:
                if id(channel) in self._started:
                    await self._stop_channel(channel)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except Exception as e:
                logger.warning("渠道停止失败 %s: %s", channel.name, e)
        if cancellation is not None:
            raise cancellation

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
        swap = self.prepare_plugin_swap(plugin_id, old_channels, new_channels)
        await self.stop_plugin_swap(swap)
        try:
            await self.start_plugin_swap(swap)
        except BaseException:
            await self.restore_plugin_swap(swap)
            raise
        self.commit_plugin_swap(swap)

    def prepare_plugin_swap(
        self,
        plugin_id: str,
        old_channels: tuple[Channel, ...],
        new_channels: tuple[Channel, ...],
    ) -> ChannelSwap:
        current = self._plugin_channels.get(plugin_id, ())
        if current != old_channels:
            raise RuntimeError(f"插件 Channel 代际不一致: {plugin_id}")
        retained_names = {
            channel.name for channel in self._channels if channel not in old_channels
        }
        new_names = [channel.name for channel in new_channels]
        if len(new_names) != len(set(new_names)) or retained_names.intersection(new_names):
            raise RuntimeError(f"Channel 名称冲突: {', '.join(new_names)}")
        old_positions = tuple(
            self._channels.index(channel)
            for channel in old_channels
            if channel in self._channels
        )
        return ChannelSwap(plugin_id, old_channels, new_channels, old_positions)

    async def stop_plugin_swap(self, swap: ChannelSwap) -> None:
        try:
            for channel in reversed(swap.old_channels):
                await self._stop_channel(channel)
                swap.stopped_old.append(channel)
        except BaseException as stop_error:
            restore_errors = await self._restore_channels(reversed(swap.stopped_old))
            swap.stopped_old.clear()
            if restore_errors:
                raise RuntimeError(
                    "旧 Channel 恢复失败: " + "; ".join(restore_errors)
                ) from stop_error
            raise stop_error

    async def start_plugin_swap(self, swap: ChannelSwap) -> None:
        try:
            for channel in swap.new_channels:
                swap.started_new.append(channel)
                await self._start_channel(channel)
        except BaseException as start_error:
            for channel in reversed(swap.started_new):
                try:
                    if id(channel) in self._started:
                        await self._stop_channel(channel)
                except Exception:
                    logger.exception("候选 Channel 清理失败: %s", channel.name)
            swap.started_new.clear()
            raise start_error

    async def restore_plugin_swap(self, swap: ChannelSwap) -> None:
        for channel in reversed(swap.started_new):
            if id(channel) in self._started:
                await self._stop_channel(channel)
        swap.started_new.clear()
        restore_errors = await self._restore_channels(reversed(swap.stopped_old))
        swap.stopped_old.clear()
        if restore_errors:
            raise RuntimeError("旧 Channel 恢复失败: " + "; ".join(restore_errors))

    def commit_plugin_swap(self, swap: ChannelSwap) -> None:
        for channel in swap.old_channels:
            if channel in self._channels:
                self._channels.remove(channel)
        insert_at = min(swap.old_positions, default=len(self._channels))
        for offset, channel in enumerate(swap.new_channels):
            self._channels.insert(insert_at + offset, channel)
        self._plugin_channels[swap.plugin_id] = swap.new_channels

    async def _restore_channels(self, channels: Iterable[Channel]) -> list[str]:
        errors: list[str] = []
        for channel in channels:
            try:
                await self._start_channel(channel)
            except Exception as error:
                errors.append(f"{channel.name}: {error}")
        return errors

    async def _start_channel(self, channel: Channel) -> None:
        resources = _ChannelResources(self._ctx_factory(channel))
        self._resources[id(channel)] = resources
        try:
            await channel.start(resources.context)
        except BaseException:
            try:
                await channel.stop()
            except (asyncio.CancelledError, Exception):
                logger.exception("Channel 部分启动清理失败: %s", channel.name)
            finally:
                try:
                    resources.close()
                finally:
                    _ = self._resources.pop(id(channel), None)
            raise
        self._started.add(id(channel))

    async def _stop_channel(self, channel: Channel) -> None:
        try:
            await channel.stop()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            raise RuntimeError(
                "Channel stop 未确认 ingress 收束和 ownership 释放: "
                f"channel={channel.name}: {error}"
            ) from error
        finally:
            resources = self._resources.pop(id(channel), None)
            try:
                if resources is not None:
                    resources.close()
            finally:
                self._started.discard(id(channel))

    @property
    def channels(self) -> list[Channel]:
        return list(self._channels)


class _ChannelResources:
    def __init__(self, context: ChannelContext) -> None:
        self._closeables: list[object] = []
        self.context = ChannelContext(
            bus=_ScopedBus(context.bus, self._closeables),  # type: ignore[arg-type]
            session_manager=context.session_manager,
            event_bus=_ScopedEventBus(context.event_bus, self._closeables),  # type: ignore[arg-type]
            push_tool=_ScopedPushTool(context.push_tool, self._closeables),  # type: ignore[arg-type]
            attachment_store=context.attachment_store,
            http_resources=context.http_resources,
            interrupt_controller=context.interrupt_controller,
            mobile_bot_commands=context.mobile_bot_commands,
            log=context.log,
        )

    def close(self) -> None:
        first_error: Exception | None = None
        for closeable in reversed(self._closeables):
            close = getattr(closeable, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        self._closeables.clear()
        if first_error is not None:
            raise first_error


class _ScopedBus:
    def __init__(self, target: object, closeables: list[object]) -> None:
        self._target = target
        self._closeables = closeables

    def subscribe_outbound(self, channel: str, callback: object) -> object:
        subscription = self._target.subscribe_outbound(channel, callback)  # type: ignore[attr-defined]
        self._closeables.append(subscription)
        return subscription

    def __getattr__(self, name: str) -> object:
        return getattr(self._target, name)


class _ScopedEventBus:
    def __init__(self, target: object, closeables: list[object]) -> None:
        self._target = target
        self._closeables = closeables

    def on(self, event_type: type[object], handler: object) -> object:
        subscription = self._target.on(event_type, handler)  # type: ignore[attr-defined]
        self._closeables.append(subscription)
        return subscription

    def __getattr__(self, name: str) -> object:
        return getattr(self._target, name)


class _ScopedPushTool:
    def __init__(self, target: object, closeables: list[object]) -> None:
        self._target = target
        self._closeables = closeables

    def register_channel(self, channel: str, **senders: object) -> object:
        registration = self._target.register_channel(channel, **senders)  # type: ignore[attr-defined]
        self._closeables.append(registration)
        return registration

    def __getattr__(self, name: str) -> object:
        return getattr(self._target, name)
