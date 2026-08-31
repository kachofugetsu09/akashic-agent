from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import cast

from infra.channels.contract import Channel, ChannelContext

logger = logging.getLogger(__name__)

class ChannelHost:
    def __init__(
        self,
        ctx_factory: Callable[[Channel], ChannelContext],
    ) -> None:
        self._ctx_factory = ctx_factory
        self._channels: list[Channel] = []
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

    async def swap_command_catalog(
        self,
        old_commands: tuple[tuple[str, str], ...],
        new_commands: tuple[tuple[str, str], ...],
    ) -> None:
        """Publish discovery metadata and restore the old catalog on failure."""

        # 1. Only started adapters that own an external command catalog participate.
        attempted: list[Channel] = []
        try:
            for channel in self._channels:
                replace_catalog = cast(
                    Callable[
                        [tuple[tuple[str, str], ...]],
                        Awaitable[None],
                    ]
                    | None,
                    getattr(channel, "replace_command_catalog", None),
                )
                if id(channel) not in self._started or not callable(replace_catalog):
                    continue
                attempted.append(channel)
                await replace_catalog(new_commands)
        except BaseException as error:
            # 2. A failed remote call may have applied before reporting failure.
            restore_errors: list[str] = []
            for channel in reversed(attempted):
                replace_catalog = cast(
                    Callable[
                        [tuple[tuple[str, str], ...]],
                        Awaitable[None],
                    ]
                    | None,
                    getattr(channel, "replace_command_catalog", None),
                )
                assert callable(replace_catalog)
                try:
                    await replace_catalog(old_commands)
                except BaseException as restore_error:
                    restore_errors.append(f"{channel.name}: {restore_error}")
            if restore_errors:
                raise RuntimeError(
                    "旧命令目录恢复失败: " + "; ".join(restore_errors)
                ) from error
            raise

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
            bus=context.bus,
            session_manager=context.session_manager,
            event_bus=_ScopedEventBus(context.event_bus, self._closeables),  # type: ignore[arg-type]
            push_tool=context.push_tool,
            attachment_store=context.attachment_store,
            http_resources=context.http_resources,
            interrupt_controller=context.interrupt_controller,
            log=context.log,
            command_catalog_provider=context.command_catalog_provider,
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
