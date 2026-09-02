from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.tools.message_push import MessagePushTool
from bootstrap.channel_host import ChannelHost
from bus.event_bus import EventBus
from bus.queue import MessageBus


class _Channel:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_start: bool = False,
        fail_stop: bool = False,
    ) -> None:
        self.name = name
        self._events = events
        self._fail_start = fail_start
        self._fail_stop = fail_stop

    async def start(self, ctx: object) -> None:
        self._events.append(f"start:{self.name}:{ctx.log}")
        if self._fail_start:
            raise RuntimeError("start failed")

    async def stop(self) -> None:
        self._events.append(f"stop:{self.name}")
        if self._fail_stop:
            raise RuntimeError("stop failed")


class _Event:
    pass


class _RegisteredChannel:
    def __init__(self, *, fail_start: bool = False) -> None:
        self.name = "registered"
        self._fail_start = fail_start

    async def start(self, ctx: object) -> None:
        ctx.event_bus.on(_Event, lambda event: event)
        if self._fail_start:
            raise RuntimeError("registered start failed")

    async def stop(self) -> None:
        return None


class _CommandCatalogChannel(_Channel):
    def __init__(self, events: list[str]) -> None:
        super().__init__("catalog", events)
        self.catalog: tuple[tuple[str, str], ...] = (("old", "old"),)
        self.fail_next = False

    async def replace_command_catalog(
        self,
        commands: tuple[tuple[str, str], ...],
    ) -> None:
        self._events.append(f"commands:{commands[0][0] if commands else 'empty'}")
        self.catalog = commands
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("command publish failed")


def _context(channel: _Channel) -> SimpleNamespace:
    return SimpleNamespace(
        bus=SimpleNamespace(),
        session_manager=None,
        event_bus=SimpleNamespace(),
        push_tool=SimpleNamespace(),
        attachment_store=None,
        http_resources=None,
        interrupt_controller=None,
        command_catalog_provider=None,
        log=f"ctx:{channel.name}",
    )


@pytest.mark.asyncio
async def test_channel_host_start_failure_does_not_block_others():
    events: list[str] = []
    host = ChannelHost(_context)  # type: ignore[arg-type]
    host.add(_Channel("a", events))  # type: ignore[arg-type]
    host.add(_Channel("b", events, fail_start=True))  # type: ignore[arg-type]
    host.add(_Channel("c", events))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="start failed"):
        await host.start_all()

    assert events == [
        "start:a:ctx:a",
        "start:b:ctx:b",
        "stop:b",
        "start:c:ctx:c",
    ]


@pytest.mark.asyncio
async def test_channel_host_command_catalog_failure_restores_old_remote_state():
    events: list[str] = []
    channel = _CommandCatalogChannel(events)
    host = ChannelHost(_context)  # type: ignore[arg-type]
    host.add(channel)  # type: ignore[arg-type]
    await host.start_all()
    events.clear()
    channel.fail_next = True

    with pytest.raises(RuntimeError, match="command publish failed"):
        await host.swap_command_catalog(
            (("old", "old"),),
            (("new", "new"),),
        )

    assert channel.catalog == (("old", "old"),)
    assert events == ["commands:new", "commands:old"]
    await host.stop_all()


@pytest.mark.asyncio
async def test_channel_host_stops_in_reverse_order():
    events: list[str] = []
    host = ChannelHost(_context)  # type: ignore[arg-type]
    host.add(_Channel("a", events))  # type: ignore[arg-type]
    host.add(_Channel("b", events, fail_stop=True))  # type: ignore[arg-type]
    host.add(_Channel("c", events))  # type: ignore[arg-type]
    await host.start_all()
    events.clear()

    await host.stop_all()

    assert events == ["stop:c", "stop:b", "stop:a"]


@pytest.mark.asyncio
async def test_channel_host_continues_after_cancelled_stop():
    events: list[str] = []

    class _CancelledStopChannel(_Channel):
        async def stop(self) -> None:
            events.append(f"stop:{self.name}")
            raise asyncio.CancelledError

    host = ChannelHost(_context)  # type: ignore[arg-type]
    host.add(_CancelledStopChannel("cancel", events))  # type: ignore[arg-type]
    host.add(_Channel("other", events))  # type: ignore[arg-type]
    await host.start_all()

    with pytest.raises(asyncio.CancelledError):
        await host.stop_all()

    assert events[-2:] == ["stop:other", "stop:cancel"]


@pytest.mark.asyncio
async def test_channel_host_scopes_event_handlers_without_legacy_registrations():
    event_bus = EventBus()
    message_bus = MessageBus()
    push_tool = MessagePushTool()
    channel = _RegisteredChannel()
    context = SimpleNamespace(
        bus=message_bus,
        session_manager=None,
        event_bus=event_bus,
        push_tool=push_tool,
        attachment_store=None,
        http_resources=None,
        interrupt_controller=None,
        command_catalog_provider=None,
        log="ctx:registered",
    )
    host = ChannelHost(lambda _channel: context)  # type: ignore[arg-type]
    host.add(channel)  # type: ignore[arg-type]

    await host.start_all()

    assert event_bus.handler_count() == 1
    assert not hasattr(message_bus, "_subscribers")
    assert not hasattr(push_tool, "register_channel")

    await host.stop_all()

    assert event_bus.handler_count() == 0
