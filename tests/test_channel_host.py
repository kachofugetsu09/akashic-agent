from __future__ import annotations

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
        async def on_outbound(_message: object) -> None:
            return None

        async def send_text(_chat_id: str, _message: str) -> None:
            return None

        ctx.event_bus.on(_Event, lambda event: event)
        ctx.bus.subscribe_outbound(self.name, on_outbound)
        ctx.push_tool.register_channel(self.name, text=send_text)
        if self._fail_start:
            raise RuntimeError("registered start failed")

    async def stop(self) -> None:
        return None


def _context(channel: _Channel) -> SimpleNamespace:
    return SimpleNamespace(
        bus=SimpleNamespace(),
        session_manager=None,
        event_bus=SimpleNamespace(),
        push_tool=SimpleNamespace(),
        attachment_store=None,
        http_resources=None,
        interrupt_controller=None,
        bot_commands=[],
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
async def test_channel_host_swaps_plugin_generation():
    events: list[str] = []
    host = ChannelHost(_context)  # type: ignore[arg-type]
    old = _Channel("old", events)
    new = _Channel("new", events)
    host.add(old)  # type: ignore[arg-type]
    host.bind_plugin_channels({"chat": (old,)})  # type: ignore[arg-type]
    await host.start_all()
    events.clear()

    await host.swap_plugin_channels("chat", (old,), (new,))  # type: ignore[arg-type]

    assert host.channels == [new]
    assert events == ["stop:old", "start:new:ctx:new"]


@pytest.mark.asyncio
async def test_channel_host_restores_old_generation_when_start_fails():
    events: list[str] = []
    host = ChannelHost(_context)  # type: ignore[arg-type]
    old = _Channel("old", events)
    failed = _Channel("new", events, fail_start=True, fail_stop=True)
    host.add(old)  # type: ignore[arg-type]
    host.bind_plugin_channels({"chat": (old,)})  # type: ignore[arg-type]
    await host.start_all()
    events.clear()

    with pytest.raises(RuntimeError, match="start failed"):
        await host.swap_plugin_channels("chat", (old,), (failed,))  # type: ignore[arg-type]

    assert host.channels == [old]
    assert events == [
        "stop:old",
        "start:new:ctx:new",
        "stop:new",
        "start:old:ctx:old",
    ]


@pytest.mark.asyncio
async def test_channel_host_keeps_failed_stop_channel_and_restores_stopped_ones():
    events: list[str] = []
    host = ChannelHost(_context)  # type: ignore[arg-type]
    first = _Channel("first", events, fail_stop=True)
    second = _Channel("second", events)
    replacement = _Channel("replacement", events)
    host.add(first)  # type: ignore[arg-type]
    host.add(second)  # type: ignore[arg-type]
    host.bind_plugin_channels({"chat": (first, second)})  # type: ignore[arg-type]
    await host.start_all()
    events.clear()

    with pytest.raises(RuntimeError, match="stop failed"):
        await host.swap_plugin_channels(  # type: ignore[arg-type]
            "chat",
            (first, second),
            (replacement,),
        )

    assert host.channels == [first, second]
    assert events == [
        "stop:second",
        "stop:first",
        "start:second:ctx:second",
    ]


@pytest.mark.asyncio
async def test_channel_host_rejects_name_conflict_before_stopping_old():
    events: list[str] = []
    host = ChannelHost(_context)  # type: ignore[arg-type]
    core = _Channel("core", events)
    old = _Channel("old", events)
    conflict = _Channel("core", events)
    host.add(core)  # type: ignore[arg-type]
    host.add(old)  # type: ignore[arg-type]
    host.bind_plugin_channels({"chat": (old,)})  # type: ignore[arg-type]
    await host.start_all()
    events.clear()

    with pytest.raises(RuntimeError, match="名称冲突"):
        await host.swap_plugin_channels(  # type: ignore[arg-type]
            "chat",
            (old,),
            (conflict,),
        )

    assert events == []


@pytest.mark.asyncio
async def test_channel_host_revokes_shared_registrations_on_stop():
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
        bot_commands=[],
        log="ctx:registered",
    )
    host = ChannelHost(lambda _channel: context)  # type: ignore[arg-type]
    host.add(channel)  # type: ignore[arg-type]

    await host.start_all()

    assert event_bus.handler_count() == 1
    assert "registered" in message_bus._subscribers
    assert await push_tool.execute(
        channel="registered",
        chat_id="1",
        message="hello",
    ) == "文本已发送"

    await host.stop_all()

    assert event_bus.handler_count() == 0
    assert "registered" not in message_bus._subscribers
    assert "未注册" in await push_tool.execute(
        channel="registered",
        chat_id="1",
        message="hello",
    )


@pytest.mark.asyncio
async def test_channel_host_restores_shared_registrations_after_failed_swap():
    event_bus = EventBus()
    message_bus = MessageBus()
    push_tool = MessagePushTool()
    old = _RegisteredChannel()
    failed = _RegisteredChannel(fail_start=True)
    context = SimpleNamespace(
        bus=message_bus,
        session_manager=None,
        event_bus=event_bus,
        push_tool=push_tool,
        attachment_store=None,
        http_resources=None,
        interrupt_controller=None,
        bot_commands=[],
        log="ctx:registered",
    )
    host = ChannelHost(lambda _channel: context)  # type: ignore[arg-type]
    host.add(old)  # type: ignore[arg-type]
    host.bind_plugin_channels({"chat": (old,)})  # type: ignore[arg-type]
    await host.start_all()

    with pytest.raises(RuntimeError, match="registered start failed"):
        await host.swap_plugin_channels(  # type: ignore[arg-type]
            "chat",
            (old,),
            (failed,),
        )

    assert event_bus.handler_count() == 1
    assert len(message_bus._subscribers["registered"]) == 1
    assert await push_tool.execute(
        channel="registered",
        chat_id="1",
        message="hello",
    ) == "文本已发送"
