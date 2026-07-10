from __future__ import annotations

import pytest

from bootstrap.channel_host import ChannelHost


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
        self._events.append(f"start:{self.name}:{ctx}")
        if self._fail_start:
            raise RuntimeError("start failed")

    async def stop(self) -> None:
        self._events.append(f"stop:{self.name}")
        if self._fail_stop:
            raise RuntimeError("stop failed")


@pytest.mark.asyncio
async def test_channel_host_start_failure_does_not_block_others():
    events: list[str] = []
    host = ChannelHost(lambda channel: f"ctx:{channel.name}")  # type: ignore[arg-type]
    host.add(_Channel("a", events))  # type: ignore[arg-type]
    host.add(_Channel("b", events, fail_start=True))  # type: ignore[arg-type]
    host.add(_Channel("c", events))  # type: ignore[arg-type]

    await host.start_all()

    assert events == [
        "start:a:ctx:a",
        "start:b:ctx:b",
        "start:c:ctx:c",
    ]


@pytest.mark.asyncio
async def test_channel_host_stops_in_reverse_order():
    events: list[str] = []
    host = ChannelHost(lambda channel: f"ctx:{channel.name}")  # type: ignore[arg-type]
    host.add(_Channel("a", events))  # type: ignore[arg-type]
    host.add(_Channel("b", events, fail_stop=True))  # type: ignore[arg-type]
    host.add(_Channel("c", events))  # type: ignore[arg-type]

    await host.stop_all()

    assert events == ["stop:c", "stop:b", "stop:a"]


@pytest.mark.asyncio
async def test_channel_host_swaps_plugin_generation():
    events: list[str] = []
    host = ChannelHost(lambda channel: f"ctx:{channel.name}")  # type: ignore[arg-type]
    old = _Channel("old", events)
    new = _Channel("new", events)
    host.add(old)  # type: ignore[arg-type]
    host.bind_plugin_channels({"chat": (old,)})  # type: ignore[arg-type]

    await host.swap_plugin_channels("chat", (old,), (new,))  # type: ignore[arg-type]

    assert host.channels == [new]
    assert events == ["stop:old", "start:new:ctx:new"]


@pytest.mark.asyncio
async def test_channel_host_restores_old_generation_when_start_fails():
    events: list[str] = []
    host = ChannelHost(lambda channel: f"ctx:{channel.name}")  # type: ignore[arg-type]
    old = _Channel("old", events)
    failed = _Channel("new", events, fail_start=True)
    host.add(old)  # type: ignore[arg-type]
    host.bind_plugin_channels({"chat": (old,)})  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="start failed"):
        await host.swap_plugin_channels("chat", (old,), (failed,))  # type: ignore[arg-type]

    assert host.channels == [old]
    assert events == [
        "stop:old",
        "start:new:ctx:new",
        "stop:new",
        "start:old:ctx:old",
    ]
