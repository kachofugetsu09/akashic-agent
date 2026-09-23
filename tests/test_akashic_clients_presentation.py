from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import WebSocket
from starlette.websockets import WebSocketState

from agent.plugin_composition.channels import ProviderClientFactory
from agent.plugin_composition.context import Context
from agent.plugin_composition.requests import RequestContext
from agent.plugin_composition.channels import (
    ChannelCapability,
    ChannelDefinition,
    ChannelPresentationPorts,
    DeliveryStatus,
    StreamDeltaPresentation,
    TurnOutputCompletedPresentation,
    TurnStartedPresentation,
    TurnStreamEvent,
    TurnStreamEventKind,
)
from plugins.akashic_clients import plugin
from plugins.akashic_clients.channel import (
    build_akashic_channel_factory,
)
from plugins.akashic_clients.config import AkashicClientsConfig
from plugins.akashic_clients.web_chat import WebChatChannel, WebNativeChannelAdapter


class _Socket:
    def __init__(self) -> None:
        self.application_state = WebSocketState.CONNECTED
        self.frames: list[dict[str, Any]] = []

    async def send_json(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    async def close(self, **_: Any) -> None:
        self.application_state = WebSocketState.DISCONNECTED


class _Subscription:
    def __init__(self) -> None:
        self.callback: Any = None
        self.admission_closed = False
        self.closed = False

    def close_admission(self) -> None:
        self.admission_closed = True

    async def await_quiescence(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class _TurnStream:
    def __init__(self) -> None:
        self.subscription = _Subscription()

    def subscribe(self, callback: Any) -> _Subscription:
        self.subscription.callback = callback
        return self.subscription


class _MessageScope:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.entered = 0
        self.exited = 0

    @asynccontextmanager
    async def __call__(self):
        self.entered += 1
        try:
            yield self.value
        finally:
            self.exited += 1


class _FollowingReader:
    session_id = "akashic:follow"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def head(self) -> int:
        return 0

    async def follow(self, *, after_seq: int = -1):
        _ = after_seq
        self.started.set()
        await self.release.wait()
        if False:
            yield None


class _FollowingCatalog:
    def __init__(self, reader: _FollowingReader) -> None:
        self.reader_value = reader

    def reader(self, _session_id: str) -> _FollowingReader:
        return self.reader_value


def test_client_plugin_declares_independent_capabilities() -> None:
    names = tuple(key.name for key in plugin.inject)

    assert "plugin.channels" in names
    assert "channel.input.v1" in names
    assert "core.message_catalog" in names
    assert "reply.status.v2" in names
    assert "models.selection.v1" in names
    assert "akashic.clients.services.v1" not in names


class _ChannelRegistry:
    def __init__(self) -> None:
        self.definition: ChannelDefinition | None = None

    async def register(self, _ctx: Any, definition: ChannelDefinition) -> None:
        self.definition = definition


class _ApplyContext:
    generation_id = "generation-1"
    runtime = SimpleNamespace(
        generation_id=generation_id,
        workspace=Path("/tmp/akashic-clients-test"),
    )

    def __init__(self, registry: _ChannelRegistry) -> None:
        self.config = {}
        self.registry = registry

    def require(self, key: Any) -> Any:
        _ = key
        return self.registry

    async def effect(self, setup: Any, *, label: str) -> Any:
        _ = (setup, label)
        return None


@pytest.mark.asyncio
async def test_apply_registers_one_formal_channel_definition() -> None:
    registry = _ChannelRegistry()
    await plugin.apply(cast(Context, _ApplyContext(registry)))

    assert registry.definition is not None
    assert registry.definition.name == "akashic"
    assert callable(registry.definition.factory)
    assert registry.definition.capabilities == frozenset(
        {
            ChannelCapability.INBOUND,
            ChannelCapability.DURABLE_INBOUND,
            ChannelCapability.OUTBOUND,
            ChannelCapability.TURN_STREAM,
        }
    )


@pytest.mark.asyncio
async def test_reply_status_sequence_keeps_channel_snapshot_identity(tmp_path: Path) -> None:
    """Active reply status must name the snapshot that owns the scoped reader."""

    reader_started = asyncio.Event()
    release_reader = asyncio.Event()
    reader_closed = asyncio.Event()
    scope_exited = asyncio.Event()

    class ReplyRead:
        async def follow(self, _session_id: str):
            try:
                assert scope_exited.is_set()
                reader_started.set()
                await release_reader.wait()
                yield ()
            finally:
                reader_closed.set()

    class ReplyScope:
        def require(self, _key: Any) -> ReplyRead:
            return ReplyRead()

    @asynccontextmanager
    async def open_scope():
        try:
            yield cast(RequestContext, ReplyScope())
        finally:
            scope_exited.set()

    generation = "reply-status-generation"
    build_akashic_channel = build_akashic_channel_factory(AkashicClientsConfig(), tmp_path)
    adapter = None
    try:
        from agent.plugin_composition.channels import ChannelFactoryContext

        context = ChannelFactoryContext(
            snapshot_id="reply-status-snapshot",
            generation_id=generation,
            boot_id="boot-1",
            binding_token="binding-1",
            config={},
            ingress=None,
            identity=None,
            open_scope=open_scope,
        )
        adapter = build_akashic_channel(context)
        stream = adapter._follow_reply_status("akashic:session-1")
        frame_task = asyncio.create_task(anext(stream))
        try:
            await reader_started.wait()
            assert scope_exited.is_set()
            release_reader.set()
            assert await frame_task == {
                "version": 2,
                "session_id": "akashic:session-1",
                "snapshot_id": "reply-status-snapshot",
                "available": True,
                "items": [],
            }
        finally:
            release_reader.set()
            if not frame_task.done():
                frame_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await frame_task
            await stream.aclose()
            await reader_closed.wait()
    finally:
        if adapter is not None:
            await adapter.stop()


@pytest.mark.asyncio
async def test_reply_status_reader_failure_is_not_hidden(tmp_path: Path) -> None:
    """A real client reply follower preserves an underlying reader failure."""

    class ReplyRead:
        async def follow(self, _session_id: str):
            raise RuntimeError("reply reader failed")
            yield {}

    class ReplyScope:
        def require(self, _key: Any) -> ReplyRead:
            return ReplyRead()

    @asynccontextmanager
    async def open_scope():
        yield cast(RequestContext, ReplyScope())

    from agent.plugin_composition.channels import ChannelFactoryContext

    factory = build_akashic_channel_factory(AkashicClientsConfig(), tmp_path)
    adapter = factory(
        ChannelFactoryContext(
            snapshot_id="reply-error-snapshot",
            generation_id="reply-error-generation",
            boot_id="boot-1",
            binding_token="reply-error-binding",
            config={},
            ingress=None,
            identity=None,
            open_scope=open_scope,
        )
    )
    stream = adapter._follow_reply_status("akashic:reply-error")
    try:
        with pytest.raises(RuntimeError, match="reply reader failed"):
            await anext(stream)
    finally:
        await stream.aclose()
        await adapter.stop()


def test_each_apply_owns_its_adapter_until_stop(tmp_path: Path) -> None:
    from agent.plugin_composition.channels import ChannelFactoryContext

    context = ChannelFactoryContext(
        snapshot_id="snapshot", generation_id="generation", boot_id="boot",
        binding_token="binding", config={}, ingress=None, identity=None,
    )
    factory = build_akashic_channel_factory(AkashicClientsConfig(), tmp_path)
    first = factory(context)
    with pytest.raises(RuntimeError, match="尚未释放"):
        factory(context)
    second_factory = build_akashic_channel_factory(AkashicClientsConfig(), tmp_path)
    second = second_factory(context)
    assert second is not first
    asyncio.run(first.stop())
    asyncio.run(second.stop())



@pytest.mark.asyncio
async def test_web_presents_formal_turn_stream_by_inbound_message_id() -> None:
    channel = WebChatChannel()
    socket = _Socket()
    session_key = "akashic:chat-1"
    assert await channel._add_connection(session_key, cast(WebSocket, socket)) is True
    channel._client_sessions["client-1"] = session_key
    stream = _TurnStream()
    channel.attach_presentation(ChannelPresentationPorts(control=None, turn_stream=stream))

    started = await stream.subscription.callback(
        TurnStreamEvent(
            presentation_id="presentation-1",
            kind=TurnStreamEventKind.TURN_STARTED,
            payload=TurnStartedPresentation(
                turn_id="turn-1",
                client_message_id="client-1",
            ),
        )
    )
    delta = await stream.subscription.callback(
        TurnStreamEvent(
            presentation_id="presentation-2",
            kind=TurnStreamEventKind.STREAM_DELTA,
            payload=StreamDeltaPresentation(
                turn_id="turn-1",
                sequence=1,
                text_delta="answer",
                reasoning_delta="",
            ),
        )
    )
    completed = await stream.subscription.callback(
        TurnStreamEvent(
            presentation_id="presentation-3",
            kind=TurnStreamEventKind.TURN_OUTPUT_COMPLETED,
            payload=TurnOutputCompletedPresentation(turn_id="turn-1", sequence=2),
        )
    )

    assert started.status is DeliveryStatus.DELIVERED
    assert delta.status is DeliveryStatus.DELIVERED
    assert completed.status is DeliveryStatus.DELIVERED
    assert [frame["type"] for frame in socket.frames] == [
        "turn.started",
        "answer.delta",
        "turn.output.completed",
    ]
    assert socket.frames[1]["turn_id"] == "turn-1"

    await channel.stop()
    assert stream.subscription.admission_closed is True
    assert stream.subscription.closed is True


@pytest.mark.asyncio
async def test_web_rejects_formal_turn_without_known_inbound_mapping() -> None:
    channel = WebChatChannel()
    stream = _TurnStream()
    channel.attach_presentation(ChannelPresentationPorts(control=None, turn_stream=stream))

    receipt = await stream.subscription.callback(
        TurnStreamEvent(
            presentation_id="presentation-unknown",
            kind=TurnStreamEventKind.TURN_STARTED,
            payload=TurnStartedPresentation(
                turn_id="turn-unknown",
                client_message_id="missing-client",
            ),
        )
    )

    assert receipt.status is DeliveryStatus.FAILED
    assert "session 映射" in (receipt.error or "")


@pytest.mark.asyncio
async def test_web_message_catalog_is_held_only_inside_request_scope() -> None:
    channel = WebChatChannel()
    catalog = object()
    scope = _MessageScope(catalog)
    channel.bind_message_scope(scope)

    async with channel._open_message_catalog() as resolved:
        assert resolved is catalog
        assert scope.entered == 1
        assert scope.exited == 0

    assert scope.entered == 1
    assert scope.exited == 1


@pytest.mark.asyncio
async def test_web_admission_close_cancels_follow_and_releases_scope() -> None:
    channel = WebChatChannel()
    reader = _FollowingReader()
    scope = _MessageScope(_FollowingCatalog(reader))
    channel.bind_message_scope(scope)
    socket = _Socket()
    task = asyncio.create_task(channel._follow(cast(WebSocket, socket), reader.session_id, -1))
    await reader.started.wait()
    assert scope.entered == scope.exited == 1
    channel._followers[cast(WebSocket, socket)] = (reader.session_id, task)

    adapter = cast(WebNativeChannelAdapter, SimpleNamespace(binding_token="binding"))
    channel._v3_adapters["binding"] = adapter
    channel._close_v3_binding(adapter)
    result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)

    assert task.cancelled()
    assert isinstance(result[0], asyncio.CancelledError)
    assert scope.exited == 1
