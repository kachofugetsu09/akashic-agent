from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.websockets import WebSocketState

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
from plugins.akashic_clients.config import AkashicClientsConfig
from plugins.akashic_clients.web_chat import WebChatChannel


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


def test_client_plugin_declares_independent_capabilities() -> None:
    names = tuple(key.name for key in plugin.inject)

    assert "core.channels" in names
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
    await plugin.apply(_ApplyContext(registry), AkashicClientsConfig())

    assert registry.definition is not None
    assert registry.definition.name == "akashic"
    assert registry.definition.factory_export == "build_akashic_channel"
    assert registry.definition.capabilities == frozenset(
        {
            ChannelCapability.INBOUND,
            ChannelCapability.OUTBOUND,
            ChannelCapability.TURN_STREAM,
        }
    )
    plugin.unregister_generation("generation-1")


@pytest.mark.asyncio
async def test_web_presents_formal_turn_stream_by_inbound_message_id() -> None:
    channel = WebChatChannel()
    socket = _Socket()
    session_key = "akashic:chat-1"
    assert await channel._add_connection(session_key, socket) is True
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
