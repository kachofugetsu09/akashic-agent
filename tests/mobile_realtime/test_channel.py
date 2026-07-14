from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from agent.config_models import MobileRealtimeConfig
from bus.events import OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from infra.mobile_realtime.channel import MobileRealtimeChannel
from infra.mobile_realtime.gateway import MobileGatewayRuntime
from infra.mobile_realtime.protocol import MessageSendCommand, parse_frame
from infra.mobile_realtime.storage import DeviceRecord, MobileRealtimeStorage
from session.manager import SessionManager


class _Runtime:
    def __init__(self, storage: MobileRealtimeStorage) -> None:
        self.storage = storage
        self.events: list[dict[str, object]] = []

    async def publish_event(self, **event: object) -> None:
        self.events.append(dict(event))


class _Bus:
    def __init__(self) -> None:
        self.inbound: list[object] = []
        self.outbound: dict[str, object] = {}

    async def publish_inbound(self, message: object) -> None:
        self.inbound.append(message)

    def subscribe_outbound(self, channel: str, callback: object) -> None:
        self.outbound[channel] = callback


class _EventBus:
    def __init__(self) -> None:
        self.handlers: dict[type[object], object] = {}

    def on(self, event_type: type[object], handler: object) -> None:
        self.handlers[event_type] = handler


class _PushTool:
    def __init__(self) -> None:
        self.registered: dict[str, dict[str, object]] = {}

    def register_channel(self, channel: str, **senders: object) -> None:
        self.registered[channel] = senders


def _register_device(storage: MobileRealtimeStorage, device_id: str) -> None:
    storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key="test-public-key",
            display_name=device_id,
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )


def _message_frame(
    *,
    frame_id: str,
    session_id: str,
    epoch: int = 1,
) -> MessageSendCommand:
    frame = parse_frame(
        json.dumps(
            {
                "v": 1,
                "kind": "command",
                "type": "message.send",
                "id": frame_id,
                "connection_epoch": epoch,
                "session_id": session_id,
                "payload": {
                    "client_message_id": frame_id,
                    "session_id": session_id,
                    "text": "你好",
                    "media_refs": [],
                    "client_created_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        )
    )
    assert isinstance(frame, MessageSendCommand)
    return frame


@pytest.mark.asyncio
async def test_message_send_is_idempotent_and_session_is_device_owned(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    first_device = uuid4().hex
    second_device = uuid4().hex
    _register_device(storage, first_device)
    _register_device(storage, second_device)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
            ),
        )
    )
    session_id = f"mobile:{uuid4()}"
    original = _message_frame(
        frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        session_id=session_id,
    )

    first = await channel.handle_command(device_id=first_device, frame=original)
    duplicate = await channel.handle_command(
        device_id=first_device,
        frame=original.model_copy(update={"connection_epoch": 2}),
    )
    forbidden = await channel.handle_command(
        device_id=second_device,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
            session_id=session_id,
        ),
    )

    assert first == duplicate
    assert first.type == "message.send.ok"
    assert len(bus.inbound) == 1
    assert forbidden.type == "message.send.error"
    assert forbidden.payload["code"] == "session_forbidden"
    storage.close()


@pytest.mark.asyncio
async def test_stream_deltas_batch_at_50ms_and_flush_before_tool_and_final(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
            ),
        )
    )
    session_id = f"mobile:{uuid4()}"
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    turn_id = uuid4().hex
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            content="帮我检查",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
    )

    for delta in ("思", "考", "中"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="mobile",
                chat_id=session_id.removeprefix("mobile:"),
                turn_id=turn_id,
                thinking_delta=delta,
            )
        )
    assert [event["event_type"] for event in runtime.events] == ["turn.started"]
    await asyncio.sleep(0.07)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "react.thinking.delta",
    ]
    first_thinking = cast(dict[str, object], runtime.events[1]["payload"])
    assert first_thinking["delta"] == "思考中"
    assert first_thinking["ordinal"] == 0

    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            turn_id=turn_id,
            content_delta="A" * 4096,
        )
    )
    await channel._on_tool_call_started(
        ToolCallStarted(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            iteration=1,
            call_id="call-1",
            tool_name="shell",
            arguments={"command": "pwd"},
            turn_id=turn_id,
        )
    )
    await channel._on_tool_call_completed(
        ToolCallCompleted(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            iteration=1,
            call_id="call-1",
            tool_name="shell",
            arguments={"command": "pwd"},
            final_arguments={"command": "pwd"},
            status="completed",
            result_preview="ok",
            turn_id=turn_id,
        )
    )
    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            turn_id=turn_id,
            thinking_delta="继续思考",
        )
    )
    await channel._on_response(
        OutboundMessage(
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            content="完成",
            thinking="思考中",
            control_turn_id=turn_id,
        )
    )

    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "react.thinking.delta",
        "answer.delta",
        "react.tool.started",
        "react.tool.completed",
        "react.thinking.delta",
        "message.final",
    ]
    tool_started = cast(dict[str, object], runtime.events[3]["payload"])
    tool_completed = cast(dict[str, object], runtime.events[4]["payload"])
    second_thinking = cast(dict[str, object], runtime.events[5]["payload"])
    assert tool_started["block_id"] == tool_completed["block_id"]
    assert tool_started["ordinal"] == tool_completed["ordinal"] == 1
    assert second_thinking["ordinal"] == 2
    assert second_thinking["block_id"] != first_thinking["block_id"]
    await channel.stop()
    storage.close()


@pytest.mark.asyncio
async def test_proactive_sender_uses_mobile_event_path(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    push = _PushTool()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=push,
                interrupt_controller=None,
            ),
        )
    )
    chat_id = str(uuid4())
    storage.claim_session(
        device_id=device_id,
        session_id=f"mobile:{chat_id}",
        created_at=datetime.now(timezone.utc),
    )
    sender = cast(Any, push.registered["mobile"]["text"])

    await sender(chat_id, "该休息一下了")

    assert runtime.events == [
        {
            "event_type": "message.proactive",
            "session_id": f"mobile:{chat_id}",
            "payload": {
                "content": "该休息一下了",
                "media": [],
                "metadata": {"source": "message_push"},
            },
        }
    ]
    await channel.stop()
    storage.close()
