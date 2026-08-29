from __future__ import annotations

import argparse
import asyncio
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import uvicorn

from bootstrap.chat_api import create_chat_app
from bus.event_bus import EventBus
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnOutputCompleted,
    TurnStarted,
)
from infra.channels.base import AttachmentStore
from infra.channels.web_chat_channel import WebChatChannel
from session.manager import SessionManager


class DeterministicRuntimeBus:
    """Persist one real session and drive the real Web channel lifecycle."""

    def __init__(self, sessions: SessionManager, events: EventBus) -> None:
        self.sessions = sessions
        self.events = events
        self.outbound: Any = None

    def subscribe_outbound(self, channel: str, callback: Any) -> None:
        if channel != "web":
            raise ValueError(f"unexpected channel: {channel}")
        self.outbound = callback

    async def publish_inbound(self, message: InboundMessage) -> None:
        """Append the request, stream deterministic rich output, and commit it."""

        # 1. Persist the same append-only session rows the formal runtime exposes.
        session = self.sessions.get_or_create(message.session_key)
        user = {"role": "user", "content": message.content, "timestamp": message.timestamp.isoformat()}
        answer = "隔离 Runtime 流式响应：" + "片" * 120
        assistant = {
            "role": "assistant",
            "content": answer,
            "reasoning_content": "先检查真实持久层，再验证 WebSocket 到 React 的可见链路。",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self.sessions.append_messages(session, [user, assistant])

        # 2. Exercise the real lifecycle-to-WebSocket adapter, including rich blocks.
        turn_id = f"turn:isolated:{message.chat_id}"
        await self.events.emit(TurnStarted(
            session_key=message.session_key,
            channel="web",
            chat_id=message.chat_id,
            content=message.content,
            timestamp=message.timestamp,
            turn_id=turn_id,
            control_turn_id=turn_id,
        ))
        await self.events.emit(StreamDeltaReady(
            session_key=message.session_key,
            channel="web",
            chat_id=message.chat_id,
            thinking_delta="先检查真实持久层，再验证可见链路。",
        ))
        await self.events.emit(ToolCallStarted(
            session_key=message.session_key,
            channel="web",
            chat_id=message.chat_id,
            iteration=1,
            call_id="runtime-fixture-tool",
            tool_name="runtime_e2e_probe",
            arguments={"scope": "isolated"},
        ))
        await self.events.emit(ToolCallCompleted(
            session_key=message.session_key,
            channel="web",
            chat_id=message.chat_id,
            iteration=1,
            call_id="runtime-fixture-tool",
            tool_name="runtime_e2e_probe",
            arguments={"scope": "isolated"},
            final_arguments={"scope": "isolated"},
            status="success",
            result_preview="真实 Runtime adapter 已连通",
        ))
        for delta in answer:
            await self.events.emit(StreamDeltaReady(
                session_key=message.session_key,
                channel="web",
                chat_id=message.chat_id,
                content_delta=delta,
            ))
            await asyncio.sleep(0.001)
        await self.events.emit(TurnOutputCompleted(
            session_key=message.session_key,
            channel="web",
            chat_id=message.chat_id,
            turn_id=turn_id,
        ))

        # 3. Deliver the authoritative terminal frame through the registered channel callback.
        if self.outbound is None:
            raise RuntimeError("Web outbound callback was not registered")
        await self.outbound(OutboundMessage(
            channel="web",
            chat_id=message.chat_id,
            content=answer,
            thinking=assistant["reasoning_content"],
            control_turn_id=turn_id,
        ))


class FixturePushTool:
    def register_channel(self, channel: str, **senders: Any) -> None:
        if channel != "web" or "deliver" not in senders:
            raise ValueError("isolated runtime expected the Web deliver adapter")


class FixtureModelRegistry:
    current = SimpleNamespace(generation_id=1, role_runtime_ids={"default": "runtime/isolated"})

    async def refresh(self) -> Any:
        return self.current

    def list_runtimes(self) -> list[dict[str, object]]:
        return [{
            "id": "runtime/isolated",
            "provider": "fixture",
            "model": "deterministic",
            "sourceId": "isolated-runtime",
            "sourceName": "隔离 Runtime",
            "reasoningEffort": "medium",
            "supportedReasoningEfforts": ["medium"],
            "roles": ["default"],
        }]


class EmptyPluginUi:
    def catalog(self) -> dict[str, object]:
        return {"catalog_revision": "0" * 64, "items": []}


async def serve(port: int, workspace: Path) -> None:
    """Start an isolated real persistence/API/channel stack until interrupted."""

    # 1. Assemble only real owners on the path under test.
    sessions = SessionManager(workspace)
    events = EventBus()
    channel = WebChatChannel()
    bus = DeterministicRuntimeBus(sessions, events)
    await channel.start(SimpleNamespace(
        bus=bus,
        session_manager=sessions,
        event_bus=events,
        push_tool=FixturePushTool(),
        attachment_store=AttachmentStore(workspace / "uploads"),
        interrupt_controller=None,
    ))
    app = create_chat_app(
        workspace=workspace,
        channel=channel,
        plugin_ui_provider=EmptyPluginUi(),
        model_registry=FixtureModelRegistry(),
    )

    @app.get("/api/shell/state")
    def shell_state() -> dict[str, object]:
        return {"status": "ready", "configured": True, "chatReady": True}

    # 2. Serve until the process receives a normal termination signal.
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False))
    try:
        await server.serve()
    finally:
        await channel.stop()
        sessions.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4174)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    temporary = args.workspace is None
    workspace = args.workspace or Path(tempfile.mkdtemp(prefix="akashic-webui-runtime-"))
    try:
        print(f'{{"event":"webui.runtime_fixture_starting","workspace":"{workspace}","port":{args.port}}}', flush=True)
        try:
            asyncio.run(serve(args.port, workspace))
        except KeyboardInterrupt:
            pass
    finally:
        if temporary:
            shutil.rmtree(workspace)


if __name__ == "__main__":
    main()
