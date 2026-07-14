from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from infra.control.socket import SocketAppServer
from session.manager import SessionManager

from akashic_sdk import Akashic, AsyncAkashic


@pytest.mark.asyncio
async def test_async_sdk_runs_against_real_socket_router(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)

    async def execute(request: TurnRequest) -> str:
        return f"sdk:{request.input}"

    runtime = ConversationRuntime(sessions.control_store, execute)
    server = SocketAppServer(tmp_path / "control.sock", ControlService(runtime, sessions, tmp_path))
    await server.start()
    try:
        async with await AsyncAkashic.connect(str(server.endpoint)) as client:
            thread = await client.thread_start()
            handle = await thread.turn("hello")
            events = [event async for event in handle.stream()]
            result = await handle.result()
            assert [event["method"] for event in events if event["method"].startswith("turn/")] == [
                "turn/queued",
                "turn/started",
                "turn/completed",
            ]
            assert result["finalResponse"] == "sdk:hello"
    finally:
        await server.stop()
        await runtime.shutdown()
        sessions.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_mode", "expected_status"),
    (
        ("completed", "completed"),
        ("failed", "failed"),
        ("interrupted", "interrupted"),
        ("cancelled", "cancelled"),
    ),
)
async def test_sdk_result_leaves_no_duplicate_terminal_in_turn_queue(
    tmp_path: Path,
    terminal_mode: str,
    expected_status: str,
) -> None:
    """等待 result 和连接 barrier 后，turn queue 不得残留第二个终态。"""

    sessions = SessionManager(tmp_path / terminal_mode)
    started = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if terminal_mode == "failed":
            raise RuntimeError("sdk failure")
        if terminal_mode in {"interrupted", "cancelled"}:
            started.set()
            await asyncio.Event().wait()
        return request.input

    runtime = ConversationRuntime(sessions.control_store, execute)
    server = SocketAppServer(
        tmp_path / f"sdk-{terminal_mode}.sock",
        ControlService(runtime, sessions, tmp_path),
    )
    await server.start()
    try:
        async with await AsyncAkashic.connect(str(server.endpoint)) as client:
            thread = await client.thread_start()
            handle = await thread.turn(terminal_mode)
            if terminal_mode == "interrupted":
                await started.wait()
                _ = await handle.interrupt()
            elif terminal_mode == "cancelled":
                await started.wait()
                await runtime.shutdown()

            result = await handle.result()
            assert result["status"] == expected_status

            # turn/read 是同连接 barrier；响应到达后，之前的通知已经进入 SDK reader。
            _ = await client.turn_read(thread.id, handle.id)
            await asyncio.sleep(0)
            assert handle._wire.turn_queues[handle.id].empty()
    finally:
        await server.stop()
        await runtime.shutdown()
        sessions.close()


@pytest.mark.asyncio
async def test_sync_sdk_has_turn_handle_and_thread_management_parity(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)

    async def execute(request: TurnRequest) -> str:
        return f"sync:{request.input}"

    runtime = ConversationRuntime(sessions.control_store, execute)
    server = SocketAppServer(tmp_path / "sync-control.sock", ControlService(runtime, sessions, tmp_path))
    await server.start()

    def exercise() -> None:
        with Akashic.connect(str(server.endpoint)) as client:
            thread = client.thread_start()
            handle = thread.turn("hello")
            events = list(handle.events())
            result = handle.result()
            assert events[-1]["method"] == "turn/completed"
            assert result["finalResponse"] == "sync:hello"
            assert client.turn_read(thread.id, handle.id)["status"] == "completed"
            assert client.thread_read(thread.id)["id"] == thread.id
            assert any(item["id"] == thread.id for item in client.thread_list()["data"])
            assert client.thread_delete(thread.id)["deleted"] is True

    try:
        await asyncio.to_thread(exercise)
    finally:
        await server.stop()
        await runtime.shutdown()
        sessions.close()
