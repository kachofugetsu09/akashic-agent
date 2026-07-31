from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest

from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from infra.control.socket import SocketAppServer
from session.manager import SessionManager

from akashic_sdk import Akashic, AsyncAkashic, SlowConsumerError, TurnHandle
from akashic_sdk.client import _WireClient


def _buffered_notification_reader(
    count: int,
    *,
    turn_id: str | None = None,
) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    for index in range(count):
        params: dict[str, object] = {"index": index}
        if turn_id is not None:
            params["turnId"] = turn_id
        payload: dict[str, object] = {
            "jsonrpc": "2.0",
            "method": "test/notification",
            "params": params,
        }
        reader.feed_data(
            (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        )
    reader.feed_eof()
    return reader


@pytest.mark.asyncio
async def test_sdk_reader_yields_to_active_turn_consumer() -> None:
    reader = _buffered_notification_reader(600, turn_id="turn-1")
    wire = _WireClient(reader, cast(asyncio.StreamWriter, object()))
    handle = TurnHandle(wire, "thread-1", "turn-1")

    notifications = handle.events()
    received = [
        await asyncio.wait_for(anext(notifications), timeout=1)
        for _ in range(600)
    ]
    await wire.reader_task

    assert [event["params"]["index"] for event in received] == list(range(600))


@pytest.mark.asyncio
async def test_sdk_reader_fails_loud_for_unconsumed_notification_queue() -> None:
    reader = _buffered_notification_reader(513)
    wire = _WireClient(reader, cast(asyncio.StreamWriter, object()))
    client = AsyncAkashic(wire)

    await wire.reader_task

    notifications = client.notifications()
    for _ in range(511):
        _ = await anext(notifications)
    with pytest.raises(
        SlowConsumerError,
        match="global notification queue overflow",
    ):
        _ = await anext(notifications)


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
async def test_async_sdk_reads_terminal_frame_larger_than_streamreader_default(
    tmp_path: Path,
) -> None:
    sessions = SessionManager(tmp_path)
    response = "x" * (128 * 1024)

    async def execute(_request: TurnRequest) -> str:
        return response

    runtime = ConversationRuntime(sessions.control_store, execute)
    server = SocketAppServer(
        tmp_path / "large-terminal.sock",
        ControlService(runtime, sessions, tmp_path),
    )
    await server.start()
    try:
        async with await AsyncAkashic.connect(str(server.endpoint)) as client:
            thread = await client.thread_start()
            result = await thread.run("large")
            assert result["status"] == "completed"
            assert result["finalResponse"] == response
    finally:
        await server.stop()
        await runtime.shutdown()
        sessions.close()


@pytest.mark.asyncio
async def test_64k_streamreader_ablation_rejects_large_ndjson_frame() -> None:
    reader = asyncio.StreamReader(limit=64 * 1024)
    reader.feed_data(b"x" * (128 * 1024) + b"\n")
    reader.feed_eof()

    with pytest.raises(ValueError, match="chunk"):
        _ = await reader.readline()


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
