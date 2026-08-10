from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from agent.control.client import ClientTurnHandle, ConnectionClosedError, ControlClient
from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from infra.control.socket import SocketAppServer
from session.manager import SessionManager


@pytest.mark.asyncio
async def test_control_client_unblocks_turn_stream_when_server_closes(
    tmp_path: Path,
) -> None:
    endpoint = tmp_path / "closing.sock"

    async def close_after_turn_start(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        initialize = json.loads(await reader.readline())
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": initialize["id"],
                        "result": {"protocolVersion": "1.0"},
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        _ = await reader.readline()
        thread_start = json.loads(await reader.readline())
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": thread_start["id"],
                        "result": {"id": "programmatic:closing"},
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        turn_start = json.loads(await reader.readline())
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": turn_start["id"],
                        "result": {
                            "id": "turn:closing",
                            "threadId": "programmatic:closing",
                            "status": "queued",
                        },
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(close_after_turn_start, path=endpoint)
    client = await ControlClient.connect(str(endpoint))
    try:
        thread = await client.start_thread()
        handle = await client.start_turn(str(thread["id"]), "verify")

        with pytest.raises(ConnectionClosedError, match="server closed"):
            _ = await asyncio.wait_for(anext(handle.events()), 1)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_control_client_preserves_buffered_terminal_on_disconnect() -> None:
    reader = asyncio.StreamReader()
    client = ControlClient(reader, object())  # type: ignore[arg-type]
    for index in range(511):
        reader.feed_data(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "item/completed",
                        "params": {"turnId": "turn:full", "index": index},
                    }
                )
                + "\n"
            ).encode()
        )
    reader.feed_data(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "turn/completed",
                    "params": {
                        "turnId": "turn:full",
                        "turn": {"id": "turn:full", "status": "completed"},
                    },
                }
            )
            + "\n"
        ).encode()
    )
    reader.feed_eof()

    await client._read_loop()
    handle = ClientTurnHandle(client, "programmatic:full", "turn:full", {})

    assert await handle.result() == {"id": "turn:full", "status": "completed"}


@pytest.mark.asyncio
async def test_control_client_reads_terminal_larger_than_asyncio_default(
    tmp_path: Path,
) -> None:
    """长递归 turn 的 terminal frame 不得触发默认 64 KiB 断连。"""

    endpoint = tmp_path / "large-terminal.sock"
    large_preview = "x" * (80 * 1024)

    async def send_large_terminal(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        # 1. 完成 initialize 与 thread/turn start 握手。
        initialize = json.loads(await reader.readline())
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": initialize["id"],
                        "result": {"protocolVersion": "1.0"},
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        _ = await reader.readline()
        thread_start = json.loads(await reader.readline())
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": thread_start["id"],
                        "result": {"id": "programmatic:large"},
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        turn_start = json.loads(await reader.readline())
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": turn_start["id"],
                        "result": {
                            "id": "turn:large",
                            "threadId": "programmatic:large",
                            "status": "queued",
                        },
                    }
                )
                + "\n"
            ).encode()
        )

        # 2. 单帧超过 asyncio 默认值，但仍在 control 合同上限内。
        terminal = {
            "jsonrpc": "2.0",
            "method": "turn/completed",
            "params": {
                "turnId": "turn:large",
                "turn": {
                    "id": "turn:large",
                    "status": "completed",
                    "finalResponse": "done",
                    "items": [{"resultPreview": large_preview}],
                },
            },
        }
        writer.write((json.dumps(terminal) + "\n").encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(send_large_terminal, path=endpoint)
    client = await ControlClient.connect(str(endpoint))
    try:
        thread = await client.start_thread()
        handle = await client.start_turn(str(thread["id"]), "verify")

        result = await asyncio.wait_for(handle.result(), 2)

        assert result["status"] == "completed"
        assert result["items"][0]["resultPreview"] == large_preview
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_exec_remote_error_exits_two_without_traceback(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)

    async def execute(request: TurnRequest) -> str:
        return request.input

    runtime = ConversationRuntime(sessions.control_store, execute)
    server = SocketAppServer(
        tmp_path / "control.sock",
        ControlService(runtime, sessions, tmp_path),
    )
    await server.start()
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "main.py",
            "exec",
            "--thread",
            "programmatic:missing",
            "--endpoint",
            str(server.endpoint),
            "--workspace",
            str(tmp_path),
            "hello",
            cwd=Path(__file__).resolve().parents[2],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 2
        assert stdout == b""
        assert "thread" in stderr.decode()
        assert "Traceback" not in stderr.decode()
    finally:
        await server.stop()
        await runtime.shutdown()
        sessions.close()


@pytest.mark.asyncio
async def test_exec_new_rejects_unbound_latest_and_defaults_to_stable(
    tmp_path: Path,
) -> None:
    sessions = SessionManager(tmp_path)
    seen: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> str:
        seen.append(request)
        return request.input

    runtime = ConversationRuntime(sessions.control_store, execute)
    server = SocketAppServer(
        tmp_path / "control.sock",
        ControlService(runtime, sessions, tmp_path),
    )
    await server.start()

    async def run(
        *extra: str,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "main.py",
            "exec",
            "--new",
            "--endpoint",
            str(server.endpoint),
            "--workspace",
            str(tmp_path),
            *extra,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode is not None
        return process.returncode, stdout.decode(), stderr.decode()

    try:
        latest = await run("--runtime", "latest", "--final-only", "verify")
        forged_env = os.environ.copy()
        forged_env["AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN"] = "turn:forged"
        forged_owner = await run("--final-only", "forged-owner", env=forged_env)
        persistent = await run("--persist-memory", "--final-only", "remember")

        assert latest[0] == 2
        assert latest[1] == ""
        assert "attached 插件验证子 turn" in latest[2]
        assert forged_owner == (0, "forged-owner\n", "")
        assert persistent == (0, "remember\n", "")
        rows = sessions.list_sessions()
        metadata = [
            sessions.control_store.get_session_meta(str(row["key"]))["metadata"]
            for row in rows
        ]
        assert len(metadata) == 2
        assert {"skip_post_memory": True} in metadata
        assert {} in metadata
        assert all(
            not any(key.startswith("_pluginRollout") for key in item)
            for item in metadata
        )
        assert [request.metadata["runtime"] for request in seen] == [
            "stable",
            "stable",
        ]
    finally:
        await server.stop()
        await runtime.shutdown()
        sessions.close()
