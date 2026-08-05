from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from agent.control.models import TurnRequest
from agent.control.protocol.router import ConnectionRouter
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from infra.control.socket import SocketAppServer
from session.manager import SessionManager


@pytest.mark.asyncio
async def test_router_requires_full_handshake_and_routes_turn(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)

    async def execute(request: TurnRequest) -> str:
        return request.input.upper()

    runtime = ConversationRuntime(sessions.control_store, execute)
    service = ControlService(runtime, sessions, tmp_path)
    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    router = ConnectionRouter(service, send)
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":1,"method":"server/status","params":{}}\n'
    )
    assert sent[-1]["error"] == {"code": -32002, "message": "Client must complete initialize/initialized"}

    await router.handle_line(
        b'{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"1.0","clientInfo":{"name":"test","version":"1"},"capabilities":{}}}\n'
    )
    initialize = next(item for item in sent if item.get("id") == 2)
    initialize_result = initialize["result"]
    assert isinstance(initialize_result, dict)
    capabilities = initialize_result["capabilities"]
    assert isinstance(capabilities, dict)
    assert capabilities["reasoningEvents"] is False
    await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized","params":{}}\n')
    await router.handle_line(b'{"jsonrpc":"2.0","id":3,"method":"thread/start","params":{"metadata":{}}}\n')
    response = next(item for item in sent if item.get("id") == 3)
    thread = response["result"]
    assert isinstance(thread, dict)
    assert sent[-1] == {
        "jsonrpc": "2.0",
        "method": "thread/started",
        "params": {"thread": thread},
    }
    thread_id = thread["id"]
    await router.handle_line(
        (
            '{"jsonrpc":"2.0","id":4,"method":"turn/start","params":'
            f'{{"threadId":"{thread_id}","input":"hello","metadata":{{}}}}}}\n'
        ).encode()
    )
    await asyncio.wait_for(_wait_terminal(sent), timeout=1)
    terminal = [item for item in sent if item.get("method") == "turn/completed"]
    assert len(terminal) == 1
    await router.close()
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
async def test_raw_ndjson_connection_emits_exactly_one_terminal(
    tmp_path: Path,
    terminal_mode: str,
    expected_status: str,
) -> None:
    """持续读取到 barrier，确认原始 NDJSON 只含一个 turn 终态。"""

    sessions = SessionManager(tmp_path / terminal_mode)
    started = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if terminal_mode == "failed":
            raise RuntimeError("raw failure")
        if terminal_mode in {"interrupted", "cancelled"}:
            started.set()
            await asyncio.Event().wait()
        return request.input.upper()

    runtime = ConversationRuntime(sessions.control_store, execute)
    server = SocketAppServer(
        tmp_path / f"{terminal_mode}.sock",
        ControlService(runtime, sessions, tmp_path),
    )
    await server.start()
    reader, writer = await asyncio.open_unix_connection(str(server.endpoint))
    frames: list[dict[str, Any]] = []
    next_id = 0

    async def exchange(method: str, params: dict[str, object]) -> dict[str, Any]:
        nonlocal next_id
        next_id += 1
        request_id = next_id
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params,
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        while True:
            frame = cast(dict[str, Any], json.loads(await reader.readline()))
            frames.append(frame)
            if frame.get("id") == request_id:
                return frame

    try:
        _ = await exchange(
            "initialize",
            {
                "protocolVersion": "1.0",
                "clientInfo": {"name": "raw-terminal-test", "version": "1"},
                "capabilities": {"reasoningEvents": False},
            },
        )
        writer.write(b'{"jsonrpc":"2.0","method":"initialized","params":{}}\n')
        await writer.drain()
        thread_response = await exchange("thread/start", {"metadata": {}})
        thread_id = str(thread_response["result"]["id"])
        turn_response = await exchange(
            "turn/start",
            {"threadId": thread_id, "input": terminal_mode, "metadata": {}},
        )
        turn_id = str(turn_response["result"]["id"])

        if terminal_mode == "interrupted":
            await started.wait()
            _ = await exchange(
                "turn/interrupt",
                {"threadId": thread_id, "turnId": turn_id},
            )
        elif terminal_mode == "cancelled":
            await started.wait()
            await runtime.shutdown()

        while not any(
            frame.get("method") == "turn/completed"
            and frame.get("params", {}).get("turnId") == turn_id
            for frame in frames
        ):
            frames.append(cast(dict[str, Any], json.loads(await reader.readline())))

        # barrier response 之后继续短暂 drain，避免首个 terminal 掩盖队列中的重复项。
        _ = await exchange("server/status", {})
        while True:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=0.02)
            except TimeoutError:
                break
            if not line:
                break
            frames.append(cast(dict[str, Any], json.loads(line)))

        terminal = [
            frame
            for frame in frames
            if frame.get("method") == "turn/completed"
            and frame.get("params", {}).get("turnId") == turn_id
        ]
        assert len(terminal) == 1
        assert terminal[0]["params"]["turn"]["status"] == expected_status
    finally:
        writer.close()
        await writer.wait_closed()
        await server.stop()
        await runtime.shutdown()
        sessions.close()


@pytest.mark.asyncio
async def test_router_maps_incompatible_version_to_stable_error(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, _echo)
    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    router = ConnectionRouter(ControlService(runtime, sessions, tmp_path), send)
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2.0","clientInfo":{"name":"test","version":"1"}}}\n'
    )
    assert sent[-1]["error"] == {
        "code": -32003,
        "message": "Unsupported protocol version",
        "data": {"supported": ["1.0"]},
    }
    await router.close()
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
async def test_failed_token_does_not_advance_handshake_state(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, _echo)
    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    service = ControlService(runtime, sessions, tmp_path, workspace_token="secret")
    router = ConnectionRouter(service, send)
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0","clientInfo":{"name":"test","version":"1"},"workspaceToken":"wrong"}}\n'
    )
    assert sent[-1]["error"] == {"code": -32004, "message": "Invalid workspace token"}
    await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized","params":{}}\n')
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":2,"method":"server/status","params":{}}\n'
    )
    assert sent[-1]["error"] == {
        "code": -32002,
        "message": "Client must complete initialize/initialized",
    }
    await router.close()
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
async def test_thread_consolidation_returns_operation_and_notification(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, _echo)
    consolidated: list[str] = []

    async def consolidate(thread_id: str) -> bool:
        consolidated.append(thread_id)
        return True

    service = ControlService(runtime, sessions, tmp_path, consolidate=consolidate)
    thread = service.start_thread({})
    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    router = ConnectionRouter(service, send)
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0","clientInfo":{"name":"test","version":"1"}}}\n'
    )
    await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized","params":{}}\n')
    await router.handle_line(
        (
            '{"jsonrpc":"2.0","id":2,"method":"thread/consolidate/start","params":'
            f'{{"threadId":"{thread["id"]}"}}}}\n'
        ).encode()
    )
    response = next(item for item in sent if item.get("id") == 2)
    operation = response["result"]
    assert isinstance(operation, dict)
    assert operation["status"] == "in_progress"
    await asyncio.wait_for(_wait_method(sent, "operation/completed"), 1)
    completed = next(item for item in sent if item.get("method") == "operation/completed")
    assert completed["params"] == {
        "operation": {
            "id": operation["id"],
            "threadId": thread["id"],
            "status": "completed",
            "result": {"consolidated": True},
        }
    }
    assert consolidated == [thread["id"]]
    await router.handle_line(
        (
            '{"jsonrpc":"2.0","id":3,"method":"thread/delete","params":'
            f'{{"threadId":"{thread["id"]}"}}}}\n'
        ).encode()
    )
    assert sent[-1]["method"] == "thread/deleted"
    await router.close()
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
async def test_plugin_uninstall_returns_before_old_turn_drain_completes(
    tmp_path: Path,
) -> None:
    """context_pressure 卸载请求不能等待调用 turn 自己持有的 lease。"""

    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, _echo)
    started = asyncio.Event()
    old_turn_released = asyncio.Event()

    async def uninstall(plugin_id: str) -> dict[str, object]:
        started.set()
        await old_turn_released.wait()
        return {
            "pluginId": plugin_id,
            "cachePath": str(tmp_path / "plugins/cache/context_pressure"),
            "dataPath": str(tmp_path / "workspace/plugin-data/context_pressure-github"),
        }

    service = ControlService(
        runtime,
        sessions,
        tmp_path,
        plugin_uninstall=uninstall,
    )
    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    router = ConnectionRouter(service, send)
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0","clientInfo":{"name":"test","version":"1"}}}\n'
    )
    await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized","params":{}}\n')
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":2,"method":"plugin/uninstall/start","params":{"pluginId":"context_pressure@github"}}\n'
    )

    response = next(item for item in sent if item.get("id") == 2)
    operation = response["result"]
    assert isinstance(operation, dict)
    assert operation["status"] == "in_progress"
    assert operation["pluginId"] == "context_pressure@github"
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not any(item.get("method") == "operation/completed" for item in sent)

    old_turn_released.set()
    await asyncio.wait_for(_wait_method(sent, "operation/completed"), timeout=1)
    completed = next(item for item in sent if item.get("method") == "operation/completed")
    assert completed["params"] == {
        "operation": {
            "id": operation["id"],
            "pluginId": "context_pressure@github",
            "status": "completed",
            "result": {
                "pluginId": "context_pressure@github",
                "cachePath": str(tmp_path / "plugins/cache/context_pressure"),
                "dataPath": str(
                    tmp_path / "workspace/plugin-data/context_pressure-github"
                ),
            },
        }
    }
    await router.close()
    await service.shutdown()
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
async def test_plugin_uninstall_survives_deferred_client_disconnect(
    tmp_path: Path,
) -> None:
    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, _echo)
    release_old_turn = asyncio.Event()
    completed = asyncio.Event()

    async def uninstall(_plugin_id: str) -> dict[str, object]:
        await release_old_turn.wait()
        completed.set()
        return {"cachePath": "removed", "dataPath": "retained"}

    service = ControlService(
        runtime,
        sessions,
        tmp_path,
        plugin_uninstall=uninstall,
    )

    async def send(_message: dict[str, object]) -> None:
        return None

    router = ConnectionRouter(service, send)
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0","clientInfo":{"name":"test","version":"1"}}}\n'
    )
    await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized","params":{}}\n')
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":2,"method":"plugin/uninstall/start","params":{"pluginId":"context_pressure@github"}}\n'
    )

    await router.close()
    release_old_turn.set()
    await asyncio.wait_for(completed.wait(), timeout=1)
    await service.shutdown()
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
async def test_control_service_attaches_utc_to_legacy_naive_session_times(
    tmp_path: Path,
) -> None:
    sessions = SessionManager(tmp_path)
    sessions.control_store.upsert_session(
        "legacy:1",
        created_at="2026-07-14T08:00:00",
        updated_at="2026-07-14T09:30:00",
        last_consolidated=0,
        metadata={},
    )

    async def execute(request: TurnRequest) -> str:
        return request.input

    runtime = ConversationRuntime(sessions.control_store, execute)
    service = ControlService(runtime, sessions, tmp_path)

    thread = service.resume_thread("legacy:1")

    assert thread["createdAt"] == "2026-07-14T08:00:00Z"
    assert thread["updatedAt"] == "2026-07-14T09:30:00Z"
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
async def test_start_thread_persists_memory_exclusion_marker(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)

    async def execute(request: TurnRequest) -> str:
        return request.input

    runtime = ConversationRuntime(sessions.control_store, execute)
    service = ControlService(runtime, sessions, tmp_path)

    thread = service.start_thread({"skip_post_memory": True})
    thread_id = cast(str, thread["id"])

    assert thread["metadata"] == {"skip_post_memory": True}
    assert sessions.control_store.get_session_meta(thread_id)["metadata"] == {
        "skip_post_memory": True
    }
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
async def test_start_thread_rejects_non_boolean_memory_marker(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)

    async def execute(request: TurnRequest) -> str:
        return request.input

    runtime = ConversationRuntime(sessions.control_store, execute)
    service = ControlService(runtime, sessions, tmp_path)

    with pytest.raises(ValueError, match="必须是 boolean"):
        service.start_thread({"skip_post_memory": "false"})

    # 1. 非法标记不创建 session。
    assert sessions.list_sessions() == []
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
async def test_thread_runtime_selector_is_strict_and_inherited_by_turn(
    tmp_path: Path,
) -> None:
    sessions = SessionManager(tmp_path)
    seen: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> str:
        seen.append(request)
        return request.input

    runtime = ConversationRuntime(sessions.control_store, execute)
    service = ControlService(runtime, sessions, tmp_path)

    thread = service.start_thread({"skip_post_memory": True}, "latest")
    thread_id = cast(str, thread["id"])
    result = await (
        await service.start_turn(thread_id, "verify", {}, None)
    ).result()

    assert thread["metadata"] == {
        "skip_post_memory": True,
        "runtime": "latest",
    }
    assert result.status.value == "completed"
    assert seen[0].metadata["runtime"] == "latest"
    with pytest.raises(ValueError, match="stable 或 latest"):
        service.start_thread({}, "candidate")
    assert len(sessions.list_sessions()) == 1
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("detached", [False, True])
async def test_router_disconnect_interrupts_only_attached_turn(
    tmp_path: Path,
    detached: bool,
) -> None:
    sessions = SessionManager(tmp_path / str(detached))
    started = asyncio.Event()

    async def execute(_request: TurnRequest) -> str:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    runtime = ConversationRuntime(sessions.control_store, execute)
    service = ControlService(runtime, sessions, tmp_path)
    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    router = ConnectionRouter(service, send)
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0","clientInfo":{"name":"test","version":"1"},"capabilities":{}}}\n'
    )
    await router.handle_line(
        b'{"jsonrpc":"2.0","method":"initialized","params":{}}\n'
    )
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":2,"method":"thread/start","params":{"metadata":{},"runtime":"latest"}}\n'
    )
    thread = cast(dict[str, object], next(item for item in sent if item.get("id") == 2)["result"])
    thread_id = cast(str, thread["id"])
    await router.handle_line(
        (
            '{"jsonrpc":"2.0","id":3,"method":"turn/start","params":'
            f'{{"threadId":"{thread_id}","input":"verify","metadata":{{}},"detached":{str(detached).lower()}}}}}\n'
        ).encode()
    )
    await started.wait()
    turn = cast(dict[str, object], next(item for item in sent if item.get("id") == 3)["result"])
    turn_id = cast(str, turn["id"])

    await router.close()

    if detached:
        assert runtime.is_thread_active(thread_id)
        await runtime.interrupt_turn(thread_id, turn_id)
    else:
        assert runtime.read_turn(thread_id, turn_id).status.value == "interrupted"
        assert not runtime.is_thread_active(thread_id)
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
async def test_plugin_candidate_control_methods_use_runtime_owner(
    tmp_path: Path,
) -> None:
    sessions = SessionManager(tmp_path)

    async def execute(request: TurnRequest) -> str:
        return request.input

    calls: list[tuple[str, object]] = []

    async def install(
        source: str,
        marketplace: str,
        ref: str,
        sparse: list[str],
    ) -> dict[str, object]:
        calls.append(("install", (source, marketplace, ref, sparse)))
        return {"publicationState": "latest_ready"}

    async def promote(plugin_id: str) -> dict[str, object]:
        calls.append(("promote", plugin_id))
        return {"publication_state": "promoted"}

    async def discard(plugin_id: str) -> dict[str, object]:
        calls.append(("discard", plugin_id))
        return {"publication_state": "discarded"}

    runtime = ConversationRuntime(sessions.control_store, execute)
    service = ControlService(
        runtime,
        sessions,
        tmp_path,
        plugin_install=install,
        plugin_status=lambda: {"candidateState": "latest_ready"},
        plugin_promote=promote,
        plugin_discard=discard,
    )

    assert await service.install_plugin("repo", "lab", "main", ["plugin"]) == {
        "publicationState": "latest_ready"
    }
    assert service.plugin_status() == {"candidateState": "latest_ready"}
    assert await service.promote_plugin("feed@lab") == {
        "publication_state": "promoted"
    }
    assert await service.discard_plugin("feed@lab") == {
        "publication_state": "discarded"
    }
    assert calls == [
        ("install", ("repo", "lab", "main", ["plugin"])),
        ("promote", "feed@lab"),
        ("discard", "feed@lab"),
    ]
    await runtime.shutdown()
    sessions.close()


async def _wait_terminal(sent: list[dict[str, object]]) -> None:
    while not any(item.get("method") == "turn/completed" for item in sent):
        await asyncio.sleep(0)


async def _wait_method(sent: list[dict[str, object]], method: str) -> None:
    while not any(item.get("method") == method for item in sent):
        await asyncio.sleep(0)


async def _echo(request: TurnRequest) -> str:
    return request.input
