from __future__ import annotations

import json
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from docker.debug.programmatic_control_probe import (
    GateFailure,
    JsonRpcSocketClient,
    _extract_id,
    _is_terminal_event,
    _prepare_host_sandbox,
    _recorded_turn_notifications,
    _wait_socket,
    _turn_projection,
    _tool_lifecycle,
)


def test_control_gate_prepares_external_static_mount_without_repo_static(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "clean-checkout"
    repo.mkdir()
    (repo / "main.py").write_text("print('clean')\n", encoding="utf-8")
    template = repo / "prompts/VEDA.md"
    template.parent.mkdir()
    template.write_text("gate veda\n", encoding="utf-8")
    assert not (repo / "static").exists()

    sandbox = tmp_path / "sandbox"
    _prepare_host_sandbox(sandbox, repo)
    compose = (
        Path(__file__).parents[1] / "docker/debug/docker-compose.control-gate.yml"
    ).read_text(encoding="utf-8")

    assert (sandbox / "static/dashboard").is_dir()
    assert (sandbox / "static/chat").is_dir()
    assert (sandbox / "app/main.py").read_text(encoding="utf-8") == "print('clean')\n"
    assert (sandbox / "app/static").is_dir()
    assert (sandbox / "workspace/memory/VEDA.md").read_text(
        encoding="utf-8"
    ) == "gate veda\n"
    head = subprocess.run(
        ["git", "-C", str(sandbox / "app"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert (sandbox / "config.toml.migration-cursor").read_text().strip() == head
    assert "read_only: true" in compose
    assert (
        "${AKASHIC_CONTROL_SANDBOX:?set by programmatic_control_probe.py}"
        "/app:/app:ro"
    ) in compose
    assert (
        "${AKASHIC_CONTROL_SANDBOX:?set by programmatic_control_probe.py}"
        "/static:/app/static"
    ) in compose


def test_socket_client_correlates_response_and_buffers_notifications(
    tmp_path: Path,
) -> None:
    endpoint = tmp_path / "control.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(endpoint))
    server.listen(1)

    def serve() -> None:
        connection, _ = server.accept()
        with connection, connection.makefile("rb") as reader:
            request = json.loads(reader.readline())
            notification = {
                "jsonrpc": "2.0",
                "method": "turn/completed",
                "params": {"threadId": "thread-1", "turnId": "turn-1"},
            }
            connection.sendall(json.dumps(notification).encode() + b"\n")
            response = {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}
            connection.sendall(json.dumps(response).encode() + b"\n")

    worker = threading.Thread(target=serve)
    worker.start()
    client = JsonRpcSocketClient(endpoint, tmp_path / "events.jsonl")
    try:
        response = client.request("server/status", {})
        terminal = client.wait_terminal("turn-1")
    finally:
        client.close()
        worker.join(timeout=2)
        server.close()

    assert response["result"] == {"ok": True}
    assert terminal["method"] == "turn/completed"
    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["direction"] for record in records] == [
        "client",
        "server",
        "server",
    ]


def test_wait_socket_rejects_stale_uds_path(tmp_path: Path) -> None:
    endpoint = tmp_path / "stale.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(endpoint))
    try:
        with pytest.raises(GateFailure, match="等待 UDS 文件超时"):
            _wait_socket(endpoint, 0.05)

        server.listen(1)
        _wait_socket(endpoint, 0.1)
        connection, _ = server.accept()
        connection.close()
    finally:
        server.close()


@pytest.mark.parametrize(
    ("payload", "resource", "expected"),
    [
        ({"result": {"id": "thread-1"}}, "thread", "thread-1"),
        ({"result": {"turn": {"id": "turn-1"}}}, "turn", "turn-1"),
    ],
)
def test_extract_id_accepts_flat_and_nested_result(
    payload: dict[str, object], resource: str, expected: str
) -> None:
    assert _extract_id(payload, resource) == expected


def test_terminal_event_requires_matching_turn() -> None:
    event = {
        "jsonrpc": "2.0",
        "method": "turn/completed",
        "params": {"threadId": "thread-1", "turnId": "turn-1"},
    }
    assert _is_terminal_event(event, "turn-1")
    assert not _is_terminal_event(event, "turn-old")


def test_recorded_turn_notifications_filters_other_turns(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    records = [
        {
            "direction": "server",
            "message": {
                "jsonrpc": "2.0",
                "method": "item/completed",
                "params": {"threadId": "thread-1", "turnId": "turn-1"},
            },
        },
        {
            "direction": "server",
            "message": {
                "jsonrpc": "2.0",
                "method": "turn/completed",
                "params": {"threadId": "thread-2", "turnId": "turn-2"},
            },
        },
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records))

    notifications = _recorded_turn_notifications(path, "turn-1")

    assert [item["method"] for item in notifications] == ["item/completed"]


def test_turn_projection_compares_domain_fields_without_random_ids() -> None:
    turn = {
        "id": "turn-1",
        "threadId": "thread-1",
        "status": "failed",
        "finalResponse": None,
        "items": [
            {"id": "item-random", "type": "userMessage", "data": {"content": "x"}}
        ],
        "usage": None,
        "error": {"type": "ProviderError", "message": "secret", "retryable": True},
    }

    assert _turn_projection(turn) == {
        "status": "failed",
        "finalResponse": None,
        "items": [{"type": "userMessage", "data": {"content": "x"}}],
        "usage": None,
        "error": {"type": "ProviderError", "retryable": True},
    }


def test_turn_projection_normalizes_transport_identity_metadata() -> None:
    turn = {
        "status": "completed",
        "finalResponse": "done",
        "items": [
            {
                "type": "assistantMessage",
                "data": {
                    "content": "done",
                    "thinking": "reasoning",
                    "media": [],
                    "sessionMessageId": "web:random:1",
                    "metadata": {
                        "client_request_id": "request-random",
                        "control_turn_id": "turn:random",
                        "turn_duration_ms": 17,
                        "context_retry": {"request_time": "random"},
                        "persisted_user_message_id": "web:random:0",
                        "render": "card",
                    },
                },
            }
        ],
        "usage": None,
        "error": None,
    }

    assert _turn_projection(turn)["items"] == [
        {
            "type": "assistantMessage",
            "data": {
                "content": "done",
                "thinking": "reasoning",
                "media": [],
                "sessionMessageId": "<session-message-id>",
                "metadata": {
                    "persisted_user_message_id": "<persisted-user-message-id>",
                    "render": "card",
                },
            },
        }
    ]


def test_tool_lifecycle_requires_matching_started_and_completed_id() -> None:
    started = {
        "method": "item/started",
        "params": {
            "item": {
                "id": "item-1",
                "type": "toolCall",
                "data": {"name": "shell", "status": "in_progress"},
            }
        },
    }
    completed = {
        "method": "item/completed",
        "params": {
            "item": {
                "id": "item-1",
                "type": "toolCall",
                "data": {"name": "shell", "status": "interrupted"},
            }
        },
    }

    assert _tool_lifecycle([started, completed], "shell") == (
        started["params"]["item"],
        completed["params"]["item"],
    )
