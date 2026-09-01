from __future__ import annotations

import argparse
import asyncio
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from docker.debug.content_wake_h5_e2e import (
    PROTECTED_REQUIRED_FILES,
    PROTECTED_SQLITE_TABLES,
    H5Error,
    _load_manifest,
    _seed_protected_fixture,
    _validate_protected_snapshot,
)
from docker.debug.wake_v3_provider_e2e import (
    _BUILDER_SYSTEM_MARKER,
    _CALLER_SYSTEM_MARKER,
    _run,
    snapshot_protected_workspace,
)


async def _start_provider(
    status: int, requests: list[dict[str, object]]
) -> asyncio.AbstractServer:
    """Serve one local Chat Completions boundary with a fixed outcome."""

    async def respond(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        header = await reader.readuntil(b"\r\n\r\n")
        if header.startswith(b"GET "):
            body = json.dumps(
                {
                    "object": "list",
                    "data": [{"id": "deepseek-v4-flash", "object": "model"}],
                }
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Content-Type: application/json\r\nConnection: close\r\n\r\n"
                + body
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        length = next(
            (
                int(line.split(b":", 1)[1].strip())
                for line in header.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            ),
            0,
        )
        request = cast(dict[str, object], json.loads(await reader.readexactly(length)))
        requests.append(request)
        if status == 200:
            tools = cast(list[dict[str, object]], request["tools"])
            tool_names = {
                cast(dict[str, object], item["function"])["name"] for item in tools
            }
            candidate = re.search(
                r"candidate_[0-9a-f]{16}",
                json.dumps(request["messages"], ensure_ascii=False),
            )
            assert candidate is not None
            if "screen_content" in tool_names:
                name = "screen_content"
                arguments = {
                    "items": [
                        {
                            "candidate_id": candidate.group(0),
                            "initial_interest": "likely_interesting",
                            "question": "Does this include a real new capability?",
                        }
                    ]
                }
            else:
                name = "share_content"
                arguments = {
                    "message": "fixture provider response",
                    "items": [candidate.group(0)],
                }
            payload: dict[str, object] = {
                "id": "fixture-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "fixture reasoning",
                            "tool_calls": [
                                {
                                    "id": "call:fixture",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        else:
            payload = {"error": {"message": "fixture-provider-body-marker"}}
        body = json.dumps(payload).encode()
        reason = {200: "OK", 400: "Bad Request", 503: "Service Unavailable"}[status]
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\n".encode()
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Content-Type: application/json\r\nConnection: close\r\n\r\n"
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    return await asyncio.start_server(respond, "127.0.0.1", 0)


async def _run_with_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> tuple[dict[str, Any], list[dict[str, object]], str]:
    protected = tmp_path / "formal"
    protected.mkdir()
    requests: list[dict[str, object]] = []
    server = await _start_provider(status, requests)
    port = int(server.sockets[0].getsockname()[1])
    endpoint = f"127.0.0.1:{port}"
    monkeypatch.setenv("PR_G_DEEPSEEK_API_KEY", "fixture-secret-marker")
    monkeypatch.setenv("PR_G_DEEPSEEK_BASE_URL", f"http://{endpoint}/v1")
    try:
        result = cast(
            dict[str, Any],
            await _run(
                argparse.Namespace(protected_workspace=str(protected), report="unused")
            ),
        )
    finally:
        server.close()
        await server.wait_closed()
    return result, requests, endpoint


@pytest.mark.asyncio
async def test_wake_provider_200_keeps_v3_request_and_delivery_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, requests, _ = await _run_with_provider(tmp_path, monkeypatch, 200)

    assert payload["status"] == "passed"
    assert payload["selected"]["final_state"] == "settled"
    assert len(requests) == 2
    for request in requests:
        assert request["model"] == "deepseek-v4-flash"
        assert request["reasoning_effort"] == "max"
        assert "max_tokens" not in request
    tool_sets = [
        {
            cast(dict[str, object], tool["function"])["name"]
            for tool in cast(list[dict[str, object]], request["tools"])
        }
        for request in requests
    ]
    assert tool_sets[0] == {"screen_content"}
    assert tool_sets[1] == {
        "recall_fixture",
        "web_fetch",
        "share_content",
        "skip_content",
    }
    first_system = str(cast(list[dict[str, object]], requests[0]["messages"])[0])
    assert _CALLER_SYSTEM_MARKER in first_system
    assert _BUILDER_SYSTEM_MARKER not in first_system
    assert "fixture-secret-marker" not in json.dumps(payload, sort_keys=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "attempts", "retryable", "content_status"),
    ((400, 1, "false", "invalidated"), (503, 4, "true", "deferred")),
)
async def test_wake_provider_error_is_terminal_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    attempts: int,
    retryable: str,
    content_status: str,
) -> None:
    payload, requests, endpoint = await _run_with_provider(
        tmp_path, monkeypatch, status
    )

    evidence = cast(dict[str, Any], payload["selected_evidence"])
    assert payload["status"] == "failed"
    assert payload["failure_code"] == "SELECTED_DELIVERY_NOT_TERMINAL"
    assert len(requests) == attempts
    assert evidence["provider_terminal_counts"]["call_error"] == 1
    assert evidence["turn_status_counts"] == {"failed": 1}
    assert evidence["turn_retryable_counts"][retryable] == 1
    assert evidence["content_counts"] == {content_status: 1}
    assert evidence["delivery_count"] == 0
    encoded = json.dumps(payload, sort_keys=True)
    assert endpoint not in encoded
    assert "fixture-secret-marker" not in encoded
    assert "fixture-provider-body-marker" not in encoded


def test_h5_manifest_and_protected_workspace_contract(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    _seed_protected_fixture(protected)
    snapshot = snapshot_protected_workspace(protected)

    _validate_protected_snapshot(snapshot)
    assert set(cast(dict[str, object], snapshot["files"])) >= PROTECTED_REQUIRED_FILES
    assert set(cast(dict[str, object], snapshot["sqlite"])) >= set(
        PROTECTED_SQLITE_TABLES
    )
    missing_file = deepcopy(snapshot)
    del cast(dict[str, object], missing_file["files"])["sessions.db"]
    with pytest.raises(H5Error, match="缺少非空fixture"):
        _validate_protected_snapshot(missing_file)
    empty_rows = deepcopy(snapshot)
    sessions = cast(
        dict[str, object],
        cast(dict[str, object], empty_rows["sqlite"])["sessions.db"],
    )
    cast(dict[str, object], sessions["rows"])["messages"] = 0
    with pytest.raises(H5Error, match="SQLite fixture 无效"):
        _validate_protected_snapshot(empty_rows)

    manifest = _load_manifest(
        Path("docker/debug/content-wake-h5.manifest.json").resolve()
    )
    assert manifest.real_provider["status"] == "PENDING"
    assert all(suite.cases for suite in manifest.suites)
