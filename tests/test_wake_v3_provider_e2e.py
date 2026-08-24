from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from agent.provider import LLMProvider

from docker.debug.wake_v3_provider_e2e import (
    ScriptedProvider,
    _BUILDER_SYSTEM_MARKER,
    _CALLER_SYSTEM_MARKER,
    _build_selected_provider,
    _formal_evidence,
    _main_fallback_report,
    _process_isolation_evidence,
    _run,
    main,
    run_quiet_suite,
    run_suite,
    snapshot_protected_workspace,
)


async def _start_provider_fixture(
    status: int,
    requests: list[dict[str, object]],
) -> asyncio.AbstractServer:
    """Start one loopback Chat Completions fixture and retain parsed requests."""

    async def respond(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        # 1. Parse the local request only inside the test process.
        header = await reader.readuntil(b"\r\n\r\n")
        content_length = next(
            (
                int(line.split(b":", 1)[1].strip())
                for line in header.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            ),
            0,
        )
        request_body = await reader.readexactly(content_length)
        requests.append(cast(dict[str, object], json.loads(request_body)))

        # 2. Return one explicit provider layer outcome.
        if status == 200:
            request_tools = requests[-1].get("tools")
            decision_request = isinstance(request_tools, list) and bool(request_tools)
            message: dict[str, object]
            finish_reason: str
            if decision_request:
                prompt = json.dumps(requests[-1].get("messages"), ensure_ascii=False)
                candidate = re.search(r"candidate_[0-9a-f]{16}", prompt)
                if candidate is None:
                    raise AssertionError("provider fixture prompt missing candidate_id")
                message = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "fixture reasoning",
                    "tool_calls": [
                        {
                            "id": "call:fixture-share",
                            "type": "function",
                            "function": {
                                "name": "share_content",
                                "arguments": json.dumps(
                                    {
                                        "message": "fixture provider response",
                                        "items": [candidate.group(0)],
                                    }
                                ),
                            },
                        }
                    ],
                }
                finish_reason = "tool_calls"
            else:
                message = {
                    "role": "assistant",
                    "content": "Wake decision recorded.",
                    "reasoning_content": "fixture summary reasoning",
                }
                finish_reason = "stop"
            payload: dict[str, object] = {
                "id": "fixture-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
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
        reason = {
            200: b"OK",
            400: b"Bad Request",
            503: b"Service Unavailable",
        }[status]
        writer.write(
            f"HTTP/1.1 {status} ".encode()
            + reason
            + b"\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Content-Type: application/json\r\nConnection: close\r\n\r\n"
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    return await asyncio.start_server(respond, "127.0.0.1", 0)


def _provider_shape(provider: LLMProvider) -> dict[str, object]:
    """Read only non-secret provider wiring fields for a builder mutant test."""

    value = cast(Any, provider)
    backend = cast(Any, value._backend)
    return {
        "runtime_id": provider.runtime_id,
        "context_window": provider.context_window,
        "has_system_prompt": bool(value._system),
        "extra_body": dict(backend._extra_body),
    }


@pytest.mark.asyncio
async def test_selected_chain_uses_one_react_delivery_projection_and_ack(
    tmp_path: Path,
) -> None:
    result = await run_suite(tmp_path, provider=ScriptedProvider())

    assert result["logical_provider_requests"] == 2
    assert result["delivery_count"] == 1
    assert result["session_projection_count"] == 1
    assert result["content_counts"] == {"settled": 1}
    assert result["source_ack_count"] == 1
    assert result["source_ack_attempts"] == 1
    assert result["content_submission_count"] == 1
    assert result["final_state"] == "settled"


@pytest.mark.asyncio
async def test_restart_and_ack_retry_never_repeat_provider_or_history(
    tmp_path: Path,
) -> None:
    result = await run_suite(
        tmp_path,
        provider=ScriptedProvider(),
        inject_settlement_failure=True,
        ack_failures=1,
    )

    assert result["logical_provider_requests"] == 2
    assert result["restart_count"] == 1
    assert result["settlement_failure_count"] == 1
    assert result["delivery_count"] == 1
    assert result["session_projection_count"] == 1
    assert result["source_ack_count"] == 1
    assert result["source_ack_attempts"] == 2
    assert result["content_submission_count"] == 1
    assert result["final_state"] == "settled"


@pytest.mark.asyncio
async def test_quiet_and_empty_poll_create_no_external_effect_or_fake_history(
    tmp_path: Path,
) -> None:
    result = await run_quiet_suite(tmp_path)

    assert result == {
        "logical_provider_requests": 0,
        "control_turn_count": 1,
        "session_projection_count": 0,
        "delivery_count": 0,
        "content_submission_count": 1,
        "source_poll_count": 2,
    }


def test_protected_workspace_snapshot_is_read_only_and_complete(tmp_path: Path) -> None:
    protected = tmp_path / "formal"
    protected.mkdir()
    archive = protected / "proactive.db"
    archive_connection = sqlite3.connect(archive)
    archive_connection.execute("CREATE TABLE archive(id INTEGER PRIMARY KEY)")
    archive_connection.commit()
    archive_connection.close()
    database = protected / "sessions.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, body TEXT)")
    connection.execute("INSERT INTO messages(body) VALUES ('kept')")
    connection.commit()
    connection.close()

    before = cast(dict[str, Any], snapshot_protected_workspace(protected))
    after = cast(dict[str, Any], snapshot_protected_workspace(protected))

    assert before == after
    assert before["sqlite"]["sessions.db"] == {
        "integrity": "ok",
        "quick_check": "ok",
        "rows": {"messages": 1},
    }
    assert set(before["old_island"]) == {"proactive.db"}


def test_live_formal_change_is_reported_without_blame_on_isolated_chain(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "formal"
    protected.mkdir()
    database = protected / "sessions.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    before_a = snapshot_protected_workspace(protected)
    before_b = snapshot_protected_workspace(protected)
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO messages DEFAULT VALUES")
    connection.commit()
    connection.close()
    after = snapshot_protected_workspace(protected)

    evidence = _formal_evidence(before_a, before_b, after)

    assert evidence["status"] == "formal_concurrent_change"
    assert evidence["deployment_gate_verified"] is False
    assert evidence["baseline_stable"] is True
    assert evidence["after_change_count"] == 1
    assert evidence["after_changes"] == (
        {"path": "sessions.db", "types": ["digest_or_size", "sqlite_state"]},
    )


def test_live_formal_baseline_change_keeps_its_own_path_and_type(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "formal"
    protected.mkdir()
    database = protected / "sessions.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    before_a = snapshot_protected_workspace(protected)
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO messages DEFAULT VALUES")
    connection.commit()
    connection.close()
    before_b = snapshot_protected_workspace(protected)
    after = snapshot_protected_workspace(protected)

    evidence = _formal_evidence(before_a, before_b, after)

    assert evidence["status"] == "formal_concurrent_change"
    assert evidence["baseline_stable"] is False
    assert evidence["baseline_change_count"] == 1
    assert evidence["baseline_changes"] == (
        {"path": "sessions.db", "types": ["digest_or_size", "sqlite_state"]},
    )
    assert evidence["after_change_count"] == 0
    assert evidence["after_changes"] == ()


def test_process_isolation_reports_only_counts_and_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "formal"
    isolated = tmp_path / "isolated"
    protected.mkdir()
    isolated.mkdir()
    monkeypatch.delenv("AKASIC_WORKSPACE", raising=False)

    assert _process_isolation_evidence(protected, isolated) == {
        "formal_fd_reference_count": 0,
        "formal_env_reference_count": 0,
        "isolated_data_root_count": 3,
        "isolated_root_digest": hashlib.sha256(
            str(isolated).encode("utf-8")
        ).hexdigest(),
    }


def test_formal_provider_builder_preserves_profile_shape_and_manual_mutant_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PR_G_DEEPSEEK_API_KEY", "fixture-key")
    monkeypatch.setenv("PR_G_DEEPSEEK_BASE_URL", "http://127.0.0.1:1/v1")
    formal, loop_config, config = _build_selected_provider(
        tmp_path,
        tmp_path / "workspace",
    )
    manual = LLMProvider(
        api_key="fixture-key",
        base_url="http://127.0.0.1:1/v1",
        provider_name="deepseek",
    )

    assert _provider_shape(formal) == {
        "runtime_id": "main",
        "context_window": 1_000_000,
        "has_system_prompt": True,
        "extra_body": {"enable_thinking": True, "reasoning_effort": "max"},
    }
    assert _provider_shape(manual) != _provider_shape(formal)
    assert loop_config.model == "deepseek-v4-flash"
    assert loop_config.max_tokens == 0
    assert loop_config.max_iterations == 1
    assert config.extra_body == {"enable_thinking": True, "reasoning_effort": "max"}


def test_missing_secret_writes_only_a_redacted_nonzero_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "formal"
    protected.mkdir()
    report = tmp_path / "report.json"
    monkeypatch.delenv("PR_G_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("PR_G_DEEPSEEK_BASE_URL", "https://secret-endpoint.invalid")
    monkeypatch.setattr(
        "sys.argv",
        [
            "wake_v3_provider_e2e.py",
            "--protected-workspace",
            str(protected),
            "--report",
            str(report),
        ],
    )

    assert main() == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_category"] == "contract"
    assert payload["error_type"] == "GateFailure"
    assert payload["failure_code"] == "MISSING_DEEPSEEK_CREDENTIAL"
    assert payload["failure_stage"] == "credential"
    assert payload["selected_evidence"]["logical_provider_requests"] == 0
    assert payload["selected_evidence"]["http_attempts"] == 0
    assert payload["protected_workspace"]["deployment_gate_verified"] is True
    assert "secret-endpoint" not in report.read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "attempts", "retryable", "content_status"),
    [(400, 1, "false", "invalidated"), (503, 4, "true", "deferred")],
)
async def test_provider_non_2xx_keeps_safe_turn_and_terminal_oracles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    attempts: int,
    retryable: str,
    content_status: str,
) -> None:
    protected = tmp_path / "formal"
    protected.mkdir()
    requests: list[dict[str, object]] = []
    server = await _start_provider_fixture(status, requests)
    port = int(server.sockets[0].getsockname()[1])
    endpoint_marker = f"127.0.0.1:{port}"
    monkeypatch.setenv("PR_G_DEEPSEEK_API_KEY", "fixture-secret-marker")
    monkeypatch.setenv("PR_G_DEEPSEEK_BASE_URL", f"http://{endpoint_marker}/v1")
    try:
        payload = cast(
            dict[str, Any],
            await _run(
                argparse.Namespace(protected_workspace=str(protected), report="unused")
            ),
        )
    finally:
        server.close()
        await server.wait_closed()

    evidence = cast(dict[str, Any], payload["selected_evidence"])
    assert payload["status"] == "failed"
    assert payload["failure_stage"] == "selected_chain"
    assert payload["failure_code"] == "SELECTED_DELIVERY_NOT_TERMINAL"
    assert evidence["logical_provider_requests"] == 1
    assert evidence["http_attempts"] == attempts
    assert len(requests) == attempts
    assert evidence["provider_call_identity_count"] == 1
    assert evidence["provider_control_identity_count"] == 1
    assert evidence["provider_terminal_counts"] == {
        "call_done": 0,
        "call_error": 1,
        "call_cancelled": 0,
        "nonstream_done": 0,
        "nonstream_error": 1,
        "nonstream_cancelled": 0,
    }
    assert evidence["turn_count"] == 1
    assert evidence["turn_status_counts"] == {"failed": 1}
    assert evidence["turn_error_type_count"] == 1
    assert evidence["turn_error_type_digest"] is not None
    assert evidence["turn_retryable_counts"] == {
        "none": 0,
        "false": int(retryable == "false"),
        "true": int(retryable == "true"),
    }
    assert evidence["turn_final_response_present_count"] == 0
    assert evidence["turn_identity_count"] == 1
    assert evidence["turn_id_digest"] == evidence["provider_control_id_digest"]
    assert evidence["delivery_count"] == 0
    assert evidence["content_counts"] == {content_status: 1}
    assert evidence["source_ack_count"] == 0
    assert payload["protected_workspace"]["deployment_gate_verified"] is True
    encoded = json.dumps(payload, sort_keys=True)
    assert endpoint_marker not in encoded
    assert "fixture-secret-marker" not in encoded
    assert "fixture-provider-body-marker" not in encoded


@pytest.mark.asyncio
async def test_formal_provider_200_reaches_delivery_with_production_request_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "formal"
    protected.mkdir()
    requests: list[dict[str, object]] = []
    server = await _start_provider_fixture(200, requests)
    port = int(server.sockets[0].getsockname()[1])
    monkeypatch.setenv("PR_G_DEEPSEEK_API_KEY", "fixture-secret-marker")
    monkeypatch.setenv("PR_G_DEEPSEEK_BASE_URL", f"http://127.0.0.1:{port}/v1")
    try:
        payload = cast(
            dict[str, Any],
            await _run(
                argparse.Namespace(protected_workspace=str(protected), report="unused")
            ),
        )
    finally:
        server.close()
        await server.wait_closed()

    assert payload["status"] == "passed"
    assert len(requests) == 2
    request, summary_request = requests
    assert request["model"] == "deepseek-v4-flash"
    assert request["reasoning_effort"] == "max"
    assert request["thinking"] == {"type": "enabled"}
    assert "max_tokens" not in request
    tools = cast(list[dict[str, object]], request["tools"])
    tool_names = {cast(dict[str, object], tool["function"])["name"] for tool in tools}
    assert {"share_content", "skip_content"}.issubset(tool_names)
    assert "tools" not in summary_request
    messages = cast(list[dict[str, object]], request["messages"])
    first = messages[0]
    assert first.get("role") == "system"
    first_system = str(first.get("content"))
    assert _CALLER_SYSTEM_MARKER in first_system
    assert _BUILDER_SYSTEM_MARKER not in first_system
    evidence = cast(dict[str, Any], payload["selected_evidence"])
    assert evidence["provider_terminal_counts"] == {
        "call_done": 2,
        "call_error": 0,
        "call_cancelled": 0,
        "nonstream_done": 2,
        "nonstream_error": 0,
        "nonstream_cancelled": 0,
    }
    assert evidence["turn_status_counts"] == {"completed": 1}
    assert evidence["turn_final_response_present_count"] == 1
    assert evidence["turn_id_digest"] == evidence["provider_control_id_digest"]
    assert payload["selected"]["final_state"] == "settled"
    assert payload["protected_workspace"]["deployment_gate_verified"] is True
    encoded = json.dumps(payload, sort_keys=True)
    assert "fixture-secret-marker" not in encoded


def test_main_fallback_uses_only_fixed_codes_and_zero_evidence() -> None:
    payload = cast(dict[str, Any], _main_fallback_report())

    assert payload["failure_code"] == "UNHANDLED_MAIN_ERROR"
    assert payload["failure_stage"] == "main"
    assert payload["selected_evidence"]["logical_provider_requests"] == 0
    assert payload["selected_evidence"]["http_attempts"] == 0
    assert payload["protected_workspace"] == {
        "status": "after_unavailable",
        "deployment_gate_verified": False,
    }
