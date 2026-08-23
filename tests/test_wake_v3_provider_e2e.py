from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from docker.debug.wake_v3_provider_e2e import (
    ScriptedProvider,
    _formal_evidence,
    _main_fallback_report,
    _process_isolation_evidence,
    _run,
    main,
    run_quiet_suite,
    run_suite,
    snapshot_protected_workspace,
)


@pytest.mark.asyncio
async def test_selected_chain_uses_one_react_delivery_projection_and_ack(
    tmp_path: Path,
) -> None:
    result = await run_suite(tmp_path, provider=ScriptedProvider())

    assert result["logical_provider_requests"] == 1
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

    assert result["logical_provider_requests"] == 1
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

    before = snapshot_protected_workspace(protected)
    after = snapshot_protected_workspace(protected)

    assert before == after
    assert before["sqlite"]["sessions.db"] == {
        "integrity": "ok",
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
        "isolated_root_digest": hashlib.sha256(str(isolated).encode("utf-8")).hexdigest(),
    }


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
async def test_provider_non_2xx_keeps_safe_counts_identities_and_formal_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        header = await reader.readuntil(b"\r\n\r\n")
        content_length = next(
            (
                int(line.split(b":", 1)[1].strip())
                for line in header.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            ),
            0,
        )
        if content_length:
            _ = await reader.readexactly(content_length)
        body = b'{"error":{"message":"fixture-provider-body-marker"}}'
        writer.write(
            b"HTTP/1.1 503 Service Unavailable\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Content-Type: application/json\r\nConnection: close\r\n\r\n"
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    protected = tmp_path / "formal"
    protected.mkdir()
    server = await asyncio.start_server(reject, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    endpoint_marker = f"127.0.0.1:{port}"
    monkeypatch.setenv("PR_G_DEEPSEEK_API_KEY", "fixture-secret-marker")
    monkeypatch.setenv("PR_G_DEEPSEEK_BASE_URL", f"http://{endpoint_marker}/v1")
    try:
        payload = await _run(
            argparse.Namespace(protected_workspace=str(protected), report="unused")
        )
    finally:
        server.close()
        await server.wait_closed()

    evidence = payload["selected_evidence"]
    assert payload["status"] == "failed"
    assert payload["failure_stage"] == "selected_chain"
    assert payload["failure_code"] == "SELECTED_DELIVERY_NOT_TERMINAL"
    assert evidence["logical_provider_requests"] == 1
    assert evidence["http_attempts"] == 3
    assert evidence["provider_call_identity_count"] == 1
    assert evidence["provider_control_identity_count"] == 1
    assert evidence["delivery_count"] == 0
    assert evidence["source_ack_count"] == 0
    assert payload["protected_workspace"]["deployment_gate_verified"] is True
    encoded = json.dumps(payload, sort_keys=True)
    assert endpoint_marker not in encoded
    assert "fixture-secret-marker" not in encoded
    assert "fixture-provider-body-marker" not in encoded


def test_main_fallback_uses_only_fixed_codes_and_zero_evidence() -> None:
    payload = _main_fallback_report()

    assert payload["failure_code"] == "UNHANDLED_MAIN_ERROR"
    assert payload["failure_stage"] == "main"
    assert payload["selected_evidence"]["logical_provider_requests"] == 0
    assert payload["selected_evidence"]["http_attempts"] == 0
    assert payload["protected_workspace"] == {
        "status": "after_unavailable",
        "deployment_gate_verified": False,
    }
