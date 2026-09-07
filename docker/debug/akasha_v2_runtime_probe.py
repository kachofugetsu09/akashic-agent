#!/usr/bin/env python3
"""Exercise Akasha learning through the public programmatic Message API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from docker.debug.model_plugin_fixture import add_openai_models
from docker.debug.programmatic_control_probe import (
    CheckResult,
    GateFailure,
    JsonRpcSocketClient,
    _connect_client,
    _configure_model_gate,
    _http_json,
    _model_requests,
    _prepare_host_sandbox,
    _repository_digest,
    _wait_barrier,
    _wait_http_ready,
    _wait_programmatic_result,
    _wait_socket,
)
from plugins.akasha.infrastructure.consumption import Consumption
from plugins.akasha.infrastructure.persistence import (
    load_consumption,
    logical_state_sha256,
)

READINESS_DEADLINE_S = 30.0
TURN_DEADLINE_S = 60.0
SESSION_ID = "programmatic:akasha-v2"
FIRST_INPUT_ID = "akv2-first-input"
SECOND_INPUT_ID = "akv2-second-input"


def _load_scripts(model_url: str, scripts: object) -> None:
    """Load deterministic model responses through the fixture control API."""

    _http_json("PUT", f"{model_url}/control/script", scripts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "sha256": _sha256(path),
        "size": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
    }


def _formal_identity(workspace: Path) -> dict[str, dict[str, object]]:
    return {
        relative: _file_identity(workspace / relative)
        for relative in ("sessions.db", "memory/akasha.db")
    }


def _write_runtime_config(sandbox: Path) -> None:
    """Write a private config whose runtime state stays in the sandbox."""

    config = """\
[runtime]
workspace = "/sandbox/workspace"

[agent]
system_prompt = "Use memory when relevant and follow the scripted response."
max_iterations = 4

[agent.plugins]
disabled_builtin = ["subagent"]

[agent.context]
[agent.context.compaction]
keep_recent_tokens = 20000

[app_server]
enabled = true
listen = "/sandbox/akashic.sock"
max_connections = 8
ingress_queue_size = 32
outbound_queue_size = 64

[channels.chat]
enabled = true

[channels.telegram]
token = ""

[channels.qq]
bot_uin = ""

"""
    path = sandbox / "config.toml"
    path.write_text(config, encoding="utf-8")
    path.chmod(0o600)


def _embedding_environment() -> tuple[str, str, str]:
    """Require the real embedding provider before a container is started."""

    values = tuple(
        os.environ.get(name, "").strip()
        for name in (
            "AKASHIC_E2E_EMBEDDING_API_KEY",
            "AKASHIC_E2E_EMBEDDING_BASE_URL",
            "AKASHIC_E2E_EMBEDDING_MODEL",
        )
    )
    if not all(values):
        raise GateFailure(
            "Akasha E2E embedding environment is incomplete; "
            "set AKASHIC_E2E_EMBEDDING_API_KEY, "
            "AKASHIC_E2E_EMBEDDING_BASE_URL, and "
            "AKASHIC_E2E_EMBEDDING_MODEL before starting Docker"
        )
    return cast(tuple[str, str, str], values)


def _configure_embedding(settings_url: str) -> None:
    """Register the external embedding binding without exposing its credential."""

    key, url, model = _embedding_environment()
    add_openai_models(
        settings_url,
        connection_id="akasha-embedding",
        endpoint=url,
        api_key=key,
        embedding_model=model,
        embedding_dimensions=1024,
    )


def _wait_learning(
    memory_path: Path,
    *,
    minimum_applied: int,
    timeout: float = TURN_DEADLINE_S,
) -> tuple[str, Consumption]:
    """Wait until the learning owner publishes the requested graph nodes."""

    deadline = time.monotonic() + timeout
    last_state: Consumption | None = None
    while time.monotonic() < deadline:
        if memory_path.exists():
            state = load_consumption(memory_path)
            last_state = state
            if state is not None and len(state.applied) >= minimum_applied:
                return logical_state_sha256(memory_path), state
        threading.Event().wait(0.05)
    raise GateFailure(
        "Akasha learning did not settle: "
        f"expected applied>={minimum_applied}, state={last_state!r}"
    )


def _message_text(row: object) -> str:
    if not isinstance(row, dict):
        return ""
    body = row.get("body")
    if not isinstance(body, dict):
        return ""
    return "".join(
        str(part.get("value", ""))
        for part in body.get("parts", [])
        if isinstance(part, dict) and part.get("kind") == "text"
    )


def _page_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    items = page.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise GateFailure(f"message/read items 非法：{page!r}")
    return cast(list[dict[str, Any]], items)


def _message_rows(database: Path) -> list[dict[str, object]]:
    """Read canonical Message rows, including the encoded body."""

    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, session_key, seq, ts, author, source, body "
            "FROM messages ORDER BY session_key, seq"
        ).fetchall()
    return [dict(row) for row in rows]


def _embedding_rows(database: Path) -> list[dict[str, object]]:
    """Read all durable embedding bytes without decoding provider data."""

    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT message_id, content_hash, model, hex(embedding) AS embedding_hex, "
            "dim, created_at, updated_at FROM message_embeddings "
            "ORDER BY message_id, model"
        ).fetchall()
    return [dict(row) for row in rows]


def _database_snapshot(sessions_db: Path, memory_db: Path) -> dict[str, object]:
    """Capture the complete published graph and its source-side durable facts."""

    state = load_consumption(memory_db)
    if state is None:
        raise GateFailure("Akasha memory snapshot lacks consumer_state_json")
    return {
        "logicalState": logical_state_sha256(memory_db),
        "consumerProgress": state.model_dump(mode="json"),
        "messages": _message_rows(sessions_db),
        "embeddings": _embedding_rows(sessions_db),
    }


def _runtime_identity(
    compose: list[str], repo: Path, env: dict[str, str]
) -> dict[str, object]:
    """Read the actual gateway PID, proc starttime, and command line."""

    script = r'''
import json
import pathlib


def read(pid, name):
    return pathlib.Path('/proc', str(pid), name).read_text()


def children(pid):
    raw = read(pid, 'task/' + str(pid) + '/children').split()
    return [int(value) for value in raw]

roots = [1, *children(1)]
stack = list(roots)
seen = set()
candidates = []
while stack:
    pid = stack.pop()
    if pid in seen:
        continue
    seen.add(pid)
    cmd = read(pid, 'cmdline').replace(chr(0), ' ').strip()
    if '/main.py' in cmd and ' gateway' in (' ' + cmd):
        candidates.append((pid, cmd))
    stack.extend(children(pid))
if len(candidates) != 1:
    raise RuntimeError('gateway identity ambiguous: %r' % (candidates,))
pid, cmd = candidates[0]
stat = read(pid, 'stat')
starttime = stat.rsplit(')', 1)[1].split()[19]
print(json.dumps({'pid': pid, 'starttime': int(starttime), 'cmdline': cmd}))
'''
    deadline = time.monotonic() + READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        completed = subprocess.run(
            [*compose, "exec", "-T", "akashic-control-gate", "python", "-c", script],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            try:
                payload = json.loads(completed.stdout.splitlines()[-1])
                return {
                    "pid": int(payload["pid"]),
                    "starttime": int(payload["starttime"]),
                    "cmdline": str(payload["cmdline"]),
                }
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        time.sleep(0.05)
    raise GateFailure(f"gateway runtime identity unavailable: {completed.stderr[-1000:]}")


def _inside_scenario(report_dir: Path) -> int:
    """Run first learning, read-only recall, and second learning online."""

    report_dir.mkdir(parents=True, exist_ok=True)
    events_path = report_dir / "akasha-v2-events.jsonl"
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    settings_url = "http://akashic-control-gate:2236/api/settings/model"
    endpoint = Path("/sandbox/akashic.sock")
    memory_path = Path("/sandbox/workspace/memory/akasha.db")
    checks: list[CheckResult] = []
    client: JsonRpcSocketClient | None = None
    scenario_state: dict[str, object] = {}
    try:
        _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
        _configure_model_gate()
        _configure_embedding(settings_url)
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        client = _connect_client(endpoint, events_path)
        status_before = client.request("server/status", {}).get("result")
        if not isinstance(status_before, dict) or not isinstance(
            status_before.get("bootId"), str
        ):
            raise GateFailure(f"server/status 缺少 bootId：{status_before!r}")

        admission = client.admit_programmatic(SESSION_ID, persist_memory=True)
        if admission.get("learning") != "eligible":
            raise GateFailure(f"programmatic Session 未取得 eligible 准入：{admission!r}")
        initial_hash, initial_progress = _wait_learning(memory_path, minimum_applied=0)

        _load_scripts(model_url, {"mode": "complete", "content": "first remembered answer"})
        first_ack = client.send_programmatic(
            SESSION_ID, FIRST_INPUT_ID, "alpha medical story"
        )
        first_result = _wait_programmatic_result(
            client, SESSION_ID, FIRST_INPUT_ID, timeout=TURN_DEADLINE_S
        )
        first_page = client.read_messages(SESSION_ID)
        first_items = _page_items(first_page)
        first_hash, first_progress = _wait_learning(memory_path, minimum_applied=1)

        barrier = "akasha-v2-after-recall"
        _http_json("PUT", f"{model_url}/control/barriers/{barrier}")
        _load_scripts(
            model_url,
            [
                {
                    "mode": "complete",
                    "tool_calls": [
                        {
                            "id": "call_akasha_tool_search",
                            "name": "tool_search",
                            "arguments": {"query": "select:recall_memory"},
                        }
                    ],
                },
                {
                    "mode": "complete",
                    "tool_calls": [
                        {
                            "id": "call_akasha_recall",
                            "name": "recall_memory",
                            "arguments": {
                                "query": "alpha medical story",
                                "limit": 5,
                            },
                        }
                    ],
                },
                {
                    "mode": "complete",
                    "content": "second answer after recall",
                    "barrier": barrier,
                },
            ],
        )
        second_ack = client.send_programmatic(
            SESSION_ID, SECOND_INPUT_ID, "what happened next"
        )
        _wait_barrier(model_url, barrier)
        during_recall_hash = logical_state_sha256(memory_path)
        _http_json("POST", f"{model_url}/control/barriers/{barrier}/release")
        second_result = _wait_programmatic_result(
            client, SESSION_ID, SECOND_INPUT_ID, timeout=TURN_DEADLINE_S
        )
        second_page = client.read_messages(SESSION_ID)
        second_items = _page_items(second_page)
        second_hash, second_progress = _wait_learning(memory_path, minimum_applied=2)

        requests = _model_requests(_http_json("GET", f"{model_url}/control/requests"))
        automatic_context_seen = any(
            "# Akasha memory" in json.dumps(payload, ensure_ascii=False)
            for payload in requests[1:]
        )
        tool_calls = [
            item
            for item in second_items
            if item.get("body", {}).get("kind") == "output"
            and any(
                isinstance(part, dict) and part.get("kind") == "tool_call"
                for part in item.get("body", {}).get("parts", [])
            )
        ]
        tool_results = [
            item
            for item in second_items
            if item.get("body", {}).get("kind") == "tool_result"
        ]
        tool_result_success = len(tool_results) == 2 and all(
            item.get("body", {}).get("outcome") == "success"
            for item in tool_results
        )
        scripted_tools = [
            call.get("name")
            for request in requests
            if isinstance(request, dict)
            for script in [request.get("script")]
            if isinstance(script, dict)
            for call in script.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        output_texts = [
            _message_text(item)
            for item in second_items
            if item.get("body", {}).get("kind") == "output"
        ]

        checks.extend(
            [
                CheckResult(
                    "AKV2-01",
                    first_ack.get("message_id") == FIRST_INPUT_ID
                    and first_result.get("status") == "complete"
                    and any(
                        text == "first remembered answer"
                        for text in (_message_text(item) for item in first_items)
                    ),
                    {
                        "admission": admission,
                        "ack": first_ack,
                        "result": first_result,
                        "messagePage": first_page,
                        "learning": first_progress.model_dump(mode="json"),
                    },
                ),
                CheckResult(
                    "AKV2-02",
                    automatic_context_seen,
                    {"modelRequestCount": len(requests)},
                ),
                CheckResult(
                    "AKV2-03",
                    first_hash == during_recall_hash
                    and scripted_tools == ["tool_search", "recall_memory"]
                    and len(tool_calls) == 2
                    and tool_result_success,
                    {
                        "beforeRecall": first_hash,
                        "duringRecall": during_recall_hash,
                        "scriptedTools": scripted_tools,
                        "toolCalls": tool_calls,
                        "toolResults": tool_results,
                    },
                ),
                CheckResult(
                    "AKV2-04",
                    second_ack.get("message_id") == SECOND_INPUT_ID
                    and second_result.get("status") == "complete"
                    and "second answer after recall" in output_texts
                    and second_hash != first_hash,
                    {
                        "ack": second_ack,
                        "result": second_result,
                        "messagePage": second_page,
                        "beforeCommit": first_hash,
                        "afterCommit": second_hash,
                        "learning": second_progress.model_dump(mode="json"),
                    },
                ),
            ]
        )
        scenario_state = {
            "sessionId": SESSION_ID,
            "inputIds": [FIRST_INPUT_ID, SECOND_INPUT_ID],
            "messagePage": second_page,
            "learningHash": second_hash,
            "consumerProgress": second_progress.model_dump(mode="json"),
            "providerRequestCount": len(requests),
            "initialLearningHash": initial_hash,
            "initialConsumerProgress": initial_progress.model_dump(mode="json"),
            "bootIdBefore": status_before["bootId"],
        }
    except Exception as error:
        checks.append(
            CheckResult(
                "AKV2-controller",
                False,
                {"type": type(error).__name__, "message": str(error)},
            )
        )
    finally:
        if client is not None:
            client.close()

    report = {
        "checks": [asdict(check) for check in checks],
        "passed": bool(checks) and all(check.passed for check in checks),
        "scenario": scenario_state,
    }
    (report_dir / "akasha-v2-inside.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["passed"] else 1


def _inside_restart_check(report_dir: Path) -> int:
    """Reload the published graph and read it without sending or embedding."""

    state = json.loads((report_dir / "akasha-v2-state.json").read_text(encoding="utf-8"))
    endpoint = Path("/sandbox/akashic.sock")
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    client: JsonRpcSocketClient | None = None
    try:
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        client = _connect_client(endpoint, report_dir / "restart-events.jsonl")
        status = client.request("server/status", {}).get("result")
        session_id = state.get("sessionId")
        input_ids = state.get("inputIds")
        expected_page = state.get("messagePage")
        if (
            not isinstance(session_id, str)
            or not isinstance(input_ids, list)
            or not isinstance(expected_page, dict)
        ):
            raise GateFailure(f"restart state fields invalid: {state!r}")
        page = client.read_messages(session_id)
        results = [
            client.programmatic_result(session_id, input_id)
            for input_id in input_ids
            if isinstance(input_id, str)
        ]
        request_count = len(_model_requests(_http_json("GET", f"{model_url}/control/requests")))
        passed = (
            isinstance(status, dict)
            and status.get("ready") is True
            and status.get("protocolVersion") == "2.0"
            and isinstance(state.get("bootIdBefore"), str)
            and isinstance(status.get("bootId"), str)
            and state.get("bootIdBefore") != status.get("bootId")
            and page == expected_page
            and all(result.get("status") == "complete" for result in results)
            and request_count == state.get("providerRequestCount")
        )
        evidence = {
            "status": status,
            "bootIdBefore": state.get("bootIdBefore"),
            "bootIdAfter": status.get("bootId") if isinstance(status, dict) else None,
            "messagePageBefore": expected_page,
            "messagePageAfter": page,
            "results": results,
            "providerRequestCountBefore": state.get("providerRequestCount"),
            "providerRequestCountAfter": request_count,
        }
    except Exception as error:
        passed = False
        evidence = {"type": type(error).__name__, "message": str(error)}
    finally:
        if client is not None:
            client.close()
    result = CheckResult("AKV2-05-read", passed, evidence)
    (report_dir / "akasha-v2-restart.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if passed else 1


def _run_controller(repo: Path, formal_workspace: Path | None) -> int:
    """Run the isolated online scenario, restart it, and compare durable state."""

    _embedding_environment()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    report_dir = repo / "docker/debug/reports/akasha-v2-runtime" / run_id
    report_dir.mkdir(parents=True)
    sandbox = Path(tempfile.mkdtemp(prefix="akashic-akasha-v2-gate-", dir="/tmp"))
    _prepare_host_sandbox(sandbox, repo)
    _write_runtime_config(sandbox)
    formal_before = _formal_identity(formal_workspace) if formal_workspace else None
    repository_before = _repository_digest(repo)
    env = {
        **os.environ,
        "AKASHIC_CONTROL_SANDBOX": str(sandbox),
        "UID": str(os.getuid()),
        "GID": str(os.getgid()),
    }
    project = f"akashic-akasha-v2-{run_id.lower()}"
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(repo / "docker/debug/docker-compose.control-gate.yml"),
    ]
    checks: list[CheckResult] = []
    controller_error = ""
    cleanup_returncode = -1
    residual: list[str] = []
    try:
        build = subprocess.run([*compose, "build", "model-gate"], cwd=repo, env=env, check=False)
        if build.returncode != 0:
            raise GateFailure(f"control-gate image build failed: {build.returncode}")
        up = subprocess.run(
            [*compose, "up", "-d", "model-gate", "akashic-control-gate"],
            cwd=repo,
            env=env,
            check=False,
        )
        if up.returncode != 0:
            raise GateFailure(f"compose up failed: {up.returncode}")
        inside = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "control-probe",
                "python",
                "docker/debug/akasha_v2_runtime_probe.py",
                "--inside-container",
                "--report-dir",
                "/sandbox/reports",
            ],
            cwd=repo,
            env=env,
            check=False,
        )
        inside_report = sandbox / "reports/akasha-v2-inside.json"
        if not inside_report.exists():
            raise GateFailure("inside Akasha V2 report missing")
        inside_payload = json.loads(inside_report.read_text(encoding="utf-8"))
        checks.extend(CheckResult(**item) for item in inside_payload.get("checks", []))
        scenario = inside_payload.get("scenario")
        if not isinstance(scenario, dict):
            raise GateFailure("inside Akasha V2 scenario state missing")
        (report_dir / "akasha-v2-inside.json").write_text(
            json.dumps(inside_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if inside.returncode != 0:
            raise GateFailure(f"inside Akasha V2 probe failed: {inside.returncode}")

        sessions_db = sandbox / "workspace/sessions.db"
        memory_db = sandbox / "workspace/memory/akasha.db"
        before_state = _database_snapshot(sessions_db, memory_db)
        scenario_state = dict(scenario)
        if (
            before_state.get("logicalState") != scenario_state.get("learningHash")
            or before_state.get("consumerProgress")
            != scenario_state.get("consumerProgress")
        ):
            raise GateFailure("inside learning evidence differs from host snapshot")
        scenario_state["beforeRestart"] = before_state
        (sandbox / "reports/akasha-v2-state.json").write_text(
            json.dumps(scenario_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        runtime_before = _runtime_identity(compose, repo, env)
        stop = subprocess.run(
            [*compose, "stop", "-t", "15", "akashic-control-gate"],
            cwd=repo,
            env=env,
            check=False,
        )
        if stop.returncode != 0:
            raise GateFailure(f"gateway stop failed: {stop.returncode}")
        start = subprocess.run(
            [*compose, "start", "akashic-control-gate"],
            cwd=repo,
            env=env,
            check=False,
        )
        if start.returncode != 0:
            raise GateFailure(f"gateway restart failed: {start.returncode}")
        runtime_after = _runtime_identity(compose, repo, env)
        restart = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "control-probe",
                "python",
                "docker/debug/akasha_v2_runtime_probe.py",
                "--inside-container",
                "--phase",
                "restart-check",
                "--report-dir",
                "/sandbox/reports",
            ],
            cwd=repo,
            env=env,
            check=False,
        )
        restart_path = sandbox / "reports/akasha-v2-restart.json"
        if not restart_path.exists():
            raise GateFailure("restart Akasha V2 report missing")
        restart_result = CheckResult(**json.loads(restart_path.read_text(encoding="utf-8")))
        checks.append(restart_result)
        (report_dir / "akasha-v2-restart.json").write_text(
            restart_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        after_state = _database_snapshot(sessions_db, memory_db)
        state_equal = before_state == after_state
        runtime_changed = (
            runtime_before.get("pid") != runtime_after.get("pid")
            or runtime_before.get("starttime") != runtime_after.get("starttime")
        )
        logical_equal = before_state.get("logicalState") == after_state.get("logicalState")
        progress_equal = before_state.get("consumerProgress") == after_state.get("consumerProgress")
        messages_equal = before_state.get("messages") == after_state.get("messages")
        embeddings_equal = before_state.get("embeddings") == after_state.get("embeddings")
        restart_evidence = restart_result.evidence
        boot_changed = (
            isinstance(restart_evidence, dict)
            and isinstance(restart_evidence.get("bootIdBefore"), str)
            and isinstance(restart_evidence.get("bootIdAfter"), str)
            and restart_evidence.get("bootIdBefore")
            != restart_evidence.get("bootIdAfter")
        )
        checks.append(
            CheckResult(
                "AKV2-05",
                restart.returncode == 0
                and restart_result.passed
                and runtime_changed
                and boot_changed
                and logical_equal
                and progress_equal
                and messages_equal
                and embeddings_equal
                and state_equal,
                {
                    "runtimeBefore": runtime_before,
                    "runtimeAfter": runtime_after,
                    "runtimeChanged": runtime_changed,
                    "bootIdBefore": (
                        restart_evidence.get("bootIdBefore")
                        if isinstance(restart_evidence, dict)
                        else None
                    ),
                    "bootIdAfter": (
                        restart_evidence.get("bootIdAfter")
                        if isinstance(restart_evidence, dict)
                        else None
                    ),
                    "bootChanged": boot_changed,
                    "logicalStateBefore": before_state.get("logicalState"),
                    "logicalStateAfter": after_state.get("logicalState"),
                    "logicalStateUnchanged": logical_equal,
                    "consumerProgressUnchanged": progress_equal,
                    "rawMessagesUnchanged": messages_equal,
                    "embeddingsUnchanged": embeddings_equal,
                    "restartProbeReturncode": restart.returncode,
                },
            )
        )
        (report_dir / "akasha-v2-state-before.json").write_text(
            json.dumps(before_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (report_dir / "akasha-v2-state-after.json").write_text(
            json.dumps(after_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as error:
        controller_error = f"{type(error).__name__}: {error}"
    finally:
        logs = subprocess.run(
            [*compose, "logs", "--no-color"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (report_dir / "compose.log").write_text(logs.stdout, encoding="utf-8")
        cleanup = subprocess.run(
            [*compose, "down", "--remove-orphans", "--volumes"],
            cwd=repo,
            env=env,
            check=False,
        )
        cleanup_returncode = cleanup.returncode
        residual = subprocess.run(
            [*compose, "ps", "-aq"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        ).stdout.split()
        formal_after = _formal_identity(formal_workspace) if formal_workspace else None
        repository_after = _repository_digest(repo)
        checks.append(
            CheckResult(
                "AKV2-06",
                cleanup_returncode == 0
                and not residual
                and formal_before == formal_after
                and repository_before == repository_after,
                {
                    "cleanupReturncode": cleanup_returncode,
                    "residualContainers": residual,
                    "formalWorkspaceUnchanged": formal_before == formal_after,
                    "repositoryUnchanged": repository_before == repository_after,
                },
            )
        )
        shutil.rmtree(sandbox, ignore_errors=True)

    passed = not controller_error and bool(checks) and all(check.passed for check in checks)
    report = {
        "runId": run_id,
        "status": "passed" if passed else "failed",
        "checks": [asdict(check) for check in checks],
        "controllerError": controller_error,
        "reportDir": str(report_dir),
    }
    (report_dir / "gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--phase", choices=("scenario", "restart-check"), default="scenario")
    parser.add_argument("--report-dir", type=Path, default=Path("/sandbox/reports"))
    parser.add_argument("--formal-workspace", type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.inside_container:
        if arguments.phase == "restart-check":
            return _inside_restart_check(arguments.report_dir)
        return _inside_scenario(arguments.report_dir)
    formal_workspace = (
        arguments.formal_workspace.resolve(strict=True)
        if arguments.formal_workspace
        else None
    )
    return _run_controller(Path(__file__).resolve().parents[2], formal_workspace)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1) from error
