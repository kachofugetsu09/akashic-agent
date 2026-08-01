#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docker.debug.programmatic_control_probe import (
    CheckResult,
    GateFailure,
    JsonRpcSocketClient,
    _connect_client,
    _event_turn,
    _extract_id,
    _http_json,
    _initialize_current_workspace,
    _model_requests,
    _prepare_host_sandbox,
    _recorded_turn_notifications,
    _repository_digest,
    _start_thread,
    _terminal_status,
    _wait_barrier,
    _wait_http_ready,
    _wait_socket,
    _write_json,
)


READINESS_DEADLINE_S = 30.0
MODEL_URL = "http://model-gate:8090"
ENDPOINT = Path("/sandbox/akashic.sock")
WORKSPACE = Path("/sandbox/workspace")

MCP_SERVER_SOURCE = r'''from __future__ import annotations
import json
import os
from pathlib import Path
import sys

log = Path(os.environ["LIFECYCLE_LOG"])
version = os.environ["VERSION"]
pid = os.getpid()

def record(event: str) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "pid": pid, "version": version}) + "\n")

record("started")
try:
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method = message.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2025-11-25"}
        elif method == "tools/list":
            result = {"tools": [{"name": "version", "description": "Return version", "inputSchema": {"type": "object", "properties": {}}}]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": version}]}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
finally:
    record("stopped")
'''


def _load_scripts(scripts: list[dict[str, object]]) -> int:
    payload = _http_json("PUT", f"{MODEL_URL}/control/script", scripts)
    loaded = cast(dict[str, object], payload)["loaded"]
    if not isinstance(loaded, int):
        raise GateFailure(f"model gate loaded 响应非法: {payload!r}")
    return loaded


def _requests() -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        _model_requests(_http_json("GET", f"{MODEL_URL}/control/requests")),
    )


def _start_probe_turn(
    client: JsonRpcSocketClient,
    thread_id: str,
    text: str,
) -> str:
    return _extract_id(
        client.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": text,
                "metadata": {"skip_memory_context_guard": True},
            },
        ),
        "turn",
    )


def _tool_names(request: dict[str, Any]) -> set[str]:
    payload = request.get("payload")
    if not isinstance(payload, dict):
        raise GateFailure(f"model request 缺少 payload: {request!r}")
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return set()
    return {
        str(item.get("function", {}).get("name"))
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }


def _read_ready() -> dict[str, Any]:
    path = WORKSPACE / ".runtime-ready.json"
    deadline = time.monotonic() + READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
            continue
        if payload.get("state") == "ready":
            return cast(dict[str, Any], payload)
        time.sleep(0.02)
    raise GateFailure("runtime readiness 超时")


def _connect_new_boot(old_boot: str, events_path: Path) -> tuple[JsonRpcSocketClient, dict[str, Any]]:
    deadline = time.monotonic() + READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        ready = _read_ready()
        if ready.get("bootId") != old_boot:
            try:
                return _connect_client(ENDPOINT, events_path), ready
            except (ConnectionError, OSError, GateFailure):
                pass
        time.sleep(0.02)
    raise GateFailure(f"等待新 boot 超时: old={old_boot}")


def _restart_scripts(index: int, barrier: str) -> list[dict[str, object]]:
    return [
        {
            "mode": "stream",
            "deltas": [],
            "tool_calls": [
                {
                    "id": f"call_search_{index}",
                    "name": "tool_search",
                    "arguments": {"query": "select:agent_restart"},
                }
            ],
        },
        {
            "mode": "stream",
            "deltas": [],
            "tool_calls": [
                {
                    "id": f"call_restart_{index}",
                    "name": "agent_restart",
                    "arguments": {"reason": f"restart gate iteration {index}"},
                }
            ],
        },
        {
            "mode": "complete",
            "content": f"restart-complete-{index}",
            "barrier": barrier,
        },
    ]


def _mcp_scripts(version: str) -> list[dict[str, object]]:
    return [
        {
            "mode": "stream",
            "deltas": [],
            "tool_calls": [{
                "id": f"call_mcp_search_{version}",
                "name": "tool_search",
                "arguments": {"query": "select:mcp_restart_probe__version"},
            }],
        },
        {
            "mode": "stream",
            "deltas": [],
            "tool_calls": [{
                "id": f"call_mcp_version_{version}",
                "name": "mcp_restart_probe__version",
                "arguments": {},
            }],
        },
        {"mode": "complete", "content": f"mcp-{version}"},
    ]


def _process_identity(pid: int) -> dict[str, int]:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = stat[stat.rfind(")") + 2 :].split()
    return {"pid": pid, "starttime": int(fields[19])}


def _identity_alive(identity: dict[str, int]) -> bool:
    try:
        stat = Path(f"/proc/{identity['pid']}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    fields = stat[stat.rfind(")") + 2 :].split()
    return fields[0] != "Z" and {
        "pid": identity["pid"],
        "starttime": int(fields[19]),
    } == identity


def _running_mcp_identity(
    version: str,
    *,
    workspace: Path = WORKSPACE,
    previous: dict[str, int] | None = None,
) -> dict[str, int]:
    deadline = time.monotonic() + READINESS_DEADLINE_S
    lifecycle = workspace / "mcp/restart-probe-lifecycle.jsonl"
    while time.monotonic() < deadline:
        records = (
            [json.loads(line) for line in lifecycle.read_text().splitlines() if line]
            if lifecycle.exists()
            else []
        )
        started = [
            int(item["pid"])
            for item in records
            if item["event"] == "started" and item["version"] == version
        ]
        for pid in reversed(started):
            try:
                identity = _process_identity(pid)
            except FileNotFoundError:
                continue
            if identity != previous:
                return identity
        time.sleep(0.02)
    raise GateFailure(f"MCP {version} 子进程未启动")


def _wait_identity_exit(identity: dict[str, int]) -> None:
    deadline = time.monotonic() + READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        if not _identity_alive(identity):
            return
        time.sleep(0.02)
    raise GateFailure(f"旧进程 identity 未退出: {identity}")


def _write_mcp_declaration(version: str, *, workspace: Path = WORKSPACE) -> None:
    root = workspace / "mcp"
    declarations = root / "servers"
    declarations.mkdir(parents=True, exist_ok=True)
    server = root / "restart_probe_server.py"
    server.write_text(MCP_SERVER_SOURCE, encoding="utf-8")
    lifecycle = root / "restart-probe-lifecycle.jsonl"
    (declarations / "restart_probe.toml").write_text(
        "schema_version = 1\n"
        'name = "restart_probe"\n'
        f'command = ["{sys.executable}", "{server}"]\n'
        "[env]\n"
        f'VERSION = "{version}"\n'
        f'LIFECYCLE_LOG = "{lifecycle}"\n',
        encoding="utf-8",
    )


def _run_mcp_call(
    client: JsonRpcSocketClient,
    version: str,
    label: str,
) -> CheckResult:
    before = len(_requests())
    _load_scripts(_mcp_scripts(version))
    thread_id = _start_thread(client, f"mcp-{label}")
    turn_id = _start_probe_turn(client, thread_id, f"call MCP {version}")
    terminal = client.wait_terminal(turn_id)
    requests = _requests()[before:]
    payload = _event_turn(terminal)
    calls = [
        item for item in payload.get("items", [])
        if isinstance(item, dict) and item.get("type") == "toolCall"
    ]
    version_call = next(
        (item for item in calls if item.get("data", {}).get("name") == "mcp_restart_probe__version"),
        None,
    )
    passed = (
        len(requests) == 3
        and "mcp_restart_probe__version" not in _tool_names(requests[0])
        and "mcp_restart_probe__version" in _tool_names(requests[1])
        and version_call is not None
        and version_call.get("data", {}).get("status") == "success"
        and version_call.get("data", {}).get("resultPreview") == version
        and _terminal_status(terminal) == "completed"
    )
    return CheckResult(
        f"MCP-{label}",
        passed,
        {
            "threadId": thread_id,
            "turnId": turn_id,
            "version": version,
            "initialTools": sorted(_tool_names(requests[0])) if requests else [],
            "postSearchTools": sorted(_tool_names(requests[1])) if len(requests) > 1 else [],
            "toolCall": version_call,
        },
    )


def _database_turn(turn_id: str) -> dict[str, Any]:
    with sqlite3.connect(WORKSPACE / "sessions.db") as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT id, status, final_response FROM turns WHERE id = ?",
            (turn_id,),
        ).fetchone()
    if row is None:
        raise GateFailure(f"DB 缺少 turn: {turn_id}")
    return dict(row)


def _sample_supervisor_children(
    supervisor_pid: int,
    stop: threading.Event,
    samples: list[int],
) -> None:
    path = Path(f"/proc/{supervisor_pid}/task/{supervisor_pid}/children")
    while not stop.is_set():
        try:
            pids = path.read_text(encoding="utf-8").split()
        except FileNotFoundError:
            pids = []
        samples.append(sum(Path(f"/proc/{pid}").exists() for pid in pids))
        stop.wait(0.002)


def _run_restart_iteration(
    index: int,
    client: JsonRpcSocketClient,
    thread_id: str,
    report_dir: Path,
) -> tuple[JsonRpcSocketClient, CheckResult]:
    before = _requests()
    ready_before = _read_ready()
    supervisor_pid = int((WORKSPACE / ".supervisor.pid").read_text())
    child_samples: list[int] = []
    stop_sampling = threading.Event()
    sampler = threading.Thread(
        target=_sample_supervisor_children,
        args=(supervisor_pid, stop_sampling, child_samples),
        daemon=True,
    )
    sampler.start()
    barrier = f"restart-final-{index}-{uuid.uuid4().hex[:8]}"
    _http_json("PUT", f"{MODEL_URL}/control/barriers/{barrier}")
    _load_scripts(_restart_scripts(index, barrier))
    turn_id = _start_probe_turn(client, thread_id, f"restart iteration {index}")
    terminal_holder: dict[str, Any] = {}

    def wait_terminal() -> None:
        terminal_holder["event"] = client.wait_terminal(
            turn_id,
            timeout=READINESS_DEADLINE_S,
        )

    waiter = threading.Thread(target=wait_terminal, daemon=True)
    waiter.start()
    _wait_barrier(MODEL_URL, barrier)

    # admission 已由 agent_restart 冻结，另一连接必须拿到 retryable error。
    concurrent = _connect_client(ENDPOINT, report_dir / f"events-{index}-concurrent.jsonl")
    try:
        other_thread = _start_thread(concurrent, f"restart-concurrent-{index}")
        rejected = concurrent.request_raw(
            "turn/start",
            {"threadId": other_thread, "input": "must retry", "metadata": {}},
        )
    finally:
        concurrent.close()
    _http_json("POST", f"{MODEL_URL}/control/barriers/{barrier}/release")
    waiter.join(timeout=READINESS_DEADLINE_S)
    if waiter.is_alive() or "event" not in terminal_holder:
        raise GateFailure(f"restart terminal 超时: {turn_id}")
    terminal = cast(dict[str, Any], terminal_holder["event"])
    terminal_payload = _event_turn(terminal)
    old_child_alive_at_terminal = Path(f"/proc/{ready_before['pid']}").exists()
    client.close()

    new_client, ready_after = _connect_new_boot(
        str(ready_before["bootId"]),
        report_dir / f"events-{index}-after.jsonl",
    )
    stop_sampling.set()
    sampler.join(timeout=2)
    max_concurrent_child = max(child_samples, default=0)
    time.sleep(0.1)
    stable_ready = _read_ready()
    after = _requests()
    iteration_requests = after[len(before):]
    if len(iteration_requests) != 3:
        raise GateFailure(f"restart model request 数量异常: {len(iteration_requests)}")
    called_tools = {
        str(item.get("data", {}).get("name"))
        for item in terminal_payload.get("items", [])
        if isinstance(item, dict) and item.get("type") == "toolCall"
    }
    turn_read = new_client.request(
        "turn/read",
        {"threadId": thread_id, "turnId": turn_id},
    )
    db_turn = _database_turn(turn_id)
    error = rejected.get("error")
    passed = (
        "agent_restart" not in _tool_names(iteration_requests[0])
        and "agent_restart" in _tool_names(iteration_requests[1])
        and {"tool_search", "agent_restart"} <= called_tools
        and _terminal_status(terminal) == "completed"
        and terminal_payload.get("finalResponse") == f"restart-complete-{index}"
        and turn_read.get("result") == terminal_payload
        and db_turn["status"] == "completed"
        and db_turn["final_response"] == f"restart-complete-{index}"
        and old_child_alive_at_terminal
        and ready_after["bootId"] != ready_before["bootId"]
        and ready_after["pid"] != ready_before["pid"]
        and int((WORKSPACE / ".supervisor.pid").read_text()) == supervisor_pid
        and max_concurrent_child == 1
        and stable_ready["bootId"] == ready_after["bootId"]
        and stable_ready["pid"] == ready_after["pid"]
        and isinstance(error, dict)
        and error.get("data", {}).get("retryable") is True
    )
    return new_client, CheckResult(
        f"RESTART-{index}",
        passed,
        {
            "threadId": thread_id,
            "turnId": turn_id,
            "before": ready_before,
            "after": ready_after,
            "supervisorPid": supervisor_pid,
            "maxConcurrentChild": max_concurrent_child,
            "restartCount": 1,
            "stableAfterRestart": stable_ready,
            "oldChildAliveAtTerminal": old_child_alive_at_terminal,
            "initialTools": sorted(_tool_names(iteration_requests[0])),
            "postSearchTools": sorted(_tool_names(iteration_requests[1])),
            "calledTools": sorted(called_tools),
            "concurrentResponse": rejected,
            "database": db_turn,
        },
    )


def _disconnect_before_terminal_check(report_dir: Path) -> CheckResult:
    client = _connect_client(ENDPOINT, report_dir / "events-disconnect.jsonl")
    ready_before = _read_ready()
    thread_id = _start_thread(client, "restart-disconnect")
    barrier = f"restart-disconnect-{uuid.uuid4().hex[:8]}"
    _http_json("PUT", f"{MODEL_URL}/control/barriers/{barrier}")
    _load_scripts(_restart_scripts(999, barrier))
    turn_id = _start_probe_turn(client, thread_id, "disconnect before terminal")
    _wait_barrier(MODEL_URL, barrier)
    client.close()
    disconnected_at = time.monotonic()
    _http_json("POST", f"{MODEL_URL}/control/barriers/{barrier}/release")

    recovery = _connect_client(ENDPOINT, report_dir / "events-recovery.jsonl")
    recovery_thread = _start_thread(recovery, "restart-disconnect-recovery")
    _load_scripts([{"mode": "complete", "content": "admission-restored"}])
    deadline = time.monotonic() + 3
    rejected: list[dict[str, Any]] = []
    recovery_turn = ""
    while time.monotonic() < deadline:
        response = recovery.request_raw(
            "turn/start",
            {
                "threadId": recovery_thread,
                "input": "recovery after disconnect",
                "metadata": {},
            },
        )
        if "result" in response:
            recovery_turn = _extract_id(response, "turn")
            break
        rejected.append(response)
        time.sleep(0.02)
    if not recovery_turn:
        raise GateFailure("disconnect 后 admission 未在 3 秒内恢复")
    terminal = recovery.wait_terminal(recovery_turn)
    recovery.close()
    elapsed = time.monotonic() - disconnected_at
    ready_after = _read_ready()
    return CheckResult(
        "RESTART-DISCONNECT",
        _terminal_status(terminal) == "completed"
        and _event_turn(terminal).get("finalResponse") == "admission-restored"
        and ready_after["bootId"] == ready_before["bootId"]
        and ready_after["pid"] == ready_before["pid"]
        and elapsed < 3,
        {
            "restartTurnId": turn_id,
            "recoveryTurnId": recovery_turn,
            "before": ready_before,
            "after": ready_after,
            "recoverySeconds": elapsed,
            "rejectedBeforeRecovery": rejected,
        },
    )


def _process_metrics(pid: int) -> dict[str, int]:
    status = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"VmRSS", "VmHWM"}:
            status[key] = int(value.split()[0])
    return {
        "pid": pid,
        "fds": len(list(Path(f"/proc/{pid}/fd").iterdir())),
        "threads": len(list(Path(f"/proc/{pid}/task").iterdir())),
        "vmRssKiB": status["VmRSS"],
        "vmHwmKiB": status["VmHWM"],
    }


def _descendant_pids(pid: int) -> set[int]:
    descendants: set[int] = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        path = Path(f"/proc/{parent}/task/{parent}/children")
        if not path.exists():
            continue
        children = {int(item) for item in path.read_text().split()}
        new = children - descendants
        descendants.update(new)
        pending.extend(new)
    return descendants


def _zombie_descendants(supervisor_pid: int) -> list[int]:
    zombies: list[int] = []
    for pid in _descendant_pids(supervisor_pid):
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            zombies.append(pid)
    return sorted(zombies)


def _peak_memory_deltas(
    supervisor_before: dict[str, int],
    child_before: dict[str, int],
    resource_samples: list[dict[str, dict[str, int]]],
) -> dict[str, int]:
    """计算各进程跨轮次 RSS 与 HWM 最大增量。"""

    return {
        "supervisorRssKiB": max(
            sample["supervisor"]["vmRssKiB"] - supervisor_before["vmRssKiB"]
            for sample in resource_samples
        ),
        "supervisorHwmKiB": max(
            sample["supervisor"]["vmHwmKiB"] - supervisor_before["vmHwmKiB"]
            for sample in resource_samples
        ),
        "childRssKiB": max(
            sample["child"]["vmRssKiB"] - child_before["vmRssKiB"]
            for sample in resource_samples
        ),
        "childHwmKiB": max(
            sample["child"]["vmHwmKiB"] - child_before["vmHwmKiB"]
            for sample in resource_samples
        ),
    }


def _memory_within_limit(deltas: dict[str, int], limit_kib: int = 64 * 1024) -> bool:
    return all(delta <= limit_kib for delta in deltas.values())


def _active_turn_count() -> int:
    with sqlite3.connect(WORKSPACE / "sessions.db") as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM turns WHERE status IN ('queued', 'in_progress')"
        ).fetchone()
    return int(row[0])


def _resource_check(
    supervisor_before: dict[str, int],
    child_before: dict[str, int],
    zombie_samples: list[list[int]],
    resource_samples: list[dict[str, dict[str, int]]],
) -> CheckResult:
    time.sleep(0.2)
    supervisor_after = _process_metrics(supervisor_before["pid"])
    child_after = _process_metrics(int(_read_ready()["pid"]))
    supervisor_delta = {
        "fds": supervisor_after["fds"] - supervisor_before["fds"],
        "threads": supervisor_after["threads"] - supervisor_before["threads"],
    }
    child_delta = {
        "fds": child_after["fds"] - child_before["fds"],
        "threads": child_after["threads"] - child_before["threads"],
    }
    memory_deltas = _peak_memory_deltas(
        supervisor_before, child_before, resource_samples
    )
    zombies = _zombie_descendants(supervisor_before["pid"])
    active_turns = _active_turn_count()
    return CheckResult(
        "RESTART-SOAK-RESOURCES",
        supervisor_delta["fds"] <= 2
        and supervisor_delta["threads"] <= 0
        and child_delta["fds"] <= 4
        and child_delta["threads"] <= 2
        and _memory_within_limit(memory_deltas)
        and not zombies
        and not any(zombie_samples)
        and active_turns == 0,
        {
            "supervisorBefore": supervisor_before,
            "supervisorAfter": supervisor_after,
            "supervisorDelta": supervisor_delta,
            "childBefore": child_before,
            "childAfter": child_after,
            "childDelta": child_delta,
            "resourceSamples": resource_samples,
            "peakMemoryDeltasKiB": memory_deltas,
            "zombies": zombies,
            "zombieSamples": zombie_samples,
            "activeTurns": active_turns,
            "thresholds": {
                "supervisorFds": 2,
                "supervisorThreads": 0,
                "childFds": 4,
                "childThreads": 2,
                "supervisorRssKiB": 64 * 1024,
                "supervisorHwmKiB": 64 * 1024,
                "childRssKiB": 64 * 1024,
                "childHwmKiB": 64 * 1024,
            },
        },
    )


def _unsupervised_tool_absence_check(report_dir: Path) -> CheckResult:
    config = Path("/sandbox/config.toml").read_text(encoding="utf-8")
    config = config.replace(
        'listen = "/sandbox/akashic.sock"',
        'listen = "/sandbox/unsupervised.sock"',
    ).replace(
        "[channels.chat]\nenabled = true",
        "[channels.chat]\nenabled = false",
    )
    config_path = Path("/sandbox/unsupervised.toml")
    workspace = Path("/sandbox/unsupervised-workspace")
    endpoint = Path("/sandbox/unsupervised.sock")
    config_path.write_text(config, encoding="utf-8")
    _copy_gate_migration_cursor(config_path)
    _initialize_current_workspace(workspace, Path("/app"))
    process = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "gateway",
            "--config",
            str(config_path),
            "--workspace",
            str(workspace),
        ]
    )
    try:
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        client = _connect_client(endpoint, report_dir / "events-unsupervised.jsonl")
        before = len(_requests())
        _load_scripts(
            [
                {
                    "mode": "stream",
                    "deltas": [],
                    "tool_calls": [
                        {
                            "id": "call_unsupervised_search",
                            "name": "tool_search",
                            "arguments": {"query": "select:agent_restart"},
                        }
                    ],
                },
                {"mode": "complete", "content": "unsupervised-complete"},
            ]
        )
        thread_id = _start_thread(client, "restart-unsupervised")
        turn_id = _start_probe_turn(client, thread_id, "find restart")
        terminal = client.wait_terminal(turn_id)
        client.close()
        requests = _requests()[before:]
        notifications = _recorded_turn_notifications(
            report_dir / "events-unsupervised.jsonl",
            turn_id,
        )
        previews = [
            str(event.get("params", {}).get("item", {}).get("data", {}).get("resultPreview") or "")
            for event in notifications
            if event.get("method") == "item/completed"
        ]
        passed = (
            _terminal_status(terminal) == "completed"
            and all("agent_restart" not in _tool_names(request) for request in requests)
            and any("未找到工具: agent_restart" in preview for preview in previews)
        )
        return CheckResult(
            "RESTART-UNSUPERVISED",
            passed,
            {
                "turnId": turn_id,
                "requestTools": [sorted(_tool_names(item)) for item in requests],
                "previews": previews,
            },
        )
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=15)
        endpoint.unlink(missing_ok=True)


def _inside(iterations: int, report_dir: Path, *, resource_gate: bool) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    _wait_http_ready(f"{MODEL_URL}/readyz", READINESS_DEADLINE_S)
    _wait_socket(ENDPOINT, READINESS_DEADLINE_S)
    events_path = report_dir / "events.jsonl"
    client = _connect_client(ENDPOINT, events_path)
    checks: list[CheckResult] = []
    try:
        isolation = {
            "extraPluginDirs": os.environ.get("AKASHIC_EXTRA_PLUGIN_DIRS"),
            "pluginCacheExists": Path("/sandbox/home/.akashic-plugin/cache").exists(),
        }
        checks.append(
            CheckResult(
                "RESTART-ISOLATION",
                isolation["extraPluginDirs"] is None
                and isolation["pluginCacheExists"] is False,
                isolation,
            )
        )
        previous_mcp_identity: dict[str, int] | None = None
        supervisor_baseline: dict[str, int] | None = None
        child_baseline: dict[str, int] | None = None
        zombie_samples: list[list[int]] = []
        resource_samples: list[dict[str, dict[str, int]]] = []
        for index in range(iterations):
            version = f"v{index + 1}"
            _write_mcp_declaration(version)
            mcp_identity = _running_mcp_identity(
                version, previous=previous_mcp_identity
            )
            if previous_mcp_identity is not None:
                _wait_identity_exit(previous_mcp_identity)
            checks.append(_run_mcp_call(client, version, f"{index}-HOT"))

            if supervisor_baseline is None:
                supervisor_pid = int((WORKSPACE / ".supervisor.pid").read_text())
                supervisor_baseline = _process_metrics(supervisor_pid)
                child_baseline = _process_metrics(int(_read_ready()["pid"]))

            thread_id = _start_thread(client, f"restart-gate-{index}")
            client, result = _run_restart_iteration(
                index,
                client,
                thread_id,
                report_dir,
            )
            checks.append(result)
            recovered_mcp_identity = _running_mcp_identity(
                version, previous=mcp_identity
            )
            _wait_identity_exit(mcp_identity)
            checks.append(
                CheckResult(
                    f"MCP-{index}-RECOVERED",
                    recovered_mcp_identity != mcp_identity,
                    {
                        "version": version,
                        "oldIdentity": mcp_identity,
                        "oldIdentityAlive": _identity_alive(mcp_identity),
                        "newIdentity": recovered_mcp_identity,
                    },
                )
            )
            checks.append(_run_mcp_call(client, version, f"{index}-AFTER-RESTART"))
            previous_mcp_identity = recovered_mcp_identity
            zombie_samples.append(_zombie_descendants(supervisor_baseline["pid"]))

            # 同 thread resume，同时证明新 turn 首次 payload 再次不含 restart。
            before = len(_requests())
            _load_scripts([{"mode": "complete", "content": f"resume-{index}"}])
            resumed = client.request("thread/resume", {"threadId": thread_id})
            if resumed.get("result", {}).get("id") != thread_id:
                raise GateFailure(f"thread resume 不一致: {resumed!r}")
            resume_turn = _start_probe_turn(client, thread_id, f"resume {index}")
            resume_terminal = client.wait_terminal(resume_turn)
            request = _requests()[before]
            checks.append(
                CheckResult(
                    f"RESTART-{index}-RESUME",
                    _terminal_status(resume_terminal) == "completed"
                    and _event_turn(resume_terminal).get("finalResponse") == f"resume-{index}"
                    and "agent_restart" not in _tool_names(request),
                    {
                        "threadId": thread_id,
                        "turnId": resume_turn,
                        "initialTools": sorted(_tool_names(request)),
                    },
                )
            )
            resource_samples.append(
                {
                    "supervisor": _process_metrics(supervisor_baseline["pid"]),
                    "child": _process_metrics(int(_read_ready()["pid"])),
                }
            )
        checks.append(_disconnect_before_terminal_check(report_dir))
        if resource_gate:
            if supervisor_baseline is None or child_baseline is None:
                raise GateFailure("soak resource baseline 缺失")
            checks.append(
                _resource_check(
                    supervisor_baseline,
                    child_baseline,
                    zombie_samples,
                    resource_samples,
                )
            )
    finally:
        client.close()

    report = {
        "gate": "restart",
        "iterations": iterations,
        "status": "passed" if all(check.passed for check in checks) else "failed",
        "checks": [asdict(check) for check in checks],
    }
    _write_json(report_dir / "restart-gate.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


def _inside_unsupervised(report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    check = _unsupervised_tool_absence_check(report_dir)
    _write_json(report_dir / "unsupervised.json", asdict(check))
    print(json.dumps(asdict(check), ensure_ascii=False))
    return 0 if check.passed else 1


def _copy_gate_migration_cursor(config_path: Path) -> None:
    source = Path("/sandbox/config.toml.migration-cursor")
    target = config_path.with_name(f"{config_path.name}.migration-cursor")
    shutil.copyfile(source, target)


def _isolated_config(name: str) -> tuple[Path, Path, Path]:
    source = Path("/sandbox/config.toml").read_text(encoding="utf-8")
    endpoint = Path(f"/sandbox/{name}.sock")
    source = source.replace(
        'listen = "/sandbox/akashic.sock"',
        f'listen = "{endpoint}"',
    ).replace(
        "[channels.chat]\nenabled = true",
        "[channels.chat]\nenabled = false",
    )
    config = Path(f"/sandbox/{name}.toml")
    workspace = Path(f"/sandbox/{name}-workspace")
    config.write_text(source, encoding="utf-8")
    _copy_gate_migration_cursor(config)
    _initialize_current_workspace(workspace, Path("/app"))
    return config, workspace, endpoint


def _install_startup_plugin(home: Path, name: str, source: str) -> None:
    cache = home / f".akashic-plugin/cache/gate/{name}/1.0.0"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "plugin.py").write_text(source, encoding="utf-8")
    manifest = home / ".akashic-plugin/manifest.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        f'[plugins."{name}@gate"]\nenabled = true\n',
        encoding="utf-8",
    )


def _wait_scenario_ready(path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + READINESS_DEADLINE_S
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
            continue
        if payload.get("state") == "ready":
            return cast(dict[str, Any], payload)
        time.sleep(0.02)
    raise GateFailure(f"场景 readiness 超时: {path}")


def _failure_mode_checks(report_dir: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []

    # 1. child 裸 75 没有私有 commit，supervisor 必须以 70 失败且只启动一次。
    naked_config, naked_workspace, _ = _isolated_config("naked75")
    naked_home = Path("/sandbox/naked75-home")
    count_path = Path("/sandbox/naked75-count.txt")
    _install_startup_plugin(
        naked_home,
        "naked75",
        "import os, pathlib\n"
        f"p=pathlib.Path({str(count_path)!r})\n"
        "p.write_text((p.read_text() if p.exists() else '') + '1\\n')\n"
        "os._exit(75)\n",
    )
    naked = subprocess.run(
        [
            sys.executable,
            "main.py",
            "supervise",
            "--config",
            str(naked_config),
            "--workspace",
            str(naked_workspace),
        ],
        env={**os.environ, "HOME": str(naked_home)},
        timeout=20,
    )
    naked_starts = count_path.read_text().splitlines() if count_path.exists() else []
    checks.append(
        CheckResult(
            "RESTART-NAKED-75",
            naked.returncode == 70 and len(naked_starts) == 1,
            {"returncode": naked.returncode, "childStarts": len(naked_starts)},
        )
    )

    # 2. stale readiness 不能满足新 boot；故障场景单独使用 15 秒失败门。
    stale_config, stale_workspace, _ = _isolated_config("stale-ready")
    stale_home = Path("/sandbox/stale-ready-home")
    stale_payload = {"bootId": "stale", "pid": 1, "state": "ready"}
    (stale_workspace / ".runtime-ready.json").write_text(json.dumps(stale_payload))
    _install_startup_plugin(stale_home, "stale_ready", "import time\ntime.sleep(30)\n")
    stale_started = time.monotonic()
    stale = subprocess.run(
        [
            sys.executable,
            "main.py",
            "supervise",
            "--config",
            str(stale_config),
            "--workspace",
            str(stale_workspace),
        ],
        env={
            **os.environ,
            "HOME": str(stale_home),
            "AKASHIC_READINESS_TIMEOUT_S": "15",
        },
        timeout=25,
    )
    stale_duration = time.monotonic() - stale_started
    stale_after = json.loads(
        (stale_workspace / ".runtime-ready.json").read_text(encoding="utf-8")
    )
    checks.append(
        CheckResult(
            "RESTART-STALE-READY",
            stale.returncode == 70
            and stale_duration >= 14
            and stale_after == stale_payload,
            {
                "returncode": stale.returncode,
                "durationSeconds": stale_duration,
                "readiness": stale_after,
            },
        )
    )

    # 3. supervisor SIGTERM 必须清理 gateway、MCP 及全部既有后代。
    stop_config, stop_workspace, _ = _isolated_config("supervisor-stop")
    stop_home = Path("/sandbox/supervisor-stop-home")
    stop_home.mkdir(exist_ok=True)
    _write_mcp_declaration("sigterm", workspace=stop_workspace)
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "supervise",
            "--config",
            str(stop_config),
            "--workspace",
            str(stop_workspace),
        ],
        env={**os.environ, "HOME": str(stop_home)},
    )
    ready_path = stop_workspace / ".runtime-ready.json"
    ready = _wait_scenario_ready(ready_path)
    child_identity = _process_identity(int(ready["pid"]))
    mcp_identity = _running_mcp_identity("sigterm", workspace=stop_workspace)
    descendant_identities = [
        _process_identity(pid) for pid in _descendant_pids(supervisor.pid)
    ]
    supervisor.send_signal(signal.SIGTERM)
    stop_exit = supervisor.wait(timeout=15)
    _wait_identity_exit(child_identity)
    _wait_identity_exit(mcp_identity)
    live_descendants = [
        identity for identity in descendant_identities if _identity_alive(identity)
    ]
    checks.append(
        CheckResult(
            "RESTART-SUPERVISOR-SIGTERM",
            stop_exit == 0
            and not _identity_alive(child_identity)
            and not _identity_alive(mcp_identity)
            and not live_descendants
            and not ready_path.exists(),
            {
                "returncode": stop_exit,
                "childIdentity": child_identity,
                "childAlive": _identity_alive(child_identity),
                "mcpIdentity": mcp_identity,
                "mcpAlive": _identity_alive(mcp_identity),
                "descendantIdentities": descendant_identities,
                "liveDescendants": live_descendants,
                "readinessExists": ready_path.exists(),
            },
        )
    )

    # 4. Supervisor SIGKILL 后，Guardian 必须通过 lease EOF 清空完整 boot。
    kill_config, kill_workspace, _ = _isolated_config("supervisor-kill")
    kill_home = Path("/sandbox/supervisor-kill-home")
    kill_home.mkdir(exist_ok=True)
    _write_mcp_declaration("supervisor-kill", workspace=kill_workspace)
    killed_supervisor = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "supervise",
            "--config",
            str(kill_config),
            "--workspace",
            str(kill_workspace),
        ],
        env={**os.environ, "HOME": str(kill_home)},
    )
    kill_ready_path = kill_workspace / ".runtime-ready.json"
    _ = _wait_scenario_ready(kill_ready_path)
    kill_descendants = [
        _process_identity(pid) for pid in _descendant_pids(killed_supervisor.pid)
    ]
    os.kill(killed_supervisor.pid, signal.SIGKILL)
    kill_exit = killed_supervisor.wait(timeout=5)
    for identity in kill_descendants:
        _wait_identity_exit(identity)
    kill_live = [
        identity for identity in kill_descendants if _identity_alive(identity)
    ]
    checks.append(
        CheckResult(
            "RESTART-SUPERVISOR-SIGKILL",
            kill_exit == -signal.SIGKILL
            and not kill_live
            and not kill_ready_path.exists(),
            {
                "returncode": kill_exit,
                "descendantIdentities": kill_descendants,
                "liveDescendants": kill_live,
                "readinessExists": kill_ready_path.exists(),
            },
        )
    )

    # 5. Guardian SIGKILL 后，Supervisor 必须兜底清空 boot 并非零退出。
    guardian_config, guardian_workspace, _ = _isolated_config("guardian-kill")
    guardian_home = Path("/sandbox/guardian-kill-home")
    guardian_home.mkdir(exist_ok=True)
    _write_mcp_declaration("guardian-kill", workspace=guardian_workspace)
    guardian_supervisor = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "supervise",
            "--config",
            str(guardian_config),
            "--workspace",
            str(guardian_workspace),
        ],
        env={**os.environ, "HOME": str(guardian_home)},
    )
    guardian_ready_path = guardian_workspace / ".runtime-ready.json"
    _ = _wait_scenario_ready(guardian_ready_path)
    guardian_descendants = _descendant_pids(guardian_supervisor.pid)
    guardian_children_path = Path(
        f"/proc/{guardian_supervisor.pid}/task/"
        f"{guardian_supervisor.pid}/children"
    )
    guardian_children = [
        int(pid) for pid in guardian_children_path.read_text().split()
    ]
    if len(guardian_children) != 1:
        raise GateFailure(f"Guardian 数量异常: {guardian_children}")
    guardian_identities = [
        _process_identity(pid) for pid in guardian_descendants
    ]
    os.kill(guardian_children[0], signal.SIGKILL)
    guardian_supervisor_exit = guardian_supervisor.wait(timeout=15)
    for identity in guardian_identities:
        _wait_identity_exit(identity)
    guardian_live = [
        identity for identity in guardian_identities if _identity_alive(identity)
    ]
    checks.append(
        CheckResult(
            "RESTART-GUARDIAN-SIGKILL",
            guardian_supervisor_exit != 0
            and not guardian_live
            and not guardian_ready_path.exists(),
            {
                "returncode": guardian_supervisor_exit,
                "guardianPid": guardian_children[0],
                "descendantIdentities": guardian_identities,
                "liveDescendants": guardian_live,
                "readinessExists": guardian_ready_path.exists(),
            },
        )
    )
    _write_json(
        report_dir / "failure-modes.json",
        {"checks": [asdict(check) for check in checks]},
    )
    return checks


def _inside_failures(report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    checks = _failure_mode_checks(report_dir)
    passed = all(check.passed for check in checks)
    print(json.dumps({"status": "passed" if passed else "failed", "checks": [asdict(check) for check in checks]}, ensure_ascii=False))
    return 0 if passed else 1


def _configure_restart_gate(sandbox: Path) -> None:
    config = sandbox / "config.toml"
    text = config.read_text(encoding="utf-8")
    text = text.replace("max_iterations = 2", "max_iterations = 5")
    text += "\n[agent.tools]\nsearch_enabled = true\n"
    config.write_text(text, encoding="utf-8")


def _digest_summary(files: dict[str, str]) -> dict[str, object]:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "fileCount": len(files)}


def _copied_source_digests(
    source: dict[str, str],
    app: Path,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    """按同一 source manifest 比较宿主源码与 sandbox app。"""

    manifest = {path: digest for path, digest in source.items() if not path.startswith("static/")}
    app_files = {
        path: hashlib.sha256((app / path).read_bytes()).hexdigest()
        for path in manifest
        if (app / path).is_file() and not (app / path).is_symlink()
    }
    missing = sorted(set(manifest) - set(app_files))
    return _digest_summary(manifest), _digest_summary(app_files), missing


def _host(iterations: int, *, soak: bool) -> int:
    repo = Path(__file__).resolve().parents[2]
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    report_dir = repo / "docker/debug/reports/restart" / run_id
    report_dir.mkdir(parents=True)
    sandbox = Path(tempfile.mkdtemp(prefix="akashic-restart-gate-", dir="/tmp"))
    _prepare_host_sandbox(sandbox, repo)
    _configure_restart_gate(sandbox)
    before = _repository_digest(repo)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty_status = subprocess.run(
        ["git", "-C", str(repo), "status", "--short"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    source_digest, app_digest, app_missing = _copied_source_digests(
        before, sandbox / "app"
    )
    env = {
        **os.environ,
        "AKASHIC_CONTROL_SANDBOX": str(sandbox),
        "UID": str(os.getuid()),
        "GID": str(os.getgid()),
    }
    env.pop("AKASHIC_EXTRA_PLUGIN_DIRS", None)
    project = f"akashic-restart-{run_id.lower()}"
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(repo / "docker/debug/docker-compose.control-gate.yml"),
    ]
    error = ""
    inside_returncode = -1
    unsupervised_returncode = -1
    failures_returncode = -1
    cleanup_returncode = -1
    residual: dict[str, list[str]] = {
        "containers": [],
        "networks": [],
        "volumes": [],
    }
    image: dict[str, object] = {}
    try:
        build = subprocess.run([*compose, "build", "model-gate"], cwd=repo, env=env)
        if build.returncode != 0:
            raise GateFailure(f"image build failed: {build.returncode}")
        image_inspect = subprocess.run(
            [
                "docker", "image", "inspect", "akashic-agent-control-gate:latest",
                "--format", '{{json .}}',
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        inspected = json.loads(image_inspect.stdout)
        image = {
            "name": "akashic-agent-control-gate:latest",
            "id": inspected["Id"],
            "repoDigests": inspected.get("RepoDigests", []),
        }
        up = subprocess.run(
            [*compose, "up", "-d", "model-gate", "akashic-control-gate"],
            cwd=repo,
            env=env,
        )
        if up.returncode != 0:
            raise GateFailure(f"compose up failed: {up.returncode}")
        inside_command = [
                *compose,
                "exec",
                "-T",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "akashic-control-gate",
                "python",
                "docker/debug/restart_probe.py",
                "--inside",
                "--iterations",
                str(iterations),
                "--report-dir",
                "/sandbox/reports/restart",
            ]
        if soak:
            inside_command.append("--resource-gate")
        inside = subprocess.run(
            inside_command,
            cwd=repo,
            env=env,
        )
        inside_returncode = inside.returncode
        if inside.returncode != 0:
            raise GateFailure(f"inside gate failed: {inside.returncode}")
        unsupervised = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "--no-deps",
                "control-probe",
                "python",
                "docker/debug/restart_probe.py",
                "--inside-unsupervised",
                "--report-dir",
                "/sandbox/reports/restart",
            ],
            cwd=repo,
            env=env,
        )
        unsupervised_returncode = unsupervised.returncode
        if unsupervised.returncode != 0:
            raise GateFailure(
                f"unsupervised gate failed: {unsupervised.returncode}"
            )
        failures = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "--no-deps",
                "control-probe",
                "python",
                "docker/debug/restart_probe.py",
                "--inside-failures",
                "--report-dir",
                "/sandbox/reports/restart",
            ],
            cwd=repo,
            env=env,
        )
        failures_returncode = failures.returncode
        if failures.returncode != 0:
            raise GateFailure(f"failure modes failed: {failures.returncode}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        logs = subprocess.run(
            [*compose, "logs", "--no-color"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (report_dir / "compose.log").write_text(logs.stdout, encoding="utf-8")
        if (sandbox / "reports/restart").exists():
            shutil.copytree(
                sandbox / "reports/restart",
                report_dir,
                dirs_exist_ok=True,
            )
        cleanup = subprocess.run(
            [*compose, "down", "--remove-orphans", "--volumes"],
            cwd=repo,
            env=env,
        )
        cleanup_returncode = cleanup.returncode
        for kind, command in {
            "containers": ["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"],
            "networks": ["docker", "network", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
            "volumes": ["docker", "volume", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
        }.items():
            result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
            residual[kind] = result.stdout.split()

    after = _repository_digest(repo)
    passed = (
        not error
        and inside_returncode == 0
        and unsupervised_returncode == 0
        and failures_returncode == 0
        and cleanup_returncode == 0
        and not any(residual.values())
        and before == after
        and source_digest == app_digest
        and not app_missing
    )
    report = {
        "runId": run_id,
        "gate": "restart",
        "head": head,
        "dirtyStatus": dirty_status,
        "sourceDigest": source_digest,
        "sandboxAppDigest": app_digest,
        "sandboxMissingSourceFiles": app_missing,
        "composeProject": project,
        "image": image,
        "iterations": iterations,
        "status": "passed" if passed else "failed",
        "insideReturncode": inside_returncode,
        "unsupervisedReturncode": unsupervised_returncode,
        "failuresReturncode": failures_returncode,
        "cleanupReturncode": cleanup_returncode,
        "residualResources": residual,
        "repositoriesUnchanged": before == after,
        "error": error,
        "reportDir": str(report_dir),
    }
    _write_json(report_dir / "gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    shutil.rmtree(sandbox)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="restart Docker 真实验收 Gate")
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--inside-unsupervised", action="store_true")
    parser.add_argument("--inside-failures", action="store_true")
    parser.add_argument("--resource-gate", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--soak", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=Path("/sandbox/reports/restart"))
    args = parser.parse_args()
    iterations = 20 if args.soak else args.iterations
    if iterations < 1:
        raise SystemExit("--iterations 必须大于 0")
    if args.inside:
        return _inside(iterations, args.report_dir, resource_gate=args.resource_gate)
    if args.inside_unsupervised:
        return _inside_unsupervised(args.report_dir)
    if args.inside_failures:
        return _inside_failures(args.report_dir)
    return _host(iterations, soak=args.soak)


if __name__ == "__main__":
    raise SystemExit(main())
