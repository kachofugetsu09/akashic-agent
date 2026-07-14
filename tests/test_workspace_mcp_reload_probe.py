from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from docker.debug.workspace_mcp_reload_probe import (
    _build_parser,
    _compose_command,
    _compose_environment,
    _pid_starttime,
    _running_pids,
)


def test_workspace_mcp_reload_probe_runs_real_app_runtime(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "docker/debug/workspace_mcp_reload_probe.py",
            "--internal",
            "--workspace",
            str(tmp_path / "gate"),
            "--report",
            str(report),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    checks = {item["name"]: item for item in payload["checks"]}
    assert all(item["passed"] for item in checks.values())
    assert {
        "initial-v1",
        "old-new-isolation",
        "watch-content-reload",
        "bad-toml-rollback",
        "partial-candidate-cleanup",
        "automatic-recovery",
        "delete-all-drains",
        "plugin-name-conflict-fail-loud",
        "watcher-fault-supervised",
        "shutdown-no-residual",
        "legacy-json-ignored",
    } <= set(checks)
    assert checks["watch-content-reload"]["evidence"]["toolResult"] == "v2:two"
    assert checks["legacy-json-ignored"]["evidence"]["lifecycle"] == []
    assert "mcp_docs__version" not in checks["delete-all-drains"]["evidence"][
        "emptyRegistryNames"
    ]


def test_running_pids_uses_proc_identity_not_stopped_event(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        starttime = _pid_starttime(process.pid)
        assert starttime is not None
        lifecycle = tmp_path / "lifecycle.jsonl"
        records = [
            {
                "event": event,
                "pid": process.pid,
                "starttime": starttime,
                "version": "test",
                "instance": "docs",
            }
            for event in ("started", "stopped")
        ]
        lifecycle.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        assert _running_pids(lifecycle) == [process.pid]
    finally:
        process.terminate()
        process.wait(timeout=5)
    assert _running_pids(lifecycle) == []


def test_workspace_mcp_probe_separates_host_and_internal_arguments(
    tmp_path: Path,
) -> None:
    parser = _build_parser()
    host = parser.parse_args([])
    assert host.internal is False
    assert host.workspace is None
    internal = parser.parse_args(
        ["--internal", "--workspace", str(tmp_path), "--report", str(tmp_path / "r.json")]
    )
    assert internal.internal is True
    assert internal.workspace == tmp_path
    compose = _compose_command(tmp_path, "gate-project")
    assert compose[:5] == ["docker", "compose", "-p", "gate-project", "-f"]
    assert compose[-1].endswith("docker/debug/docker-compose.control-gate.yml")
    environment = _compose_environment(tmp_path / "sandbox")
    assert "AKASHIC_EXTRA_PLUGIN_DIRS" not in environment
