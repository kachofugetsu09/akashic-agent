from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from docker.debug.programmatic_control_probe import _prepare_host_sandbox
from docker.debug.restart_probe import (
    _copied_source_digests,
    _configure_restart_gate,
    _mcp_scripts,
    _identity_alive,
    _memory_within_limit,
    _peak_memory_deltas,
    _process_identity,
    _process_metrics,
    _restart_scripts,
    _tool_names,
)


def test_restart_gate_enables_search_and_multi_step_tool_loop(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("", encoding="utf-8")
    template = source / "prompts/VEDA.md"
    template.parent.mkdir()
    template.write_text("gate veda\n", encoding="utf-8")
    sandbox = tmp_path / "sandbox"
    _prepare_host_sandbox(sandbox, source)
    (sandbox / "config.toml").write_text(
        "max_iterations = 2\n",
        encoding="utf-8",
    )

    _configure_restart_gate(sandbox)

    config = (sandbox / "config.toml").read_text(encoding="utf-8")
    assert "max_iterations = 5" in config
    assert "[agent.tools]\nsearch_enabled = true" in config


def test_restart_scripts_unlock_before_restart_and_gate_final_reply() -> None:
    scripts = _restart_scripts(3, "final-barrier")

    assert scripts[0]["tool_calls"] == [
        {
            "id": "call_search_3",
            "name": "tool_search",
            "arguments": {"query": "select:agent_restart"},
        }
    ]
    assert scripts[1]["tool_calls"] == [
        {
            "id": "call_restart_3",
            "name": "agent_restart",
            "arguments": {"reason": "restart gate iteration 3"},
        }
    ]
    assert scripts[2]["barrier"] == "final-barrier"
    assert scripts[2]["content"] == "restart-complete-3"


def test_mcp_scripts_unlock_call_and_complete() -> None:
    scripts = _mcp_scripts("v7")

    assert scripts[0]["tool_calls"][0]["name"] == "tool_search"  # type: ignore[index]
    assert scripts[1]["tool_calls"][0]["name"] == "mcp_restart_probe__version"  # type: ignore[index]
    assert scripts[2]["content"] == "mcp-v7"


def test_sandbox_digest_uses_same_source_manifest(tmp_path: Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.py").write_text("same", encoding="utf-8")
    source = {
        "main.py": hashlib.sha256(b"same").hexdigest(),
        "static/generated.js": hashlib.sha256(b"ignored").hexdigest(),
    }

    source_digest, app_digest, missing = _copied_source_digests(source, app)

    assert source_digest == app_digest
    assert missing == []


def test_process_identity_rejects_reused_pid_counterexample() -> None:
    identity = _process_identity(os.getpid())

    assert _identity_alive(identity)
    assert not _identity_alive(
        {"pid": identity["pid"], "starttime": identity["starttime"] + 1}
    )


def test_process_identity_treats_proc_lookup_race_as_exited(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = {"pid": 123456, "starttime": 1}

    def read_text(_path: Path, **_kwargs: object) -> str:
        raise ProcessLookupError("process exited during /proc read")

    monkeypatch.setattr(Path, "read_text", read_text)

    assert not _identity_alive(identity)


def test_process_identity_propagates_unexpected_proc_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = {"pid": 123456, "starttime": 1}

    def read_text(_path: Path, **_kwargs: object) -> str:
        raise PermissionError("unexpected /proc read failure")

    monkeypatch.setattr(Path, "read_text", read_text)

    with pytest.raises(PermissionError, match="unexpected /proc read failure"):
        _identity_alive(identity)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="依赖 Linux zombie 状态")
def test_process_identity_treats_zombie_as_exited() -> None:
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    try:
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        identity = _process_identity(pid)

        assert not _identity_alive(identity)
    finally:
        os.waitpid(pid, 0)


def test_process_metrics_include_rss_and_high_water_mark() -> None:
    metrics = _process_metrics(os.getpid())

    assert metrics["vmRssKiB"] > 0
    assert metrics["vmHwmKiB"] >= metrics["vmRssKiB"]
    assert metrics["fds"] > 0
    assert metrics["threads"] > 0


def test_hwm_peak_over_limit_fails_after_rss_recovers() -> None:
    supervisor = {"vmRssKiB": 100, "vmHwmKiB": 200}
    child = {"vmRssKiB": 300, "vmHwmKiB": 400}
    samples = [
        {
            "supervisor": {"vmRssKiB": 110, "vmHwmKiB": 200 + 200 * 1024},
            "child": {"vmRssKiB": 310, "vmHwmKiB": 410},
        }
    ]

    deltas = _peak_memory_deltas(supervisor, child, samples)

    assert deltas["supervisorRssKiB"] == 10
    assert deltas["supervisorHwmKiB"] == 200 * 1024
    assert not _memory_within_limit(deltas)


def test_tool_names_reads_real_openai_payload_shape() -> None:
    request = {
        "payload": {
            "tools": [
                {"type": "function", "function": {"name": "tool_search"}},
                {"type": "function", "function": {"name": "agent_restart"}},
            ]
        }
    }

    assert _tool_names(request) == {"tool_search", "agent_restart"}


def test_restart_probe_defaults_are_pr_gate_and_soak_compatible() -> None:
    source = Path("docker/debug/restart_probe.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--iterations", type=int, default=3)' in source
    assert "iterations = 20 if args.soak else args.iterations" in source
    assert 'inside_command.append("--resource-gate")' in source
    assert '"supervisorRssKiB": 64 * 1024' in source
    assert '"supervisorHwmKiB": 64 * 1024' in source
    assert '"childRssKiB": 64 * 1024' in source
    assert '"childHwmKiB": 64 * 1024' in source
    assert '"AKASHIC_READINESS_TIMEOUT_S": "15"' in source
    json.dumps({"status": "passed", "iterations": 3})
