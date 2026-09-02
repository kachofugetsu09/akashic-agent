from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agent.core.passive_turn import _host_runtime_execution_hint
from agent.host_bridge.server import _materialize_runtime_cli


def test_runtime_cli_is_bound_to_materialized_release(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "main.py").write_text("# runtime entry\n", encoding="utf-8")
    fake_python = tmp_path / "bridge-python"
    fake_python.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$PYTHONPATH" "$@"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    bridge_python = tmp_path / "bridge-venv" / "bin" / "python"
    bridge_python.parent.mkdir(parents=True)
    bridge_python.symlink_to(fake_python)
    stale = (
        tmp_path
        / "artifacts"
        / "runtime-cli"
        / ("a" * 40)
        / f".akashic-runtime.{os.getpid()}.tmp"
    )
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")
    stale.chmod(0o500)
    launcher = _materialize_runtime_cli(
        tmp_path / "artifacts",
        checkout,
        bridge_python,
        "a" * 40,
    )

    result = subprocess.run(
        [str(launcher), "plugin-doctor", "demo@github"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "AKASHIC_BRIDGE_PYTHON": "/attacker/python",
            "AKASHIC_RUNTIME_CHECKOUT": "/attacker/checkout",
        },
    )

    assert result.stdout.splitlines() == [
        str(checkout),
        str(checkout / "main.py"),
        "plugin-doctor",
        "demo@github",
    ]
    assert str(bridge_python) in launcher.read_text(encoding="utf-8")
    assert str(fake_python) not in launcher.read_text(encoding="utf-8")


def test_host_runtime_hint_is_explicit_and_local_mode_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_EXECUTION_MODE", "host-bridge")
    monkeypatch.setenv("AKASHIC_RUNTIME_COMMIT", "a" * 40)
    monkeypatch.setenv("AKASHIC_RUNTIME_CHECKOUT", "/srv/runtime")

    hint = _host_runtime_execution_hint()

    assert "a" * 40 in hint
    assert "/srv/runtime" in hint
    assert "akashic-runtime" in hint
    assert "不要用 host 的 python" in hint
    monkeypatch.setenv("AKASHIC_EXECUTION_MODE", "local")
    assert _host_runtime_execution_hint() == ""
