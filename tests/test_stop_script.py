from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOP_SCRIPT = PROJECT_ROOT / "scripts" / "stop-runtime.sh"
LOCK_HOLDER_CODE = """
import fcntl
import signal
import sys

stream = open(sys.argv[1], "a+")
fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.pause()
"""
SUPERVISOR_CODE = f"""
import fcntl
import signal
import subprocess
import sys

stream = open(sys.argv[1], "a+")
fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
child = subprocess.Popen([sys.executable, "-c", {LOCK_HOLDER_CODE!r}, sys.argv[2]])

def stop(*_args):
    child.terminate()
    child.wait(timeout=5)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
child.wait()
"""


def _wait_until_locked(lock_path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("fixture process exited before acquiring its lock")
        probe = subprocess.run(
            ["flock", "-n", str(lock_path), "-c", "true"],
            check=False,
        )
        if probe.returncode != 0:
            return
        time.sleep(0.05)
    process.terminate()
    process.wait(timeout=5)
    raise AssertionError("fixture process did not acquire its lock")


def _start_lock_holder(lock_path: Path) -> subprocess.Popen[str]:
    lock_path.parent.mkdir(parents=True)
    process = subprocess.Popen(
        [sys.executable, "-c", LOCK_HOLDER_CODE, str(lock_path)],
        text=True,
    )
    _wait_until_locked(lock_path, process)
    return process


def test_stops_runtime_lock_owner(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    process = _start_lock_holder(workspace / ".instance.lock")
    try:
        result = subprocess.run(
            [str(STOP_SCRIPT), "--workspace", str(workspace)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "workspace 已停止" in result.stdout
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_keeps_stale_lock_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lock_path = workspace / ".instance.lock"
    lock_path.write_text("stale-owner", encoding="utf-8")

    result = subprocess.run(
        [str(STOP_SCRIPT), "--workspace", str(workspace)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "AKASHIC_WORKSPACE": ""},
    )

    assert result.returncode == 0, result.stderr
    assert "workspace 未运行" in result.stdout
    assert lock_path.read_text(encoding="utf-8") == "stale-owner"


def test_stops_supervisor_before_runtime_child(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    supervisor_lock = workspace / ".supervisor.lock"
    runtime_lock = workspace / ".instance.lock"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            SUPERVISOR_CODE,
            str(supervisor_lock),
            str(runtime_lock),
        ],
        text=True,
    )
    try:
        _wait_until_locked(supervisor_lock, process)
        _wait_until_locked(runtime_lock, process)
        result = subprocess.run(
            [str(STOP_SCRIPT), "--workspace", str(workspace)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert f"pid={process.pid}" in result.stdout
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
