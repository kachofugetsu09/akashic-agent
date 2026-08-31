from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.request import urlopen

import pytest

from agent.plugin_composition import ManagedProcessDefinition
import agent.plugins.managed_process_host as managed_process_host
from agent.plugins.managed_process_host import (
    ManagedProcessGenerationHost,
    _Generation,
    _LogRing,
    _ProcessEpoch,
)
from utils.process_group import OwnedProcessGroup


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _http_definition(
    script: Path,
    *,
    formal_port: int,
    startup_timeout_seconds: float = 3.0,
    env: dict[str, str] | None = None,
) -> ManagedProcessDefinition:
    return ManagedProcessDefinition(
        name="calendar_api",
        command=(sys.executable, str(script)),
        cwd=str(script.parent),
        env=env or {},
        port_env="PORT",
        formal_port=formal_port,
        readiness_path="/health",
        startup_timeout_seconds=startup_timeout_seconds,
    )


def _write_http_server(
    script: Path,
    *,
    exit_first: bool = False,
    ready_status: int = 200,
    redirect_location: str | None = None,
) -> None:
    first_exit = (
        "from pathlib import Path\n"
        "counter = Path('attempts')\n"
        "attempt = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(attempt))\n"
        "if attempt == 1:\n"
        "    import threading\n"
        "    threading.Thread(target=server.serve_forever, daemon=True).start()\n"
        "    time.sleep(0.15)\n"
        "    raise SystemExit(17)\n"
        if exit_first
        else ""
    )
    redirect_header = (
        f"        self.send_header('Location', {redirect_location!r})\n"
        if redirect_location is not None
        else ""
    )
    script.write_text(
        "import os, sys, time\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        f"        self.send_response({ready_status})\n"
        + redirect_header
        + "        self.end_headers(); self.wfile.write(b'ready')\n"
        "    def log_message(self, *args): pass\n"
        "server = HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler)\n"
        "if os.environ.get('PID_LOG'):\n"
        "    with open(os.environ['PID_LOG'], 'a', encoding='utf-8') as output:\n"
        "        output.write(str(os.getpid()) + '\\n')\n"
        "print('managed stdout', flush=True)\n"
        "print('managed stderr', file=sys.stderr, flush=True)\n"
        + first_exit
        + "server.serve_forever()\n",
        encoding="utf-8",
    )


def _write_slow_process(script: Path) -> None:
    script.write_text(
        "import os, signal, socket, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "Path(os.environ['PID_FILE']).write_text(str(os.getpid()))\n"
        "Path(os.environ['CHILD_PID_FILE']).write_text(str(child.pid))\n"
        "Path(os.environ['PORT_FILE']).write_text(os.environ['PORT'])\n"
        "listener = socket.socket()\n"
        "listener.bind(('127.0.0.1', int(os.environ['PORT'])))\n"
        "listener.listen()\n"
        "def ignore_term(signum, frame):\n"
        "    time.sleep(30)\n"
        "signal.signal(signal.SIGTERM, ignore_term)\n"
        "while True:\n"
        "    time.sleep(1)\n",
        encoding="utf-8",
    )


def _write_exhausting_recovery_server(script: Path) -> None:
    script.write_text(
        "import os, threading, time\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "from pathlib import Path\n"
        "counter = Path('recovery-attempts')\n"
        "attempt = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(attempt))\n"
        "if attempt == 1:\n"
        "    class Handler(BaseHTTPRequestHandler):\n"
        "        def do_GET(self):\n"
        "            self.send_response(200); self.end_headers(); self.wfile.write(b'ready')\n"
        "        def log_message(self, *args): pass\n"
        "    server = HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler)\n"
        "    threading.Thread(target=server.serve_forever, daemon=True).start()\n"
        "    time.sleep(0.1)\n"
        "raise SystemExit(19)\n",
        encoding="utf-8",
    )


def _slow_definition(
    script: Path,
    *,
    pid_file: Path,
    child_pid_file: Path,
    port_file: Path,
) -> ManagedProcessDefinition:
    return ManagedProcessDefinition(
        name="slow_process",
        command=(sys.executable, str(script)),
        cwd=str(script.parent),
        env={
            "PID_FILE": str(pid_file),
            "CHILD_PID_FILE": str(child_pid_file),
            "PORT_FILE": str(port_file),
        },
        port_env="PORT",
        formal_port=_free_port(),
        readiness_path="/never-ready",
        startup_timeout_seconds=5.0,
    )


def _pid_live(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    command_end = stat.rfind(")")
    fields = stat[command_end + 2 :].split()
    return bool(fields) and fields[0] != "Z"


def _port_live(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.1)
        try:
            probe.connect(("127.0.0.1", port))
        except OSError:
            return False
        return True


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        await asyncio.sleep(0.02)


def test_log_ring_enforces_utf8_byte_cap() -> None:
    ring = _LogRing(max_bytes=4, max_lines=4)
    ring.append("😀😀".encode("utf-8"))

    lines = ring.snapshot()
    assert sum(len(line.encode("utf-8")) for line in lines) <= 4
    assert all(line.encode("utf-8").decode("utf-8") == line for line in lines)
    assert lines == ("😀",)


@pytest.mark.asyncio
async def test_readiness_poll_sleep_respects_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = ManagedProcessDefinition(
        name="deadline",
        command=(sys.executable, "-c"),
        formal_port=_free_port(),
        startup_timeout_seconds=0.01,
    )
    entry = _ProcessEpoch(
        generation_id="deadline-generation",
        definition=definition,
        mode="candidate",
        artifact_root=None,
    )
    entry.process = cast(Any, SimpleNamespace(returncode=None))
    generation = _Generation(
        generation_id="deadline-generation",
        mode="candidate",
        artifact_root=None,
        entries={definition.name: entry},
    )

    async def direct_to_thread(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", direct_to_thread)
    monkeypatch.setattr(managed_process_host, "_url_ready", lambda *_args: False)
    host = ManagedProcessGenerationHost()
    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await host._wait_ready(generation, entry, _free_port())
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.04


@pytest.mark.asyncio
async def test_candidate_uses_temporary_port_and_bounded_logs(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_http_server(script)
    health: list[tuple[str, str, bool, str]] = []
    incidents: list[tuple[str, str, str, str]] = []

    def record_health(
        generation_id: str, process_name: str, healthy: bool, reason: str
    ) -> None:
        health.append((generation_id, process_name, healthy, reason))

    def record_incident(
        generation_id: str, process_name: str, kind: str, message: str
    ) -> None:
        incidents.append((generation_id, process_name, kind, message))

    host = ManagedProcessGenerationHost(
        on_health=record_health,
        on_incident=record_incident,
        log_max_bytes=64,
        log_max_lines=4,
    )
    formal_port = _free_port()
    generation = await host.start_generation(
        "candidate-1",
        {"calendar_api": _http_definition(script, formal_port=formal_port)},
    )

    endpoint = generation.endpoint("calendar_api")
    assert endpoint.mode == "candidate"
    assert endpoint.port != formal_port
    with urlopen(endpoint.readiness_url, timeout=1) as response:
        assert response.read() == b"ready"
    await asyncio.sleep(0.05)
    logs = generation.logs("calendar_api")
    assert any("managed stdout" in line for line in logs.stdout)
    assert any("managed stderr" in line for line in logs.stderr)
    assert host.health("candidate-1", "calendar_api")
    assert ("candidate-1", "calendar_api", True, "ready") in health
    assert incidents == []

    await host.stop_generation("candidate-1")
    assert host.get("candidate-1") is None
    with pytest.raises(OSError):
        urlopen(endpoint.readiness_url, timeout=0.2)


@pytest.mark.asyncio
async def test_formal_fixed_port_and_candidate_are_isolated(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_http_server(script)
    host = ManagedProcessGenerationHost()
    formal_port = _free_port()
    definition = _http_definition(script, formal_port=formal_port)

    formal = await host.start_generation(
        "formal-1", {definition.name: definition}, mode="formal"
    )
    candidate = await host.start_generation("candidate-1", {definition.name: definition})
    assert formal.endpoint("calendar_api").port == formal_port
    assert candidate.endpoint("calendar_api").port != formal_port

    await host.stop_generation("candidate-1")
    with urlopen(formal.endpoint("calendar_api").readiness_url, timeout=1) as response:
        assert response.status == 200
    await host.stop_generation("formal-1")


@pytest.mark.asyncio
async def test_process_exit_recovers_with_new_epoch_without_stale_resurrection(
    tmp_path: Path,
) -> None:
    script = tmp_path / "recover.py"
    pid_log = tmp_path / "recover-pids.txt"
    _write_http_server(script, exit_first=True)
    incidents: list[tuple[str, str, str, str]] = []

    def record_incident(
        generation_id: str, process_name: str, kind: str, message: str
    ) -> None:
        incidents.append((generation_id, process_name, kind, message))

    host = ManagedProcessGenerationHost(
        on_incident=record_incident,
        recovery_backoff_seconds=(0.01, 0.01),
        recovery_stable_seconds=60,
    )
    generation = await host.start_generation(
        "candidate-recover",
        {
            "calendar_api": _http_definition(
                script,
                formal_port=_free_port(),
                env={"PID_LOG": str(pid_log)},
            )
        },
    )
    initial = generation.endpoint("calendar_api")

    def recovered() -> bool:
        try:
            return generation.endpoint("calendar_api").epoch > initial.epoch
        except RuntimeError:
            return False

    await _wait_until(
        recovered,
    )
    recovered_endpoint = generation.endpoint("calendar_api")
    assert recovered_endpoint.epoch > initial.epoch
    assert recovered_endpoint.port != initial.port
    assert any(item[2] == "process_exit" for item in incidents)
    assert host.health("candidate-recover", "calendar_api")

    await host.stop_generation("candidate-recover")
    process_ids = tuple(
        int(value)
        for value in pid_log.read_text(encoding="utf-8").splitlines()
    )
    assert len(process_ids) >= 2
    await _wait_until(lambda: all(not _pid_live(pid) for pid in process_ids))
    assert host.get("candidate-recover") is None


@pytest.mark.asyncio
async def test_cancel_start_drains_real_process_group_after_repeated_cancellation(
    tmp_path: Path,
) -> None:
    script = tmp_path / "slow.py"
    pid_file = tmp_path / "leader.pid"
    child_pid_file = tmp_path / "child.pid"
    port_file = tmp_path / "port"
    _write_slow_process(script)
    host = ManagedProcessGenerationHost(stop_timeout_seconds=0.15)
    definition = _slow_definition(
        script,
        pid_file=pid_file,
        child_pid_file=child_pid_file,
        port_file=port_file,
    )
    start_task = asyncio.create_task(
        host.start_generation("cancel-start", {definition.name: definition})
    )
    try:
        await _wait_until(
            lambda: pid_file.exists() and child_pid_file.exists() and port_file.exists()
        )
        leader_pid = int(pid_file.read_text(encoding="utf-8"))
        child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        port = int(port_file.read_text(encoding="utf-8"))
        await _wait_until(lambda: _port_live(port))

        start_task.cancel()
        await asyncio.sleep(0.02)
        start_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start_task

        await _wait_until(
            lambda: not _pid_live(leader_pid) and not _pid_live(child_pid),
            timeout=3.0,
        )
        assert host.get("cancel-start") is None
        assert host.tombstone("cancel-start") is None
        assert not _port_live(port)
    finally:
        if not start_task.done():
            start_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await start_task
        await host.close()


@pytest.mark.asyncio
async def test_health_ready_callback_cancellation_cleans_started_process(
    tmp_path: Path,
) -> None:
    script = tmp_path / "server.py"
    _write_http_server(script)

    async def cancel_ready(
        generation_id: str, process_name: str, healthy: bool, reason: str
    ) -> None:
        if healthy:
            raise asyncio.CancelledError

    host = ManagedProcessGenerationHost(on_health=cancel_ready)
    with pytest.raises(asyncio.CancelledError):
        await host.start_generation(
            "health-cancel",
            {"calendar_api": _http_definition(script, formal_port=_free_port())},
        )
    assert host.get("health-cancel") is None
    assert host.tombstone("health-cancel") is None


@pytest.mark.asyncio
async def test_incident_callback_cancellation_cleans_readiness_process(
    tmp_path: Path,
) -> None:
    script = tmp_path / "unready.py"
    _write_http_server(script, ready_status=503)

    async def cancel_incident(
        generation_id: str,
        process_name: str,
        kind: str,
        message: str,
    ) -> None:
        raise asyncio.CancelledError

    host = ManagedProcessGenerationHost(on_incident=cancel_incident)
    with pytest.raises(asyncio.CancelledError):
        await host.start_generation(
            "incident-cancel",
            {
                "calendar_api": _http_definition(
                    script,
                    formal_port=_free_port(),
                    startup_timeout_seconds=0.15,
                )
            },
        )
    assert host.get("incident-cancel") is None
    assert host.tombstone("incident-cancel") is None


@pytest.mark.asyncio
async def test_health_callback_failure_is_fail_loud_and_cleans_process(
    tmp_path: Path,
) -> None:
    script = tmp_path / "server.py"
    _write_http_server(script)

    def fail_ready(
        generation_id: str, process_name: str, healthy: bool, reason: str
    ) -> None:
        if healthy:
            raise RuntimeError("health bridge unavailable")

    host = ManagedProcessGenerationHost(on_health=fail_ready)
    with pytest.raises(RuntimeError, match="health bridge unavailable"):
        await host.start_generation(
            "health-failure",
            {"calendar_api": _http_definition(script, formal_port=_free_port())},
        )
    assert host.get("health-failure") is None
    assert host.tombstone("health-failure") is None


@pytest.mark.asyncio
async def test_readiness_rejects_redirect_and_strict_timeout(
    tmp_path: Path,
) -> None:
    script = tmp_path / "redirect.py"
    _write_http_server(
        script,
        ready_status=302,
        redirect_location="http://127.0.0.1:9/unrelated",
    )
    host = ManagedProcessGenerationHost()
    with pytest.raises(TimeoutError):
        await host.start_generation(
            "redirected",
            {
                "calendar_api": _http_definition(
                    script,
                    formal_port=_free_port(),
                    startup_timeout_seconds=0.15,
                )
            },
        )
    assert host.get("redirected") is None
    assert host.tombstone("redirected") is None


@pytest.mark.asyncio
async def test_recovery_exhaustion_retains_tombstone_until_explicit_retry(
    tmp_path: Path,
) -> None:
    script = tmp_path / "exhaust.py"
    _write_exhausting_recovery_server(script)
    host = ManagedProcessGenerationHost(
        recovery_backoff_seconds=(0.01, 0.01),
        recovery_stable_seconds=60.0,
    )
    generation_id = "recovery-exhausted"
    await host.start_generation(
        generation_id,
        {"calendar_api": _http_definition(script, formal_port=_free_port())},
    )

    await _wait_until(lambda: host.tombstone(generation_id) is not None, timeout=3.0)
    tombstone = host.tombstone(generation_id)
    assert tombstone is not None
    assert tombstone.state == "degraded"
    assert tombstone.action == "retry_runtime_recovery"
    assert host.generation_state(generation_id) == "degraded"

    await host.retry_runtime_recovery(generation_id)
    assert host.get(generation_id) is None
    assert host.tombstone(generation_id) is None


@pytest.mark.asyncio
async def test_cleanup_failure_retains_tombstone_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "server.py"
    _write_http_server(script)
    host = ManagedProcessGenerationHost()
    generation_id = "candidate-cleanup"
    await host.start_generation(
        generation_id,
        {"calendar_api": _http_definition(script, formal_port=_free_port())},
    )
    original_terminate = OwnedProcessGroup.terminate
    calls = 0

    async def fail_once(self: OwnedProcessGroup, *, timeout_s: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected terminate failure")
        await original_terminate(self, timeout_s=timeout_s)

    monkeypatch.setattr(OwnedProcessGroup, "terminate", fail_once)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await host.stop_generation(generation_id)
    tombstone = host.tombstone(generation_id)
    assert tombstone is not None
    assert tombstone.action == "retry_generation_cleanup"
    assert host.get(generation_id) is not None

    monkeypatch.setattr(OwnedProcessGroup, "terminate", original_terminate)
    await host.retry_generation_cleanup(generation_id)
    assert host.tombstone(generation_id) is None
    assert host.get(generation_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback_error",
    [RuntimeError("root disposed"), asyncio.CancelledError()],
)
async def test_stopped_observer_failure_cannot_retain_cleaned_process(
    tmp_path: Path,
    callback_error: BaseException,
) -> None:
    script = tmp_path / "server.py"
    _write_http_server(script)

    def fail_after_root_dispose(
        generation_id: str,
        process_name: str,
        healthy: bool,
        reason: str,
    ) -> None:
        if not healthy and reason == "stopped":
            raise callback_error

    host = ManagedProcessGenerationHost(on_health=fail_after_root_dispose)
    generation_id = "stopped-observer"
    generation = await host.start_generation(
        generation_id,
        {"calendar_api": _http_definition(script, formal_port=_free_port())},
    )
    port = generation.endpoint("calendar_api").port

    await host.stop_generation(generation_id)

    assert host.get(generation_id) is None
    assert host.tombstone(generation_id) is None
    assert not _port_live(port)
