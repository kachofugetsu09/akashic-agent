from __future__ import annotations

import asyncio
import os
import socket
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest

import agent.plugins.service_host as service_host_module
from agent.plugins.service_host import PluginServiceHost


def _free_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _service_spec(script: Path, port: int, version: str) -> dict[str, object]:
    return {
        "command": [sys.executable, str(script)],
        "cwd": str(script.parent),
        "env": {"PORT": str(port), "VERSION": version},
        "readiness_url": f"http://127.0.0.1:{port}/",
        "startup_timeout_seconds": 3,
        "revision": version,
    }


def _read(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
        return response.read().decode()


def _read_or_none(port: int) -> str | None:
    try:
        return _read(port)
    except OSError:
        return None


@pytest.mark.asyncio
async def test_managed_service_swaps_generation(tmp_path: Path) -> None:
    script = tmp_path / "service.py"
    _ = script.write_text(
        "import os\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers()\n"
        "        self.wfile.write(os.environ['VERSION'].encode())\n"
        "    def log_message(self, *args): pass\n"
        "HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    port = _free_port()
    first = {"monitor": _service_spec(script, port, "v1")}
    second = {"monitor": _service_spec(script, port, "v2")}
    host = PluginServiceHost()
    host.bind_plugin_services({"health": first})  # type: ignore[arg-type]
    await host.start_all()
    assert _read(port) == "v1"

    await host.swap_plugin_services("health", first, second)  # type: ignore[arg-type]

    assert _read(port) == "v2"
    await host.stop_all()


@pytest.mark.asyncio
async def test_candidate_service_runs_beside_formal_service_and_stops_by_generation(
    tmp_path: Path,
) -> None:
    script = tmp_path / "service.py"
    _ = script.write_text(
        "import os\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers()\n"
        "        self.wfile.write(os.environ['VERSION'].encode())\n"
        "    def log_message(self, *args): pass\n"
        "HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    formal_port = _free_port()
    candidate_port = _free_port()
    host = PluginServiceHost()
    host.bind_plugin_services(
        {"health": {"monitor": _service_spec(script, formal_port, "stable")}}
    )
    await host.start_all()

    await host.start_candidate(
        "generation-2",
        {"monitor": _service_spec(script, candidate_port, "candidate")},
    )

    assert _read(formal_port) == "stable"
    assert _read(candidate_port) == "candidate"
    await host.assert_candidate_healthy("generation-2")
    await host.stop_candidate("generation-2")
    assert _read(formal_port) == "stable"
    with pytest.raises(OSError):
        _read(candidate_port)
    await host.stop_all()


@pytest.mark.asyncio
async def test_managed_service_restores_old_generation_on_failure(
    tmp_path: Path,
) -> None:
    service = tmp_path / "service.py"
    failed = tmp_path / "failed.py"
    _ = service.write_text(
        "import os\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers()\n"
        "        self.wfile.write(os.environ['VERSION'].encode())\n"
        "    def log_message(self, *args): pass\n"
        "HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    _ = failed.write_text("raise RuntimeError('failed')\n", encoding="utf-8")
    port = _free_port()
    old = {"monitor": _service_spec(service, port, "v1")}
    new = {"monitor": _service_spec(failed, port, "v2")}
    host = PluginServiceHost()
    host.bind_plugin_services({"health": old})  # type: ignore[arg-type]
    await host.start_all()

    with pytest.raises(RuntimeError, match="启动失败"):
        await host.swap_plugin_services("health", old, new)  # type: ignore[arg-type]

    assert _read(port) == "v1"
    await host.stop_all()


@pytest.mark.asyncio
async def test_managed_service_rejects_occupied_readiness_endpoint(
    tmp_path: Path,
) -> None:
    service = tmp_path / "service.py"
    failed = tmp_path / "failed.py"
    _ = service.write_text(
        "import os\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self): self.send_response(200); self.end_headers()\n"
        "    def log_message(self, *args): pass\n"
        "HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    _ = failed.write_text("raise SystemExit(7)\n", encoding="utf-8")
    port = _free_port()
    existing = {"server": _service_spec(service, port, "existing")}
    collision = {"server": _service_spec(failed, port, "candidate")}
    host = PluginServiceHost()
    host.bind_plugin_services({"existing": existing})  # type: ignore[arg-type]
    await host.start_all()

    with pytest.raises(RuntimeError, match="已被占用"):
        await host.swap_plugin_services(  # type: ignore[arg-type]
            "candidate",
            {},
            collision,
        )

    assert _read(port) == ""
    await host.stop_all()


@pytest.mark.asyncio
async def test_managed_service_rejects_occupied_port_without_http_readiness(
    tmp_path: Path,
) -> None:
    """未知 listener 即使不返回健康 HTTP，也不能被启动流程接管。"""
    failed = tmp_path / "failed.py"
    _ = failed.write_text("raise AssertionError('must not spawn')\n", encoding="utf-8")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = int(listener.getsockname()[1])
    host = PluginServiceHost()
    try:
        with pytest.raises(RuntimeError, match="监听端口已被占用"):
            await host.swap_plugin_services(
                "candidate",
                {},
                {"server": _service_spec(failed, port, "candidate")},
            )
    finally:
        listener.close()
        await host.stop_all()


@pytest.mark.skipif(sys.platform == "win32", reason="依赖 POSIX 进程组")
@pytest.mark.asyncio
async def test_managed_service_cleans_listener_after_leader_exit(
    tmp_path: Path,
) -> None:
    """ready 后 leader 意外退出时必须回收仍监听端口的同组后代。"""
    wrapper = tmp_path / "wrapper.py"
    child_pid_file = tmp_path / "child.pid"
    _ = wrapper.write_text(
        "import os, time, urllib.request\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "from pathlib import Path\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    class Handler(BaseHTTPRequestHandler):\n"
        "        def do_GET(self):\n"
        "            self.send_response(200); self.end_headers()\n"
        "            self.wfile.write(b'child')\n"
        "        def log_message(self, *args): pass\n"
        "    HTTPServer(('127.0.0.1', int(os.environ['PORT'])), "
        "Handler).serve_forever()\n"
        "Path(os.environ['CHILD_PID_FILE']).write_text(str(pid))\n"
        "url = 'http://127.0.0.1:' + os.environ['PORT'] + '/'\n"
        "for _ in range(100):\n"
        "    try:\n"
        "        urllib.request.urlopen(url, timeout=0.1).close()\n"
        "        break\n"
        "    except OSError:\n"
        "        time.sleep(0.02)\n"
        "time.sleep(0.5)\n"
        "raise SystemExit(17)\n",
        encoding="utf-8",
    )
    port = _free_port()
    spec = _service_spec(wrapper, port, "wrapper")
    env = spec["env"]
    assert isinstance(env, dict)
    env["CHILD_PID_FILE"] = str(child_pid_file)
    host = PluginServiceHost()
    host.bind_plugin_services({"wrapper": {"server": spec}})  # type: ignore[arg-type]
    try:
        await host.start_all()
        assert _read(port) == "child"
        child_pid = int(child_pid_file.read_text())

        await _wait_until(lambda: not _process_exists(child_pid))
        await _wait_until(lambda: _read_or_none(port) == "child")
        assert int(child_pid_file.read_text()) != child_pid
    finally:
        await host.stop_all()


def _process_exists(pid: int) -> bool:
    try:
        stat_fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat_fields[0] == "Z":
            return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 5.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_managed_service_stop_finishes_when_cancelled(tmp_path: Path) -> None:
    service = tmp_path / "slow_stop.py"
    _ = service.write_text(
        "import os, signal, time\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "def stop(*args): time.sleep(0.2); raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self): self.send_response(200); self.end_headers()\n"
        "    def log_message(self, *args): pass\n"
        "HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    port = _free_port()
    services = {"server": _service_spec(service, port, "v1")}
    host = PluginServiceHost()
    host.bind_plugin_services({"slow": services})  # type: ignore[arg-type]
    await host.start_all()
    stopping = asyncio.create_task(host.stop_all())
    await asyncio.sleep(0.05)
    stopping.cancel()

    with pytest.raises(asyncio.CancelledError):
        await stopping

    assert host._running == {}
    with pytest.raises(OSError):
        _read(port)


@pytest.mark.asyncio
async def test_managed_service_without_readiness_rejects_fast_exit(
    tmp_path: Path,
) -> None:
    failed = tmp_path / "failed.py"
    _ = failed.write_text("raise SystemExit(7)\n", encoding="utf-8")
    spec = {
        "worker": {
            "command": [sys.executable, str(failed)],
            "cwd": str(tmp_path),
            "env": {},
            "readiness_url": "",
            "startup_timeout_seconds": 1,
            "revision": "failed",
        }
    }
    host = PluginServiceHost()
    host.bind_plugin_services({"failed": {}})

    with pytest.raises(RuntimeError, match="exit=7"):
        await host.swap_plugin_services("failed", {}, spec)  # type: ignore[arg-type]

    assert host._running == {}


@pytest.mark.asyncio
async def test_start_all_preserves_start_error_when_rollback_fails() -> None:
    host = PluginServiceHost()
    host.bind_plugin_services({"plugin": {"a": {}, "b": {}}})
    start_error = RuntimeError("start failed")
    rollback_error = RuntimeError("rollback failed")

    async def _start(plugin_id: str, service_id: str, spec: object) -> None:
        if service_id == "b":
            raise start_error

    async def _stop(plugin_id: str, service_id: str) -> None:
        raise rollback_error

    host._start = _start  # type: ignore[method-assign]
    host._stop = _stop  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="start failed") as caught:
        await host.start_all()

    assert caught.value is start_error
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "rollback failed" in str(caught.value.__cause__)


def _exiting_worker_spec(
    script: Path,
    counter: Path,
    *,
    lifetime_seconds: float = 0.22,
) -> dict[str, object]:
    return {
        "command": [sys.executable, str(script)],
        "cwd": str(script.parent),
        "env": {
            "COUNTER": str(counter),
            "LIFETIME_SECONDS": str(lifetime_seconds),
        },
        "readiness_url": "",
        "startup_timeout_seconds": 1,
        "revision": "exiting",
    }


def _write_exiting_worker(script: Path) -> None:
    _ = script.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "counter = Path(os.environ['COUNTER'])\n"
        "value = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(value))\n"
        "time.sleep(float(os.environ['LIFETIME_SECONDS']))\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_active_service_exhausts_three_restarts_into_stable_fatal_future(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "exiting.py"
    counter = tmp_path / "counter"
    _write_exiting_worker(script)
    monkeypatch.setattr(
        service_host_module,
        "_RECOVERY_BACKOFF_SECONDS",
        (0.01, 0.01, 0.01),
    )
    host = PluginServiceHost()
    host.bind_plugin_services(
        {"unstable": {"worker": _exiting_worker_spec(script, counter)}}
    )
    await host.start_all()

    with pytest.raises(RuntimeError, match="recovery 耗尽") as first:
        await asyncio.wait_for(host.wait_fatal_failure(), timeout=2)
    with pytest.raises(RuntimeError, match="recovery 耗尽") as second:
        await host.wait_fatal_failure()

    assert first.value is second.value
    assert counter.read_text() == "4"
    assert host._running == {}
    assert host._epochs == {}


@pytest.mark.asyncio
async def test_stable_runtime_resets_recovery_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "exiting.py"
    counter = tmp_path / "counter"
    _write_exiting_worker(script)
    monkeypatch.setattr(service_host_module, "_RECOVERY_BACKOFF_SECONDS", (0.01,))
    monkeypatch.setattr(service_host_module, "_RECOVERY_STABLE_SECONDS", 0.01)
    host = PluginServiceHost()
    host.bind_plugin_services(
        {"stable": {"worker": _exiting_worker_spec(script, counter)}}
    )
    await host.start_all()

    await _wait_until(lambda: counter.exists() and int(counter.read_text()) >= 3)
    assert host._fatal_failure is None
    await host.stop_all()


@pytest.mark.asyncio
async def test_explicit_stop_during_backoff_never_resurrects_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "exiting.py"
    counter = tmp_path / "counter"
    _write_exiting_worker(script)
    monkeypatch.setattr(
        service_host_module,
        "_RECOVERY_BACKOFF_SECONDS",
        (0.3, 0.3, 0.3),
    )
    host = PluginServiceHost()
    host.bind_plugin_services(
        {"stopped": {"worker": _exiting_worker_spec(script, counter)}}
    )
    await host.start_all()
    await _wait_until(lambda: host._running == {})

    await host.stop_all()
    await asyncio.sleep(0.35)

    assert counter.read_text() == "1"
    assert host._running == {}
    assert host._epochs == {}


@pytest.mark.asyncio
async def test_swap_during_backoff_cannot_revive_or_kill_new_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exiting = tmp_path / "exiting.py"
    healthy = tmp_path / "healthy.py"
    counter = tmp_path / "counter"
    _write_exiting_worker(exiting)
    _ = healthy.write_text(
        "import time\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(service_host_module, "_RECOVERY_BACKOFF_SECONDS", (0.3,))
    old = {"worker": _exiting_worker_spec(exiting, counter)}
    new = {
        "worker": {
            "command": [sys.executable, str(healthy)],
            "cwd": str(tmp_path),
            "env": {},
            "readiness_url": "",
            "startup_timeout_seconds": 1,
            "revision": "healthy",
        }
    }
    host = PluginServiceHost()
    host.bind_plugin_services({"swapped": old})
    await host.start_all()
    await _wait_until(lambda: host._running == {})

    await host.swap_plugin_services("swapped", old, new)
    replacement = host._running[("swapped", "worker")]
    await asyncio.sleep(0.35)

    assert host._running[("swapped", "worker")] is replacement
    assert replacement.process.returncode is None
    assert counter.read_text() == "1"
    await host.stop_all()


@pytest.mark.asyncio
async def test_candidate_exhaustion_rejects_promotion_without_active_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "exiting.py"
    counter = tmp_path / "counter"
    _write_exiting_worker(script)
    monkeypatch.setattr(service_host_module, "_RECOVERY_BACKOFF_SECONDS", (0.01,))
    host = PluginServiceHost()
    await host.start_candidate(
        "candidate-1",
        {"worker": _exiting_worker_spec(script, counter)},
    )
    await _wait_until(lambda: "candidate-1" in host._candidate_failures)

    with pytest.raises(RuntimeError, match="recovery 耗尽"):
        await host.assert_candidate_healthy("candidate-1")
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(host.wait_fatal_failure(), timeout=0.05)

    await host.stop_candidate("candidate-1")


@pytest.mark.asyncio
async def test_candidate_cleanup_during_backoff_never_resurrects_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "exiting.py"
    counter = tmp_path / "counter"
    _write_exiting_worker(script)
    monkeypatch.setattr(service_host_module, "_RECOVERY_BACKOFF_SECONDS", (0.3,))
    host = PluginServiceHost()
    await host.start_candidate(
        "candidate-cleanup",
        {"worker": _exiting_worker_spec(script, counter)},
    )
    await _wait_until(lambda: host._running == {})

    await host.stop_candidate("candidate-cleanup")
    await asyncio.sleep(0.35)

    assert counter.read_text() == "1"
    assert host._running == {}
    assert host._epochs == {}


@pytest.mark.asyncio
async def test_candidate_recovering_without_live_attempt_rejects_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "exiting.py"
    counter = tmp_path / "counter"
    _write_exiting_worker(script)
    monkeypatch.setattr(service_host_module, "_RECOVERY_BACKOFF_SECONDS", (0.3,))
    host = PluginServiceHost()
    await host.start_candidate(
        "candidate-recovering",
        {"worker": _exiting_worker_spec(script, counter)},
    )
    await _wait_until(lambda: host._running == {})

    with pytest.raises(RuntimeError, match="没有健康的当前 process epoch"):
        await host.assert_candidate_healthy("candidate-recovering")

    await host.stop_candidate("candidate-recovering")


@pytest.mark.asyncio
async def test_candidate_readiness_is_reprobed_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "service.py"
    _ = script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    host = PluginServiceHost()
    await host.start_candidate(
        "candidate-readiness",
        {
            "worker": {
                "command": [sys.executable, str(script)],
                "cwd": str(tmp_path),
                "env": {},
                "readiness_url": "",
                "startup_timeout_seconds": 1,
                "revision": "candidate",
            }
        },
    )
    epoch = host._epochs[("validation:candidate-readiness", "worker")]
    epoch.spec["readiness_url"] = "http://127.0.0.1:1/ready"
    monkeypatch.setattr(service_host_module, "_url_ready", lambda _url: False)

    with pytest.raises(RuntimeError, match="readiness 失败"):
        await host.assert_candidate_healthy("candidate-readiness")

    await host.stop_candidate("candidate-readiness")
