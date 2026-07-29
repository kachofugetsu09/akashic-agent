from __future__ import annotations

import asyncio
import os
import socket
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest

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

        await _wait_until(lambda: host._running == {})
        await _wait_until(lambda: not _process_exists(child_pid))
        with pytest.raises(OSError):
            _read(port)
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
