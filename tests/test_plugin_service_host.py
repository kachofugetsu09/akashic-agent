from __future__ import annotations

import socket
import asyncio
import sys
import urllib.request
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
