from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from websockets.sync.server import serve
from websockets.typing import Subprotocol

from agent.plugin_composition import CompositionError, DashboardContext
from agent.plugins.manager import PluginManager
from agent.plugins.static_manifest import load_static_plugin_manifest
from agent.tools.registry import ToolRegistry
from agent.workloads.model import (
    WorkloadEndpoint,
    WorkloadLease,
    WorkloadStartReceipt,
    WorkloadStopReceipt,
)
from bus.event_bus import EventBus
from plugins.computer.dashboard import register as register_computer_dashboard
from plugins.computer.mcp_server import save_screenshot

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "computer"
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _Gateway(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send("application/json", b'{"status":"ready"}')
            return
        if self.path == "/activity":
            self._send(
                "application/json",
                b'{"revision":2,"noticeId":1,"active":false}',
            )
            return
        if self.path.startswith("/screenshot"):
            self._send("image/png", _PNG)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(size))
        if self.path == "/input":
            self._send("application/json", json.dumps(payload).encode())
            return
        if self.path == "/browser/observe":
            value = (
                {
                    "mimeType": "image/png",
                    "data": base64.b64encode(_PNG).decode("ascii"),
                }
                if payload["observe"] == "screenshot"
                else {"url": "https://example.com", "title": "Example Domain"}
            )
            self._send("application/json", json.dumps(value).encode())
            return
        if self.path == "/browser/action":
            self._send("application/json", json.dumps({"ok": True, **payload}).encode())
            return
        self.send_error(404)

    def _send(self, media_type: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("content-type", media_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        _ = args


class _Controller:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.starts = []
        self.stops = []

    async def start(self, request) -> WorkloadStartReceipt:
        self.starts.append(request)
        lease = WorkloadLease(
            workspace_id=request.workspace_id,
            plugin_id=request.plugin_id,
            workload=request.workload,
            mode=request.mode,
            transaction_id=request.transaction_id,
            generation_id=request.generation_id,
            container_id=f"computer-{len(self.starts)}",
            spec_digest=request.spec_digest,
        )
        return WorkloadStartReceipt(
            lease,
            tuple(
                WorkloadEndpoint(name, self.endpoint) for name, _number in request.ports
            ),
            None,
        )

    async def stop(self, lease: WorkloadLease) -> WorkloadStopReceipt:
        self.stops.append(lease)
        return WorkloadStopReceipt(lease, True, True)

    async def cleanup_candidates(
        self, workspace_id: str
    ) -> tuple[WorkloadStopReceipt, ...]:
        _ = workspace_id
        return ()


def test_computer_static_manifest_owns_workload_mcp_and_data() -> None:
    manifest = load_static_plugin_manifest(PLUGIN)

    assert manifest.name == "computer"
    assert manifest.exclude_data_paths == (
        "state/config/pulse",
        "state/config/xfce4/desktop/icons.screen.latest.rc",
        "state/home/.opencli/node_modules",
        "state/profile/SingletonCookie",
        "state/profile/SingletonLock",
        "state/profile/SingletonSocket",
    )
    assert manifest.workloads[0].ports == (
        ("gateway", 8080),
        ("display", 6080),
        ("opencli", 19826),
    )
    assert manifest.workloads[0].loopback_ports == (("opencli", 19825),)
    assert manifest.workloads[0].data == (("state", "/data", True),)
    assert manifest.workloads[0].user_namespaces is True
    assert manifest.mcp_servers[0].workload_env == (
        ("COMPUTER_URL", "computer", "gateway"),
    )
    assert manifest.mcp_servers[0].required_tools == (
        "browser_observe",
        "browser_action",
        "computer_observe",
        "computer_action",
    )


def test_opencli_stays_a_skill_for_the_ordinary_shell() -> None:
    skill = (PLUGIN / "skills" / "opencli" / "SKILL.md").read_text(encoding="utf-8")

    assert "name: opencli" in skill
    assert 'shell({"command":"opencli ' in skill
    assert "OPENCLI_DAEMON_PORT" not in skill
    assert 'browser({"args"' not in skill
    assert not (PLUGIN / "skills" / "computer" / "SKILL.md").exists()


def test_browser_ref_click_scrolls_before_reading_click_coordinates() -> None:
    gateway = (ROOT / "docker" / "computer" / "gateway.mjs").read_text(encoding="utf-8")
    click = gateway[gateway.index("async function clickNode") :]

    assert click.index("DOM.scrollIntoViewIfNeeded") < click.index("DOM.getBoxModel")


def test_browser_ref_focus_keeps_the_backend_node_across_cdp_sessions() -> None:
    gateway = (ROOT / "docker" / "computer" / "gateway.mjs").read_text(encoding="utf-8")
    focus = gateway[
        gateway.index("async function focusNode") : gateway.index(
            "async function clickNode"
        )
    ]

    assert '"DOM.focus", { backendNodeId }' in focus
    assert "DOM.resolveNode" not in focus
    assert "Runtime.callFunctionOn" not in focus


def test_browser_fill_uses_chromiums_select_all_edit_command() -> None:
    gateway = (ROOT / "docker" / "computer" / "gateway.mjs").read_text(encoding="utf-8")
    fill = gateway[
        gateway.index('if (action === "fill")') : gateway.index(
            'await cdp(selected.target, "Input.insertText"'
        )
    ]

    assert 'commands: ["SelectAll"]' in fill
    assert "windowsVirtualKeyCode: 65" in fill
    assert "windowsVirtualKeyCode: 8" in fill


def test_computer_screenshot_files_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "computer-data"
    monkeypatch.setenv("AKA_PLUGIN_DATA_DIR", str(data_dir))

    last = ""
    for _index in range(40):
        last = save_screenshot(_PNG, "image/png")

    screenshots = tuple((data_dir / "screenshots").glob("computer-*.png"))
    assert len(screenshots) == 32
    assert Path(last).is_file()


def test_visual_drag_sends_intermediate_pointer_positions() -> None:
    gateway = (ROOT / "docker" / "computer" / "gateway.mjs").read_text(encoding="utf-8")

    assert "const dragSteps = 8;" in gateway
    assert "for (let step = 1; step <= dragSteps; step += 1)" in gateway
    assert 'step === dragSteps ? "0.15" : "0.04"' in gateway


def test_dashboard_context_exposes_only_declared_workload_port(tmp_path: Path) -> None:
    context = DashboardContext(
        plugin_id="computer",
        plugin_dir=PLUGIN,
        data_root=tmp_path,
        validation=False,
        _workload_urls=MappingProxyType(
            {
                ("computer", "gateway"): "http://computer.internal:8080",
                ("computer", "display"): "http://computer.internal:6080",
            }
        ),
    )

    assert (
        context.workload_url("computer", "gateway") == "http://computer.internal:8080"
    )
    with pytest.raises(CompositionError, match="未声明 Workload port"):
        context.workload_url("computer", "desktop")


def test_computer_dashboard_proxies_activity_and_live_display(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    display_closed = threading.Event()

    def echo_display(connection) -> None:
        try:
            for message in connection:
                connection.send(message)
        finally:
            display_closed.set()

    display_server = serve(
        echo_display,
        sock=listener,
        subprotocols=[Subprotocol("binary")],
        compression=None,
    )
    display_thread = threading.Thread(
        target=display_server.serve_forever,
        daemon=True,
    )
    display_thread.start()
    display_port = listener.getsockname()[1]
    context = DashboardContext(
        plugin_id="computer",
        plugin_dir=PLUGIN,
        data_root=tmp_path,
        validation=False,
        _workload_urls=MappingProxyType(
            {
                ("computer", "gateway"): (f"http://127.0.0.1:{server.server_port}"),
                ("computer", "display"): f"http://127.0.0.1:{display_port}",
            }
        ),
    )
    app = FastAPI()
    gateway_client = register_computer_dashboard(app, context)
    try:
        with TestClient(app) as client:
            assert (
                client.get("/api/dashboard/computer/activity").json()["noticeId"] == 1
            )
            with client.websocket_connect(
                "/api/dashboard/computer/display",
                subprotocols=["binary"],
            ) as display:
                assert display.accepted_subprotocol == "binary"
                display.send_bytes(b"RFB 003.008\n")
                assert display.receive_bytes() == b"RFB 003.008\n"
            assert display_closed.wait(timeout=5)
    finally:
        gateway_client.close()
        display_server.shutdown()
        display_thread.join(timeout=5)
        server.shutdown()
        server.server_close()


def test_computer_mcp_calls_exact_workload_gateway(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = dict(os.environ)
    env["COMPUTER_URL"] = f"http://127.0.0.1:{server.server_port}"
    env["AKA_PLUGIN_DATA_DIR"] = str(tmp_path / "computer-data")
    process = subprocess.Popen(
        [str(PLUGIN / "mcp_server.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        messages = (
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "browser_observe",
                    "arguments": {"observe": "get_title"},
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "missing", "arguments": {}},
            },
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "browser_observe",
                    "arguments": {"observe": "screenshot"},
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "computer_observe",
                    "arguments": {"observe": "screenshot"},
                },
            },
        )
        for message in messages:
            process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

        initialized = json.loads(process.stdout.readline())
        tools = json.loads(process.stdout.readline())
        call = json.loads(process.stdout.readline())
        failed_call = json.loads(process.stdout.readline())
        browser_screenshot = json.loads(process.stdout.readline())
        desktop_screenshot = json.loads(process.stdout.readline())
        assert initialized["result"]["protocolVersion"] == "2025-11-25"
        assert [item["name"] for item in tools["result"]["tools"]] == [
            "browser_observe",
            "browser_action",
            "computer_observe",
            "computer_action",
        ]
        browser_action = tools["result"]["tools"][1]
        browser_schema = browser_action["inputSchema"]["properties"]
        assert "navigate" in browser_schema["action"]["enum"]
        assert browser_schema["ref"]["pattern"].startswith("^e")
        assert browser_schema["snapshot_id"]["maxLength"] == 64
        action_tool = tools["result"]["tools"][3]
        action_schema = action_tool["inputSchema"]["properties"]
        assert "drag" in action_schema["action"]["enum"]
        assert action_schema["to_x"]["maximum"] == 1279
        assert action_schema["to_y"]["maximum"] == 799
        assert "Example Domain" in call["result"]["content"][0]["text"]
        assert failed_call["id"] == 4
        assert "unknown tool" in failed_call["error"]["message"]
        screenshot_dir = tmp_path / "computer-data" / "screenshots"
        screenshot_paths = []
        for response in (browser_screenshot, desktop_screenshot):
            assert response["result"]["content"][0]["type"] == "text"
            reference = json.loads(response["result"]["content"][0]["text"])
            assert reference["kind"] == "screenshot_file"
            assert reference["mime_type"] == "image/png"
            screenshot_path = Path(reference["path"])
            assert screenshot_path.parent == screenshot_dir
            assert screenshot_path.read_bytes() == _PNG
            assert "read_image_vision" in reference["next"]
            screenshot_paths.append(screenshot_path)
        assert screenshot_paths[0] != screenshot_paths[1]
    finally:
        process.terminate()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()
        assert process.stderr is not None
        process.stderr.close()
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_builtin_computer_loads_through_public_plugin_contract(
    tmp_path: Path,
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    controller = _Controller(f"http://127.0.0.1:{server.server_port}")
    manager = PluginManager(
        plugin_dirs=[ROOT / "plugins" / "conversation_ui", PLUGIN],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(validate_semantic_schema=False),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "plugin-home" / "cache",
        workload_controller=controller,
    )
    try:
        await manager.load_all()
        generation = manager.generation("computer")
        assert generation is not None
        assert manager.workload_urls(generation.generation_id) == {
            ("computer", "gateway"): controller.endpoint,
            ("computer", "display"): controller.endpoint,
            ("computer", "opencli"): controller.endpoint,
        }
        snapshot = manager.current_snapshot
        assert snapshot is not None and snapshot.tool_registry is not None
        assert snapshot.tool_registry.get_tool_names_by_source("mcp", "computer") == {
            "mcp_computer__browser_observe",
            "mcp_computer__browser_action",
            "mcp_computer__computer_observe",
            "mcp_computer__computer_action",
        }
    finally:
        await manager.terminate_all()
        server.shutdown()
        server.server_close()
    assert len(controller.starts) == len(controller.stops) == 1
