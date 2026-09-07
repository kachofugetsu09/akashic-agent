from pathlib import Path
from collections.abc import Mapping
from typing import Literal, cast
import asyncio
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import pytest

from agent.plugin_composition import (
    MCP_SERVERS,
    TOOL_CATALOG,
    WORKLOADS,
    CompositionRoot,
    PluginRuntime,
    PluginTools,
    PluginWorkloads,
)
from agent.plugin_composition.bindings import BINDINGS, Bindings
from agent.plugin_composition.messages import MESSAGE_CATALOG, OWNER_STATE, MessageCatalog, OwnerState
from agent.plugins.archive import PluginArchive
from agent.plugin_composition.mcp_slots import PluginMcpServers, _freeze_plugin_mcp_servers
from plugins.tools import plugin as tools_plugin
from plugins.tools.api import MessageReply
from plugins.tools.plugin import TOOLS
from plugins.turn_projection import plugin as turn_projection_plugin
from session.log import MessageLog
from agent.plugin_composition.tool_catalog import _freeze_plugin_tools
from agent.plugins.static_manifest import (
    load_static_plugin_manifest,
    validate_module_exports,
)
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from agent.plugins.workload_generation_host import WorkloadGenerationHost
from agent.workloads.model import (
    WorkloadEndpoint,
    WorkloadLease,
    WorkloadStartReceipt,
    WorkloadStartRequest,
    WorkloadStopReceipt,
)
from bus.event_bus import EventBus
from plugins.computer import plugin
from plugins.computer.control import endpoint_name, request
from session.message import CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult


@pytest.mark.asyncio
async def test_computer_plugin_mounts_with_static_manifest(tmp_path: Path) -> None:
    """静态入口只声明 Message Tool 与空 discovery 的 MCP。"""
    path = Path(plugin.__file__).parent
    manifest = load_static_plugin_manifest(path)
    validate_module_exports(manifest, plugin, plugin_root=path)
    assert TOOL_CATALOG not in plugin.inject
    assert any(key.name == "tools.v1" for key in plugin.inject)
    assert manifest.mcp_servers[0].required_tools == ()
    assert manifest.mcp_servers[0].candidate_read_only_tools == ()


@pytest.mark.asyncio
async def test_computer_plugin_mounts_real_tools_and_mcp_services(tmp_path: Path) -> None:
    """真实 Composition Root 接纳新 Tool 与 workload/MCP 声明。"""
    log = MessageLog(tmp_path / "sessions.db")
    root = CompositionRoot("computer-services")
    mcp = PluginMcpServers(root.instance_token)
    workloads = PluginWorkloads(root.instance_token)
    archive = PluginArchive(tmp_path / "archives")

    from contextlib import asynccontextmanager
    from agent.plugin_composition.bindings import BindingScope

    @asynccontextmanager
    async def unused_open(_components):
        yield BindingScope(root)

    bindings = Bindings(log, archive, unused_open)
    for key, value in (
        (MCP_SERVERS, mcp),
        (WORKLOADS, workloads),
        (BINDINGS, bindings),
        (MESSAGE_CATALOG, MessageCatalog(log)),
        (OWNER_STATE, OwnerState(log)),
    ):
        await root.context.provide(key, value)
    path = Path(plugin.__file__).parent
    try:
        await root.mount(
            lambda ctx: tools_plugin.apply(ctx, {}),
            name="tools",
            inject=tools_plugin.inject,
            runtime=PluginRuntime(
                plugin_id="tools", generation_id="tools-services", plugin_dir=path.parent / "tools",
                data_dir=tmp_path / "tools-data", workspace=tmp_path, config={},
            ),
        )
        await root.mount(
            lambda ctx: turn_projection_plugin.apply(ctx, {}),
            name="turn_projection",
            inject=turn_projection_plugin.inject,
            runtime=PluginRuntime(
                plugin_id="turn_projection", generation_id="projection-services", plugin_dir=path.parent / "turn_projection",
                data_dir=tmp_path / "projection-data", workspace=tmp_path, config={},
            ),
        )
        await root.mount(
            lambda ctx: plugin.apply(ctx, {}),
            name="computer",
            inject=plugin.inject,
            runtime=PluginRuntime(
                plugin_id="computer", generation_id="computer-services", plugin_dir=path,
                data_dir=tmp_path / "computer-data", workspace=tmp_path, config={},
            ),
        )
        assert [item["name"] for item in root.context.require(TOOLS).descriptions()] == ["computer"]
        registry = _freeze_plugin_mcp_servers(mcp, root.instance_token)
        assert registry["computer"].definition.workload_env[0].env == "COMPUTER_URL"
    finally:
        await root.dispose()
        log.close()


@pytest.mark.asyncio
async def test_computer_mcp_subprocess_keeps_isolated_control_boundary(tmp_path: Path) -> None:
    """独立 MCP 进程只依赖插件目录，并把 driver context 原样送到 loopback。"""
    seen: list[dict[str, object]] = []

    class Gateway(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"version":2,"source":true,"ready":true}')

        def do_POST(self) -> None:
            size = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(size))
            seen.append(body)
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"content": [{"type": "text", "text": "ok"}], "call_id": body["context"]["call_id"]}
                ).encode()
            )

        def log_message(self, *_args: object) -> None:
            return

    gateway = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    Thread(target=gateway.serve_forever, daemon=True).start()
    path = Path(plugin.__file__).parent
    data_root = tmp_path / "plugin-data"
    data_root.mkdir()
    scope_id = "bound-subprocess-test"
    assert endpoint_name(data_root, scope_id) != endpoint_name(data_root, "other-generation")
    environment = {
        "PATH": os.environ["PATH"],
        "COMPUTER_URL": f"http://127.0.0.1:{gateway.server_port}",
        "AKA_PLUGIN_DATA_DIR": str(data_root),
        "AKASHIC_MCP_SCOPE_ID": scope_id,
    }
    code = f"import sys; sys.path.insert(0, {str(path)!r}); import mcp_server; mcp_server.main()"
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        code,
        env=environment,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert child.stdin is not None and child.stdout is not None
        child.stdin.write(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n')
        await child.stdin.drain()
        assert json.loads(await child.stdout.readline())["result"]["capabilities"]["tools"] == {}
        child.stdin.write(b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n')
        await child.stdin.drain()
        assert json.loads(await child.stdout.readline())["result"]["tools"] == []
        reader, writer = await asyncio.open_unix_connection(
            "\0" + endpoint_name(data_root, scope_id)
        )
        writer.write(
            json.dumps(
                {
                    "op": "run",
                    "context": {"session_id": "s", "turn_id": "t", "call_id": "c"},
                    "code": "nodeRepl.write('ok')",
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        assert json.loads(await reader.readline())["call_id"] == "c"
        writer.close()
        await writer.wait_closed()
        assert seen[0]["context"] == {"session_id": "s", "turn_id": "t", "call_id": "c"}
    finally:
        child.stdin.close()
        await asyncio.wait_for(child.wait(), 10)
        gateway.shutdown()
        gateway.server_close()


@pytest.mark.asyncio
async def test_computer_control_against_optional_container_oracle(tmp_path: Path) -> None:
    """保留真实 Computer 容器 oracle，并按当前 scope/control 协议调用。"""
    gateway = os.environ.get("COMPUTER_TEST_GATEWAY")
    if not gateway:
        pytest.skip("requires a disposable Computer container")
    httpx = pytest.importorskip("httpx")
    path = Path(plugin.__file__).parent
    data_root = tmp_path / "plugin-data"
    data_root.mkdir()
    scope_id = "computer-container-oracle"
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        str(path / "mcp_server.py"),
        env={
            **os.environ,
            "COMPUTER_URL": gateway,
            "AKA_PLUGIN_DATA_DIR": str(data_root),
            "AKASHIC_MCP_SCOPE_ID": scope_id,
        },
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert child.stdin is not None and child.stdout is not None

        async def read_json() -> dict[str, object]:
            raw = await asyncio.wait_for(child.stdout.readline(), 30)
            assert raw
            value = json.loads(raw)
            assert isinstance(value, dict)
            return value

        child.stdin.write(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n')
        await child.stdin.drain()
        initialize = await read_json()
        initialize_result = cast(Mapping[str, object], initialize["result"])
        server_info = cast(Mapping[str, object], initialize_result["serverInfo"])
        assert server_info["name"] == "akashic-computer"
        child.stdin.write(b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n')
        await child.stdin.drain()
        listing = await read_json()
        assert cast(Mapping[str, object], listing["result"])["tools"] == []
        child.stdin.write(
            b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"browser_action","arguments":{"action":"navigate","url":"about:blank"}}}\n'
        )
        await child.stdin.drain()
        error = cast(Mapping[str, object], (await read_json())["error"])
        assert error["code"] == -32601

        async def driver_call(
            call_id: str,
            code: str,
            *,
            op: str = "run",
            timeout_ms: int = 60000,
        ) -> dict[str, object]:
            value = await request(
                endpoint_name(data_root, scope_id),
                {
                    "op": op,
                    "context": {
                        "generation_id": scope_id,
                        "session_id": "container-oracle",
                        "source": "test",
                        "turn_id": "turn:container-oracle",
                        "turn_input_id": "oracle-input",
                        "call_id": call_id,
                    },
                    "code": code,
                    "timeoutMs": timeout_ms,
                },
            )
            assert isinstance(value, dict)
            return value

        output = await driver_call(
            "first",
            "nodeRepl.write(41+1); await nodeRepl.emitImage((await sky.get_screenshot())[0].bytes);",
        )
        assert "42" in json.dumps(output)
        screenshots = list((data_root / "screenshots").glob("*"))
        assert screenshots

        async with httpx.AsyncClient(base_url=gateway) as client:
            running = asyncio.create_task(
                driver_call("task-cancel", "await new Promise(()=>{});", timeout_ms=30000)
            )
            async with asyncio.timeout(10):
                while not (await client.get("/activity")).json()["active"]:
                    await asyncio.sleep(0.05)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running

            await driver_call("after-cancel", "nodeRepl.write('after cancel');")
            new = await client.post(
                "/browser/action", json={"action": "tab_new", "url": "about:blank"}
            )
            new.raise_for_status()
            closed = await client.post(
                "/browser/action",
                json={"action": "tab_close", "target_id": new.json()["target_id"]},
            )
            closed.raise_for_status()

        scratch_result = await driver_call(
            "scratch",
            "const anchor = await browser.tabs.new(); await anchor.markDeliverable(); const scratch = await browser.tabs.new(); nodeRepl.write(JSON.stringify({anchor: anchor.id, scratch: scratch.id}));",
        )
        scratch_content = cast(list[object], scratch_result["content"])
        scratch_item = cast(Mapping[str, object], scratch_content[0])
        tabs = json.loads(cast(str, scratch_item["text"]))
        assert isinstance(tabs, dict)
        anchor = cast(str, tabs["anchor"])
        scratch = cast(str, tabs["scratch"])
        await driver_call("end-turn", "", op="end_turn")
        listing = await driver_call("listing", "nodeRepl.write(await browser.tabs.list());")
        listing_text = json.dumps(listing)
        assert scratch not in listing_text
        assert anchor in listing_text
    finally:
        if child.stdin is not None:
            child.stdin.close()
        await asyncio.wait_for(child.wait(), 25)
        assert child.returncode == 0, (await child.stderr.read()).decode() if child.stderr else ""



class _ComputerGatewayState:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.routed_calls: list[tuple[str, dict[str, object]]] = []
        self.events: list[tuple[str, dict[str, object]]] = []
        self.fail_runs = 0
        self.fail_ends = 0
        self.hold_release = Event()


def _call_context(call: Mapping[str, object]) -> Mapping[str, object]:
    """Return the driver context from one recorded gateway request."""
    value = call.get("context")
    assert isinstance(value, Mapping)
    return value


class _WorkloadController:
    def __init__(self, port: int) -> None:
        self.port = port
        self.ports_by_image: dict[str, int] = {}
        self.started: list[WorkloadStartRequest] = []
        self.stopped: list[WorkloadLease] = []

    async def cleanup_candidates(self, workspace_id: str) -> tuple[WorkloadStopReceipt, ...]:
        _ = workspace_id
        return ()

    async def start(self, request: WorkloadStartRequest) -> WorkloadStartReceipt:
        self.started.append(request)
        lease = WorkloadLease(
            request.workspace_id,
            request.plugin_id,
            request.workload,
            request.mode,
            request.transaction_id,
            request.generation_id,
            "computer-test-container",
            request.spec_digest,
        )
        port = self.ports_by_image.get(request.image, self.port)
        url = f"http://127.0.0.1:{port}"
        return WorkloadStartReceipt(
            lease,
            (
                WorkloadEndpoint("gateway", url),
                WorkloadEndpoint("display", url),
                WorkloadEndpoint("opencli", url),
            ),
            None,
        )

    async def stop(self, lease: WorkloadLease) -> WorkloadStopReceipt:
        self.stopped.append(lease)
        return WorkloadStopReceipt(lease, True, True)


class _ComputerHarness:
    def __init__(self, root: Path, workspace: Path, log: MessageLog,
                 manager: PluginManager, gateway: ThreadingHTTPServer,
                 gateway_state: _ComputerGatewayState,
                 controller: _WorkloadController) -> None:
        self.root = root
        self.workspace = workspace
        self.log = log
        self.manager = manager
        self.gateway = gateway
        self.gateway_state = gateway_state
        self.controller = controller
        snapshot = manager.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        self.composition_root = snapshot.composition_root
        workload_host = manager.composition_generation_host._workload_host
        assert workload_host is not None
        self.workload_host: WorkloadGenerationHost = workload_host
        self.tools = self.composition_root.context.require(TOOLS)
        self.bindings = self.composition_root.context.require(BINDINGS)
        self.binding: str | None = None

    async def bind_computer(self) -> str:
        async with lease_runtime_snapshot(self.manager.snapshot_store):
            self.binding = self.tools.bind("computer", self.bindings)
        assert self.binding is not None
        return self.binding

    def _writer(self, session_id: str, author: str, body_types: tuple[type, ...], *, call_ref=None):
        return self.log.writer(
            session_id,
            author=author,
            source="chat",
            body_types=body_types,
            content={"text": lambda _part: ContentReferences()},
            call_ref=call_ref,
            check_call=lambda _call: None,
        )

    def add_call(self, name: str, code: str = "nodeRepl.write('ok');") -> MessageReply:
        assert self.binding is not None
        session_id = f"computer-session:{name}"
        input_id, output_id, result_id = f"{name}-input", f"{name}-output", f"{name}-result"
        self._writer(session_id, "user", (Input,)).append(
            input_id, Input((ContentPart("text", "run computer"),))
        )
        self._writer(session_id, "assistant", (Output,)).append(
            output_id,
            Output((ToolCall(self.binding, {"code": code}),), "continue"),
        )
        call_ref = CallRef(output_id, 0)
        result_writer = self._writer(session_id, "tool", (ToolResult,), call_ref=call_ref)
        return MessageReply(result_id, call_ref, self.log.reader(session_id), result_writer, lambda: None)

    async def execute(self, reply: MessageReply):
        async with lease_runtime_snapshot(self.manager.snapshot_store):
            tools = self.composition_root.context.require(TOOLS)
            async def authorize(_binding: str, _arguments: Mapping[str, object]):
                return {}
            return await tools.execution(authorize).execute_call(reply)

    def finish(self, reply: MessageReply, status: Literal["complete", "quiet", "abandoned"]) -> None:
        session_id = reply.reader.session_id
        if status in {"complete", "quiet"}:
            self._writer(session_id, "assistant", (Output,)).append(
                f"{session_id}-finish",
                Output(
                    (ContentPart("text", status),),
                    cast(Literal["complete", "quiet"], status),
                ),
            )
            return
        assert status == "abandoned"
        result = reply.reader.get(reply.message_id)
        assert result is not None
        self._writer(session_id, "system", (Control,)).append(
            f"{session_id}-abandon",
            Control("abandon", result.seq),
        )

    def control(self, reply: MessageReply, action: Literal["pause", "failure"]) -> None:
        """Append a non-terminal Control while leaving its Turn open."""
        result = reply.reader.get(reply.message_id)
        assert result is not None
        self._writer(reply.reader.session_id, "system", (Control,)).append(
            f"{reply.reader.session_id}-{action}",
            Control(action, result.seq),
        )

    def owner(self, reply: MessageReply):
        return self.log.owner("plugin:computer").read(
            "computer-use:" + "message:" + json.dumps(
                [reply.call_ref.message_id, reply.call_ref.part_index],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    async def close(self) -> None:
        await self.manager.terminate_all()
        self.log.close()
        self.gateway.shutdown()
        self.gateway.server_close()


def _start_test_gateway(
    state: _ComputerGatewayState,
    label: str,
) -> ThreadingHTTPServer:
    """Start one loopback driver target with an observable route label."""

    class Gateway(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"version":2,"source":true,"ready":true}')

        def do_POST(self) -> None:
            size = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(size))
            if self.path == "/driver/cancel":
                state.events.append(("cancel", body))
                state.hold_release.set()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"cancelled":true}')
                return
            state.calls.append(body)
            state.routed_calls.append((label, body))
            state.events.append(("run", body))
            if body.get("endTurn") and state.fail_ends:
                state.fail_ends -= 1
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"temporary end failure")
                return
            if body.get("code") == "hold":
                state.hold_release.wait(10)
                state.events.append(("released", body))
            if body.get("code") == "fail" and state.fail_runs:
                state.fail_runs -= 1
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"temporary run failure")
                return
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "content": [{"type": "text", "text": "ok"}],
                "call_id": body["context"]["call_id"],
            }).encode())

        def log_message(self, *_args: object) -> None:
            return

    gateway = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    Thread(target=gateway.serve_forever, daemon=True).start()
    return gateway


async def _computer_harness(tmp_path: Path, *, log: MessageLog | None = None,
                            gateway: ThreadingHTTPServer | None = None,
                            gateway_state: _ComputerGatewayState | None = None,
                            gateway_label: str = "gateway",
                            controller: _WorkloadController | None = None,
                            ) -> _ComputerHarness:
    """Start the real manager, workload host, MCP process, and Message Tool catalog."""
    import shutil

    if gateway is None:
        if gateway_state is None:
            gateway_state = _ComputerGatewayState()
        gateway = _start_test_gateway(gateway_state, gateway_label)
    assert gateway_state is not None
    source_root = tmp_path / "computer-plugins"
    repo_plugins = Path(plugin.__file__).parent.parent
    source_root.mkdir(exist_ok=True)
    for name in ("tools", "turn_projection", "computer"):
        destination = source_root / name
        if not destination.exists():
            shutil.copytree(
                repo_plugins / name,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules"),
            )
    workspace = tmp_path / "computer-workspace"
    workspace.mkdir(exist_ok=True)
    if log is None:
        log = MessageLog(tmp_path / "computer-sessions.db")
    if controller is None:
        controller = _WorkloadController(gateway.server_port)
    manager = PluginManager(
        [source_root],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "plugin-cache",
        message_log=log,
        workload_controller=controller,
    )
    await manager.load_all()
    await manager.start_runtime()
    harness = _ComputerHarness(
        source_root, workspace, log, manager, gateway, gateway_state, controller
    )
    await harness.bind_computer()
    return harness


async def _wait_until(predicate, *, timeout: float = 10) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_computer_message_tool_and_follower_closes_turn_statuses(tmp_path: Path) -> None:
    """真实 Message CallRef 执行后，follower 只收尾 complete/quiet/abandoned Turn。"""
    harness = await _computer_harness(tmp_path)
    try:
        replies: list[tuple[MessageReply, Literal["complete", "quiet", "abandoned"]]] = []
        cases: tuple[tuple[str, Literal["complete", "quiet", "abandoned"]], ...] = (
            ("complete", "complete"),
            ("quiet", "quiet"),
            ("abandon", "abandoned"),
        )
        for name, status in cases:
            reply = harness.add_call(name)
            result = await harness.execute(reply)
            assert result.outcome == "success"
            owner = harness.owner(reply)
            assert owner is not None
            binding = harness.binding
            assert binding is not None
            binding_state = harness.bindings.describe(binding, TOOLS)["state"]
            assert isinstance(binding_state, Mapping)
            control_binding = binding_state["control_binding"]
            assert isinstance(control_binding, str)
            assert owner.value == {
                "v": 1,
                "phase": "started",
                "control_binding": control_binding,
                "session_id": reply.reader.session_id,
                "source": "chat",
                "turn_input_id": f"{name}-input",
            }
            replies.append((reply, status))
            harness.finish(reply, status)
        for reply, _status in replies:
            await _wait_until(lambda reply=reply: harness.owner(reply).value["phase"] == "ended")
        assert sum(1 for call in harness.gateway_state.calls if call.get("endTurn")) == 3
        end_contexts = [
            _call_context(call)
            for call in harness.gateway_state.calls
            if call.get("endTurn")
        ]
        assert all(context["session_id"].startswith("computer-session:") for context in end_contexts)

        pending_controls: list[MessageReply] = []
        for action in ("pause", "failure"):
            reply = harness.add_call(action)
            await harness.execute(reply)
            assert harness.owner(reply).value["phase"] == "started"
            harness.control(reply, action)
            pending_controls.append(reply)

        # The barrier is in a separate Session, so it wakes the follower
        # without closing either non-terminal Control Turn.
        barrier_reply = harness.add_call("control-barrier")
        await harness.execute(barrier_reply)
        harness.finish(barrier_reply, "complete")
        await _wait_until(lambda: harness.owner(barrier_reply).value["phase"] == "ended")
        assert all(harness.owner(reply).value["phase"] == "started" for reply in pending_controls)
        assert sum(1 for call in harness.gateway_state.calls if call.get("endTurn")) == 4
        for reply in pending_controls:
            harness.finish(reply, "abandoned")
        for reply in pending_controls:
            await _wait_until(lambda reply=reply: harness.owner(reply).value["phase"] == "ended")
        assert sum(1 for call in harness.gateway_state.calls if call.get("endTurn")) == 6

        open_reply = harness.add_call("open")
        await harness.execute(open_reply)
        assert harness.owner(open_reply).value["phase"] == "started"
        assert sum(1 for call in harness.gateway_state.calls if call.get("endTurn")) == 6

        # A later complete Turn is the follower barrier: it proves the open
        # Turn was scanned and skipped, instead of merely yielding the loop.
        barrier_reply = harness.add_call("barrier")
        await harness.execute(barrier_reply)
        harness.finish(barrier_reply, "complete")
        await _wait_until(lambda: harness.owner(barrier_reply).value["phase"] == "ended")
        assert harness.owner(open_reply).value["phase"] == "started"
        assert sum(1 for call in harness.gateway_state.calls if call.get("endTurn")) == 7

        harness.finish(open_reply, "abandoned")
        await _wait_until(lambda: harness.owner(open_reply).value["phase"] == "ended")
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_computer_cancel_releases_driver_and_follower_ends_unknown_call(tmp_path: Path) -> None:
    """取消先取得 driver released，再把已 started 的效果持久为 unknown 并可收尾。"""
    harness = await _computer_harness(tmp_path)
    try:
        reply = harness.add_call("cancel", code="hold")
        running = asyncio.create_task(harness.execute(reply))
        await _wait_until(lambda: any(call.get("code") == "hold" for call in harness.gateway_state.calls))
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        owner = harness.owner(reply)
        assert owner is not None and owner.value["phase"] == "started"
        assert any(call.get("code") == "hold" for call in harness.gateway_state.calls)
        event_names = [name for name, _body in harness.gateway_state.events]
        assert "cancel" in event_names
        assert event_names.index("cancel") < event_names.index("released")
        cancel_body = next(body for name, body in harness.gateway_state.events if name == "cancel")
        held_body = next(body for name, body in harness.gateway_state.events if name == "run")
        assert cancel_body["call_id"] == _call_context(held_body)["call_id"]
        harness.finish(reply, "complete")
        await _wait_until(lambda: harness.owner(reply).value["phase"] == "ended")
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_computer_failure_retries_started_owner_after_restart_and_source_change(tmp_path: Path) -> None:
    """启动失败保留 owner；重启后的旧 binding 仍命中旧 driver。"""
    state = _ComputerGatewayState()
    harness = await _computer_harness(
        tmp_path, gateway_state=state, gateway_label="old"
    )
    restarted: _ComputerHarness | None = None
    new_gateway: ThreadingHTTPServer | None = None
    try:
        state.fail_runs = 1
        reply = harness.add_call("failure", code="fail")
        with pytest.raises(RuntimeError, match="503"):
            await harness.execute(reply)
        assert harness.owner(reply).value["phase"] == "started"
        state.fail_ends = 1
        harness.finish(reply, "complete")
        await _wait_until(lambda: sum(1 for call in state.calls if call.get("endTurn")) == 1)
        assert harness.owner(reply).value["phase"] == "started"
        old_revision = harness.manager.current_snapshot.generations["computer"].source_revision
        old_mcp_command = harness.manager.current_snapshot.mcp_server_registry["computer"].descriptor.command
        old_end_generation = next(
            _call_context(call)["generation_id"]
            for call in state.calls
            if call.get("endTurn")
        )
        await _wait_until(lambda: harness.manager.current_snapshot.lease_count == 0)
        incidents = harness.composition_root.receipt().incidents
        assert any(incident.kind == "computer-end-turn" for incident in incidents)

        computer_source = harness.root / "computer" / "plugin.py"
        manifest_source = harness.root / "computer" / "akashic.plugin.toml"
        old_digest = "9bd4f6e215b4848e91f0dbfea75a7b227faeba96268c422d62e81a9b64d5ac92"
        old_image = "ghcr.io/kachofugetsu09/akashic-computer@sha256:" + old_digest
        new_image = old_image[:-1] + "a"
        new_digest = new_image.rsplit(":", 1)[1]
        source_text = computer_source.read_text()
        manifest_text = manifest_source.read_text()
        assert source_text.count(old_digest) == 1
        assert manifest_text.count(old_image) == 1
        changed_source_text = (
            source_text.replace(old_digest, new_digest)
            .replace('command=("mcp_server.py",),', 'command=("mcp_server.py", "--new-target"),')
            + "\n# source revision after binding capture\n"
        )
        changed_manifest_text = manifest_text.replace(old_image, new_image).replace(
            'command = ["mcp_server.py"]',
            'command = ["mcp_server.py", "--new-target"]',
        )
        # Boot from the durable old stable source; the changed source is made
        # visible after restart and then published through reconcile/promote.
        computer_source.write_text(source_text)
        manifest_source.write_text(manifest_text)
        # A real restart releases the old manager's process owners.  The new
        # manager then restores its archived stable workload before publishing
        # the changed source as a fresh candidate.
        new_gateway = _start_test_gateway(state, "new")
        harness.controller.ports_by_image[new_image] = new_gateway.server_port
        await harness.manager.terminate_all()
        restarted = await _computer_harness(
            tmp_path,
            log=harness.log,
            gateway=new_gateway,
            gateway_state=state,
            gateway_label="new",
            controller=harness.controller,
        )
        computer_source.write_text(changed_source_text)
        manifest_source.write_text(changed_manifest_text)
        try:
            await _wait_until(lambda: harness.log.owner("plugin:computer").read(
                "computer-use:message:[\"failure-output\",0]"
            ).value["phase"] == "ended")
            assert sum(1 for call in state.calls if call.get("endTurn")) == 2
            assert [
                label
                for label, call in state.routed_calls
                if call.get("endTurn")
            ] == ["old", "old"]
            assert restarted.manager.current_snapshot is not None
            assert restarted.manager.current_snapshot.generations["computer"].source_revision == old_revision

            result = await restarted.manager.reconcile_changed()
            computer_result = next(item for item in result if item["plugin_id"] == "computer")
            assert computer_result["publication_state"] == "committed"
            assert restarted.manager.current_snapshot.generations["computer"].source_revision != old_revision
            new_mcp_command = restarted.manager.current_snapshot.mcp_server_registry["computer"].descriptor.command
            assert old_mcp_command != new_mcp_command
            assert new_mcp_command[-1] == "--new-target"
            current_root = restarted.manager.current_snapshot.composition_root
            assert current_root is not None
            restarted.composition_root = current_root
            restarted.tools = current_root.context.require(TOOLS)
            restarted.bindings = current_root.context.require(BINDINGS)
            await restarted.bind_computer()
            end_generations = [
                _call_context(call)["generation_id"]
                for call in state.calls
                if call.get("endTurn")
            ]
            assert len(end_generations) == 2
            assert end_generations[0] == old_end_generation
            assert end_generations[1] != old_end_generation

            # A fresh call proves the restarted current binding uses the new
            # gateway while the archive retry above stayed on the old one.
            fresh_reply = restarted.add_call("fresh")
            await restarted.execute(fresh_reply)
            fresh_generation = next(
                _call_context(call)["generation_id"]
                for call in state.calls
                if _call_context(call).get("session_id") == "computer-session:fresh"
            )
            assert fresh_generation != end_generations[-1]
            fresh_routes = [
                label
                for label, call in state.routed_calls
                if _call_context(call).get("session_id") == "computer-session:fresh"
            ]
            assert fresh_routes == ["new"]
            restarted.finish(fresh_reply, "complete")
            await _wait_until(lambda: restarted.owner(fresh_reply).value["phase"] == "ended")
        finally:
            await restarted.manager.terminate_all()
    finally:
        if restarted is not None:
            await restarted.manager.terminate_all()
        else:
            await harness.manager.terminate_all()
        harness.log.close()
        harness.gateway.shutdown()
        harness.gateway.server_close()
        if new_gateway is not None:
            new_gateway.shutdown()
            new_gateway.server_close()
