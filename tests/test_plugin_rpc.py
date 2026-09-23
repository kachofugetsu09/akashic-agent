"""外部 RPC 的方法、参数和生命周期由实际 provider 拥有。"""
import ast
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent.config_models import Config
from agent.control.protocol.router import ConnectionRouter
from agent.control.service import ControlService
from agent.plugin_composition import FiberState
from agent.plugin_composition.rpc import RpcMethod, StrictModel, rpc_method_key
from agent.plugins.install import install_git_plugin
from bootstrap.app_server import build_control_service
from bootstrap.tools import build_core_runtime
from core.net.http import SharedHttpResources
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from tests.test_plugin_install import _commit


class FirstParams(StrictModel):
    first: str


class SecondParams(StrictModel):
    second: int


RPC_LOADING_STARTED: asyncio.Event | None = None
RPC_LOADING_RELEASE: asyncio.Event | None = None


def _rpc_error(frame: dict[str, object]) -> dict[str, object]:
    error = frame["error"]
    assert isinstance(error, dict)
    return error


def _write_plugin_source(path: Path, source: str) -> None:
    """Parse and compile a generated plugin before writing it to the fixture workspace."""
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    path.write_text(source, encoding="utf-8")


@pytest.mark.asyncio
async def test_connection_uses_current_plugin_schema_and_keeps_inflight_method():
    entered, release = asyncio.Event(), asyncio.Event()
    active = []
    frames = []

    async def first(params):
        entered.set()
        await release.wait()
        assert active == ["first"]
        return params.first

    async def second(params):
        return params.second

    current = RpcMethod(FirstParams, first)

    @asynccontextmanager
    async def resolve(name: str) -> AsyncIterator[RpcMethod | None]:
        selected = current if name == "example/inspect" else None
        marker = "first" if selected is not None and selected.params is FirstParams else "second"
        active.append(marker)
        try:
            yield selected
        finally:
            active.remove(marker)

    async def send(frame):
        frames.append(frame)

    service = cast(ControlService, SimpleNamespace(
        methods={}, resolve_method=resolve, initialize=lambda _: {},
    ))
    router = ConnectionRouter(service, send)

    async def request(identity, method, params):
        await router.handle_line(json.dumps({"jsonrpc": "2.0", "id": identity,
            "method": method, "params": params}).encode())

    await request(1, "initialize", {"protocolVersion": "2.0",
        "clientInfo": {"name": "test", "version": "1"}})
    await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized"}')
    pending = asyncio.create_task(request(2, "example/inspect", {"first": "old"}))
    try:
        await entered.wait()
        current = RpcMethod(SecondParams, second)
        await request(3, "example/inspect", {"second": 42})
        await request(4, "example/inspect", {"first": "stale schema"})
        release.set()
        await pending
        current = None
        await request(5, "example/inspect", {})
    finally:
        release.set()
        await pending
        await router.close()
    responses = {frame["id"]: frame for frame in frames}
    assert responses[2]["result"] == "old"
    assert responses[3]["result"] == 42
    assert _rpc_error(responses[4])["code"] == -32602
    assert _rpc_error(responses[5])["code"] == -32601
    assert active == []


def test_plugin_method_cannot_replace_host_management():
    with pytest.raises(ValueError, match="已经存在"):
        rpc_method_key("plugin/install")


def _write_live_rpc_plugin(source: Path, *, version: str) -> None:
    """Write a real provider Fiber whose RPC schema changes across reloads."""
    source.mkdir(parents=True, exist_ok=True)
    if version == "old":
        params = "class InspectParams(BaseModel):\n    value: str\n"
        result_value = "params.value"
    else:
        params = "class InspectParams(BaseModel):\n    amount: int\n"
        result_value = "params.amount"
    _write_plugin_source(source.joinpath("plugin.py"),
        "import asyncio\n"
        "from pydantic import BaseModel\n"
        "from agent.control.protocol.method import RpcMethod\n"
        "from agent.plugin_composition.rpc import rpc_method_key\n"
        "api_version = 3\n"
        "name = 'target'\n"
        "version = '1.0.0'\n"
        f"VERSION = {version!r}\n"
        "ENTERED = None\n"
        "RELEASE = None\n"
        "CANCEL_ENTERED = None\n"
        "RAISE = False\n"
        "CANCEL = False\n"
        "CLEANUP = None\n"
        "CLOSES = 0\n"
        f"{params}\n"
        "async def inspect(params):\n"
        "    try:\n"
        "        if RAISE:\n"
        "            raise RuntimeError('handler failed')\n"
        "        if CANCEL:\n"
        "            if CANCEL_ENTERED is not None:\n"
        "                CANCEL_ENTERED.set()\n"
        "            await asyncio.Event().wait()\n"
        "        if ENTERED is not None:\n"
        "            ENTERED.set()\n"
        "        if RELEASE is not None:\n"
        "            await RELEASE.wait()\n"
        f"        return {{'version': VERSION, 'value': {result_value}}}\n"
        "    finally:\n"
        "        if CLEANUP is not None:\n"
        "            CLEANUP.append('handler')\n"
        "async def apply(ctx):\n"
        "    async def close():\n"
        "        global CLOSES\n"
        "        CLOSES += 1\n"
        "        if CLEANUP is not None:\n"
        "            CLEANUP.append('effect')\n"
        "    await ctx.effect(lambda: close)\n"
        "    await ctx.provide(\n"
        "        rpc_method_key('example/inspect'),\n"
        "        RpcMethod(InspectParams, inspect),\n"
        "    )\n",
    )


def _write_loading_rpc_plugin(source: Path) -> None:
    """Write a provider that blocks its real RUNTIME_STARTED listener."""
    source.mkdir(parents=True, exist_ok=True)
    _write_plugin_source(source.joinpath("plugin.py"),
        "from pydantic import BaseModel\n"
        "from agent.control.protocol.method import RpcMethod\n"
        "from agent.plugin_composition import RUNTIME_STARTED\n"
        "from agent.plugin_composition.rpc import rpc_method_key\n"
        "from tests.test_plugin_rpc import RPC_LOADING_RELEASE, RPC_LOADING_STARTED\n"
        "api_version = 3\n"
        "name = 'target'\n"
        "version = '1.0.0'\n"
        "class InspectParams(BaseModel):\n"
        "    value: str\n"
        "async def inspect(params):\n"
        "    return {'value': params.value}\n"
        "async def started(_event):\n"
        "    if RPC_LOADING_STARTED is not None:\n"
        "        RPC_LOADING_STARTED.set()\n"
        "    if RPC_LOADING_RELEASE is not None:\n"
        "        await RPC_LOADING_RELEASE.wait()\n"
        "async def apply(ctx):\n"
        "    await ctx.on(RUNTIME_STARTED, started)\n"
        "    await ctx.provide(\n"
        "        rpc_method_key('example/inspect'),\n"
        "        RpcMethod(InspectParams, inspect),\n"
        "    )\n",
    )


def _write_peer_rpc_plugin(source: Path) -> None:
    """Write an unrelated provider whose scope must remain independently usable."""
    source.mkdir(parents=True, exist_ok=True)
    _write_plugin_source(source.joinpath("plugin.py"),
        "from pydantic import BaseModel\n"
        "from agent.control.protocol.method import RpcMethod\n"
        "from agent.plugin_composition.rpc import rpc_method_key\n"
        "api_version = 3\n"
        "name = 'peer'\n"
        "version = '1.0.0'\n"
        "STARTS = 0\n"
        "CLOSES = 0\n"
        "class PeerParams(BaseModel):\n"
        "    value: str\n"
        "async def peer(params):\n"
        "    return {'peer': params.value}\n"
        "async def apply(ctx):\n"
        "    global STARTS\n"
        "    STARTS += 1\n"
        "    async def close():\n"
        "        global CLOSES\n"
        "        CLOSES += 1\n"
        "    await ctx.effect(lambda: close)\n"
        "    await ctx.provide(\n"
        "        rpc_method_key('example/peer'), RpcMethod(PeerParams, peer),\n"
        "    )\n",
    )


def _write_hard_rpc_consumer(source: Path) -> None:
    """Write a real dependent Fiber that signals before provider drain."""
    source.mkdir(parents=True, exist_ok=True)
    _write_plugin_source(source.joinpath("plugin.py"),
        "from agent.plugin_composition.rpc import rpc_method_key\n"
        "api_version = 3\n"
        "name = 'hard'\n"
        "version = '1.0.0'\n"
        "inject = (rpc_method_key('example/inspect'),)\n"
        "HARD_CLOSED = None\n"
        "STARTS = 0\n"
        "CLOSES = 0\n"
        "async def apply(ctx):\n"
        "    global STARTS\n"
        "    STARTS += 1\n"
        "    ctx.require(inject[0])\n"
        "    async def close():\n"
        "        global CLOSES\n"
        "        CLOSES += 1\n"
        "        if HARD_CLOSED is not None:\n"
        "            HARD_CLOSED.set()\n"
        "    await ctx.effect(lambda: close)\n",
    )


@pytest.mark.asyncio
async def test_live_rpc_resolution_holds_exact_provider_scope_during_local_replace(
    tmp_path: Path, monkeypatch,
) -> None:
    """A real live Root drains one RPC owner while a peer keeps serving."""
    workspace = tmp_path / "workspace"
    plugin_home = tmp_path / "plugin-home"
    target_source = tmp_path / "target-source"
    peer_root = tmp_path / "plugins"
    _write_live_rpc_plugin(target_source, version="old")
    _commit(target_source)
    _write_peer_rpc_plugin(peer_root / "peer")
    _write_hard_rpc_consumer(peer_root / "hard")
    initialize_plugin_workspace(workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))
    install_git_plugin(
        workspace=workspace, source=str(target_source), marketplace="lab",
        plugins_home=plugin_home,
    )

    http = SharedHttpResources()
    core = build_core_runtime(
        Config(), workspace, http, plugin_dirs=[peer_root],
    )
    service = None
    router = None
    old_request = None
    operation = None
    manager = None
    release = asyncio.Event()
    try:
        await core.start()
        service = build_control_service(core)
        frames: list[dict[str, object]] = []

        async def send(frame: dict[str, object]) -> None:
            frames.append(frame)

        router = ConnectionRouter(service, send)

        async def request(identity: int, method: str, params: dict[str, object]):
            await router.handle_line(json.dumps({
                "jsonrpc": "2.0", "id": identity,
                "method": method, "params": params,
            }).encode())
            return next(frame for frame in reversed(frames) if frame.get("id") == identity)

        await request(1, "initialize", {
            "protocolVersion": "2.0", "clientInfo": {"name": "test", "version": "1"},
        })
        await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized"}')

        manager = core.plugin_manager
        root = manager.live_root
        target = manager.generation("target@lab")
        peer = manager.generation("peer")
        hard = manager.generation("hard")
        assert root is not None and target is not None and target.fiber is not None
        assert peer is not None and peer.fiber is not None
        assert hard is not None and hard.fiber is not None
        peer_fiber = peer.fiber
        peer_context = peer.fiber.context
        peer_token = peer_context.fiber.activation_token
        peer_module = peer.instance.module
        peer_counts = (peer_module.STARTS, peer_module.CLOSES)
        hard_module = hard.instance.module
        hard_module.HARD_CLOSED = asyncio.Event()
        old_module = target.instance.module
        old_module.ENTERED = asyncio.Event()
        old_module.RELEASE = release
        old_module.CLEANUP = []

        old_request = asyncio.create_task(
            request(2, "example/inspect", {"value": "old"}),
        )
        await old_module.ENTERED.wait()

        invalid = await request(3, "example/inspect", {"amount": 42})
        assert _rpc_error(invalid)["code"] == -32602
        assert len(target.fiber._in_flight_calls) == 1
        old_module.RAISE = True
        failed = await request(4, "example/inspect", {"value": "error"})
        assert _rpc_error(failed)["code"] == -32603
        assert len(target.fiber._in_flight_calls) == 1
        old_module.RAISE = False
        old_module.CANCEL = True
        old_module.CANCEL_ENTERED = asyncio.Event()
        cancelled = asyncio.create_task(request(5, "example/inspect", {"value": "cancel"}))
        await old_module.CANCEL_ENTERED.wait()
        cancelled.cancel()
        cancelled_result = await asyncio.gather(cancelled, return_exceptions=True)
        assert isinstance(cancelled_result[0], asyncio.CancelledError)
        assert len(target.fiber._in_flight_calls) == 1
        old_module.CANCEL = False
        old_module.CLEANUP.clear()

        async def reject_snapshot_entry(*args, **kwargs):
            raise AssertionError("局部换代不得取得 snapshot lease")

        def reject_snapshot_compile(
            generations, *, snapshot_revision="", composition_root=None,
        ):
            raise AssertionError("局部换代不得编译 snapshot")

        monkeypatch.setattr(manager.snapshot_store, "acquire", reject_snapshot_entry)
        monkeypatch.setattr(manager, "_replace_formal_root", reject_snapshot_entry)
        monkeypatch.setattr(manager._snapshot_compiler, "compile", reject_snapshot_compile)

        _write_live_rpc_plugin(target_source, version="new")
        _commit(target_source)
        accepted = await manager.install(
            source=str(target_source), marketplace="lab", ref_name="", sparse_paths=[],
            update_id="target-rpc-v2",
        )
        assert accepted.state == "accepted"
        operation = manager._operation
        await hard_module.HARD_CLOSED.wait()
        assert operation is not None and target.fiber.state is FiberState.UNLOADING
        assert not old_request.done()
        assert len(target.fiber._in_flight_calls) == 1

        unavailable = await request(6, "example/inspect", {"value": "blocked"})
        assert _rpc_error(unavailable)["code"] == -32601
        peer_result = await request(7, "example/peer", {"value": "still-live"})
        assert peer_result["result"] == {"peer": "still-live"}
        assert manager.generation("peer") is peer
        assert peer.fiber is peer_fiber
        assert peer.fiber.state is FiberState.ACTIVE
        assert peer_context.fiber.activation_token is peer_token
        assert (peer_module.STARTS, peer_module.CLOSES) == peer_counts

        release.set()
        old_result = await old_request
        assert old_result["result"] == {"version": "old", "value": "old"}
        operation_result = await asyncio.gather(operation.task, return_exceptions=True)
        assert len(operation_result) == 1 and operation_result[0].state == "active"
        assert not target.fiber._in_flight_calls
        assert old_module.CLEANUP == ["handler", "effect"]
        assert old_module.CLOSES == 1

        invalid = await request(8, "example/inspect", {"value": "old schema"})
        assert _rpc_error(invalid)["code"] == -32602
        current = await request(9, "example/inspect", {"amount": 42})
        assert current["result"] == {"version": "new", "value": 42}
        assert manager.live_root is root
        assert manager.generation("peer") is peer
        assert peer.fiber is peer_fiber
        assert peer.fiber.state is FiberState.ACTIVE
        assert peer_context.fiber.activation_token is peer_token
    finally:
        release.set()
        if old_request is not None:
            await asyncio.gather(old_request, return_exceptions=True)
        if operation is None and manager is not None:
            operation = manager._operation
        if operation is not None:
            await asyncio.gather(operation.task, return_exceptions=True)
        if router is not None:
            await router.close()
        if service is not None:
            await service.shutdown()
        await core.bus.aclose()
        await core.stop()
        await http.aclose()


@pytest.mark.asyncio
async def test_live_rpc_rejects_loading_and_missing_methods_on_one_connection(
    tmp_path: Path, monkeypatch,
) -> None:
    """The live Root hides a registered RPC until its RUNTIME_STARTED owner is active."""
    global RPC_LOADING_STARTED, RPC_LOADING_RELEASE
    RPC_LOADING_STARTED = asyncio.Event()
    RPC_LOADING_RELEASE = asyncio.Event()
    workspace = tmp_path / "workspace"
    plugin_home = tmp_path / "plugin-home"
    target_source = tmp_path / "target-source"
    peer_root = tmp_path / "plugins"
    peer_root.mkdir(parents=True)
    _write_loading_rpc_plugin(target_source)
    _commit(target_source)
    initialize_plugin_workspace(workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))
    install_git_plugin(
        workspace=workspace, source=str(target_source), marketplace="lab",
        plugins_home=plugin_home,
    )

    http = SharedHttpResources()
    core = build_core_runtime(Config(), workspace, http, plugin_dirs=[peer_root])
    service = build_control_service(core)
    router = None
    startup = None
    frames: list[dict[str, object]] = []
    try:
        async def send(frame: dict[str, object]) -> None:
            frames.append(frame)

        router = ConnectionRouter(service, send)

        async def request(identity: int, method: str, params: dict[str, object]):
            await router.handle_line(json.dumps({
                "jsonrpc": "2.0", "id": identity,
                "method": method, "params": params,
            }).encode())
            return next(frame for frame in reversed(frames) if frame.get("id") == identity)

        await request(1, "initialize", {
            "protocolVersion": "2.0", "clientInfo": {"name": "test", "version": "1"},
        })
        await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized"}')
        startup = asyncio.create_task(core.start())
        await RPC_LOADING_STARTED.wait()

        manager = core.plugin_manager
        root = manager.live_root
        assert root is not None
        loading_fibers = [
            fiber for fiber in root.root_fiber.children if fiber.name == "target@lab"
        ]
        assert len(loading_fibers) == 1
        assert loading_fibers[0].state is FiberState.LOADING
        assert root.service_value(rpc_method_key("example/inspect")) is None

        loading = await request(2, "example/inspect", {"value": "loading"})
        assert _rpc_error(loading)["code"] == -32601
        missing = await request(3, "example/missing", {})
        assert _rpc_error(missing)["code"] == -32601

        RPC_LOADING_RELEASE.set()
        await startup
        target = manager.generation("target@lab")
        assert target is not None and target.fiber is not None
        assert target.fiber.state is FiberState.ACTIVE
        current = await request(4, "example/inspect", {"value": "ready"})
        assert current["result"] == {"value": "ready"}
    finally:
        RPC_LOADING_RELEASE.set()
        if startup is not None:
            await asyncio.gather(startup, return_exceptions=True)
        if router is not None:
            await router.close()
        await service.shutdown()
        await core.bus.aclose()
        await core.stop()
        await http.aclose()
        RPC_LOADING_STARTED = None
        RPC_LOADING_RELEASE = None
