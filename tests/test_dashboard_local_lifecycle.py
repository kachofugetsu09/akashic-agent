"""行为级测试用例：真实 Dashboard 制品、Root admission 和 ASGI 生命周期。"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from starlette.websockets import WebSocket

from agent.plugin_composition import (
    CompositionError,
    CompositionRoot,
    Context,
    FiberState,
    PluginRuntime,
    ServiceKey,
)
from agent.plugin_composition.host import HOST_INFO, HostInfo
from agent.plugin_composition.ui import DASHBOARD_ROUTES, UI
from agent.plugins.dashboard_host import LiveDashboardMiddleware
from plugins.ui import plugin as ui_plugin


def _write_dashboard_module(
    code: Path,
    *,
    websocket: bool = False,
    handler_error: bool = False,
    normal_end: bool = False,
    websocket_error: bool = False,
    cleanup_error: bool = False,
    request_dependency: bool = False,
) -> tuple[ModuleType, object]:
    """Create a real plugin-local module and loader with the same code origin."""

    if websocket:
        route = """
    @app.websocket('/api/dashboard/live')
    async def socket(websocket: WebSocket) -> None:
        await websocket.accept()
        ws_started.set()
"""
        if normal_end:
            route += """
        return
"""
        else:
            route += """
        try:
"""
            if websocket_error:
                route += """
            raise RuntimeError('dashboard app failure')
"""
            route += """
            await websocket.receive_text()
        finally:
            ws_cleanup_started.set()
"""
            if cleanup_error:
                route += """
            raise RuntimeError('dashboard cleanup failure')
"""
            route += """
            await ws_cleanup_release.wait()
            ws_cleanup_finished.set()
"""
    elif handler_error:
        route = """
    @app.get('/api/dashboard/live')
    async def http() -> dict[str, bool]:
        raise CompositionError('OWNER_UNAVAILABLE', 'handler failure')
"""
    else:
        route = """
    @app.get('/api/dashboard/live')
    async def http() -> dict[str, bool]:
        http_started.set()
"""
        if request_dependency:
            route += """
        await http_dependency_release.wait()
        observed_dependency.append(_context.require(DEPENDENCY))
        observed_dependency_state.append(_context.fiber.state.value)
        http_dependency_read.set()
"""
        route += """
        try:
            await http_release.wait()
        finally:
            http_finished.set()
        return {'ok': True}
"""
    source = f"""
import asyncio
from fastapi import FastAPI
from starlette.websockets import WebSocket
from agent.plugin_composition import CompositionError

http_started = asyncio.Event()
http_finished = asyncio.Event()
http_release = asyncio.Event()
ws_started = asyncio.Event()
ws_cleanup_started = asyncio.Event()
ws_cleanup_finished = asyncio.Event()
ws_cleanup_release = asyncio.Event()
observed_dependency = []
observed_dependency_state = []
http_dependency_release = asyncio.Event()
http_dependency_read = asyncio.Event()
DEPENDENCY = None

def register(app: FastAPI, _context: object) -> None:
{route}

def load_dashboard() -> object:
    return _MODULE
"""
    module_path = code / "dashboard.py"
    module_path.write_text(source, encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    compiled = compile(tree, str(module_path), "exec", dont_inherit=True)
    module = ModuleType("fixture_live_dashboard")
    module.__file__ = str(module_path)
    module.__package__ = ""
    exec(compiled, module.__dict__)
    module._MODULE = module
    return module, module.load_dashboard


async def _live_dashboard(
    tmp_path: Path,
    *,
    websocket: bool = False,
    handler_error: bool = False,
    normal_end: bool = False,
    websocket_error: bool = False,
    cleanup_error: bool = False,
    request_dependency: ServiceKey[object] | None = None,
) -> tuple[CompositionRoot, LiveDashboardMiddleware, ModuleType]:
    """Build one real host/UI/contributor graph with fixed dashboard bytes."""

    root = CompositionRoot("dashboard-local")
    host = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @host.get("/host")
    async def host_route() -> dict[str, bool]:
        return {"host": True}

    await root.context.provide(HOST_INFO, HostInfo(boot_id="test", validation=False))
    await root.context.provide(DASHBOARD_ROUTES, tuple(host.routes))
    await root.mount(ui_plugin.apply, name="ui")

    code = tmp_path / "plugin"
    code.mkdir()
    (code / "dashboard.js").write_text(
        "export function activate() { return () => {}; }\n",
        encoding="utf-8",
    )
    module, loader = _write_dashboard_module(
        code,
        websocket=websocket,
        handler_error=handler_error,
        normal_end=normal_end,
        websocket_error=websocket_error,
        cleanup_error=cleanup_error,
        request_dependency=request_dependency is not None,
    )
    if request_dependency is not None:
        module.DEPENDENCY = request_dependency

        async def dependency_provider(ctx: Context) -> None:
            await ctx.provide(request_dependency, "frozen-dependency")

        await root.mount(dependency_provider, name="dashboard-dependency")

    async def contributor(ctx: Context) -> None:
        await ctx.require(UI).register(ctx, web="dashboard.js", dashboard=loader)

    dependencies = (UI,) if request_dependency is None else (UI, request_dependency)

    await root.mount(
        contributor,
        name="dashboard",
        inject=dependencies,
        runtime=PluginRuntime(
            plugin_id="dashboard",
            generation_id="dashboard-generation",
            plugin_dir=code,
            data_dir=code / "data",
            workspace=code,
            config={},
        ),
    )
    manager = SimpleNamespace(live_root=root)
    return root, LiveDashboardMiddleware(host, manager), module


async def _asgi_call(
    app: object,
    scope: dict[str, object],
    messages: list[dict[str, object]],
    *,
    wire_release: asyncio.Event | None = None,
) -> list[dict[str, object]]:
    """Run one deterministic ASGI call with an asynchronous send boundary."""

    pending = list(messages)
    if scope.get("type") == "websocket":
        pending.insert(0, {"type": "websocket.connect"})

    async def receive() -> dict[str, object]:
        if pending:
            return pending.pop(0)
        if scope.get("type") == "websocket":
            await asyncio.Future()
        return {"type": "http.request", "body": b""}

    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        if message.get("type") == "websocket.close" and wire_release is not None:
            await wire_release.wait()
        sent.append(message)

    await app(scope, receive, send)  # type: ignore[operator]
    return sent


def _http_scope(headers: list[tuple[bytes, bytes]]) -> dict[str, object]:
    return {
        "type": "http",
        "method": "GET",
        "path": "/api/dashboard/live",
        "raw_path": b"/api/dashboard/live",
        "query_string": b"",
        "headers": headers,
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1),
        "http_version": "1.1",
    }


def _websocket_scope(query: str) -> dict[str, object]:
    return {
        "type": "websocket",
        "path": "/api/dashboard/live",
        "query_string": query.encode("ascii"),
        "headers": [(b"origin", b"http://test"), (b"host", b"test")],
        "scheme": "ws",
        "server": ("test", 80),
        "client": ("test", 1),
    }


def _identity(root: CompositionRoot, module: ModuleType) -> tuple[str, str]:
    catalog = root.context.require(UI).catalog()
    descriptor = catalog.modules[0]
    assert descriptor.plugin_id == "dashboard"
    assert descriptor.generation_id == "dashboard-generation"
    assert module.__file__ is not None
    return root.generation_id, catalog.identity


def _dashboard_query(root: CompositionRoot, module: ModuleType) -> str:
    """Build the exact live Root/catalog/module WebSocket identity."""
    snapshot_id, catalog_id = _identity(root, module)
    return urlencode({
        "__akashic_web_snapshot": snapshot_id,
        "__akashic_web_catalog": catalog_id,
        "__akashic_web_module": "dashboard",
        "__akashic_web_generation": "dashboard-generation",
    })


def _websocket_receive(*messages: dict[str, object]):
    """Provide the real connect handshake before optional WebSocket messages."""

    pending = [{"type": "websocket.connect"}, *messages]

    async def receive() -> dict[str, object]:
        if pending:
            return pending.pop(0)
        await asyncio.Future()
        return {"type": "websocket.disconnect"}

    return receive


async def _async_send(_message: dict[str, object]) -> None:
    return None


@pytest.mark.asyncio
async def test_old_http_scope_drains_and_new_fence_is_rejected(tmp_path: Path) -> None:
    dependency = ServiceKey[object]("test.dashboard.request")
    root, middleware, module = await _live_dashboard(
        tmp_path, request_dependency=dependency,
    )
    old_task: asyncio.Task[object] | None = None
    dispose_task: asyncio.Task[object] | None = None
    peer_fiber = None
    try:
        snapshot_id, catalog_id = _identity(root, module)
        headers = [
            (b"x-akashic-web-snapshot", snapshot_id.encode()),
            (b"x-akashic-web-catalog", catalog_id.encode()),
            (b"x-akashic-web-module", b"dashboard"),
            (b"x-akashic-web-generation", b"dashboard-generation"),
        ]
        root_owner = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "dashboard"
        )
        async def peer(ctx: Context) -> None:
            await ctx.effect(lambda: None)

        peer_fiber = await root.mount(peer, name="dashboard-peer")
        peer_before = (
            peer_fiber.context,
            peer_fiber._activation_token,  # pyright: ignore[reportPrivateUsage]
            peer_fiber.state,
            tuple(peer_fiber.effects),
            peer_fiber._lifecycle_started,  # pyright: ignore[reportPrivateUsage]
            peer_fiber._stopping_completed,  # pyright: ignore[reportPrivateUsage]
        )
        old_task = asyncio.create_task(
            _asgi_call(
                middleware,
                _http_scope(headers),
                [{"type": "http.request", "body": b"", "more_body": False}],
            ),
            name="old-http-owner",
        )
        await module.http_started.wait()
        dispose_task = asyncio.create_task(root_owner.dispose(), name="dashboard-dispose")
        await root_owner._admission_closed.wait()  # pyright: ignore[reportPrivateUsage]
        assert root_owner.state is FiberState.UNLOADING
        module.http_dependency_release.set()
        await module.http_dependency_read.wait()
        assert module.observed_dependency == ["frozen-dependency"]
        assert module.observed_dependency_state == [FiberState.UNLOADING.value]
        async with peer_fiber.context.runtime_scope():
            assert peer_fiber.state is FiberState.ACTIVE
        peer_after = (
            peer_fiber.context,
            peer_fiber._activation_token,  # pyright: ignore[reportPrivateUsage]
            peer_fiber.state,
            tuple(peer_fiber.effects),
            peer_fiber._lifecycle_started,  # pyright: ignore[reportPrivateUsage]
            peer_fiber._stopping_completed,  # pyright: ignore[reportPrivateUsage]
        )
        assert peer_after == peer_before
        sent = await _asgi_call(
            middleware,
            _http_scope(headers),
            [{"type": "http.request", "body": b"", "more_body": False}],
        )
        assert sent[0]["status"] == 409
        module.http_release.set()
        old_sent = await old_task
        await dispose_task
        assert old_sent[0]["status"] == 200
        assert root_owner.state is FiberState.DISPOSED
        assert module.http_finished.is_set()
    finally:
        module.http_release.set()
        if old_task is not None and not old_task.done():
            old_task.cancel()
            try:
                await old_task
            except BaseException:
                pass
        if dispose_task is not None and not dispose_task.done():
            try:
                await dispose_task
            except BaseException:
                pass
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ["http-stale", "http-forbidden", "websocket-stale"])
async def test_dashboard_rejection_releases_ui_before_backpressured_wire(
    tmp_path: Path,
    rejection: str,
) -> None:
    websocket = rejection.startswith("websocket")
    root, middleware, module = await _live_dashboard(tmp_path, websocket=websocket)
    request_task: asyncio.Task[object] | None = None
    dispose_task: asyncio.Task[object] | None = None
    wire_release = asyncio.Event()
    wire_started = asyncio.Event()
    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        wire_started.set()
        await wire_release.wait()
        sent.append(message)

    try:
        snapshot_id, catalog_id = _identity(root, module)
        if rejection == "http-stale":
            scope = _http_scope([
                (b"x-akashic-web-snapshot", snapshot_id.encode()),
                (b"x-akashic-web-catalog", b"stale-catalog"),
                (b"x-akashic-web-module", b"dashboard"),
                (b"x-akashic-web-generation", b"dashboard-generation"),
            ])
            pending = [{"type": "http.request", "body": b"", "more_body": False}]
        elif rejection == "http-forbidden":
            scope = _http_scope([])
            pending = [{"type": "http.request", "body": b"", "more_body": False}]
        else:
            query = urlencode({
                "__akashic_web_snapshot": snapshot_id,
                "__akashic_web_catalog": "stale-catalog",
                "__akashic_web_module": "dashboard",
                "__akashic_web_generation": "dashboard-generation",
            })
            scope = _websocket_scope(query)
            pending = [{"type": "websocket.connect"}]

        async def receive() -> dict[str, object]:
            if pending:
                return pending.pop(0)
            await asyncio.Future()
            return {"type": "http.disconnect"}

        request_task = asyncio.create_task(
            middleware(scope, receive, send), name=f"dashboard-{rejection}-reject",
        )
        await wire_started.wait()
        ui_fiber = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "ui"
        )
        dispose_task = asyncio.create_task(ui_fiber.dispose(), name="ui-rejection-dispose")
        await dispose_task
        assert ui_fiber.state is FiberState.DISPOSED
        assert not ui_fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
        assert not module.http_started.is_set()
        assert not module.ws_started.is_set()
        wire_release.set()
        await request_task
        if websocket:
            assert sent[0]["code"] == 4409
        else:
            assert sent[0]["status"] == (403 if rejection == "http-forbidden" else 409)
    finally:
        wire_release.set()
        if request_task is not None and not request_task.done():
            request_task.cancel()
            try:
                await request_task
            except BaseException:
                pass
        if dispose_task is not None and not dispose_task.done():
            try:
                await dispose_task
            except BaseException:
                pass
        await root.dispose()


@pytest.mark.asyncio
async def test_same_context_reregister_gets_new_registration_fence(tmp_path: Path) -> None:
    root = CompositionRoot("dashboard-reregister")
    host = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @host.get("/host")
    async def host_route() -> dict[str, bool]:
        return {"host": True}

    code = tmp_path / "plugin"
    code.mkdir()
    (code / "dashboard.js").write_text(
        "export function activate() { return () => {}; }\n",
        encoding="utf-8",
    )
    module, loader = _write_dashboard_module(code)
    contexts: list[Context] = []
    current_task: asyncio.Task[object] | None = None

    async def contributor(ctx: Context) -> None:
        contexts.append(ctx)
        await ctx.require(UI).register(ctx, web="dashboard.js", dashboard=loader)

    try:
        await root.context.provide(HOST_INFO, HostInfo(boot_id="test", validation=False))
        await root.context.provide(DASHBOARD_ROUTES, tuple(host.routes))
        await root.mount(ui_plugin.apply, name="ui")
        await root.mount(
            contributor,
            name="dashboard",
            inject=(UI,),
            runtime=PluginRuntime(
                plugin_id="dashboard", generation_id="g1", plugin_dir=code,
                data_dir=code / "data", workspace=code, config={},
            ),
        )
        registry = contexts[0].require(UI)
        middleware = LiveDashboardMiddleware(host, SimpleNamespace(live_root=root))
        registration = registry._entries["dashboard"]  # pyright: ignore[reportPrivateUsage]
        first = registration.effect
        assert first is not None
        first_catalog = registry.catalog()
        first_module = first_catalog.modules[0]
        first_id = first_module.registration_uuid
        first_catalog_id = first_catalog.identity
        await first.aclose()
        second = await registry.register(
            contexts[0], web="dashboard.js", dashboard=loader,
        )
        second_catalog = registry.catalog()
        second_module = second_catalog.modules[0]
        second_id = second_module.registration_uuid
        assert first_id != second_id
        assert first_catalog_id != second_catalog.identity
        assert first_module.asset == second_module.asset
        assert first_module.generation_id == second_module.generation_id == "g1"
        assert root.generation_id == "dashboard-reregister"
        first_headers = [
            (b"x-akashic-web-snapshot", root.generation_id.encode()),
            (b"x-akashic-web-catalog", first_catalog_id.encode()),
            (b"x-akashic-web-module", first_module.plugin_id.encode()),
            (b"x-akashic-web-generation", first_module.generation_id.encode()),
        ]
        stale = await _asgi_call(
            middleware,
            _http_scope(first_headers),
            [{"type": "http.request", "body": b"", "more_body": False}],
        )
        assert stale[0]["status"] == 409
        current_headers = [
            (b"x-akashic-web-snapshot", root.generation_id.encode()),
            (b"x-akashic-web-catalog", second_catalog.identity.encode()),
            (b"x-akashic-web-module", second_module.plugin_id.encode()),
            (b"x-akashic-web-generation", second_module.generation_id.encode()),
        ]
        current_task = asyncio.create_task(
            _asgi_call(
                middleware,
                _http_scope(current_headers),
                [{"type": "http.request", "body": b"", "more_body": False}],
            ),
        )
        await module.http_started.wait()
        module.http_release.set()
        current = await current_task
        assert current[0]["status"] == 200
        await second.aclose()
    finally:
        module.http_release.set()
        if current_task is not None:
            if not current_task.done():
                current_task.cancel()
            try:
                await current_task
            except BaseException:
                pass
        await root.dispose()


@pytest.mark.asyncio
async def test_dashboard_hard_dependency_reactivation_gets_new_context_and_fence(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("dashboard-dependency-reactivation")
    host = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    dependency = ServiceKey[object]("test.dashboard.hard-dependency")
    contexts: list[Context] = []
    seen_dependency: list[str] = []

    @host.get("/host")
    async def host_route() -> dict[str, bool]:
        return {"host": True}

    code = tmp_path / "plugin"
    code.mkdir()
    artifact = b"export function activate() { return () => {}; }\n"
    (code / "dashboard.js").write_bytes(artifact)
    module, loader = _write_dashboard_module(code)

    async def provide_a(ctx: Context) -> None:
        await ctx.provide(dependency, "A")

    async def provide_b(ctx: Context) -> None:
        await ctx.provide(dependency, "B")

    async def contributor(ctx: Context) -> None:
        contexts.append(ctx)
        seen_dependency.append(ctx.require(dependency))
        await ctx.require(UI).register(ctx, web="dashboard.js", dashboard=loader)

    middleware = None
    current_task: asyncio.Task[object] | None = None
    try:
        await root.context.provide(HOST_INFO, HostInfo(boot_id="test", validation=False))
        await root.context.provide(DASHBOARD_ROUTES, tuple(host.routes))
        provider_a = await root.mount(provide_a, name="dependency-a")
        await root.mount(ui_plugin.apply, name="ui")
        contributor = await root.mount(
            contributor,
            name="dashboard",
            inject=(UI, dependency),
            runtime=PluginRuntime(
                plugin_id="dashboard",
                generation_id="dashboard-generation",
                plugin_dir=code,
                data_dir=code / "data",
                workspace=code,
                config={},
            ),
        )
        middleware = LiveDashboardMiddleware(host, SimpleNamespace(live_root=root))
        old_context = contexts[0]
        old_registry = old_context.require(UI)
        old_catalog = old_registry.catalog()
        old_module = old_catalog.modules[0]
        root_token = root.instance_token
        generation_id = root.generation_id

        await provider_a.dispose()
        assert contributor.state is FiberState.PENDING
        provider_b = await root.mount(provide_b, name="dependency-b")
        assert contributor.state is FiberState.ACTIVE
        assert provider_b.state is FiberState.ACTIVE
        assert seen_dependency == ["A", "B"]
        new_context = contexts[1]
        new_registry = new_context.require(UI)
        new_catalog = new_registry.catalog()
        new_module = new_catalog.modules[0]
        assert new_context is not old_context
        assert new_registry is old_registry
        assert old_module.registration_uuid != new_module.registration_uuid
        assert old_catalog.identity != new_catalog.identity
        assert old_module.asset == new_module.asset
        assert (code / "dashboard.js").read_bytes() == artifact
        assert root.instance_token is root_token
        assert root.generation_id == generation_id

        old_headers = [
            (b"x-akashic-web-snapshot", generation_id.encode()),
            (b"x-akashic-web-catalog", old_catalog.identity.encode()),
            (b"x-akashic-web-module", old_module.plugin_id.encode()),
            (b"x-akashic-web-generation", old_module.generation_id.encode()),
        ]
        stale = await _asgi_call(
            middleware,
            _http_scope(old_headers),
            [{"type": "http.request", "body": b"", "more_body": False}],
        )
        assert stale[0]["status"] == 409

        new_headers = [
            (b"x-akashic-web-snapshot", generation_id.encode()),
            (b"x-akashic-web-catalog", new_catalog.identity.encode()),
            (b"x-akashic-web-module", new_module.plugin_id.encode()),
            (b"x-akashic-web-generation", new_module.generation_id.encode()),
        ]
        current_task = asyncio.create_task(
            _asgi_call(
                middleware,
                _http_scope(new_headers),
                [{"type": "http.request", "body": b"", "more_body": False}],
            ),
            name="dashboard-reactivated-http",
        )
        await module.http_started.wait()
        module.http_release.set()
        current = await current_task
        assert current[0]["status"] == 200
    finally:
        module.http_release.set()
        if current_task is not None and not current_task.done():
            current_task.cancel()
            try:
                await current_task
            except BaseException:
                pass
        await root.dispose()


@pytest.mark.asyncio
async def test_websocket_owner_cancel_settles_before_backpressured_wire_close(
    tmp_path: Path,
) -> None:
    root, middleware, module = await _live_dashboard(tmp_path, websocket=True)
    task: asyncio.Task[object] | None = None
    dispose_task: asyncio.Task[object] | None = None
    try:
        snapshot_id, catalog_id = _identity(root, module)
        query = urlencode({
            "__akashic_web_snapshot": snapshot_id,
            "__akashic_web_catalog": catalog_id,
            "__akashic_web_module": "dashboard",
            "__akashic_web_generation": "dashboard-generation",
        })
        wire_release = asyncio.Event()

        pending = [{"type": "websocket.connect"}]

        async def receive() -> dict[str, object]:
            if pending:
                return pending.pop(0)
            await asyncio.Future()
            return {"type": "websocket.disconnect"}

        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            if message.get("type") == "websocket.close":
                await wire_release.wait()
            sent.append(message)

        task = asyncio.create_task(
            middleware(_websocket_scope(query), receive, send), name="dashboard-ws",
        )
        await module.ws_started.wait()
        root_owner = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "dashboard"
        )
        dispose_task = asyncio.create_task(root_owner.dispose(), name="dashboard-ws-dispose")
        await module.ws_cleanup_started.wait()
        assert not wire_release.is_set()
        assert not dispose_task.done(), "app finally 尚未放行时不得等待 wire close"
        module.ws_cleanup_release.set()
        await dispose_task
        assert root_owner.state is FiberState.DISPOSED
        assert not task.done(), "wire backpressure 不能阻塞 app/owner 物理结算"
        wire_release.set()
        await task
        assert any(message.get("code") == 1012 for message in sent)
        assert module.ws_cleanup_finished.is_set()
    finally:
        wire_release.set()
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if dispose_task is not None and not dispose_task.done():
            module.ws_cleanup_release.set()
            await dispose_task
        await root.dispose()


@pytest.mark.asyncio
async def test_websocket_caller_cancel_is_observable_and_cleanup_is_joined(
    tmp_path: Path,
) -> None:
    root, middleware, module = await _live_dashboard(tmp_path, websocket=True)
    task: asyncio.Task[object] | None = None
    try:
        snapshot_id, catalog_id = _identity(root, module)
        query = urlencode({
            "__akashic_web_snapshot": snapshot_id,
            "__akashic_web_catalog": catalog_id,
            "__akashic_web_module": "dashboard",
            "__akashic_web_generation": "dashboard-generation",
        })
        release = asyncio.Event()

        pending = [{"type": "websocket.connect"}]

        async def receive() -> dict[str, object]:
            if pending:
                return pending.pop(0)
            await release.wait()
            return {"type": "websocket.disconnect"}

        async def send(_message: dict[str, object]) -> None:
            return None

        task = asyncio.create_task(
            middleware(_websocket_scope(query), receive, send),
            name="dashboard-caller-cancel",
        )
        await module.ws_started.wait()
        task.cancel()
        await module.ws_cleanup_started.wait()
        task.cancel()
        module.ws_cleanup_release.set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert module.ws_cleanup_started.is_set()
    finally:
        release.set()
        module.ws_cleanup_release.set()
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", [1, 2])
async def test_websocket_create_task_failure_closes_unentered_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: int,
) -> None:
    root, middleware, module = await _live_dashboard(tmp_path, websocket=True)
    query = _dashboard_query(root, module)
    owner = next(
        fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
        if fiber.name == "dashboard"
    )
    calls = 0
    real_create_task = asyncio.create_task
    failed_coroutines = []

    def create_task(coro, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls >= failure_at:
            failed_coroutines.append(coro)
            raise RuntimeError("dashboard task creation failure")
        return real_create_task(coro, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", create_task)
    try:
        with pytest.raises(RuntimeError, match="task creation failure"):
            await middleware(
                _websocket_scope(query),
                _websocket_receive(),
                _async_send,
            )
        assert calls >= failure_at
        assert failed_coroutines
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for coro in failed_coroutines
        )
        assert not module.ws_started.is_set()
        assert not owner._in_flight_calls  # pyright: ignore[reportPrivateUsage]
    finally:
        monkeypatch.undo()
        await root.dispose()


@pytest.mark.asyncio
async def test_websocket_cancel_before_app_first_instruction_closes_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, middleware, module = await _live_dashboard(tmp_path, websocket=True)
    query = _dashboard_query(root, module)
    owner = next(
        fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
        if fiber.name == "dashboard"
    )
    real_create_task = asyncio.create_task
    first = True

    def cancel_first(coro, *args, **kwargs):
        nonlocal first
        task = real_create_task(coro, *args, **kwargs)
        if first:
            first = False
            task.cancel()
        return task

    monkeypatch.setattr(asyncio, "create_task", cancel_first)
    try:
        await middleware(
            _websocket_scope(query),
            _websocket_receive(),
            _async_send,
        )
        assert not module.ws_started.is_set()
        assert not owner._in_flight_calls  # pyright: ignore[reportPrivateUsage]
    finally:
        monkeypatch.undo()
        await root.dispose()


@pytest.mark.asyncio
async def test_websocket_normal_end_cancels_monitor_without_generation_close(
    tmp_path: Path,
) -> None:
    root, middleware, module = await _live_dashboard(
        tmp_path, websocket=True, normal_end=True,
    )
    query = _dashboard_query(root, module)
    try:
        sent = await _asgi_call(
            middleware,
            _websocket_scope(query),
            [{"type": "websocket.receive", "text": "start"}],
        )
        assert module.ws_started.is_set()
        assert not any(message.get("code") == 1012 for message in sent)
    finally:
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["app", "cleanup", "wire", "cleanup+wire"])
async def test_websocket_app_and_cleanup_failures_remain_observable(
    tmp_path: Path,
    failure: str,
) -> None:
    root, middleware, module = await _live_dashboard(
        tmp_path,
        websocket=True,
        websocket_error=failure == "app",
        cleanup_error=failure in {"cleanup", "cleanup+wire"},
    )
    query = _dashboard_query(root, module)
    task: asyncio.Task[object] | None = None
    dispose_task: asyncio.Task[object] | None = None

    async def send(message: dict[str, object]) -> None:
        if failure in {"wire", "cleanup+wire"} and message.get("type") == "websocket.close":
            raise RuntimeError("dashboard wire failure")

    try:
        task = asyncio.create_task(
            middleware(
                _websocket_scope(query),
                _websocket_receive(),
                send,
            ),
            name=f"dashboard-{failure}-failure",
        )
        await module.ws_started.wait()
        if failure in {"cleanup", "cleanup+wire", "wire"}:
            owner = next(
                fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
                if fiber.name == "dashboard"
            )
            dispose_task = asyncio.create_task(owner.dispose(), name="dashboard-failure-dispose")
            await module.ws_cleanup_started.wait()
            module.ws_cleanup_release.set()
        else:
            await module.ws_cleanup_started.wait()
            module.ws_cleanup_release.set()
        if dispose_task is not None:
            await dispose_task
        if failure == "cleanup+wire":
            with pytest.raises(BaseExceptionGroup) as raised:
                await task
            child_messages = [str(error) for error in raised.value.exceptions]
            assert any("dashboard cleanup failure" in message for message in child_messages)
            assert any("dashboard wire failure" in message for message in child_messages)
        else:
            expected = "dashboard wire failure" if failure == "wire" else f"dashboard {failure} failure"
            with pytest.raises(RuntimeError, match=expected):
                await task
    finally:
        module.ws_cleanup_release.set()
        if dispose_task is not None and not dispose_task.done():
            try:
                await dispose_task
            except BaseException:
                pass
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await root.dispose()


@pytest.mark.asyncio
async def test_dashboard_handler_composition_error_is_not_rewritten(tmp_path: Path) -> None:
    root, middleware, _module = await _live_dashboard(tmp_path, handler_error=True)
    try:
        snapshot_id, catalog_id = _identity(root, _module)
        headers = [
            (b"x-akashic-web-snapshot", snapshot_id.encode()),
            (b"x-akashic-web-catalog", catalog_id.encode()),
            (b"x-akashic-web-module", b"dashboard"),
            (b"x-akashic-web-generation", b"dashboard-generation"),
        ]
        with pytest.raises(CompositionError, match="handler failure"):
            await _asgi_call(
                middleware,
                _http_scope(headers),
                [{"type": "http.request", "body": b"", "more_body": False}],
            )
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_core_route_passthrough_without_ui_and_identity_is_stale(tmp_path: Path) -> None:
    root = CompositionRoot("core-only")
    host = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @host.get("/api/dashboard/core")
    async def core_route() -> dict[str, bool]:
        return {"ok": True}

    middleware = LiveDashboardMiddleware(host, SimpleNamespace(live_root=root))
    try:
        sent = await _asgi_call(
            middleware,
            {**_http_scope([]), "path": "/api/dashboard/core", "raw_path": b"/api/dashboard/core"},
            [{"type": "http.request", "body": b"", "more_body": False}],
        )
        assert sent[0]["status"] == 200
        stale = await _asgi_call(
            middleware,
            {**_http_scope([(b"x-akashic-web-snapshot", b"old")]),
             "path": "/api/dashboard/core", "raw_path": b"/api/dashboard/core"},
            [{"type": "http.request", "body": b"", "more_body": False}],
        )
        assert stale[0]["status"] == 409
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_active_ui_core_passthrough_preserves_core_composition_error(
    tmp_path: Path,
) -> None:
    root, middleware, _module = await _live_dashboard(tmp_path)

    async def core_failure() -> dict[str, bool]:
        raise CompositionError("OWNER_UNAVAILABLE", "core route failure")

    middleware._app.add_api_route(  # pyright: ignore[reportPrivateUsage]
        "/api/dashboard/core",
        core_failure,
    )
    try:
        with pytest.raises(CompositionError, match="core route failure"):
            await _asgi_call(
                middleware,
                {
                    **_http_scope([]),
                    "path": "/api/dashboard/core",
                    "raw_path": b"/api/dashboard/core",
                },
                [{"type": "http.request", "body": b"", "more_body": False}],
            )
    finally:
        await root.dispose()
