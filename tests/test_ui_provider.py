"""Web/Dashboard provider tests for the one live local UI graph."""

import ast
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import FastAPI

from agent.plugin_composition import (
    CompositionError,
    CompositionRoot,
    Context,
    FiberState,
    PluginRuntime,
    ServiceKey,
)
from agent.plugin_composition.host import HOST_INFO, HostInfo
from agent.plugin_composition.ui import DASHBOARD_ROUTES, UI, WEB_UI
from agent.plugin_composition.ui import WebModuleDescriptor
from plugins.ui import plugin as ui_plugin
from plugins.ui.dashboard import _plugin_routes, _require_routes_available


class _DashboardModule(ModuleType):
    load_dashboard: Callable[[], ModuleType]


def _module_from_source(code: Path, source: str) -> tuple[ModuleType, Callable[[], ModuleType]]:
    """Compile a dashboard loader with a real plugin-local code origin."""

    code.mkdir(parents=True, exist_ok=True)
    path = code / "dashboard.py"
    path.write_text(source, encoding="utf-8")
    compiled = compile(ast.parse(source, filename=str(path)), str(path), "exec")
    module = _DashboardModule("fixture_dashboard")
    module.__file__ = str(path)
    exec(compiled, module.__dict__)
    module._MODULE = module
    return module, module.load_dashboard


async def _mount_owner(
    root: CompositionRoot,
    code: Path,
    *,
    name: str = "view",
    register: bool = True,
    web: str | None = None,
    dashboard: Callable[[], ModuleType] | None = None,
    requires: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    contract_digests: Mapping[str, str] | None = None,
) -> Context:
    """Mount one real contributor Fiber and return its original Context."""

    contexts: list[Context] = []

    async def apply(ctx: Context) -> None:
        contexts.append(ctx)
        if register:
            await ctx.require(UI).register(
                ctx, web=web, dashboard=dashboard, requires=requires,
                provides=provides, contract_digests=contract_digests,
            )

    await root.mount(
        apply,
        name=name,
        inject=(UI,),
        runtime=PluginRuntime(
            plugin_id=name,
            generation_id=f"{name}-generation",
            plugin_dir=code,
            data_dir=code / "data",
            workspace=code,
            config={},
        ),
    )
    return contexts[0]


async def _mount_ui_and_host(
    root: CompositionRoot,
    *,
    runtime_dir: Path,
    host_path: str = "/host",
    validation: bool = False,
    ui_inject: tuple[ServiceKey[object], ...] = (),
) -> FastAPI:
    """Install the provider and give it the real application route tuple."""

    await root.context.provide(HOST_INFO, HostInfo(boot_id="test", validation=validation))
    host = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @host.get(host_path)
    def host_route() -> dict[str, bool]:
        return {"host": True}

    await root.context.provide(DASHBOARD_ROUTES, tuple(host.routes))
    await root.mount(
        ui_plugin.apply,
        name="ui",
        inject=ui_inject,
        runtime=PluginRuntime(
            plugin_id="ui",
            generation_id="ui-generation",
            plugin_dir=Path(__file__).parents[1] / "plugins" / "ui",
            data_dir=runtime_dir / "data",
            workspace=runtime_dir / "workspace",
            config={},
        ),
    )
    return host


@pytest.mark.asyncio
async def test_web_registration_is_live_and_wire_bytes_omit_internal_fence(tmp_path: Path) -> None:
    root = CompositionRoot("ui")
    code = tmp_path / "view"
    code.mkdir()
    (code / "view.js").write_text(
        "export function activate() { return () => {}; }\n",
        encoding="utf-8",
    )
    try:
        await _mount_ui_and_host(root, runtime_dir=tmp_path / "ui-runtime")
        ctx = await _mount_owner(root, code, web="view.js")
        registry = ctx.require(UI)
        catalog = registry.catalog()
        descriptor = catalog.modules[0]
        assert isinstance(descriptor, WebModuleDescriptor)
        assert descriptor.registration_uuid
        assert b"registrationUuid" not in catalog.encode_bootstrap("live")
        assert ctx.fiber.state is FiberState.ACTIVE
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_contract_conflict_is_rejected_before_contributor_becomes_active(tmp_path: Path) -> None:
    root = CompositionRoot("contracts")
    (tmp_path / "view.js").write_text(
        "export function activate() { return () => {}; }\n",
        encoding="utf-8",
    )
    try:
        await _mount_ui_and_host(root, runtime_dir=tmp_path / "ui-runtime")
        await _mount_owner(root, tmp_path, name="left", web="view.js", provides=("panel.v1",))
        right = await root.mount(
            lambda ctx: ctx.require(UI).register(
                ctx, web="view.js", provides=("panel.v1",),
            ),
            name="right",
            inject=(UI,),
            runtime=PluginRuntime(
                plugin_id="right", generation_id="right-generation",
                plugin_dir=tmp_path, data_dir=tmp_path / "data",
                workspace=tmp_path, config={},
            ),
        )
        assert right.state is FiberState.FAILED
        assert isinstance(right.error, RuntimeError)
        assert "重复提供" in str(right.error)
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_dashboard_uses_host_routes_and_original_contributor_context(tmp_path: Path) -> None:
    root = CompositionRoot("dashboard")
    module, loader = _module_from_source(
        tmp_path / "dashboard",
        """
from fastapi import FastAPI
observed = []
def register(app: FastAPI, context: object) -> None:
    observed.append(context)
    @app.get('/api/dashboard/view')
    def view() -> dict[str, bool]:
        return {'ok': True}
def load_dashboard() -> object:
    return _MODULE
""",
    )
    try:
        await _mount_ui_and_host(root, runtime_dir=tmp_path / "ui-runtime")
        ctx = await _mount_owner(
            root, tmp_path / "dashboard", dashboard=loader,
        )
        binding = ctx.require(UI).bindings()[0]
        assert binding.context is ctx
        assert module.observed and module.observed[0] is not None
        _require_routes_available(binding, list(_plugin_routes(tuple(ctx.require(DASHBOARD_ROUTES)))))
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_dashboard_route_conflict_is_rejected_before_active(tmp_path: Path) -> None:
    root = CompositionRoot("dashboard-route-conflict")
    _module, loader = _module_from_source(
        tmp_path / "dashboard",
        """
from fastapi import FastAPI
def register(app: FastAPI, _context: object) -> None:
    @app.get('/host')
    def conflict() -> dict[str, bool]:
        return {'conflict': True}
def load_dashboard() -> object:
    return _MODULE
""",
    )
    try:
        await _mount_ui_and_host(root, runtime_dir=tmp_path / "ui-runtime")
        fiber = await root.mount(
            lambda ctx: ctx.require(UI).register(ctx, dashboard=loader),
            name="conflict",
            inject=(UI,),
            runtime=PluginRuntime(
                plugin_id="conflict", generation_id="conflict-generation",
                plugin_dir=tmp_path / "dashboard", data_dir=tmp_path / "dashboard" / "data",
                workspace=tmp_path / "dashboard", config={},
            ),
        )
        assert fiber.state is FiberState.FAILED
        assert isinstance(fiber.error, RuntimeError)
        assert "dashboard route 冲突" in str(fiber.error)
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_dashboard_resource_close_failure_retries_same_effect_without_rebuild(
    tmp_path: Path,
) -> None:
    """The registration Effect retains all closeables across a failed close."""

    source = "\n".join([
        "from fastapi import FastAPI",
        "builds = 0",
        "closed = []",
        "class Resource:",
        "    def __init__(self, name: str, fail_once: bool = False) -> None:",
        "        self.name = name",
        "        self.fail_once = fail_once",
        "    def close(self) -> None:",
        "        closed.append(self.name)",
        "        if self.fail_once:",
        "            self.fail_once = False",
        "            raise RuntimeError('close failure')",
        "def register(_app: FastAPI, _context: object) -> list[object]:",
        "    global builds",
        "    builds += 1",
        "    return [Resource('first'), Resource('second', fail_once=True)]",
        "def load_dashboard() -> object:",
        "    return _MODULE",
    ])
    module, loader = _module_from_source(tmp_path / "dashboard", source)
    root = CompositionRoot("dashboard-resource-close")
    try:
        await _mount_ui_and_host(root, runtime_dir=tmp_path / "ui-runtime")
        fiber = await root.mount(
            lambda ctx: ctx.require(UI).register(ctx, dashboard=loader),
            name="resources",
            inject=(UI,),
            runtime=PluginRuntime(
                plugin_id="resources", generation_id="resources-generation",
                plugin_dir=tmp_path / "dashboard", data_dir=tmp_path / "dashboard" / "data",
                workspace=tmp_path / "dashboard", config={},
            ),
        )
        assert fiber.state is FiberState.ACTIVE
        with pytest.raises(RuntimeError, match="close failure"):
            await fiber.dispose()
        assert fiber.state is FiberState.UNLOADING
        assert module.builds == 1
        assert module.closed == ["second"]
        await fiber.dispose()
        assert fiber.state is FiberState.DISPOSED
        assert module.builds == 1
        assert module.closed == ["second", "second", "first"]
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_dashboard_build_validation_retains_acquired_resources_for_retry(
    tmp_path: Path,
) -> None:
    """A bad returned item keeps earlier closeables owned by the registration Effect."""

    source = "\n".join([
        "from fastapi import FastAPI",
        "builds = 0",
        "closed = []",
        "class Resource:",
        "    def __init__(self) -> None:",
        "        self.attempts = 0",
        "    def close(self) -> None:",
        "        self.attempts += 1",
        "        closed.append(self.attempts)",
        "        if self.attempts == 1:",
        "            raise RuntimeError('retained close failure')",
        "def register(_app: FastAPI, _context: object) -> list[object]:",
        "    global builds",
        "    builds += 1",
        "    return [Resource(), object()]",
        "def load_dashboard() -> object:",
        "    return _MODULE",
    ])
    module, loader = _module_from_source(tmp_path / "dashboard", source)
    root = CompositionRoot("dashboard-build-validation")
    try:
        await _mount_ui_and_host(root, runtime_dir=tmp_path / "ui-runtime")
        with pytest.raises(BaseExceptionGroup) as mount_failure:
            await root.mount(
                lambda ctx: ctx.require(UI).register(ctx, dashboard=loader),
                name="invalid-resource",
                inject=(UI,),
                runtime=PluginRuntime(
                    plugin_id="invalid-resource",
                    generation_id="invalid-resource-generation",
                    plugin_dir=tmp_path / "dashboard",
                    data_dir=tmp_path / "dashboard" / "data",
                    workspace=tmp_path / "dashboard",
                    config={},
                ),
            )
        child_errors = mount_failure.value.exceptions
        assert len(child_errors) == 2
        assert any("indexes=[1]" in str(error) for error in child_errors)
        assert any("retained close failure" in str(error) for error in child_errors)
        fiber = next(
            item for item in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if item.name == "invalid-resource"
        )
        ui = root.context.require(UI)
        registration = ui._entries["invalid-resource"]  # pyright: ignore[reportPrivateUsage]
        assert registration.effect is not None
        assert registration.resources is not None
        assert module.builds == 1
        assert fiber.state is FiberState.UNLOADING
        assert registration.resources.closeables  # pyright: ignore[reportPrivateUsage]
        assert module.closed == [1]
        await fiber.dispose()
        assert fiber.state is FiberState.DISPOSED
        assert module.builds == 1
        assert module.closed == [1, 2]
    finally:
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("validation", [False, True])
async def test_dashboard_context_uses_real_host_validation(
    tmp_path: Path, validation: bool,
) -> None:
    source = "\n".join([
        "from fastapi import FastAPI",
        "seen = []",
        "def plugin_enabled(context: object) -> bool:",
        "    seen.append(context.validation)",
        "    return True",
        "def register(_app: FastAPI, _context: object) -> None:",
        "    return None",
        "def load_dashboard() -> object:",
        "    return _MODULE",
    ])
    module, loader = _module_from_source(tmp_path / "dashboard", source)
    root = CompositionRoot("dashboard-validation")
    try:
        await _mount_ui_and_host(
            root, runtime_dir=tmp_path / "ui-runtime", validation=validation,
        )
        fiber = await root.mount(
            lambda ctx: ctx.require(UI).register(ctx, dashboard=loader),
            name="validation",
            inject=(UI,),
            runtime=PluginRuntime(
                plugin_id="validation", generation_id="validation-generation",
                plugin_dir=tmp_path / "dashboard", data_dir=tmp_path / "dashboard" / "data",
                workspace=tmp_path / "dashboard", config={},
            ),
        )
        assert fiber.state is FiberState.ACTIVE
        assert module.seen == [validation]
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_web_ui_bootstrap_and_state_open_their_own_ui_scope(tmp_path: Path) -> None:
    root = CompositionRoot("web-ui-scope")
    dependency = ServiceKey[object]("test.ui.dependency")

    async def dependency_provider(ctx: Context) -> None:
        await ctx.provide(dependency, object())

    code = tmp_path / "view"
    code.mkdir()
    (code / "view.js").write_text(
        "export function activate() { return () => {}; }\n",
        encoding="utf-8",
    )
    try:
        await root.mount(dependency_provider, name="ui-dependency")
        await _mount_ui_and_host(
            root,
            runtime_dir=tmp_path / "ui-runtime",
            ui_inject=(dependency,),
        )
        ui_fiber = next(fiber for fiber in root._fibers.values() if fiber.name == "ui")
        old_context = ui_fiber.context
        old_provider = old_context.require(WEB_UI)
        bootstrap = await old_provider.bootstrap()
        state = await old_provider.state()
        assert bootstrap and state["snapshotId"] == root.generation_id

        dependency_fiber = next(
            fiber for fiber in root._fibers.values() if fiber.name == "ui-dependency"
        )
        await dependency_fiber.dispose()
        assert ui_fiber.state is FiberState.PENDING
        await root.mount(dependency_provider, name="ui-dependency")
        assert ui_fiber.state is FiberState.ACTIVE
        new_context = ui_fiber.context
        new_provider = new_context.require(WEB_UI)
        assert new_context is not old_context
        assert new_provider is not old_provider

        with pytest.raises(CompositionError) as stale:
            await old_provider.bootstrap()
        assert stale.value.code == "STALE_ACTIVATION"
        with pytest.raises(CompositionError) as stale:
            await old_provider.state()
        assert stale.value.code == "STALE_ACTIVATION"
        assert await new_provider.bootstrap()
        assert (await new_provider.state())["snapshotId"] == root.generation_id
        assert not ui_fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
    finally:
        await root.dispose()
