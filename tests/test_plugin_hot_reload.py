from __future__ import annotations
import ast
import asyncio
import threading
from pathlib import Path
from typing import cast
import pytest
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from agent.plugin_composition import FiberState
from agent.plugins.manager import PluginManager
from agent.plugins import manager as manager_module

def _v3_source(
    name: str,
    *,
    version: str = "1.0.0",
    body: str = "    return None\n",
    exports: str = "",
) -> str:
    """Create one minimal v3 module with the exact Core admission contract."""

    return (
        "api_version = 3\n"
        f"name = {name!r}\n"
        f"version = {version!r}\n"
        f"{exports}"
        "async def apply(ctx):\n"
        f"{body}"
    )

def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir

def _write_checked_plugin(root: Path, name: str, source: str) -> Path:
    """Parse and compile a dynamic fixture before writing its source."""
    entry = root / name / "plugin.py"
    tree = ast.parse(source, filename=str(entry))
    compile(tree, str(entry), "exec")
    return _write_plugin(root, name, source)

def _manager(
    tmp_path: Path,
    *,
    workspace: Path | None = None,
) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        workspace=workspace or tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


@pytest.mark.asyncio
async def test_cancelled_source_preparation_drains_before_manager_reuse(tmp_path, monkeypatch):
    """O/C1：源码准备不占用 loop，撤销后先排空归档，再释放操作 owner。"""
    plugin = _write_plugin(tmp_path / "plugins", "slow", _v3_source("slow"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    release = threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = manager_module.prepare_plugin_input

    def blocked(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise RuntimeError("源码准备阻塞了事件循环，无法接收释放信号")
        return original(*args, **kwargs)

    try:
        await manager.load_all()
        before = manager._selection.read()
        old = manager.generation("slow")
        (plugin / "plugin.py").write_text(_v3_source("slow", version="2"))
        monkeypatch.setattr(manager_module, "prepare_plugin_input", blocked)
        observer = asyncio.create_task(manager.reconcile_changed())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            operation = manager._operation
            assert operation is not None and not operation.task.done()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert not operation.task.done()
            # 新操作排队等前一个退出；排队上限到了仍报忙，且没有做任何改动。
            manager.ADMISSION_WAIT_SECONDS = 0.1
            with pytest.raises(manager_module.OperationBusyError, match="没有开始"):
                await manager.reconcile_changed()
            assert manager._selection.read() == before
            assert manager.generation("slow") is old
        finally:
            release.set()
            await asyncio.gather(observer, return_exceptions=True)
        with pytest.raises(asyncio.CancelledError):
            await operation.task
        assert manager._selection.read() == before
        monkeypatch.setattr(manager_module, "prepare_plugin_input", original)
        result = await manager.reconcile_changed()
        assert result[0]["publication_state"] == "active"
        assert manager.generation("slow") is not old
    finally:
        release.set()
        await manager.terminate_all()

@pytest.mark.asyncio
async def test_live_selection_compile_abort_then_commit(tmp_path: Path) -> None:
    """A bad source keeps A; a valid update commits B in the same Root."""
    plugin = _write_plugin(
        tmp_path / "plugins", "selection", _v3_source("selection", version="release-a"),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        root = manager.live_root
        old = manager.generation("selection")
        selected_a = manager._selection.read()
        assert root is not None and old is not None and selected_a is not None

        (plugin / "plugin.py").write_text("def invalid(:\n", encoding="utf-8")
        result = await manager.reconcile_changed()
        assert result[0]["publication_state"] == "source_unavailable"
        assert manager._selection.read() == selected_a
        assert manager.generation("selection") is old
        assert old.fiber is not None and old.fiber.state == FiberState.ACTIVE

        (plugin / "plugin.py").write_text(
            _v3_source("selection", version="release-b"), encoding="utf-8",
        )
        committed = await manager.reconcile_changed()
        current = manager.generation("selection")
        selected_b = manager._selection.read()
        assert committed[0]["publication_state"] == "active"
        assert selected_b is not None and selected_b != selected_a
        assert current is not None and current is not old
        assert current.instance.version == "release-b"
        assert old.scope.closed
        assert manager.live_root is root
    finally:
        await manager.terminate_all()

@pytest.mark.asyncio
async def test_cold_selected_import_failure_retains_failed_generation_and_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected pre-Fiber import failure keeps B while a peer still starts."""
    bad_source = (
        "import os\n"
        + _v3_source(
            "cold_bad",
            exports=(
                "if os.environ.get('COLD_IMPORT_FAIL') == 'yes':\n"
                "    raise RuntimeError('cold import blocked')\n"
            ),
        )
    )
    _write_checked_plugin(tmp_path / "plugins", "cold_bad", bad_source)
    _write_checked_plugin(tmp_path / "plugins", "cold_peer", _v3_source("cold_peer"))
    initialize_plugin_workspace(tmp_path / "workspace")

    monkeypatch.setenv("COLD_IMPORT_FAIL", "no")
    first = _manager(tmp_path)
    try:
        await first.load_all()
        selected = first._selection.read()
        assert selected is not None
    finally:
        await first.terminate_all()

    monkeypatch.setenv("COLD_IMPORT_FAIL", "yes")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        failed = manager.generation("cold_bad")
        peer = manager.generation("cold_peer")
        assert failed is not None
        assert failed.state == "failed"
        assert failed.load_error is not None
        assert str(failed.load_error) == "cold import blocked"
        assert failed.fiber is None
        assert failed.scope.closed
        assert manager._selection.read() == selected
        assert not manager._draining_generations.get("cold_bad")
        assert peer is not None and peer.fiber is not None
        assert peer.fiber.state == FiberState.ACTIVE

        status = next(
            item for item in cast(list[dict[str, object]], manager.plugin_status()["plugins"])
            if item["plugin_id"] == "cold_bad"
        )
        assert status["state"] == "failed"
        assert status["load_error"] == "cold import blocked"
        assert status["cleanup_pending"] is False

        from agent.plugin_composition.runtime_catalog import build_runtime_catalog

        root = manager._live_root
        assert root is not None
        catalog = build_runtime_catalog(
            root,
            manager._active_generations,
            manager._draining_generations,
        )
        view = next(
            item for item in cast(list[dict[str, object]], catalog["plugins"])
            if item["id"] == "cold_bad"
        )
        assert view["api_version"] == 3
        assert view["input_ref"] == failed.input_ref
        assert view["state"] == "failed"
        assert view["load_error"] == "cold import blocked"
        assert view["cleanup_pending"] is False
        assert cast(dict[str, object], view["composition"])["ready"] is False
        assert cast(dict[str, object], view["composition"])["fibers"] == []
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_assets_open_skips_unloading_contributors(tmp_path: Path) -> None:
    from agent.plugin_composition import CompositionRoot, Context, PluginRuntime, ServiceKey
    from plugins.assets.contract import INSTALLED_ASSETS
    from plugins.assets.plugin import apply as apply_assets

    root = CompositionRoot("test")
    consumer_ctx: list[Context] = []
    ready_dir = tmp_path / "ready-plugin"
    (ready_dir / "skills").mkdir(parents=True)
    flaky_dir = tmp_path / "flaky-plugin"
    (flaky_dir / "skills").mkdir(parents=True)
    unload_gate = asyncio.Event()
    in_flight_entered = asyncio.Event()

    async def mount_assets(ctx: Context) -> None:
        await apply_assets(ctx)

    async def mount_ready(ctx: Context) -> None:
        _ = await ctx.require(INSTALLED_ASSETS).register(ctx, "skills", "skills")

    async def mount_flaky(ctx: Context) -> None:
        _ = await ctx.require(INSTALLED_ASSETS).register(ctx, "skills", "skills")
        _ = await ctx.provide(ServiceKey("test.flaky.ping"), lambda: ctx)

    async def mount_consumer(ctx: Context) -> None:
        consumer_ctx.append(ctx)

    await root.mount(mount_assets, name="assets")
    await root.mount(
        mount_ready,
        name="ready",
        inject=(INSTALLED_ASSETS,),
        runtime=PluginRuntime("ready@local", "local", ready_dir, tmp_path / "d1", tmp_path, {}),
    )
    await root.mount(
        mount_flaky,
        name="flaky",
        inject=(INSTALLED_ASSETS,),
        runtime=PluginRuntime("flaky@local", "local", flaky_dir, tmp_path / "d2", tmp_path, {}),
    )
    await root.mount(
        mount_consumer,
        name="consumer",
        inject=(INSTALLED_ASSETS,),
        runtime=PluginRuntime("consumer@local", "local", tmp_path, tmp_path / "dc", tmp_path, {}),
    )

    try:
        consumer = consumer_ctx[0]
        assets_svc = root.service_value(INSTALLED_ASSETS)
        assert assets_svc is not None
        flaky_fiber = next(f for f in root.fibers() if f.name == "flaky")
        ping_fn = root.service_value(ServiceKey("test.flaky.ping"))
        assert ping_fn is not None
        flaky_ctx = cast(Context, ping_fn())

        async def hold_flaky() -> None:
            async with flaky_ctx.runtime_scope():
                in_flight_entered.set()
                _ = await unload_gate.wait()

        try:
            async with asyncio.timeout(3):
                holder = asyncio.create_task(hold_flaky())
                _ = await in_flight_entered.wait()
                unloader = asyncio.create_task(flaky_fiber.dispose())
                while flaky_fiber.state is not FiberState.UNLOADING:
                    await asyncio.sleep(0.01)

                async with consumer.runtime_scope():
                    listed = assets_svc(consumer)
                    assert [a.owner_id for a in listed] == ["ready@local"]
                    async with assets_svc.open(consumer, category="skills") as opened:
                        assert [a.owner_id for a in opened] == ["ready@local"]

                unload_gate.set()
                await holder
                await unloader
        finally:
            unload_gate.set()
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_registries_filter_non_active_contributors(tmp_path: Path) -> None:
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from typing import Mapping

    from agent.plugin_composition import CompositionError, CompositionRoot, Context, PluginRuntime
    from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
    from plugins.delivery.api import Receipt, Sender
    from plugins.delivery.contract import DELIVERY_SENDERS
    from plugins.delivery.senders import Senders
    from plugins.ledger.bindings import Bindings
    from plugins.ledger.contract import BINDINGS, MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION, Message
    from plugins.ledger.log import MessageLog
    from plugins.ledger.services import MessageWriters, OwnerState, SessionAdmission
    from plugins.onboarding.contract import ONBOARDING, Step
    from plugins.onboarding.plugin import apply as apply_onboarding
    from plugins.plugin_settings.contract import SETTINGS_FORMS, SettingsField, SettingsForm
    from plugins.plugin_settings.plugin import apply as apply_plugin_settings
    from plugins.sources.contract import SOURCES_V5 as SOURCES
    from plugins.sources.plugin import apply as apply_sources
    from plugins.ui.contract import UI

    class _DummySender(Sender):
        @property
        def idempotent(self) -> bool:
            return True

        async def send(self, key: str, address: str, message: Message) -> Receipt:
            raise AssertionError("not called")

        async def query(self, key: str, address: str) -> Receipt | None:
            return None

    class _DummyConfig:
        async def read(self) -> dict[str, object]:
            return {}

        async def save(self, request_id: str, expected_input: str, values: dict[str, object]) -> dict[str, object]:
            return {}

        def receipt(self, request_id: str) -> dict[str, object]:
            return {}

    log = MessageLog(tmp_path / "sessions.db")
    root = CompositionRoot("test")
    unloading_gate = asyncio.Event()
    unloading_entered = asyncio.Event()
    try:
        async def core(ctx: Context) -> None:
            for key, val in [
                (MESSAGE_CATALOG, log.catalog()),
                (MESSAGE_WRITERS, MessageWriters(log)),
                (OWNER_STATE, OwnerState(log)),
                (SESSION_ADMISSION, SessionAdmission(log)),
                (BINDINGS, Bindings(log, ctx)),
                (RUNTIME_CATALOG, lambda _ctx: {"plugins": []}),
                (UI, SimpleNamespace(
                    register=lambda *_a, **_kw: asyncio.sleep(0),
                    register_configuration=lambda *_a, **_kw: None,
                )),
            ]:
                _ = await ctx.provide(key, val)

        async def mount_delivery(ctx: Context) -> None:
            _ = await ctx.provide(DELIVERY_SENDERS, Senders(ctx))

        async def mount_onboarding(ctx: Context) -> None:
            await apply_onboarding(ctx)

        async def mount_settings(ctx: Context) -> None:
            await apply_plugin_settings(ctx)

        async def mount_sources(ctx: Context) -> None:
            await apply_sources(ctx)

        await root.mount(core, name="core")
        await root.mount(mount_delivery, name="delivery", inject=(BINDINGS,))
        await root.mount(mount_onboarding, name="onboarding", inject=(RUNTIME_CATALOG,))
        await root.mount(mount_settings, name="plugin_settings", inject=(UI,))
        await root.mount(mount_sources, name="sources", inject=(MESSAGE_WRITERS,))

        async def _internal_err() -> dict[str, object]:
            raise CompositionError("SERVICE_SCOPE_MISMATCH", "internal")

        err_ctx_holder: list[Context] = []

        async def mount_active_err_step(ctx: Context) -> None:
            err_ctx_holder.append(ctx)
            _ = await ctx.require(ONBOARDING).register(
                ctx,
                Step(key="err_step", title="Err", group="other", route="/err", status=_internal_err),
            )

        await root.mount(
            mount_active_err_step,
            name="err_contrib",
            inject=(ONBOARDING, DELIVERY_SENDERS),
            runtime=PluginRuntime("err@local", "local", tmp_path, tmp_path / "fe", tmp_path, {}),
        )

        unloading_ctx_holder: list[Context] = []

        async def mount_unloading_contrib(ctx: Context) -> None:
            unloading_ctx_holder.append(ctx)
            senders = cast(Senders, ctx.require(DELIVERY_SENDERS))
            _ = await senders.candidate(ctx, name="flaky_chan", title="Flaky", route="/flaky", status=lambda: {"enabled": True})

            @asynccontextmanager
            async def open_sender():
                yield _DummySender()

            _ = await senders.register(ctx, name="flaky_chan", open=open_sender, idempotent=True)
            _ = await ctx.require(ONBOARDING).register(
                ctx,
                Step(
                    key="flaky_step",
                    title="Flaky Step",
                    group="other",
                    route="/flaky",
                    status=lambda: asyncio.sleep(0, result={"configured": True}),
                ),
            )
            _ = await ctx.require(SETTINGS_FORMS).register(
                ctx,
                SettingsForm("Flaky", "desc", (SettingsField("token", "Token"),)),
                _DummyConfig(),
            )
            _ = await ctx.require(SOURCES).register(
                ctx,
                name="flaky_src",
                channels=("flaky",),
                open=lambda _sid: None,  # type: ignore[arg-type,return-value]
                needs_reply=lambda _r: asyncio.sleep(0, result=False),
                accept=lambda *_a: asyncio.sleep(0, result=None),  # type: ignore[arg-type,return-value]
            )

        await root.mount(
            mount_unloading_contrib,
            name="flaky_contrib",
            inject=(DELIVERY_SENDERS, ONBOARDING, SETTINGS_FORMS, SOURCES),
            runtime=PluginRuntime("flaky@local", "local", tmp_path, tmp_path / "fd", tmp_path, {}),
        )

        flaky_ctx = unloading_ctx_holder[0]
        flaky_fiber = next(f for f in root.fibers() if f.name == "flaky_contrib")

        async def hold() -> None:
            async with flaky_ctx.runtime_scope():
                unloading_entered.set()
                _ = await unloading_gate.wait()

        try:
            async with asyncio.timeout(3):
                holder = asyncio.create_task(hold())
                _ = await unloading_entered.wait()
                unloader = asyncio.create_task(flaky_fiber.dispose())
                while flaky_fiber.state is not FiberState.UNLOADING:
                    await asyncio.sleep(0.01)

                senders = root.service_value(DELIVERY_SENDERS)
                assert senders is not None
                assert senders.candidates() == ()
                # bind_all 仍保留已登记项，避免长寿命绑定在重载窗口丢失发送者
                bindings = root.service_value(BINDINGS)
                assert bindings is not None
                async with err_ctx_holder[0].runtime_scope():
                    assert "flaky_chan" in cast(Mapping[str, str], senders.bind_all(bindings))

                onboarding = root.service_value(ONBOARDING)
                assert onboarding is not None
                catalog = await onboarding.catalog()
                assert [s["id"] for s in cast(list[dict[str, object]], catalog["steps"])] == ["err@local/err_step"]
                with pytest.raises(KeyError):
                    _ = await onboarding.status("flaky@local/flaky_step")
                # 步骤内部抛出的 CompositionError 不能被伪装成 404
                with pytest.raises(CompositionError, match="internal"):
                    _ = await onboarding.status("err@local/err_step")

                forms = root.service_value(SETTINGS_FORMS)
                assert forms is not None
                assert forms.entries() == ()
                with pytest.raises(KeyError):
                    _ = forms.entry("flaky@local")

                sources_svc = root.service_value(SOURCES)
                assert sources_svc is not None
                assert sources_svc.entries() == ()

                unloading_gate.set()
                await holder
                await unloader
        finally:
            unloading_gate.set()
    finally:
        await root.dispose()
        log.close()
