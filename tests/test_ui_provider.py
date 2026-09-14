"""普通 UI provider 的归属、封存和资源失败合同。"""

from pathlib import Path
from types import ModuleType

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime, SNAPSHOT_SEALING, SnapshotSealing
from agent.plugin_composition.ui import UI
from plugins.ui import plugin as ui_plugin


async def mount_owner(root, code, *, name="view", register=True, **options):
    """把真实 Context 交给 provider，测试不自报注册 owner。"""
    contexts = []

    async def apply(ctx):
        contexts.append(ctx)
        if register:
            await ctx.require(UI).register(ctx, **options)

    await root.mount(
        apply, name=name, inject=(UI,),
        runtime=PluginRuntime(
            plugin_id=name, generation_id=f"{name}-generation", plugin_dir=code,
            data_dir=code / "data", workspace=code, config={},
        ),
    )
    return contexts[0]


@pytest.mark.asyncio
async def test_ui_uses_explicit_provider_and_freezes_code_bytes(tmp_path):
    root = CompositionRoot("ui")
    code = tmp_path / "view"
    code.mkdir()
    script = code / "view.js"
    script.write_text("export function activate() { return () => {}; }\n")
    original = script.read_text()
    try:
        # 服务按能力连接，不依赖 provider 的安装名字。
        await root.mount(ui_plugin.apply, name="my-ui-provider")
        ctx = await mount_owner(root, code, web="view.js")
        await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
        catalog = root.context.require(UI).catalog()
        descriptor = catalog.modules[0]
        assert (descriptor.plugin_id, descriptor.generation_id) == ("view", "view-generation")
        script.write_text("changed after registration")
        assert descriptor.asset.module == original
        with pytest.raises(RuntimeError, match="已封存"):
            await ctx.require(UI).register(ctx, web="view.js")
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_ui_rejects_foreign_context_and_code_paths(tmp_path):
    roots = [CompositionRoot("left"), CompositionRoot("right")]
    code = tmp_path / "owner"
    code.mkdir()
    foreign = tmp_path / "foreign.js"
    foreign.write_text("export function activate() {}")
    (code / "linked.js").symlink_to(foreign)
    try:
        for root in roots:
            await root.mount(ui_plugin.apply, name="ui")
        left = await mount_owner(roots[0], code, register=False)
        right = await mount_owner(roots[1], code, register=False)
        registry = left.require(UI)
        with pytest.raises(ValueError, match="跨 composition Root"):
            await registry.register(right, web="linked.js")
        for path in ("../foreign.js", str(foreign), "linked.js"):
            with pytest.raises(ValueError, match="制品"):
                await registry.register(left, web=path)
        with pytest.raises(ValueError, match="loader 不属于"):
            await registry.register(left, dashboard=lambda: ModuleType("foreign"))
    finally:
        for root in roots:
            await root.dispose()


@pytest.mark.asyncio
async def test_ui_contract_collision_fails_at_provider_sealing(tmp_path):
    root = CompositionRoot("contracts")
    (tmp_path / "view.js").write_text("export function activate() {}")
    try:
        await root.mount(ui_plugin.apply, name="ui")
        for name in ("left", "right"):
            await mount_owner(root, tmp_path, name=name, web="view.js", provides=("panel.v1",))
        with pytest.raises(RuntimeError, match="重复提供"):
            await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
        with pytest.raises(RuntimeError, match="尚未封存"):
            root.context.require(UI).catalog()
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_dashboard_close_failure_retains_same_handle(tmp_path):
    """关闭失败不弹出资源；成功后不再次关闭，且不重放初始化。"""
    calls = []

    class Resource:
        def close(self):
            calls.append("close")
            if len(calls) == 1:
                raise RuntimeError("retry close")

    root = CompositionRoot("close")
    try:
        await root.mount(ui_plugin.apply, name="ui")
        ctx = await mount_owner(root, Path(__file__).parent, register=False)
        module = ModuleType("fixture_dashboard")
        module.__file__ = __file__
        handle = Resource()
        module.register = lambda app, context: [handle, object()]
        registry = ctx.require(UI)
        effect = await registry.register(ctx, dashboard=lambda: module)
        await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
        with pytest.raises(RuntimeError, match="不是 closeable"):
            registry.prepare_dashboard(
                core_routes=(), workload_urls=lambda generation: {},
                validation_owners=frozenset(), tolerate_failures=False,
            )
        with pytest.raises(RuntimeError, match="retry close"):
            await effect.aclose()
        await effect.aclose()
        await effect.aclose()
        assert calls == ["close", "close"]
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_dashboard_host_rejects_borrowed_provider():
    """即使人为把旧服务别名装入新 Root，宿主也不能使用旧目录。"""
    from agent.plugins.dashboard_host import PluginDashboardHost
    from agent.plugins.snapshot import RuntimeSnapshot

    old = CompositionRoot("old")
    new = CompositionRoot("new")
    try:
        await old.mount(ui_plugin.apply, name="ui")
        await old.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
        await new.context.provide(UI, old.context.require(UI))
        snapshot = RuntimeSnapshot("new", {}, composition_root=new)
        with pytest.raises(RuntimeError, match="实际 Root"):
            PluginDashboardHost(core_routes=()).prepare_snapshot(snapshot)
    finally:
        await new.dispose()
        await old.dispose()
