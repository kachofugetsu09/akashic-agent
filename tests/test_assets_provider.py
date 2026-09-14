"""普通资产 provider 的 owner、作用域和固定制品合同。"""

import asyncio
import shutil
from pathlib import Path
from typing import cast

import pytest

from agent.plugin_composition import CompositionRoot, Context, PluginRuntime
from agent.plugin_composition.archive import PluginArchive
from agent.plugin_composition.assets import INSTALLED_ASSETS
from agent.plugin_composition.bindings import BINDINGS
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import bind_runtime_snapshot, lease_runtime_snapshot, reset_runtime_snapshot
from bus.event_bus import EventBus
from plugins.assets.plugin import apply
from session.log import MessageLog
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


@pytest.mark.asyncio
async def test_registration_effect_rejects_other_owner_paths_and_other_roots(tmp_path):
    """同一 generation 名称不能替代实际 Root；目录和注销归贡献方。"""
    first = CompositionRoot("same-generation")
    other = CompositionRoot("same-generation")
    contexts: list[Context] = []
    own = tmp_path / "owner"
    foreign = tmp_path / "foreign"
    (own / "records").mkdir(parents=True)
    foreign.mkdir()
    (foreign / "secret").write_text("foreign bytes")
    (own / "linked").symlink_to(foreign, target_is_directory=True)
    (own / "records/escape").symlink_to(foreign / "secret")

    async def contributor(ctx):
        contexts.append(ctx)

    def runtime(name, directory):
        return PluginRuntime(name, "same-generation", directory, tmp_path / name / "data", tmp_path, {})

    try:
        for root in (first, other):
            await root.mount(apply, name="assets", runtime=runtime("assets", tmp_path))
            await root.mount(contributor, name="owner", inject=(INSTALLED_ASSETS,),
                             runtime=runtime("owner", own))
        registry = first.context.require(INSTALLED_ASSETS)
        owner, other_owner = contexts
        with pytest.raises(ValueError, match="跨 composition Root"):
            await registry.register(other_owner, "abde", "records")
        for path in ("../foreign", str(foreign), "linked", "records"):
            with pytest.raises(ValueError, match="制品"):
                await registry.register(owner, "abde", path)
        (own / "records/escape").unlink()
        effect = await registry.register(owner, "abde", "records")
        with pytest.raises(ValueError, match="重复注册"):
            await registry.register(owner, "abde", "records")
        assert first.binding_contributors(INSTALLED_ASSETS) == (owner,)
        await effect.aclose()
        assert first.binding_contributors(INSTALLED_ASSETS) == ()
        assert (foreign / "secret").read_text() == "foreign bytes"
        assert (own / "records").is_dir()
    finally:
        await other.dispose()
        await first.dispose()


def _sources(tmp_path: Path, *, provider: bool = True) -> Path:
    """建立独立贡献者；随机类别和原始字节不经过技能解析。"""
    sources = tmp_path / "plugins"
    if provider:
        shutil.copytree(Path(__file__).parents[1] / "plugins/assets", sources / "assets",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    plugin = sources / "records"
    (plugin / "records").mkdir(parents=True)
    (plugin / "records/raw.bin").write_bytes(b"\x00\xff")
    (plugin / "plugin.py").write_text('''from agent.plugin_composition.assets import INSTALLED_ASSETS
api_version = 3
name = "records"
version = "1.0.0"
inject = (INSTALLED_ASSETS,)
async def apply(ctx):
    await ctx.require(INSTALLED_ASSETS).register(ctx, "abde", "records")
''')
    return sources


@pytest.mark.asyncio
async def test_fixed_assets_keep_binding_contributors_and_require_exact_scope(tmp_path):
    """原安装消失后仍读原字节；泄漏 callable 无法绕过 lease。"""
    sources = _sources(tmp_path)
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                            installed_cache_root=tmp_path / "home/cache", message_log=log)
    other_path = tmp_path / "other"
    initialize_plugin_workspace(other_path / "workspace")
    other = PluginManager([_sources(other_path)], event_bus=EventBus(), workspace=other_path / "workspace",
                          installed_cache_root=other_path / "home/cache")
    try:
        await other.load_all()
        await manager.load_all()
        async with lease_runtime_snapshot(manager.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            registry = ctx.require(INSTALLED_ASSETS)
            asset, = registry()
            assert asset.owner_id == "records" and asset.category == "abde"
            assert asset.root_dir == snapshot.generations["records"].code_dir / "records"
            reference = ctx.require(BINDINGS).bind(INSTALLED_ASSETS, {})
            binding = log.read_binding(reference)
            archive = PluginArchive(tmp_path / "workspace/runtime/plugin-archives", create=False)
            closure = archive.read_descriptor(cast(str, binding["root_ref"]))
            owners = {archive.read_descriptor(ref)["plugin_id"]
                      for ref in cast(tuple[str, ...], closure["components"])}
            assert owners == {"assets", "records"}
            shutil.rmtree(sources)
            assert (asset.root_dir / "raw.bin").read_bytes() == b"\x00\xff"

            async def leaked_read():
                return registry()

            with pytest.raises(RuntimeError, match="scope"):
                await asyncio.create_task(leaked_read())
        with pytest.raises(RuntimeError, match="scope"):
            registry()
        lease = other.snapshot_store.lease()
        token = bind_runtime_snapshot(lease)
        try:
            with pytest.raises(RuntimeError, match="scope"):
                registry()
        finally:
            reset_runtime_snapshot(token)
            await lease.release()
    finally:
        await other.terminate_all()
        await manager.terminate_all()
        log.close()
    assert (asset.root_dir / "raw.bin").read_bytes() == b"\x00\xff"


@pytest.mark.asyncio
async def test_assets_provider_is_required_by_explicit_selection(tmp_path):
    sources = _sources(tmp_path, provider=False)
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                            installed_cache_root=tmp_path / "home/cache")
    try:
        with pytest.raises(RuntimeError):
            await manager.load_all()
    finally:
        await manager.terminate_all()
