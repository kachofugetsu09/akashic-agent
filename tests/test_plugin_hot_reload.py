from __future__ import annotations

from agent.plugin_composition.ui import UI, DashboardBinding

import asyncio
import dataclasses
import importlib
import os
import py_compile
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlencode

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition.config_input import save_config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.convertors import CONVERTOR_TYPES, StringConvertor
from starlette.websockets import WebSocketDisconnect

from agent.plugin_composition import CompositionError, CompositionRoot
from agent.plugin_composition.assets import INSTALLED_ASSETS, InstalledAsset
from agent.plugins.artifacts import ArtifactPointer, read_pointer, write_pointers
from plugins.ui.dashboard import (
    _plugin_routes,
    _require_routes_available,
)
from agent.plugins.manager import PluginManager, _source_revision
from agent.plugins.manifest import write_plugin_manifest
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    lease_runtime_snapshot,
)
from agent.plugins.watcher import PluginWatcher
from plugins.standard_tools.skill_catalog import SkillCatalogParser
from bootstrap.dashboard_api import create_dashboard_app
from bus.event_bus import EventBus


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


def _ui_source(name: str, *, dashboard: bool = True, body: str = "") -> str:
    return _v3_source(
        name,
        exports="from importlib import import_module\nfrom agent.plugin_composition import ServiceKey\nfrom agent.plugin_composition.ui import UI\ninject = (UI,)\n",
        body="    await ctx.require(UI).register(ctx, web='web_module.js'"
             + (", dashboard=lambda: import_module('.dashboard', __package__)" if dashboard else "") + ")\n" + body,
    )


def _copy_ui_provider(tmp_path: Path) -> None:
    shutil.copytree(Path(__file__).parents[1] / "plugins/ui", tmp_path / "plugins/ui",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _asset_source(name: str, relative_path: str = "skills", *, version: str = "1.0.0") -> str:
    return _v3_source(
        name, version=version,
        exports="from agent.plugin_composition.assets import INSTALLED_ASSETS\ninject = (INSTALLED_ASSETS,)\n",
        body=f"    await ctx.require(INSTALLED_ASSETS).register(ctx, 'skills', {relative_path!r})\n",
    )


def _copy_assets_provider(tmp_path: Path) -> None:
    shutil.copytree(Path(__file__).parents[1] / "plugins/assets", tmp_path / "plugins/assets",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


async def _read_assets(manager: PluginManager) -> tuple[InstalledAsset, ...]:
    async with lease_runtime_snapshot(manager.snapshot_store) as snapshot:
        return snapshot.composition_root.context.require(INSTALLED_ASSETS)()


def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir


def _manager(
    tmp_path: Path,
    *,
    workspace: Path | None = None,
) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=workspace or tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


@pytest.mark.parametrize("kind", ["missing", "symlink", "directory"])
def test_import_boundary_rejects_invalid_plugin_file(tmp_path: Path, kind: str) -> None:
    """直接加载固定制品时也拒绝坏入口，不能执行别名文件。"""
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "custom.py").write_text("raise AssertionError('must not import')\n")
    if kind == "symlink":
        (root / "plugin.py").symlink_to(root / "custom.py")
    elif kind == "directory":
        (root / "plugin.py").mkdir()
    owner = _manager(tmp_path)
    with pytest.raises(ValueError, match="plugin.py 必须是普通文件"):
        owner._import_plugin("_invalid_entrypoint_probe", root)
    assert "_invalid_entrypoint_probe" not in sys.modules


def test_import_boundary_keeps_exact_root_file_without_calling_apply(tmp_path: Path) -> None:
    """导入固定文件保留来源路径，但不替组合层调用 apply。"""
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "plugin.py").write_text(_v3_source("probe", body="    raise AssertionError('not yet')\n"))
    (root / "custom.py").write_text("raise AssertionError('wrong file')\n")
    owner = _manager(tmp_path)
    module_name = "_fixed_entrypoint_probe"
    try:
        owner._import_plugin(module_name, root)
        module = sys.modules[module_name]
        assert module.__file__ is not None
        assert Path(module.__file__) == root / "plugin.py"
        assert callable(module.apply)
    finally:
        owner._remove_module_tree(module_name)


@pytest.mark.parametrize("version", [2, 3])
def test_archived_custom_entrypoint_contract_is_rejected_before_import(tmp_path: Path, monkeypatch, version: int) -> None:
    """旧归档不被重新解释为新入口，也不改写其恢复材料。"""
    owner = _manager(tmp_path)
    record = {"version": version, "entrypoint": "custom.py"}
    monkeypatch.setattr(owner._archive, "read_descriptor", lambda ref: record)

    def forbidden(*args):
        raise AssertionError("old archive must not import")

    monkeypatch.setattr(owner, "_import_plugin", forbidden)
    with pytest.raises(RuntimeError, match="归档运行合同不兼容"):
        owner._archived_generations(
            ("old-component",), CompositionRoot("probe"), workspace=tmp_path, sources={},
        )
    assert record == {"version": version, "entrypoint": "custom.py"}


def _write_installed_artifact(
    tmp_path: Path,
    artifact_id: str,
    source: str,
    *,
    plugin_name: str = "installed_snapshot",
) -> tuple[Path, Path]:
    """Create an installed artifact with its import-free v3 identity manifest."""

    plugin_base = tmp_path / "home" / "cache" / "lab" / plugin_name
    artifact = plugin_base / ".artifacts" / artifact_id
    artifact.mkdir(parents=True)
    (artifact / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_base, artifact


def _install_assets_provider(tmp_path: Path) -> None:
    source = (Path(__file__).parents[1] / "plugins/assets/plugin.py").read_text()
    base, _ = _write_installed_artifact(tmp_path, "1.0.0-assets", source, plugin_name="assets")
    pointer = ArtifactPointer(".artifacts/1.0.0-assets")
    write_pointers(base, stable=pointer, latest=pointer)
    write_plugin_manifest({"assets@lab": True, "installed_snapshot@lab": True},
                          plugins_home=tmp_path / "home")


def _write_installed_skill(plugin_root: Path, name: str, body: str) -> Path:
    skill_dir = plugin_root / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    return skill_dir


@pytest.mark.asyncio
async def test_candidate_publishes_unique_generation(tmp_path: Path):
    _write_plugin(tmp_path / "plugins", "candidate", _v3_source("candidate"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)

    await manager.load_all()

    generation = manager.generation("candidate")
    assert generation is not None
    assert sys.modules[generation.module_path] is generation.instance.module
    assert generation.instance.module.__name__ == generation.module_path
    assert generation.instance.version == "1.0.0"
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("signature, accepted", [
    ("host", True), ("host, settings=None", True), ("*args", True),
    ("host, settings", False), ("*, host", False),
])
async def test_plugin_entry_uses_python_call_semantics(tmp_path: Path, signature: str, accepted: bool):
    """可用一个位置参数调用的入口不受参数命名限制。"""
    source = (
        'api_version = 3\nname = "ordinary"\nversion = "1.0.0"\n'
        f'async def apply({signature}):\n    return None\n'
    )
    _write_plugin(tmp_path / "plugins", "ordinary", source)
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        if accepted:
            await manager.load_all()
        else:
            with pytest.raises(RuntimeError):
                await manager.load_all()
        assert (manager.generation("ordinary") is not None) is accepted
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_invalid_source_does_not_block_next_load_attempt(tmp_path: Path):
    """坏源码直接报告加载错误；修复后可重新加载完整组合。"""
    plugin = _write_plugin(tmp_path / "plugins", "broken", "this is not python !!!\n")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)

    try:
        with pytest.raises(ValueError, match="插件身份源码无法解析"):
            await manager.load_all()
        assert manager.generation("broken") is None
        assert manager.current_snapshot is None

        (plugin / "plugin.py").write_text(_v3_source("broken"), encoding="utf-8")
        await manager.load_all()
        assert manager.generation("broken") is not None
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_compile_error_keeps_original_error_and_stable(tmp_path: Path, monkeypatch):
    """编译失败清理候选，原错误直接交给调用者，正式选择不变。"""
    plugin = _write_plugin(tmp_path / "plugins", "ordinary", _v3_source("ordinary"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        stable = manager.current_snapshot
        (plugin / "plugin.py").write_text(_v3_source("ordinary", version="2.0.0"))
        failure = ValueError("fixture compilation failed")

        def fail_compile(*args, **kwargs):
            raise failure

        monkeypatch.setattr(manager._snapshot_compiler, "compile", fail_compile)
        with pytest.raises(ValueError, match="fixture compilation failed") as caught:
            await manager.prepare_candidate("ordinary")
        assert caught.value is failure
        assert manager.current_snapshot is stable
        assert manager.prepared_generation("ordinary") is None
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_boot_failure_never_publishes_a_smaller_plugin_selection(tmp_path: Path):
    """不能删除失败插件后把剩余插件伪装成选中的 stable 组合。"""
    _write_plugin(tmp_path / "plugins", "good", _v3_source("good"))
    _write_plugin(tmp_path / "plugins", "broken", _v3_source(
        "broken", body="    raise ValueError('cannot initialize')\n",
    ))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="拓扑未就绪"):
            await manager.load_all()
        assert manager.current_snapshot is None
        assert manager.generation("good") is None
        assert manager.generation("broken") is None
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_failure_is_bound_to_requested_plugin(tmp_path: Path):
    root = tmp_path / "plugins"
    _write_plugin(root, "first", _v3_source("first"))
    _write_plugin(root, "second", _v3_source("second"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()

    for name in ("first", "second"):
        original = (root / name / "plugin.py").read_text(encoding="utf-8")
        (root / name / "plugin.py").write_text(
            f"this is not valid python for {name} !!!\n", encoding="utf-8"
        )
        # 身份源码在导入前解析；失败直接传播原错误，不产生伪造的候选记录。
        with pytest.raises(ValueError, match=f"{name}/plugin.py"):
            await manager.prepare_candidate(name)
        assert manager.candidate_status(name)["candidate_state"] is None
        assert manager.generation(name) is not None
        # 完整选择要求全部插件身份可解析；恢复后再验证下一个。
        (root / name / "plugin.py").write_text(original, encoding="utf-8")

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_same_source_gets_new_generation_namespace_after_restart(tmp_path: Path):
    _write_plugin(tmp_path / "plugins", "repeat", _v3_source("repeat"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()
    first = manager.generation("repeat")
    assert first is not None

    await manager.terminate_all()
    manager = _manager(tmp_path)
    await manager.load_all()

    second = manager.generation("repeat")
    assert second is not None
    assert first.state == "retired"
    assert second.generation_id != first.generation_id
    assert second.module_path != first.module_path
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_generation_module_tree_is_removed_on_config_failure_and_terminate(
    tmp_path: Path,
):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "module_tree",
        _v3_source(
            "module_tree",
            body="    Config.model_validate(ctx.config)\n",
            exports=(
                "from pydantic import BaseModel\n"
                "from . import child\n"
                "class Config(BaseModel):\n"
                "    required: str\n"
            ),
        ),
    )
    (plugin_dir / "child.py").write_text("value = 1\n", encoding="utf-8")
    config_dir = tmp_path / "workspace" / "plugin-data" / "module_tree-builtin"
    config_dir.mkdir(parents=True)
    save_config(config_dir, {})
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)

    modules_before = set(sys.modules)
    with pytest.raises(RuntimeError, match="拓扑未就绪"):
        await manager.load_all()
    assert manager.current_snapshot is None
    assert not any(name.startswith("_akashic_") for name in set(sys.modules) - modules_before)

    save_config(config_dir, {"required": "ok"})
    await manager.load_all()
    generation = manager.generation("module_tree")
    assert generation is not None
    assert f"{generation.module_path}.child" in sys.modules
    child = importlib.import_module(generation.module_path + ".child")
    assert child.value == 1
    assert child is generation.instance.module.child

    await manager.terminate_all()
    assert generation.module_path not in sys.modules
    assert not any(name.startswith(generation.module_path + ".") for name in sys.modules)


@pytest.mark.asyncio
async def test_source_revision_includes_helper_changes(tmp_path: Path):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "revision",
        _v3_source("revision", exports="from . import helper\n"),
    )
    helper = plugin_dir / "helper.py"
    helper.write_text("value = 1\n", encoding="utf-8")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("revision")
    assert active is not None

    helper.write_text("value = 2\n", encoding="utf-8")
    prepared = await manager.prepare_candidate("revision")

    assert prepared is not None
    assert prepared.source_revision != active.source_revision
    await manager.discard_prepared("revision")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_declared_paths_cannot_escape_plugin_root(tmp_path: Path):
    _copy_assets_provider(tmp_path)
    outside = tmp_path / "plugins" / "outside" / "skill"
    outside.mkdir(parents=True)
    (outside / "SKILL.md").write_text("# outside\n", encoding="utf-8")
    _write_plugin(
        tmp_path / "plugins",
        "escaped",
        _asset_source("escaped", "../outside"),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)

    with pytest.raises(RuntimeError, match="插件组合拓扑未就绪"):
        await manager.load_all()

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_source_symlink_cannot_escape_plugin_root(tmp_path: Path):
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "linked_source",
        _v3_source("linked_source", exports="from . import helper\n"),
    )
    (plugin_dir / "helper.py").symlink_to(outside)
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)

    try:
        with pytest.raises(RuntimeError, match="源码符号链接.*越界"):
            await manager.load_all()
        assert manager.current_snapshot is None
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_ignores_stale_bytecode_for_root_and_helper(tmp_path: Path):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "fresh_source",
        _v3_source(
            "fresh_source",
            version="release-a",
            exports="from . import helper\nhelper_value = helper.VALUE\n",
        ),
    )
    plugin_file = plugin_dir / "plugin.py"
    helper_file = plugin_dir / "helper.py"
    helper_file.write_text("VALUE = 'release-a'\n", encoding="utf-8")
    plugin_stat = plugin_file.stat()
    helper_stat = helper_file.stat()
    py_compile.compile(str(plugin_file), doraise=True)
    py_compile.compile(str(helper_file), doraise=True)
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()

    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace("release-a", "release-b"),
        encoding="utf-8",
    )
    helper_file.write_text("VALUE = 'release-b'\n", encoding="utf-8")
    os.utime(plugin_file, ns=(plugin_stat.st_atime_ns, plugin_stat.st_mtime_ns))
    os.utime(helper_file, ns=(helper_stat.st_atime_ns, helper_stat.st_mtime_ns))

    prepared = await manager.prepare_candidate("fresh_source")

    assert prepared is not None
    assert prepared.instance.version == "release-b"
    assert prepared.instance.module.helper_value == "release-b"
    await manager.discard_prepared("fresh_source")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_assets_provider_leaves_skill_duplicates_to_standard_tools(tmp_path: Path):
    _copy_assets_provider(tmp_path)
    first_dir = _write_plugin(
        tmp_path / "plugins",
        "first_skills",
        _asset_source("first_skills", "skills"),
    )
    first_skill = first_dir / "skills" / "shared"
    first_skill.mkdir(parents=True)
    (first_skill / "SKILL.md").write_text(
        "---\ndescription: first\n---\nfirst\n", encoding="utf-8"
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path, workspace=tmp_path / "workspace")
    await manager.load_all()
    first = manager.generation("first_skills")
    assert first is not None
    first_asset = next(
        asset
        for asset in await _read_assets(manager)
        if asset.category == "skills" and asset.owner_id == "first_skills"
    )
    assert (first_asset.root_dir / "shared" / "SKILL.md").is_file()

    second_dir = _write_plugin(
        tmp_path / "plugins",
        "second_skills",
        _asset_source("second_skills", "skills"),
    )
    second_skill = second_dir / "skills" / "shared"
    second_skill.mkdir(parents=True)
    (second_skill / "SKILL.md").write_text(
        "---\ndescription: second\n---\nsecond\n", encoding="utf-8"
    )

    assert await manager.prepare_candidate("second_skills") is not None
    publication = await manager.publish_prepared("second_skills")
    assert publication["publication_state"] == "committed"

    second = manager.generation("second_skills")
    assert second is not None
    second_asset = next(
        asset
        for asset in await _read_assets(manager)
        if asset.category == "skills" and asset.owner_id == "second_skills"
    )
    with pytest.raises(RuntimeError, match="Skill 名称重复"):
        SkillCatalogParser(capability_checker=None).parse(
            (
                InstalledAsset("first_skills", "skills", first_dir / "skills"),
                InstalledAsset("second_skills", "skills", second_dir / "skills"),
            )
        )
    assert (second_asset.root_dir / "shared" / "SKILL.md").is_file()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_skill_catalog_freezes_generation_and_ignores_old_root_link(
    tmp_path: Path,
):
    _copy_assets_provider(tmp_path)
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "skill_reload",
        _asset_source("skill_reload", "skills-a"),
    )
    v1_skill = plugin_dir / "skills-a" / "shared"
    v1_skill.mkdir(parents=True)
    (v1_skill / "SKILL.md").write_text(
        "---\ndescription: release a\n---\nbody a\n", encoding="utf-8"
    )
    workspace = tmp_path / "workspace"
    workspace_skill = workspace / "skills" / "personal"
    workspace_skill.mkdir(parents=True)
    (workspace_skill / "SKILL.md").write_text(
        "---\ndescription: workspace one\n---\nworkspace body a\n", encoding="utf-8"
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path, workspace=workspace)
    await manager.load_all()
    active = manager.generation("skill_reload")
    assert active is not None
    active_asset = next(
        asset
        for asset in await _read_assets(manager)
        if asset.category == "skills"
    )
    active_root = active_asset.root_dir / "shared"
    assert active_root.is_dir()
    assert active_root.joinpath("SKILL.md").read_text(encoding="utf-8").endswith(
        "body a\n"
    )

    release_b_skill = plugin_dir / "skills-b" / "shared"
    release_b_skill.mkdir(parents=True)
    (release_b_skill / "SKILL.md").write_text(
        "---\ndescription: release b\n---\nbody b\n", encoding="utf-8"
    )
    (plugin_dir / "plugin.py").write_text(
        _asset_source("skill_reload", "skills-b"),
        encoding="utf-8",
    )

    prepared = await manager.prepare_candidate("skill_reload")

    assert prepared is not None
    await manager.publish_prepared("skill_reload")
    prepared_asset = next(
        asset for asset in await _read_assets(manager) if asset.category == "skills"
    )
    prepared_root = prepared_asset.root_dir / "shared"
    assert prepared_root.joinpath("SKILL.md").read_text(encoding="utf-8").endswith(
        "body b\n"
    )
    assert active_root != prepared_root
    assert (active_root / "SKILL.md").read_text().endswith("body a\n")
    await manager.terminate_all()


def _installed_snapshot_source(
    version: str,
    *,
    skills: bool = False,
) -> str:
    if skills:
        return _asset_source("installed_snapshot", version=version)
    return _v3_source("installed_snapshot", version=version)


@pytest.mark.asyncio
async def test_disabled_installed_plugin_is_not_part_of_boot_selection(tmp_path: Path) -> None:
    """禁用插件不进入装配；另一个选中的插件仍完整启动。"""
    plugin_base, _ = _write_installed_artifact(
        tmp_path, "1.0.0-disabled",
        _v3_source("installed_snapshot", body="    raise RuntimeError('must not start')\n"),
    )
    pointer = ArtifactPointer(".artifacts/1.0.0-disabled")
    write_pointers(plugin_base, stable=pointer, latest=pointer)
    write_plugin_manifest(
        {"installed_snapshot@lab": False}, plugins_home=tmp_path / "home",
    )
    _write_plugin(tmp_path / "plugins", "selected", _v3_source("selected"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        assert manager.current_snapshot is not None
        assert set(manager.current_snapshot.generations) == {"selected"}
        assert manager.generation("installed_snapshot@lab") is None
        assert read_pointer(plugin_base, "stable") == pointer
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_installed_candidate_requires_explicit_promote_or_discard(
    tmp_path: Path,
) -> None:
    plugin_base, stable_root = _write_installed_artifact(
        tmp_path, "1.0.0-aaaa", _installed_snapshot_source("release-a")
    )
    _, latest_root = _write_installed_artifact(
        tmp_path, "2.0.0-bbbb", _installed_snapshot_source("release-b")
    )
    _, _ = _write_installed_artifact(
        tmp_path, "3.0.0-cccc", _installed_snapshot_source("release-c")
    )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    next_pointer = ArtifactPointer(".artifacts/3.0.0-cccc")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"installed_snapshot@lab": True}, plugins_home=tmp_path / "home"
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    stable_generation = manager.generation("installed_snapshot@lab")
    stable_snapshot = manager.current_snapshot
    assert stable_generation is not None and stable_snapshot is not None
    assert stable_generation.instance.version == "release-a"

    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    result = (await manager.reconcile_changed())[0]
    candidate = manager.ready_candidate
    assert result["publication_state"] == "latest_ready"
    assert candidate is not None and candidate.instance.version == "release-b"
    assert manager.generation("installed_snapshot@lab") is stable_generation
    assert manager.current_snapshot is stable_snapshot
    stable_lease = manager.snapshot_store.lease()
    latest_lease = manager.snapshot_store.lease(selector="latest")
    assert (
        stable_lease.snapshot.generations["installed_snapshot@lab"].instance.version
        == "release-a"
    )
    assert (
        latest_lease.snapshot.generations["installed_snapshot@lab"].instance.version
        == "release-b"
    )
    await stable_lease.release()
    await latest_lease.release()

    discarded = await manager.drop_candidate("installed_snapshot@lab")
    assert discarded["publication_state"] == "discarded"
    # promote/drop 不再写 per-plugin artifact 指针；stable 运行选择由 selection journal 承担。
    assert manager.generation("installed_snapshot@lab").instance.version == "release-a"  # type: ignore[union-attr]
    assert manager.current_snapshot is stable_snapshot
    assert manager.ready_candidate is None
    assert not latest_root.samefile(stable_root)

    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
    promoted = await manager.switch_ready("installed_snapshot@lab")
    assert promoted["publication_state"] == "promoted"
    assert manager.generation("installed_snapshot@lab").instance.version == "release-b"  # type: ignore[union-attr]

    write_pointers(plugin_base, stable=latest_pointer, latest=next_pointer)
    assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
    await manager.switch_ready("installed_snapshot@lab")
    assert manager.generation("installed_snapshot@lab").instance.version == "release-c"  # type: ignore[union-attr]
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_installed_promotion_uses_fixed_assets_without_touching_workspace_skills(tmp_path: Path) -> None:
    _install_assets_provider(tmp_path)
    plugin_base, stable_root = _write_installed_artifact(
        tmp_path, "1.0.0-aaaa", _installed_snapshot_source("release-a", skills=True)
    )
    _, candidate_root = _write_installed_artifact(
        tmp_path, "2.0.0-bbbb", _installed_snapshot_source("release-b", skills=True)
    )
    _write_installed_skill(stable_root, "shared", "stable body\n")
    _write_installed_skill(candidate_root, "shared", "candidate body\n")
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    candidate_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    workspace = tmp_path / "workspace"
    personal = workspace / "skills" / "shared"
    personal.mkdir(parents=True)
    (personal / "SKILL.md").write_bytes(b"user-owned bytes")
    legacy = workspace / "skills" / "old-link"
    legacy.symlink_to(stable_root / "skills" / "shared")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = PluginManager([], event_bus=EventBus(), workspace=workspace,
                            installed_cache_root=tmp_path / "home" / "cache")

    async def content():
        asset = next(item for item in await _read_assets(manager) if item.category == "skills")
        return (asset.root_dir / "shared" / "SKILL.md").read_text()

    try:
        await manager.load_all()
        assert await content() == "stable body\n"
        write_pointers(plugin_base, stable=stable_pointer, latest=candidate_pointer)
        assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
        assert await content() == "stable body\n"
        await manager.drop_candidate("installed_snapshot@lab")
        assert await content() == "stable body\n"
        write_pointers(plugin_base, stable=stable_pointer, latest=candidate_pointer)
        assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
        assert (await manager.switch_ready("installed_snapshot@lab"))["publication_state"] == "promoted"
        assert await content() == "candidate body\n"
    finally:
        await manager.terminate_all()
    assert (personal / "SKILL.md").read_bytes() == b"user-owned bytes"
    assert legacy.readlink() == stable_root / "skills" / "shared"
    assert not (workspace / "runtime" / "plugin-skill-links.json").exists()


@pytest.mark.asyncio
async def test_workspace_skill_name_does_not_block_plugin_promotion(
    tmp_path: Path,
) -> None:
    _install_assets_provider(tmp_path)
    plugin_base, _ = _write_installed_artifact(
        tmp_path, "1.0.0-aaaa", _installed_snapshot_source("release-a")
    )
    _, candidate_root = _write_installed_artifact(
        tmp_path, "2.0.0-bbbb", _installed_snapshot_source("release-b", skills=True)
    )
    _write_installed_skill(candidate_root, "personal", "candidate body\n")
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    candidate_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    workspace = tmp_path / "workspace"
    personal = workspace / "skills" / "personal"
    personal.mkdir(parents=True)
    (personal / "SKILL.md").write_text("user body\n", encoding="utf-8")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    stable_generation = manager.generation("installed_snapshot@lab")
    stable_snapshot = manager.current_snapshot
    write_pointers(plugin_base, stable=stable_pointer, latest=candidate_pointer)
    assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
    await manager.switch_ready("installed_snapshot@lab")
    assert manager.current_snapshot is not stable_snapshot
    assert manager.generation("installed_snapshot@lab") is not stable_generation
    # 晋升不写 per-plugin artifact 指针；workspace 用户资产不受影响。
    assert personal.is_dir() and not personal.is_symlink()
    assert (personal / "SKILL.md").read_text(encoding="utf-8") == "user body\n"
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_rejected_installed_candidate_keeps_latest_for_explicit_settle(
    tmp_path: Path,
) -> None:
    plugin_base, _ = _write_installed_artifact(
        tmp_path, "1.0.0-aaaa", _installed_snapshot_source("release-a")
    )
    _, _ = _write_installed_artifact(
        tmp_path,
        "2.0.0-bbbb",
        _v3_source(
            "installed_snapshot",
            version="release-b",
            body="    raise ValueError('candidate rejected during apply')\n",
        ),
    )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    write_plugin_manifest(
        {"installed_snapshot@lab": True}, plugins_home=tmp_path / "home"
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    results = await manager.reconcile_changed()
    assert results[0]["prepared_generation"] is None
    assert results[0]["preparation_state"] == "failed"
    assert "candidate rejected" in str(results[0].get("error"))
    assert manager.generation("installed_snapshot@lab").instance.version == "release-a"  # type: ignore[union-attr]
    assert read_pointer(plugin_base, "stable") == stable_pointer
    # 初始化失败不再静默回退 latest；登台指针保留，待显式 discard 结算。
    assert read_pointer(plugin_base, "latest") == latest_pointer
    # 本次登台没有 armed 更新 owner；失败如实留在结果中而不伪装回退。
    assert manager.reload_journal.armed_update_for_plugin("installed_snapshot@lab") is None
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("promoted_on_disk", [False, True])
async def test_restart_keeps_stable_when_legacy_candidate_pointers_drift(
    tmp_path: Path,
    promoted_on_disk: bool,
) -> None:
    plugin_base, stable_root = _write_installed_artifact(
        tmp_path, "1.0.0-aaaa", _installed_snapshot_source("release-a")
    )
    _, latest_root = _write_installed_artifact(
        tmp_path, "2.0.0-bbbb", _installed_snapshot_source("release-b")
    )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"installed_snapshot@lab": True}, plugins_home=tmp_path / "home"
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    await manager.terminate_all()
    # 安装指针不再决定重启选择；旧 journal 也不能自动晋升候选。
    write_pointers(
        plugin_base,
        stable=latest_pointer if promoted_on_disk else stable_pointer,
        latest=latest_pointer,
    )
    manager = PluginManager(
        plugin_dirs=[], event_bus=EventBus(), workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    tx_id = manager.reload_journal.begin(
        plugin_id="installed_snapshot@lab",
        base_snapshot_id="stable-release-a",
        generation_id="candidate-release-b",
        source_revision=_source_revision(latest_root),
        config_revision="config-release-b",
    )
    manager.reload_journal.advance(tx_id, "prepared")
    manager.reload_journal.advance(tx_id, "validating")
    manager.reload_journal.advance(tx_id, "commit_started")
    manager.reload_journal.advance(tx_id, "latest_ready")
    manager.reload_journal.advance(tx_id, "promoting")

    await manager.load_all()

    # 旧记录没有完整 selection 转换证据，保持未知，不伪造 recovered/aborted。
    assert manager.reload_journal.get(tx_id).phase == "promoting"
    assert manager.generation("installed_snapshot@lab").instance.version == "release-a"  # type: ignore[union-attr]
    assert manager.ready_candidate is None
    assert stable_root.exists()
    assert read_pointer(plugin_base, "latest") == latest_pointer
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_latest_candidate_staging_waits_for_runtime_service_start(
    tmp_path: Path,
) -> None:
    """Staging a latest-only candidate must not start the stable Root early."""

    lifecycle_exports = (
        "import asyncio\n"
        "from agent.plugin_composition import RUNTIME_STARTED\n"
        "started = asyncio.Event()\n"
        "starts = []\n"
    )
    lifecycle_body = (
        "    async def start(_event):\n"
        "        started.set()\n"
        "        starts.append('start')\n"
        "    await ctx.on(RUNTIME_STARTED, start)\n"
    )
    plugin_base, _ = _write_installed_artifact(
        tmp_path,
        "1.0.0-aaaa",
        _v3_source(
            "installed_snapshot",
            version="release-a",
            exports=lifecycle_exports,
            body=lifecycle_body,
        ),
    )
    _, _ = _write_installed_artifact(
        tmp_path,
        "2.0.0-bbbb",
        _installed_snapshot_source("release-b"),
    )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"installed_snapshot@lab": True}, plugins_home=tmp_path / "home"
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    result = await manager.reconcile_changed()

    stable = manager.generation("installed_snapshot@lab")
    assert stable is not None
    module = stable.instance.module
    started = cast(asyncio.Event, module.started)
    starts = cast(list[str], module.starts)
    assert stable.instance.version == "release-a"
    assert result[0]["publication_state"] == "latest_ready"
    assert manager.ready_candidate is not None
    # 启动已融合进提交的 closed scope；staging latest 候选不重启 stable Root。
    assert started.is_set() and starts == ["start"]

    runner = asyncio.create_task(manager.run_runtime_services())
    try:
        async with asyncio.timeout(1):
            while manager.ready_candidate is not None:
                await asyncio.sleep(0.01)
    except TimeoutError:
        pass
    assert starts == ["start"]
    runner.cancel()
    _ = await asyncio.gather(runner, return_exceptions=True)
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_snapshot_admission_waits_while_current_is_quiesced(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins", "snapshot_admission", _v3_source("snapshot_admission")
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()
    snapshot = manager.current_snapshot
    assert snapshot is not None
    held = manager.snapshot_store.lease()
    quiescing = asyncio.create_task(manager.snapshot_store.quiesce_current())
    waiting = asyncio.create_task(manager.snapshot_store.acquire())
    await asyncio.sleep(0)
    assert not quiescing.done()
    assert not waiting.done()

    await held.release()
    assert await quiescing is snapshot
    assert not waiting.done()
    await manager.snapshot_store.resume(snapshot)
    admitted = await waiting
    assert admitted.snapshot is snapshot
    await admitted.release()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_snapshot_cleanup_failure_requires_another_explicit_close() -> None:
    """失败资源留在原快照，不在同次关闭中自动重放。"""
    attempts = 0

    async def drain(snapshot):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("still open")

    store = RuntimeSnapshotStore(drain)
    snapshot = RuntimeSnapshotCompiler().compile({})
    store.install(snapshot)
    with pytest.raises(RuntimeError, match="drain 失败"):
        await store.close()
    assert attempts == 1
    assert snapshot.snapshot_id in store.retained_snapshot_ids
    await store.close()
    assert attempts == 2
    assert store.retained_snapshot_ids == ()


@pytest.mark.asyncio
async def test_snapshot_cleanup_join_survives_repeated_caller_cancel() -> None:
    """取消等待者不能取消实际 snapshot 资源回收。"""
    entered, release = asyncio.Event(), asyncio.Event()
    closed = []

    async def drain(snapshot):
        entered.set()
        await release.wait()
        closed.append(snapshot.snapshot_id)

    store = RuntimeSnapshotStore(drain)
    snapshot = RuntimeSnapshotCompiler().compile({})
    store.install(snapshot)
    closing = asyncio.create_task(store.close())
    await entered.wait()
    closing.cancel()
    asyncio.get_running_loop().call_soon(closing.cancel)
    asyncio.get_running_loop().call_soon(release.set)
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert closed == [snapshot.snapshot_id]
    assert store.retained_snapshot_ids == ()


@pytest.mark.asyncio
async def test_runtime_snapshot_lease_commit_and_abort(tmp_path: Path) -> None:
    _write_plugin(tmp_path / "plugins", "snapshot", _v3_source("snapshot"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot")
    prepared = await manager.prepare_candidate("snapshot")
    installed = manager.current_snapshot
    assert active is not None and prepared is not None and installed is not None
    compiler = RuntimeSnapshotCompiler()
    v1 = compiler.compile({"snapshot": active})
    next_snapshot = compiler.compile(
        {"snapshot": prepared}
    )
    drained: list[str] = []

    async def on_drained(snapshot: RuntimeSnapshot) -> None:
        drained.append(snapshot.snapshot_id)

    store = RuntimeSnapshotStore(on_drained)
    store.install(v1)
    v1_lease = store.lease()
    transaction = store.begin_publish(next_snapshot)
    with pytest.raises(RuntimeError, match="不可租用"):
        store.lease(next_snapshot.snapshot_id)
    await store.abort(transaction)
    assert store.current is v1
    assert drained == [next_snapshot.snapshot_id]
    await v1_lease.release()
    next_snapshot = compiler.compile(
        {"snapshot": prepared}
    )
    held_v1 = store.lease()
    await store.commit(store.begin_publish(next_snapshot))
    assert store.current is next_snapshot
    with pytest.raises(RuntimeError, match="不可租用"):
        store.lease(v1.snapshot_id)
    await held_v1.release()
    await store.retry_drains()
    assert drained == [next_snapshot.snapshot_id, v1.snapshot_id]
    await store.close()
    await manager.discard_prepared("snapshot")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_runtime_snapshot_latest_closes_before_fresh_formal_publication(
    tmp_path: Path,
) -> None:
    from agent.plugin_composition import CompositionRoot

    compiler = RuntimeSnapshotCompiler()
    drained: list[str] = []
    closed: list[str] = []

    async def build(revision: str) -> RuntimeSnapshot:
        root = CompositionRoot(revision)
        async def apply(ctx):
            await ctx.effect(lambda: lambda: closed.append(revision))
        await root.mount(apply, name="snapshot_selector")
        return compiler.compile({}, snapshot_revision=revision, composition_root=root)

    stable = await build("stable")
    latest = await build("latest")

    async def on_drained(snapshot: RuntimeSnapshot) -> None:
        await snapshot.composition_root.dispose()
        drained.append(snapshot.snapshot_id)

    store = RuntimeSnapshotStore(on_drained)
    store.install(stable)
    latest_transaction = store.begin_publish(latest)
    await store.commit_latest(latest_transaction)
    stable_lease = store.lease()
    latest_lease = store.lease(selector="latest")
    assert stable_lease.snapshot is stable
    assert latest_lease.snapshot is latest
    with pytest.raises(RuntimeError, match="等待 promote/discard"):
        store.begin_publish(
            compiler.compile({}, snapshot_revision="next")
        )
    store.pause_candidate_admission(latest)
    await latest_lease.release()
    await store.wait_for_no_leases(latest)
    store.seal_candidate_validation(latest)
    with pytest.raises(RuntimeError, match="publication target 已失效"):
        store.retain_publication_target(latest_transaction)
    assert latest.lease_count == 0
    await store.discard_latest(latest)
    assert drained == [latest.snapshot_id]
    assert closed == ["latest"]
    assert store.current is stable
    assert stable_lease.snapshot is stable
    formal = await build("fresh-formal")
    transaction = store.begin_publish(formal)
    publication_lease = store.retain_publication_target(transaction)
    assert publication_lease.snapshot is formal
    await store.commit_provisional(transaction)
    assert store.current is stable
    provisional_lease = store.retain_publication_target(transaction)
    assert provisional_lease.snapshot is formal
    await provisional_lease.release()
    await publication_lease.release()
    await store.finalize_provisional(transaction)
    with pytest.raises(RuntimeError, match="publication target 已失效"):
        store.retain_publication_target(transaction)
    assert formal.lease_count == 0
    assert transaction.previous is stable
    assert store.current is formal
    assert formal is not latest
    assert formal.composition_root is not latest.composition_root
    assert drained == [latest.snapshot_id]
    await stable_lease.release()
    await store.retry_drains()
    assert drained == [latest.snapshot_id, stable.snapshot_id]
    await store.close()
    assert closed == ["latest", "stable", "fresh-formal"]


@pytest.mark.asyncio
async def test_runtime_snapshot_discard_keeps_stable_and_waits_for_latest_lease(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins", "snapshot_discard", _v3_source("snapshot_discard")
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot_discard")
    prepared = await manager.prepare_candidate("snapshot_discard")
    assert active is not None and prepared is not None
    compiler = RuntimeSnapshotCompiler()
    stable = compiler.compile({"snapshot_discard": active}, snapshot_revision="stable")
    latest = compiler.compile(
        {"snapshot_discard": prepared}, snapshot_revision="latest"
    )
    drained: list[str] = []

    async def on_drained(snapshot: RuntimeSnapshot) -> None:
        drained.append(snapshot.snapshot_id)

    store = RuntimeSnapshotStore(on_drained)
    store.install(stable)
    await store.commit_latest(store.begin_publish(latest))
    latest_lease = store.lease(selector="latest")
    discarding = asyncio.create_task(store.discard_latest())
    await asyncio.sleep(0)
    assert not discarding.done()
    stable_lease = store.lease()
    await latest_lease.release()
    assert await discarding is latest
    assert store.current is stable
    assert store.latest is stable
    await stable_lease.release()
    await store.close()
    await manager.discard_prepared("snapshot_discard")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_reconcile_changed_adds_and_removes_discovered_plugin(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "anchor", _v3_source("anchor"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()
    added_dir = _write_plugin(plugins, "added", _v3_source("added"))

    added = await manager.reconcile_changed()
    assert added[0]["publication_state"] == "committed"
    assert manager.generation("added") is not None
    shutil.rmtree(added_dir)
    removed = await manager.reconcile_changed()
    assert removed[0]["publication_state"] == "disabled"
    assert manager.generation("added") is None
    assert manager.current_snapshot is not None
    assert set(manager.current_snapshot.generations) == {"anchor"}
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_runtime_start_owner_rejects_publication_until_started_scope_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启动尚未结束时更新立即 busy，不排队退休正在启动的 Root。"""

    source = _v3_source(
        "runner_race",
        exports=(
            "import asyncio\n"
            "from agent.plugin_composition import RUNTIME_STARTED, RUNTIME_STOPPING\n"
            "started = asyncio.Event()\n"
            "allow_finish = asyncio.Event()\n"
            "stopped = asyncio.Event()\n"
        ),
        body=(
            "    async def start(_event):\n"
            "        async with ctx.runtime_scope():\n"
            "            started.set()\n"
            "            await allow_finish.wait()\n"
            "    async def stop(_event):\n"
            "        stopped.set()\n"
            "    await ctx.on(RUNTIME_STARTED, start)\n"
            "    await ctx.on(RUNTIME_STOPPING, stop)\n"
        ),
    )
    plugin_dir = _write_plugin(tmp_path / "plugins", "runner_race", source)
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    from agent.plugins._operation import OperationBusyError
    entered, allow_start = asyncio.Event(), asyncio.Event()
    deactivate_entered = asyncio.Event()
    real_start = manager._start_closed_runtime_snapshot
    real_deactivate = manager._deactivate_plugin
    modules: dict[str, object] = {}

    async def blocked_start(lease):
        generation = lease.snapshot.generations.get("runner_race")
        if generation is None:
            return await real_start(lease)
        modules["plugin"] = generation.instance.module
        entered.set()
        await allow_start.wait()
        await real_start(lease)

    async def observed_deactivate(plugin_id: str):
        deactivate_entered.set()
        return await real_deactivate(plugin_id)

    monkeypatch.setattr(manager, "_start_closed_runtime_snapshot", blocked_start)
    monkeypatch.setattr(manager, "_deactivate_plugin", observed_deactivate)
    load = asyncio.create_task(manager.load_all())
    await entered.wait()
    module = modules["plugin"]
    shutil.rmtree(plugin_dir)
    # fused closed start 持有 load_all 操作；候选尚未开放，更新立即 busy
    with pytest.raises(OperationBusyError):
        await manager.reconcile_changed()
    assert not deactivate_entered.is_set()
    assert manager.current_snapshot is None
    allow_start.set()
    await module.started.wait()
    with pytest.raises(OperationBusyError):
        await manager.reconcile_changed()
    module.allow_finish.set()
    await load
    old_snapshot = manager.current_snapshot
    assert old_snapshot is not None
    result = await manager.reconcile_changed()
    assert result[0]["publication_state"] == "disabled"
    await module.stopped.wait()
    await manager.snapshot_store.wait_for_snapshot_drained(old_snapshot)
    assert old_snapshot.lease_count == 0
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_plugin_watcher_reloads_v3_source_without_signal(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "watched",
        _v3_source("watched", version="release-a"),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()
    baseline_revision = await asyncio.to_thread(manager.watch_revision)
    watcher = PluginWatcher(
        manager, baseline_revision=baseline_revision, interval_seconds=0.01
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    (plugin_dir / "plugin.py").write_text(
        _v3_source("watched", version="release-b"), encoding="utf-8"
    )
    for _ in range(100):
        generation = manager.generation("watched")
        if generation is not None and generation.instance.version == "release-b":
            break
        await asyncio.sleep(0.01)
    generation = manager.generation("watched")
    assert generation is not None and generation.instance.version == "release-b"
    watcher.stop()
    await task
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_plugin_toggle_changes_assets_without_creating_workspace_projections(tmp_path: Path) -> None:
    _copy_assets_provider(tmp_path)
    plugin_dir = _write_plugin(tmp_path / "plugins", "computer", _asset_source(
        "computer",
    ))
    skill_dir = plugin_dir / "skills" / "opencli"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# OpenCLI\n")
    write_plugin_manifest({"computer": True}, plugins_home=tmp_path / "home")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        assert manager.generation("computer") is not None
        write_plugin_manifest({"computer": False}, plugins_home=tmp_path / "home")
        await manager.reconcile_changed()
        assert manager.generation("computer") is None
        write_plugin_manifest({"computer": True}, plugins_home=tmp_path / "home")
        await manager.reconcile_changed()
        assert manager.generation("computer") is not None
        assert not (tmp_path / "workspace" / "skills").exists()
        assert not (tmp_path / "workspace" / "drift" / "skills").exists()
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_plugin_watcher_scans_files_outside_event_loop_thread() -> None:
    event_loop_thread = threading.get_ident()

    class Manager:
        def __init__(self) -> None:
            self.scan_threads: list[int] = []

        def watch_revision(self) -> str:
            self.scan_threads.append(threading.get_ident())
            return "stable"

        async def reconcile_changed(self) -> list[dict[str, object]]:
            return []

    manager = Manager()
    watcher = PluginWatcher(
        cast(PluginManager, manager), baseline_revision="stable", interval_seconds=0.01
    )
    task = asyncio.create_task(watcher.run())
    for _ in range(100):
        if manager.scan_threads:
            break
        await asyncio.sleep(0.01)
    watcher.stop()
    await task
    assert manager.scan_threads
    assert all(thread_id != event_loop_thread for thread_id in manager.scan_threads)


@pytest.mark.asyncio
async def test_plugin_watcher_retries_failed_reconcile_and_notifies() -> None:
    class Manager:
        def __init__(self) -> None:
            self.revision = "broken"
            self.allow_reconcile = False
            self.calls = 0
            self.failed = asyncio.Event()
            self.recovered = asyncio.Event()

        def watch_revision(self) -> str:
            return self.revision

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            if not self.allow_reconcile:
                self.failed.set()
                raise RuntimeError("callback failed")
            self.recovered.set()
            return []

    manager = Manager()
    notified = asyncio.Event()

    async def notify() -> None:
        notified.set()

    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
        after_reconcile=notify,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.wait_for(manager.failed.wait(), timeout=1)
    for _ in range(100):
        if manager.calls >= 3:
            break
        await asyncio.sleep(0.01)
    assert manager.calls == 3
    manager.allow_reconcile = True
    manager.revision = "fixed"
    await asyncio.wait_for(manager.recovered.wait(), timeout=1)
    await asyncio.wait_for(notified.wait(), timeout=1)
    watcher.stop()
    await task
    assert manager.calls == 4


@pytest.mark.asyncio
async def test_plugin_watcher_cancellation_marks_stopped() -> None:
    class Manager:
        def watch_revision(self) -> str:
            return "stable"

        async def reconcile_changed(self) -> list[dict[str, object]]:
            await asyncio.Event().wait()
            return []

    watcher = PluginWatcher(
        cast(PluginManager, Manager()),
        baseline_revision="stable",
        interval_seconds=0.01,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await watcher.wait_stopped()


@pytest.mark.asyncio
async def test_dashboard_routes_follow_snapshot_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _copy_ui_provider(tmp_path)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "home"))
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_dashboard",
        _ui_source("snapshot_dashboard"),
    )
    (plugin_dir / "web_module.js").write_text(
        "export function activate() { return () => {}; }\n",
        encoding="utf-8",
    )

    (plugin_dir / "values.py").write_text(
        "class Value:\n    def __init__(self, text): self.text = text\n"
    )

    def write_dashboard(version: str) -> None:
        (plugin_dir / "plugin.py").write_text(_ui_source(
            "snapshot_dashboard",
            body=f"    from .values import Value\n    await ctx.provide(ServiceKey('fixture.dashboard-value'), Value('{version}'))\n",
        ))
        (plugin_dir / "dashboard.py").write_text(
            "from agent.plugin_composition import ServiceKey\n"
            "from .values import Value\n"
            "VALUE = ServiceKey('fixture.dashboard-value')\n"
            "inject = (VALUE,)\n"
            "def register(app, context):\n"
            "    @app.get('/api/dashboard/undeclared')\n"
            "    async def undeclared(): return context.require(ServiceKey('core.message-writers'))\n"
            "    @app.get('/api/dashboard/snapshot-version')\n"
            "    async def version():\n"
            "        value = context.require(VALUE)\n"
            "        assert isinstance(value, Value)\n"
            "        return {'version': value.text}\n"
            "    class Closeable:\n"
            "        def close(self):\n"
            f"            (context.data_root / 'dashboard-{version}-closed').write_text('closed')\n"
            "    return Closeable()\n",
            encoding="utf-8",
        )

    write_dashboard("release-a")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    assert old_snapshot is not None
    old_generation = old_snapshot.generations["snapshot_dashboard"]
    old_catalog = old_snapshot.composition_root.context.require(UI).catalog()
    assert old_catalog is not None
    old_headers = {
        "X-Akashic-Web-Snapshot": old_snapshot.snapshot_id,
        "X-Akashic-Web-Catalog": old_catalog.identity,
        "X-Akashic-Web-Module": "snapshot_dashboard",
        "X-Akashic-Web-Generation": old_generation.generation_id,
    }
    old_lease = manager.snapshot_store.lease()
    app = create_dashboard_app(
        tmp_path / "workspace",
        plugin_manager=manager,
    )
    client = TestClient(app)
    old_binding = old_snapshot.composition_root.context.require(UI).bindings()[0]
    assert client.get("/api/dashboard/snapshot-version").json() == {
        "code": "forbidden_contract"
    }
    assert (
        client.get(
            "/api/dashboard/snapshot-version",
            headers=old_headers,
        ).status_code
        == 200
    )
    assert client.get(
        "/api/dashboard/snapshot-version",
        headers={"Sec-Fetch-Site": "same-origin"},
    ).json() == {"code": "forbidden_contract"}
    write_dashboard("release-b")
    assert await manager.prepare_candidate("snapshot_dashboard") is not None
    publication = asyncio.create_task(manager.publish_prepared("snapshot_dashboard"))
    while old_snapshot.accepting_leases:
        await asyncio.sleep(0)
    assert not publication.done()
    await old_lease.release()
    await publication
    assert client.get("/api/dashboard/snapshot-version").json() == {
        "code": "forbidden_contract"
    }
    caplog.clear()
    caplog.set_level("WARNING", logger="agent.plugins.dashboard_host")
    stale = client.get(
        "/api/dashboard/snapshot-version",
        headers=old_headers,
    )
    assert stale.json() == {"code": "stale_catalog"}
    assert stale.headers["x-akashic-web-stale"] == "1"
    assert old_snapshot.snapshot_id not in caplog.text
    assert old_catalog.identity not in caplog.text
    assert old_generation.generation_id not in caplog.text
    new_snapshot = manager.current_snapshot
    assert new_snapshot is not None and new_snapshot.composition_root.context.require(UI).catalog() is not None
    new_generation = new_snapshot.generations["snapshot_dashboard"]
    new_headers = {
        "X-Akashic-Web-Snapshot": new_snapshot.snapshot_id,
        "X-Akashic-Web-Catalog": new_snapshot.composition_root.context.require(UI).catalog().identity,
        "X-Akashic-Web-Module": "snapshot_dashboard",
        "X-Akashic-Web-Generation": new_generation.generation_id,
    }
    assert client.get(
        "/api/dashboard/snapshot-version",
        headers=new_headers,
    ).json() == {"version": "release-b"}
    with pytest.raises(CompositionError, match="未声明能力"):
        client.get("/api/dashboard/undeclared", headers=new_headers)
    with pytest.raises(RuntimeError, match="实际 runtime scope"):
        TestClient(old_binding.app).get("/api/dashboard/snapshot-version")
    import httpx
    async with lease_runtime_snapshot(manager.snapshot_store):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=old_binding.app), base_url="http://fixture") as old_client:
            with pytest.raises(RuntimeError, match="当前 runtime scope"):
                await old_client.get("/api/dashboard/snapshot-version")
    await manager.snapshot_store.retry_drains()
    assert (old_generation.data_dir / "dashboard-release-a-closed").exists()
    assert old_generation.scope.closed
    client.close()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_initial_web_module_is_not_served_without_its_dashboard_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_ui_provider(tmp_path)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "home"))
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "paired_web",
        _ui_source("paired_web"),
    )
    (plugin_dir / "web_module.js").write_text(
        "export function activate() { return () => {}; }\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "raise RuntimeError('paired API broken')\n",
        encoding="utf-8",
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()

    with pytest.raises(RuntimeError, match="paired API broken"):
        create_dashboard_app(tmp_path / "workspace", plugin_manager=manager)

    await manager.terminate_all()


def test_dashboard_rejects_custom_path_convertor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CustomConvertor(StringConvertor):
        regex = "(?:x|z)"

    monkeypatch.setitem(CONVERTOR_TYPES, "custom_gate", CustomConvertor())
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/api/dashboard/{value:custom_gate}")
    def route() -> dict[str, bool]:
        return {"ok": True}

    with pytest.raises(RuntimeError, match="内建 path converter"):
        _plugin_routes(app.routes)


@pytest.mark.parametrize("wildcard_methods", [None, set()])
def test_dashboard_treats_missing_methods_as_wildcard(
    wildcard_methods: set[str] | None,
) -> None:
    core_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    plugin_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @core_app.api_route("/api/dashboard/{rest:path}", methods=["GET"])
    def core_route() -> dict[str, bool]:
        return {"core": True}

    @plugin_app.get("/api/dashboard/sessions")
    def plugin_route() -> dict[str, bool]:
        return {"plugin": True}

    core_routes = _plugin_routes(core_app.routes)
    core_routes[0].methods = wildcard_methods
    binding = DashboardBinding(
        plugin_id="wildcard",
        app=plugin_app,
        routes=_plugin_routes(plugin_app.routes),
    )
    with pytest.raises(RuntimeError, match="dashboard route 冲突"):
        _require_routes_available(binding, list(core_routes))


def test_dashboard_allows_narrow_route_before_path_catchall() -> None:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/api/dashboard/sessions/{key:path}/messages")
    def messages() -> dict[str, bool]:
        return {"messages": True}

    @app.get("/api/dashboard/sessions/{key:path}")
    def session() -> dict[str, bool]:
        return {"session": True}

    binding = DashboardBinding(
        plugin_id="ordered-paths",
        app=app,
        routes=_plugin_routes(app.routes),
    )
    _require_routes_available(binding, [])

    binding = dataclasses.replace(binding, routes=tuple(reversed(binding.routes)))
    with pytest.raises(RuntimeError, match="dashboard route 冲突"):
        _require_routes_available(binding, [])


def test_dashboard_allows_http_and_websocket_on_the_same_path() -> None:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/api/dashboard/live")
    def live_status() -> dict[str, bool]:
        return {"ready": True}

    @app.websocket("/api/dashboard/live")
    async def live_socket() -> None:
        return None

    binding = DashboardBinding(
        plugin_id="two-protocols",
        app=app,
        routes=_plugin_routes(app.routes),
    )
    _require_routes_available(binding, [])


@pytest.mark.asyncio
async def test_dashboard_websocket_uses_exact_generation_and_closes_for_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _copy_ui_provider(tmp_path)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "home"))
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_socket",
        _ui_source("snapshot_socket"),
    )
    (plugin_dir / "web_module.js").write_text(
        "export function activate() { return () => {}; }\n",
        encoding="utf-8",
    )
    sibling_dir = _write_plugin(
        tmp_path / "plugins",
        "socket_sibling",
        _ui_source("socket_sibling", dashboard=False),
    )
    (sibling_dir / "web_module.js").write_text(
        "export function activate() { return () => {}; }\n",
        encoding="utf-8",
    )

    def write_dashboard(version: str) -> None:
        (plugin_dir / "dashboard.py").write_text(
            "from fastapi import WebSocket, WebSocketDisconnect\n"
            "def register(app, context):\n"
            "    @app.websocket('/api/dashboard/snapshot-socket')\n"
            "    async def snapshot_socket(socket: WebSocket):\n"
            "        await socket.accept(subprotocol='binary')\n"
            "        try:\n"
            "            while True:\n"
            "                value = await socket.receive_bytes()\n"
            f"                await socket.send_bytes(b'{version}:' + value)\n"
            "        except WebSocketDisconnect:\n"
            "            return\n",
            encoding="utf-8",
        )

    def socket_path(
        snapshot: RuntimeSnapshot,
        module: str = "snapshot_socket",
    ) -> str:
        catalog = snapshot.composition_root.context.require(UI).catalog()
        assert catalog is not None
        generation = snapshot.generations[module]
        query = urlencode(
            {
                "__akashic_web_snapshot": snapshot.snapshot_id,
                "__akashic_web_catalog": catalog.identity,
                "__akashic_web_module": module,
                "__akashic_web_generation": generation.generation_id,
            }
        )
        return f"/api/dashboard/snapshot-socket?{query}"

    write_dashboard("release-a")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    assert old_snapshot is not None
    app = create_dashboard_app(tmp_path / "workspace", plugin_manager=manager)

    def assert_web_identity_not_logged(snapshot: RuntimeSnapshot) -> None:
        """Keep exact Web identity values out of rejection diagnostics."""

        catalog = snapshot.composition_root.context.require(UI).catalog()
        assert catalog is not None
        identities = (
            snapshot.snapshot_id,
            catalog.identity,
            "snapshot_socket",
            "socket_sibling",
            *(item.generation_id for item in snapshot.generations.values()),
        )
        assert all(identity not in caplog.text for identity in identities)

    with TestClient(app) as client:
        with (
            pytest.raises(WebSocketDisconnect) as missing,
            client.websocket_connect(
                "/api/dashboard/snapshot-socket",
                headers={"origin": "http://testserver"},
            ),
        ):
            pass
        assert missing.value.code == 4403

        with (
            pytest.raises(WebSocketDisconnect) as cross_origin,
            client.websocket_connect(
                socket_path(old_snapshot),
                headers={"origin": "https://outside.example"},
            ),
        ):
            pass
        assert cross_origin.value.code == 4403

        caplog.clear()
        caplog.set_level("WARNING", logger="agent.plugins.dashboard_host")
        with (
            pytest.raises(WebSocketDisconnect) as sibling,
            client.websocket_connect(
                socket_path(old_snapshot, "socket_sibling"),
                headers={"origin": "http://testserver"},
            ),
        ):
            pass
        assert sibling.value.code == 4403
        assert "Web UI WebSocket plugin 身份不匹配" in caplog.text
        assert_web_identity_not_logged(old_snapshot)

        caplog.clear()
        missing_path = socket_path(old_snapshot).replace(
            "/api/dashboard/snapshot-socket",
            "/api/dashboard/missing-socket",
            1,
        )
        with (
            pytest.raises(WebSocketDisconnect) as missing_route,
            client.websocket_connect(
                missing_path,
                headers={"origin": "http://testserver"},
            ),
        ):
            pass
        assert missing_route.value.code == 4403
        assert "Web UI WebSocket 路由不存在" in caplog.text
        assert_web_identity_not_logged(old_snapshot)

        with client.websocket_connect(
            socket_path(old_snapshot),
            headers={"origin": "http://testserver"},
            subprotocols=["binary"],
        ) as live:
            live.send_bytes(b"one")
            assert live.receive_bytes() == b"release-a:one"
            write_dashboard("release-b")
            assert await manager.prepare_candidate("snapshot_socket") is not None
            publication = asyncio.create_task(
                manager.publish_prepared("snapshot_socket")
            )
            while old_snapshot.accepting_leases:
                await asyncio.sleep(0)

            def receive_restart() -> int:
                try:
                    live.receive_bytes()
                except WebSocketDisconnect as error:
                    return error.code
                raise AssertionError("old generation WebSocket stayed open")

            assert (
                await asyncio.wait_for(
                    asyncio.to_thread(receive_restart),
                    timeout=5,
                )
                == 1012
            )

        # Finish the close handshake so middleware can release its snapshot
        # lease; publication can only drain after that lifecycle boundary.
        await asyncio.wait_for(publication, timeout=5)

    await manager.snapshot_store.retry_drains()
    await manager.terminate_all()


def test_compiled_source_reuse_keeps_modules_fresh_and_observes_same_size_edits(tmp_path):
    from types import ModuleType
    from agent.plugins.importer import FreshSourceLoader
    path = tmp_path / "plugin.py"
    path.write_text("values = []\ndef value(): return 1\n")
    loader = FreshSourceLoader(path)
    first = ModuleType("first"); second = ModuleType("second")
    loader.exec_module(first); loader.exec_module(second)
    first.values.append("changed")
    assert second.values == []
    assert first.value.__globals__ is not second.value.__globals__
    stamp = path.stat()
    path.write_text("values = []\ndef value(): return 2\n")
    import os
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    third = ModuleType("third"); loader.exec_module(third)
    assert first.value() == second.value() == 1
    assert third.value() == 2
