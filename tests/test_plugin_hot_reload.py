from __future__ import annotations

import ast

from agent.plugin_composition.ui import UI, DashboardBinding

import asyncio
import dataclasses
import importlib
import os
import py_compile
import sqlite3
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlencode

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

import agent.plugins.manager as manager_module
import agent.plugins.reload_journal as reload_journal_module
from agent.plugin_composition.config_input import save_config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.convertors import CONVERTOR_TYPES, StringConvertor
from starlette.websockets import WebSocketDisconnect

from agent.plugin_composition import CompositionError, CompositionRoot, FiberState, ServiceKey
from agent.plugin_composition.assets import INSTALLED_ASSETS, InstalledAsset
from agent.plugins.artifacts import ArtifactPointer, read_pointer, write_pointers
from plugins.ui.dashboard import (
    _plugin_routes,
    _require_routes_available,
)
from agent.plugins.manager import OperationBusyError, PluginManager, _source_revision
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.selection import SelectionConflictError
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


def _write_checked_plugin(root: Path, name: str, source: str) -> Path:
    """Parse and compile a dynamic fixture before writing its source."""
    entry = root / name / "plugin.py"
    tree = ast.parse(source, filename=str(entry))
    compile(tree, str(entry), "exec")
    return _write_plugin(root, name, source)


def _peer_source(name: str, service_name: str) -> str:
    """Create a peer plugin with observable lifecycle, effect, and service state."""
    return (
        "from agent.plugin_composition import Context, ServiceKey, RUNTIME_STARTING, RUNTIME_STARTED\n"
        f"api_version = 3\nname = {name!r}\nversion = '1.0.0'\n"
        "inject = ()\n"
        f"PEER_SERVICE = ServiceKey({service_name!r})\n"
        "async def apply(ctx: Context):\n"
        "    state = {'events': [], 'effect': 0, 'cleanup': 0}\n"
        "    await ctx.on(RUNTIME_STARTING, lambda _event: state['events'].append('starting'))\n"
        "    await ctx.on(RUNTIME_STARTED, lambda _event: state['events'].append('started'))\n"
        "    async def setup():\n"
        "        state['effect'] += 1\n"
        "        async def cleanup():\n"
        "            state['cleanup'] += 1\n"
        "        return cleanup\n"
        "    await ctx.effect(setup, label='peer-effect')\n"
        "    await ctx.provide(PEER_SERVICE, state)\n"
    )


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


def _reload_journal_state(path: Path) -> tuple[object, ...]:
    """Capture journal bytes, SQL dump, schema rows, and directory state."""
    connection = sqlite3.connect(path)
    try:
        dump = tuple(connection.iterdump())
        schema = tuple(
            connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "ORDER BY type, name"
            ).fetchall()
        )
    finally:
        connection.close()
    files = tuple(
        (entry.name, entry.read_bytes())
        for entry in sorted(path.parent.iterdir())
        if entry.is_file()
    )
    return (path.exists(), path.parent.exists(), files, dump, schema)


@pytest.mark.parametrize(
    "invalid_schema",
    [False, True],
    ids=["valid-schema", "missing-required-index"],
)
def test_reload_journal_existing_schema_closes_readonly_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_schema: bool,
) -> None:
    """Existing-schema checks close their real read-only SQLite connection."""
    workspace = tmp_path / "workspace"
    created = ReloadJournal(workspace)
    database = created.path
    if invalid_schema:
        connection = sqlite3.connect(database)
        try:
            connection.execute("DROP INDEX idx_reload_events_tx")
            connection.commit()
        finally:
            connection.close()
    before = _reload_journal_state(database)

    captured: list[sqlite3.Connection] = []
    real_connect = reload_journal_module.sqlite3.connect

    def observed_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)
        captured.append(connection)
        return connection

    monkeypatch.setattr(reload_journal_module.sqlite3, "connect", observed_connect)
    try:
        if invalid_schema:
            with pytest.raises(RuntimeError, match="缺少当前索引"):
                ReloadJournal(workspace)
        else:
            ReloadJournal(workspace)
        assert len(captured) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            captured[0].execute("SELECT 1")
        assert _reload_journal_state(database) == before
    finally:
        for connection in captured:
            connection.close()


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
    """首次 null 跳过坏 source；修复未选 source 不被 watcher 偷装。"""
    plugin = _write_plugin(tmp_path / "plugins", "broken", "this is not python !!!\n")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)

    try:
        await manager.load_all()
        assert manager.generation("broken") is None
        selection = manager._selection.read()
        assert selection is not None
        assert manager._selection_components(selection) == ()
        failures = manager.plugin_status()["source_failures"]
        assert failures[0]["plugin_id"] is None
        assert failures[0]["phase"] == "identity"

        (plugin / "plugin.py").write_text(_v3_source("broken"), encoding="utf-8")
        result = await manager.reconcile_changed()
        assert result[0]["publication_state"] == "unselected_source"
        assert manager.generation("broken") is None
        assert manager.plugin_status()["source_failures"] == failures
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
    selection_before = manager._selection.read()
    added_dir = _write_plugin(plugins, "added", _v3_source("added"))

    added = await manager.reconcile_changed()
    assert added[0]["publication_state"] == "unselected_source"
    assert manager.generation("added") is None
    assert manager._selection.read() == selection_before
    shutil.rmtree(added_dir)
    removed = await manager.reconcile_changed()
    assert removed == []
    assert manager.generation("added") is None
    assert manager._selection.read() == selection_before
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_first_null_commits_only_sources_that_prepare_and_compile(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "healthy", _v3_source("healthy"))
    broken = _write_plugin(plugins, "broken", "this is not Python !!!\n")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    commit_calls: list[tuple[tuple[str, ...], str | None]] = []
    real_commit = manager._selection.commit

    def observed_commit(
        components: tuple[str, ...], *, expected_ref: str | None,
    ) -> str:
        commit_calls.append((components, expected_ref))
        return real_commit(components, expected_ref=expected_ref)

    manager._selection.commit = observed_commit  # type: ignore[method-assign]

    try:
        await manager.load_all()
        selection = manager._selection.read()
        assert selection is not None
        descriptors = [
            manager._archive.read_descriptor(ref)
            for ref in manager._selection_components(selection)
        ]
        assert [descriptor["plugin_id"] for descriptor in descriptors] == ["healthy"]
        assert manager.generation("healthy") is not None
        assert manager.generation("broken") is None
        assert len(commit_calls) == 1
        assert commit_calls[0][0] == tuple(manager._selection_components(selection))
        assert commit_calls[0][1] is None
        failures = manager.plugin_status()["source_failures"]
        assert failures[0]["source_root"] == str(broken.resolve())
        assert failures[0]["plugin_id"] is None
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_first_null_all_source_content_failures_commit_empty_once(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "first", "this is not Python !!!\n")
    _write_plugin(plugins, "second", "also not Python !!!\n")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    commit_calls: list[tuple[tuple[str, ...], str | None]] = []
    real_commit = manager._selection.commit

    def observed_commit(
        components: tuple[str, ...], *, expected_ref: str | None,
    ) -> str:
        commit_calls.append((components, expected_ref))
        return real_commit(components, expected_ref=expected_ref)

    manager._selection.commit = observed_commit  # type: ignore[method-assign]
    try:
        await manager.load_all()
        selection = manager._selection.read()
        assert selection is not None
        assert manager._selection_components(selection) == ()
        assert commit_calls == [((), None)]
        assert manager.live_root is not None
        assert len(manager.plugin_status()["source_failures"]) == 2
    finally:
        await manager.terminate_all()

    (plugins / "first" / "plugin.py").write_text(
        _v3_source("first"), encoding="utf-8",
    )
    (plugins / "second" / "plugin.py").write_text(
        _v3_source("second"), encoding="utf-8",
    )
    restarted = _manager(tmp_path)
    commit_calls = 0
    real_commit = restarted._selection.commit

    def observed_restart_commit(
        components: tuple[str, ...], *, expected_ref: str | None,
    ) -> str:
        nonlocal commit_calls
        commit_calls += 1
        return real_commit(components, expected_ref=expected_ref)

    restarted._selection.commit = observed_restart_commit  # type: ignore[method-assign]
    try:
        selection_before = restarted._selection.read()
        assert selection_before is not None
        assert restarted._selection_components(selection_before) == ()
        await restarted.load_all()
        assert restarted._selection.read() == selection_before
        assert restarted._selection_components(selection_before) == ()
        assert commit_calls == 0
        assert restarted.generation("first") is None
        assert restarted.generation("second") is None
    finally:
        await restarted.terminate_all()


@pytest.mark.asyncio
async def test_first_null_secondary_compile_failure_is_diagnostic_only(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "healthy", _v3_source("healthy"))
    broken = _write_plugin(plugins, "broken", _v3_source("broken"))
    (broken / "helper.py").write_text("this is not Python !!!\n", encoding="utf-8")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)

    try:
        await manager.load_all()
        selection = manager._selection.read()
        assert selection is not None
        descriptors = [
            manager._archive.read_descriptor(ref)
            for ref in manager._selection_components(selection)
        ]
        assert [descriptor["plugin_id"] for descriptor in descriptors] == ["healthy"]
        failures = manager.plugin_status()["source_failures"]
        assert failures[0]["plugin_id"] == "broken"
        assert failures[0]["phase"] == "compile"
        before = list(failures)
        await asyncio.to_thread(manager.watch_revision)
        assert manager.plugin_status()["source_failures"] == before
        (broken / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        await asyncio.to_thread(manager.watch_revision)
        assert manager.plugin_status()["source_failures"] == before
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_first_null_cas_conflict_keeps_competing_empty_selection(
    tmp_path: Path,
) -> None:
    _write_checked_plugin(tmp_path / "plugins", "racer", _v3_source("racer"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    real_commit = manager._selection.commit
    competed = False
    competing_ref: str | None = None

    def competing_commit(
        components: tuple[str, ...], *, expected_ref: str | None,
    ) -> str:
        nonlocal competed, competing_ref
        if not competed:
            competed = True
            competing_ref = real_commit((), expected_ref=None)
        return real_commit(components, expected_ref=expected_ref)

    manager._selection.commit = competing_commit  # type: ignore[method-assign]
    try:
        with pytest.raises(SelectionConflictError):
            await manager.load_all()
        operation = manager._operation
        assert operation is not None
        await asyncio.wait((operation.task,))
        assert operation.task.done()
        assert manager.live_root is None
        assert manager._active_generations == {}
        assert manager._draining_generations == {}
        assert manager._building_roots == {}
        selection = manager._selection.read()
        assert selection is not None
        assert selection == competing_ref
        assert manager._selection_components(selection) == ()
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_first_null_caller_cancel_keeps_selection_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_checked_plugin(tmp_path / "plugins", "cancelled", _v3_source("cancelled"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    real_load_one = manager._load_one
    cleanup_cancel: asyncio.CancelledError | None = None

    async def blocked_load_one(*args: Any, **kwargs: Any) -> Any:
        nonlocal cleanup_cancel
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError as error:
            cleanup_cancel = error
            await release.wait()
            raise
        return await real_load_one(*args, **kwargs)

    monkeypatch.setattr(manager, "_load_one", blocked_load_one)
    task = asyncio.create_task(manager.load_all(), name="first-null-caller-cancel")
    caller_retrieved = False
    operation_retrieved = False
    operation = None
    try:
        async with asyncio.timeout(10):
            await entered.wait()
        operation = manager._operation
        assert operation is not None
        exact_operation_task = operation.task
        assert not exact_operation_task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caller_cancel:
            await task
        caller_retrieved = True
        assert isinstance(caller_cancel.value, asyncio.CancelledError)
        assert manager._selection.read() is None
        assert not exact_operation_task.done()
        release.set()
        async with asyncio.timeout(10):
            await asyncio.wait((exact_operation_task,))
        with pytest.raises(asyncio.CancelledError):
            await exact_operation_task
        operation_retrieved = True
        assert cleanup_cancel is not None
        assert manager.live_root is None
        assert manager._active_generations == {}
        assert manager._draining_generations == {}
        assert manager._building_roots == {}
    finally:
        release.set()
        try:
            if not caller_retrieved:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                finally:
                    caller_retrieved = True
        finally:
            try:
                if operation is None:
                    operation = manager._operation
                if operation is not None and not operation_retrieved:
                    await asyncio.wait((operation.task,))
                    try:
                        operation.task.result()
                    except asyncio.CancelledError:
                        if not operation.task.cancelled():
                            raise
                    finally:
                        operation_retrieved = True
            finally:
                await manager.terminate_all()


@pytest.mark.asyncio
async def test_first_null_shared_source_read_error_keeps_selection_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_checked_plugin(tmp_path / "plugins", "shared-error", _v3_source("shared-error"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)

    def fail_shared_scan(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("shared source read failed")

    monkeypatch.setattr(manager_module, "scan_plugin_sources", fail_shared_scan)
    try:
        with pytest.raises(OSError, match="shared source read failed"):
            await manager.load_all()
        operation = manager._operation
        assert operation is not None
        await asyncio.wait((operation.task,))
        assert operation.task.done()
        assert manager._selection.read() is None
        assert manager.live_root is None
        assert manager._active_generations == {}
        assert manager._draining_generations == {}
        assert manager._building_roots == {}
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_selected_secondary_compile_repair_prepares_before_clearing_error(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "plugins"
    selected = _write_checked_plugin(
        plugins, "selected", _v3_source("selected", version="release-a"),
    )
    (selected / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)

    try:
        await manager.load_all()
        before_errors = list(manager.plugin_status()["source_failures"])
        before_revision = await asyncio.to_thread(manager.watch_revision)
        (selected / "helper.py").write_text(
            "def broken(:\n    return 1\n", encoding="utf-8",
        )
        broken_revision = await asyncio.to_thread(manager.watch_revision)
        assert broken_revision != before_revision
        assert manager.plugin_status()["source_failures"] == before_errors

        result = await manager.reconcile_changed()
        assert result[0]["publication_state"] == "source_unavailable"
        failure = manager.plugin_status()["source_failures"][0]
        assert failure["source_root"] == str(selected.resolve())
        assert failure["phase"] == "compile"
        assert failure["error_type"] == "SyntaxError"
        assert "line" in failure["error_text"]

        (selected / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
        fixed_revision = await asyncio.to_thread(manager.watch_revision)
        assert fixed_revision != broken_revision
        repaired = await manager.reconcile_changed()
        assert repaired[0]["publication_state"] == "active"
        assert manager.plugin_status()["source_failures"] == []
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_selected_source_disappearance_keeps_selection_and_runtime_owner(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_checked_plugin(
        tmp_path / "plugins", "selected-alias", _v3_source("selected"),
    )
    peer_dir = _write_checked_plugin(
        tmp_path / "plugins", "peer-directory", _peer_source("peer", "peer.service"),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)

    try:
        await manager.load_all()
        selection_before = manager._selection.read()
        assert selection_before is not None
        peer = manager.generation("peer")
        assert peer is not None and peer.fiber is not None
        peer_root = manager.live_root
        assert peer_root is not None
        peer_fiber = peer.fiber
        peer_context = peer_fiber.context
        peer_activation = peer_context.fiber.activation_token
        peer_effects = tuple(peer_fiber.effects)
        async with peer_context.runtime_scope():
            peer_state = peer_context.require(ServiceKey("peer.service"))
        peer_events = tuple(cast(list[str], peer_state["events"]))
        peer_effect_count = peer_state["effect"]
        peer_cleanup_count = peer_state["cleanup"]
        assert not peer_fiber._in_flight_calls

        async def assert_peer_unchanged() -> None:
            assert manager.live_root is peer_root
            assert manager.generation("peer") is peer
            assert peer.fiber is peer_fiber
            assert peer_fiber.context is peer_context
            assert peer_context.fiber.activation_token is peer_activation
            assert tuple(peer_fiber.effects) == peer_effects
            assert tuple(cast(list[str], peer_state["events"])) == peer_events
            assert peer_state["effect"] == peer_effect_count == 1
            assert peer_state["cleanup"] == peer_cleanup_count == 0
            assert not peer_fiber._in_flight_calls
            async with peer_context.runtime_scope():
                assert peer_context.require(ServiceKey("peer.service")) is peer_state
            assert not peer_fiber._in_flight_calls

        shutil.rmtree(plugin_dir)
        result = await manager.reconcile_changed()
        assert result[0]["publication_state"] == "source_unavailable"
        assert manager._selection.read() == selection_before
        assert manager.generation("selected") is not None
        failure = manager.plugin_status()["source_failures"][0]
        assert failure["plugin_id"] == "selected"
        assert failure["error_type"] == "SourceUnavailable"
        assert failure["source_root"] == str(plugin_dir.resolve())
        await assert_peer_unchanged()

        restored = _write_checked_plugin(
            tmp_path / "plugins", "selected-alias", _v3_source("selected"),
        )
        assert restored == plugin_dir
        repaired = await manager.reconcile_changed()
        assert repaired == []
        assert manager.plugin_status()["source_failures"] == []
        await assert_peer_unchanged()
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_archive_only_restart_never_invents_source_root(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_checked_plugin(
        tmp_path / "plugins", "selected-alias", _v3_source("selected"),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    first = _manager(tmp_path)
    await first.load_all()
    await first.terminate_all()

    shutil.rmtree(plugin_dir)
    (plugin_dir).mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "this is not valid source !!!\n", encoding="utf-8",
    )
    restarted = _manager(tmp_path)
    try:
        await restarted.load_all()
        selected = restarted._selection.read()
        assert selected is not None
        assert restarted.generation("selected") is not None

        missing = await restarted.reconcile_changed()
        assert missing[0]["publication_state"] == "source_unavailable"
        assert missing[0]["source_root"] is None
        failures = restarted.plugin_status()["source_failures"]
        assert len(failures) == 1
        assert failures[0]["source_root"] == str(plugin_dir.resolve())
        assert failures[0]["plugin_id"] is None

        (plugin_dir / "plugin.py").write_text(
            _v3_source("selected"), encoding="utf-8",
        )
        repaired = await restarted.reconcile_changed()
        assert repaired == []
        assert restarted.plugin_status()["source_failures"] == []
    finally:
        await restarted.terminate_all()


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
    mounted = asyncio.Event()
    deactivate_entered = asyncio.Event()
    real_mount = manager._mount_generation_composition
    real_deactivate = manager._deactivate_plugin
    observed: dict[str, Any] = {}

    async def observed_mount(root: Any, generation: Any) -> None:
        observed["root"] = root
        observed["generation"] = generation
        module = cast(Any, generation.instance.module)
        observed["module"] = module
        mount_task = asyncio.create_task(
            real_mount(root, generation), name="runtime-start-owner-mount",
        )
        observed["mount_task"] = mount_task
        await module.started.wait()
        mounted.set()
        await mount_task

    async def observed_deactivate(
        plugin_id: str, *, expected_ref: str | None = None,
        accepted: asyncio.Future[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        deactivate_entered.set()
        return await real_deactivate(
            plugin_id, expected_ref=expected_ref, accepted=accepted,
        )

    monkeypatch.setattr(manager, "_mount_generation_composition", observed_mount)
    monkeypatch.setattr(manager, "_deactivate_plugin", observed_deactivate)
    load: asyncio.Task[None] | None = None
    load_retrieved = False
    operation = None
    operation_retrieved = False
    try:
        load = asyncio.create_task(manager.load_all(), name="runtime-start-owner-load")
        async with asyncio.timeout(10):
            await mounted.wait()
        operation = manager._operation
        assert operation is not None
        assert not operation.task.done()
        root = cast(Any, observed["root"])
        generation = cast(Any, observed["generation"])
        module = cast(Any, observed["module"])
        assert manager.live_root is root
        assert manager.generation("runner_race") is generation
        selection_before = manager._selection.read()
        assert selection_before is not None
        shutil.rmtree(plugin_dir)
        with pytest.raises(OperationBusyError):
            await manager.reconcile_changed()
        assert not deactivate_entered.is_set()
        module.allow_finish.set()
        try:
            await load
        finally:
            load_retrieved = True
        try:
            await operation.task
        finally:
            operation_retrieved = True
        result = await manager.reconcile_changed()
        assert result[0]["publication_state"] == "source_unavailable"
        assert manager._selection.read() == selection_before
        assert manager.live_root is root
        assert manager.generation("runner_race") is generation
        assert not module.stopped.is_set()
        assert not deactivate_entered.is_set()
    finally:
        module = cast(Any, observed.get("module")) if observed.get("module") else None
        if module is not None:
            module.allow_finish.set()
        try:
            if load is not None and not load_retrieved:
                try:
                    await load
                finally:
                    load_retrieved = True
        finally:
            try:
                if operation is None:
                    operation = manager._operation
                if operation is not None and not operation_retrieved:
                    await asyncio.wait((operation.task,))
                    try:
                        await operation.task
                    finally:
                        operation_retrieved = True
            finally:
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
async def test_dashboard_routes_follow_live_root_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    app = create_dashboard_app(
        tmp_path / "workspace",
        plugin_manager=manager,
    )
    await manager.load_all()
    root = manager.live_root
    assert root is not None
    catalog = root.context.require(UI).catalog()
    generation = next(
        item for item in catalog.modules if item.plugin_id == "snapshot_dashboard"
    )
    headers = {
        "X-Akashic-Web-Snapshot": root.generation_id,
        "X-Akashic-Web-Catalog": catalog.identity,
        "X-Akashic-Web-Module": "snapshot_dashboard",
        "X-Akashic-Web-Generation": generation.generation_id,
    }
    client = TestClient(app)
    assert client.get("/api/dashboard/snapshot-version").json() == {
        "code": "forbidden_contract"
    }
    assert client.get("/api/dashboard/snapshot-version", headers=headers).status_code == 200
    assert client.get(
        "/api/dashboard/snapshot-version",
        headers={"Sec-Fetch-Site": "same-origin"},
    ).json() == {"code": "forbidden_contract"}
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
    _ = create_dashboard_app(tmp_path / "workspace", plugin_manager=manager)
    with pytest.raises(RuntimeError, match="paired API broken"):
        await manager.load_all()

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
        context=cast(Any, object()),
        generation_id="wildcard-generation",
        has_web=False,
        runtime_workspace=Path("."),
        runtime_data_root=Path("."),
        module_name="wildcard",
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
        context=cast(Any, object()),
        generation_id="ordered-paths-generation",
        has_web=False,
        runtime_workspace=Path("."),
        runtime_data_root=Path("."),
        module_name="ordered-paths",
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
        context=cast(Any, object()),
        generation_id="two-protocols-generation",
        has_web=False,
        runtime_workspace=Path("."),
        runtime_data_root=Path("."),
        module_name="two-protocols",
    )
    _require_routes_available(binding, [])


@pytest.mark.asyncio
async def test_dashboard_websocket_uses_live_manager_owner_switch_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_ui_provider(tmp_path)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "home"))
    plugin_dir = _write_plugin(
        tmp_path / "plugins", "snapshot_socket", _ui_source("snapshot_socket"),
    )
    (plugin_dir / "web_module.js").write_text(
        "export function activate() { return () => {}; }\n", encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "import asyncio\n"
        "from starlette.websockets import WebSocket\n"
        "started = asyncio.Event()\n"
        "cleanup_started = asyncio.Event()\n"
        "cleanup_release = asyncio.Event()\n"
        "cleanup_finished = asyncio.Event()\n"
        "def register(app, _context):\n"
        "    @app.websocket('/api/dashboard/live')\n"
        "    async def live(websocket: WebSocket):\n"
        "        await websocket.accept()\n"
        "        started.set()\n"
        "        try:\n"
        "            await asyncio.Future()\n"
        "        finally:\n"
        "            cleanup_started.set()\n"
        "            await cleanup_release.wait()\n"
        "            cleanup_finished.set()\n",
        encoding="utf-8",
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    task: asyncio.Task[object] | None = None
    reconcile: asyncio.Task[object] | None = None
    dashboard = None
    app = create_dashboard_app(tmp_path / "workspace", plugin_manager=manager)
    try:
        await manager.load_all()
        root = manager.live_root
        assert root is not None
        generation = manager.generation("snapshot_socket")
        assert generation is not None and generation.fiber is not None
        old_fiber = generation.fiber
        dashboard = sys.modules[f"{generation.instance.module.__package__}.dashboard"]  # type: ignore[union-attr]
        catalog = root.context.require(UI).catalog()
        module = next(item for item in catalog.modules if item.plugin_id == "snapshot_socket")
        query = urlencode({
            "__akashic_web_snapshot": root.generation_id,
            "__akashic_web_catalog": catalog.identity,
            "__akashic_web_module": module.plugin_id,
            "__akashic_web_generation": module.generation_id,
        })

        pending = [{"type": "websocket.connect"}]

        async def receive() -> dict[str, object]:
            if pending:
                return pending.pop(0)
            await asyncio.Future()
            return {"type": "websocket.disconnect"}

        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        task = asyncio.create_task(
            app(
                {
                    "type": "websocket",
                    "path": "/api/dashboard/live",
                    "query_string": query.encode("ascii"),
                    "headers": [(b"origin", b"http://test"), (b"host", b"test")],
                    "scheme": "ws",
                    "server": ("test", 80),
                    "client": ("test", 1),
                },
                receive,
                send,
            ),
            name="hot-reload-dashboard-ws",
        )
        await dashboard.started.wait()
        (plugin_dir / "plugin.py").write_text(
            _ui_source("snapshot_socket", body="    marker = 'release-b'\n"),
            encoding="utf-8",
        )
        reconcile = asyncio.create_task(
            manager._run_operation(manager._reconcile_changed),  # pyright: ignore[reportPrivateUsage]
            name="hot-reload-dashboard-reconcile",
        )
        await dashboard.cleanup_started.wait()
        assert old_fiber.state is FiberState.UNLOADING
        dashboard.cleanup_release.set()
        result = await reconcile
        await task
        new_generation = manager.generation("snapshot_socket")
        assert new_generation is not None and new_generation is not generation
        assert new_generation.fiber is not None
        assert new_generation.fiber.state is FiberState.ACTIVE
        assert manager.live_root is root
        assert result and result[0]["publication_state"] == "active"
        assert dashboard.cleanup_finished.is_set()
        assert any(message.get("code") == 1012 for message in sent)
    finally:
        if dashboard is not None:
            dashboard.cleanup_release.set()
        for child in (task, reconcile):
            if child is not None and not child.done():
                child.cancel()
                try:
                    await child
                except asyncio.CancelledError:
                    pass
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


@pytest.mark.asyncio
async def test_local_loader_keeps_one_live_root_during_update(tmp_path: Path) -> None:
    """A local update changes the generation while retaining the formal Root."""
    plugin = _write_plugin(tmp_path / "plugins", "local", _v3_source("local"))
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        root = manager._live_root
        assert root is not None
        (plugin / "plugin.py").write_text(_v3_source("local", version="2.0.0"), encoding="utf-8")
        await manager._run_operation(manager._reconcile_changed)
        assert manager._live_root is root
        assert manager.generation("local").instance.version == "2.0.0"  # type: ignore[union-attr]
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_local_compile_failure_keeps_selection_and_old_owner(
    tmp_path: Path,
) -> None:
    """A failed import-free compile cannot publish a new selection."""
    plugin = _write_plugin(
        tmp_path / "plugins", "compile_local",
        _v3_source(
            "compile_local",
            version="1.0.0",
            exports="from . import helper\n",
        ),
    )
    (plugin / "helper.py").write_text("value = 1\n", encoding="utf-8")
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        old = manager.generation("compile_local")
        selection = manager._selection.read()
        (plugin / "helper.py").write_text(
            "def broken(:\n    return None\n",
            encoding="utf-8",
        )
        with pytest.raises(SyntaxError):
            await manager._run_operation(manager._reconcile_changed)
        assert manager._selection.read() == selection
        assert manager.generation("compile_local") is old
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
        assert view["archive_ref"] == failed.archive_ref
        assert view["state"] == "failed"
        assert view["load_error"] == "cold import blocked"
        assert view["cleanup_pending"] is False
        assert cast(dict[str, object], view["composition"])["ready"] is False
        assert cast(dict[str, object], view["composition"])["fibers"] == []
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gate", "exports", "error_fragment"),
    [
        (
            "LIVE_IMPORT_STAGE",
            "import os\nif os.environ.get('LIVE_IMPORT_STAGE') == 'yes':\n"
            "    raise ImportError('selected import blocked')\n",
            "selected import blocked",
        ),
        (
            "LIVE_EXPORT_STAGE",
            "import os\ninject = ()\nif os.environ.get('LIVE_EXPORT_STAGE') == 'yes':\n"
            "    inject = (object(),)\n",
            "inject 必须是 ServiceKey 序列",
        ),
    ],
)
async def test_live_pre_fiber_error_stages_project_one_failed_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate: str,
    exports: str,
    error_fragment: str,
) -> None:
    """Import and dynamic-export failures use the same real live failure owner."""
    source = _v3_source("staged_failure", exports=exports)
    _write_checked_plugin(tmp_path / "plugins", "staged_failure", source)
    _write_checked_plugin(tmp_path / "plugins", "staged_peer", _v3_source("staged_peer"))
    initialize_plugin_workspace(tmp_path / "workspace")

    monkeypatch.setenv(gate, "no")
    first = _manager(tmp_path)
    try:
        await first.load_all()
        selected = first._selection.read()
        assert selected is not None
    finally:
        await first.terminate_all()

    monkeypatch.setenv(gate, "yes")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        failed = manager.generation("staged_failure")
        peer = manager.generation("staged_peer")
        assert failed is not None and peer is not None and peer.fiber is not None
        assert manager._selection.read() == selected
        assert failed.state == "failed"
        assert failed.fiber is None
        assert failed.load_error is not None
        assert error_fragment in str(failed.load_error)
        assert peer.fiber.state == FiberState.ACTIVE

        root = manager._live_root
        assert root is not None
        from agent.plugin_composition.runtime_catalog import build_runtime_catalog
        catalog = build_runtime_catalog(
            root, manager._active_generations, manager._draining_generations,
        )
        view = next(
            item for item in cast(list[dict[str, object]], catalog["plugins"])
            if item["id"] == "staged_failure"
        )
        composition = cast(dict[str, object], view["composition"])
        assert view["archive_ref"] == failed.archive_ref
        assert view["state"] == "failed"
        assert view["load_error"] == str(failed.load_error)
        assert view["cleanup_pending"] is False
        assert composition["ready"] is False
        assert composition["fibers"] == []
        assert composition["incident_count"] == 0
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_public_install_retains_failed_b_until_real_selected_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public install exposes accepted B before blocked cleanup and retry creates fresh owners."""
    from agent.plugins.install import install_git_plugin
    from tests.test_plugin_install import _commit

    source_repo = tmp_path / "public-source"
    source_repo.mkdir()
    source_entry = source_repo / "plugin.py"

    def write_checked_source(source: str) -> None:
        tree = ast.parse(source, filename=str(source_entry))
        compile(tree, str(source_entry), "exec")
        source_entry.write_text(source, encoding="utf-8")

    write_checked_source(_v3_source("public_target", version="1.0.0"))
    _commit(source_repo)
    workspace = tmp_path / "workspace"
    plugin_home = tmp_path / "home"
    initialize_plugin_workspace(workspace)
    install_git_plugin(
        workspace=workspace,
        source=str(source_repo),
        marketplace="lab",
        plugins_home=plugin_home,
    )
    _write_checked_plugin(
        tmp_path / "plugins", "public_peer", _peer_source("public_peer", "public.peer"),
    )
    event_bus = EventBus()
    manager = PluginManager(
        [tmp_path / "plugins"],
        event_bus=event_bus,
        workspace=workspace,
        installed_cache_root=plugin_home / "cache",
    )
    monkeypatch.setenv("PUBLIC_IMPORT_FAIL", "no")
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    trace: list[tuple[str, str]] = []
    observed_b: Any = None
    operation: Any = None
    operation_retrieved = False
    original_load_live = manager._load_live_generation

    async def observe_load_live(generation: Any) -> None:
        nonlocal observed_b
        if (
            generation.plugin_id == "public_target@lab"
            and generation.archive_ref != old.archive_ref
            and observed_b is None
        ):
            observed_b = generation

            async def cleanup() -> None:
                trace.append(("cleanup-start", generation.generation_id))
                cleanup_started.set()
                await release_cleanup.wait()
                trace.append(("cleanup-done", generation.generation_id))

            generation.scope.defer("public-b-cleanup", cleanup)
        if generation.plugin_id == "public_target@lab" and generation.archive_ref != old.archive_ref:
            trace.append(("load", generation.generation_id))
        await original_load_live(generation)

    try:
        await manager.load_all()
        old = manager.generation("public_target@lab")
        peer = manager.generation("public_peer")
        root = manager._live_root
        assert old is not None and peer is not None and peer.fiber is not None and root is not None
        peer_context = peer.fiber.context
        peer_identity = (peer.fiber, peer_context, peer_context.fiber.activation_token)
        peer_effects = tuple(peer.fiber.effects)
        async with peer_context.runtime_scope():
            peer_state = peer_context.require(ServiceKey("public.peer"))
        assert isinstance(peer_state, dict)
        assert peer_state["events"] == ["starting", "started"]
        assert peer_state["effect"] == 1
        assert peer_state["cleanup"] == 0
        assert not peer_context.fiber._fiber._in_flight_calls
        monkeypatch.setattr(manager, "_load_live_generation", observe_load_live)

        write_checked_source(
            _v3_source(
                "public_target",
                version="2.0.0",
                exports=(
                    "import os\n"
                    "if os.environ.get('PUBLIC_IMPORT_FAIL') == 'yes':\n"
                    "    raise ImportError('public B import blocked')\n"
                ),
            ),
        )
        _commit(source_repo)
        monkeypatch.setenv("PUBLIC_IMPORT_FAIL", "yes")

        accepted = await manager.install(
            source=str(source_repo), marketplace="lab", ref_name="", sparse_paths=[],
            update_id="public-b-failure",
        )
        assert accepted.state == "accepted"
        assert accepted.selection == "selected"
        assert accepted.error == ""
        operation = manager._operation
        assert operation is not None
        await cleanup_started.wait()
        failed = observed_b
        assert failed is not None and failed is manager.generation("public_target@lab")
        assert failed.state == "failed"
        assert failed.fiber is None
        assert failed.load_error is not None
        assert str(failed.load_error) == "public B import blocked"
        assert failed.scope.closed is False
        assert manager._active_generations["public_target@lab"] is failed
        assert manager._draining_generations["public_target@lab"] == [failed]
        assert failed.module_path in sys.modules
        assert failed.module_path in manager._fresh_importer._roots
        journal_mid = manager._reload_journal.update("public-b-failure")
        assert journal_mid.error == ""
        blocked = manager.read_update("public-b-failure")
        assert blocked.state == "failed"
        assert blocked.selection == "selected"
        assert blocked.archive_ref == failed.archive_ref
        assert blocked.generation_id == failed.generation_id
        assert blocked.error == "public B import blocked"
        assert operation.task.done() is False
        assert old.scope.closed
        assert manager._live_root is root
        assert manager.generation("public_peer") is peer
        async with peer_context.runtime_scope():
            assert peer_context.require(ServiceKey("public.peer")) is peer_state
        assert peer.fiber is peer_identity[0]
        assert peer.fiber.context is peer_identity[1]
        assert peer_context.fiber.activation_token is peer_identity[2]
        assert tuple(peer.fiber.effects) == peer_effects
        assert peer_state["events"] == ["starting", "started"]
        assert peer_state["effect"] == 1
        assert peer_state["cleanup"] == 0
        assert not peer_context.fiber._fiber._in_flight_calls

        release_cleanup.set()
        with pytest.raises(ImportError, match="public B import blocked"):
            try:
                await operation.task
            finally:
                operation_retrieved = True
        assert old.scope.closed
        assert failed.scope.closed
        assert failed.module_path not in sys.modules
        assert failed.module_path not in manager._fresh_importer._roots
        assert manager._draining_generations.get("public_target@lab") is None
        final_failed = manager.read_update("public-b-failure")
        assert final_failed.state == "failed"
        assert final_failed.selection == "selected"
        assert final_failed.archive_ref == failed.archive_ref
        assert final_failed.generation_id == failed.generation_id
        assert final_failed.error == "public B import blocked"
        assert manager._reload_journal.update("public-b-failure").error == "public B import blocked"

        monkeypatch.setenv("PUBLIC_IMPORT_FAIL", "no")
        recovered = await manager.retry_runtime_recovery("public_target@lab")
        fresh = manager.generation("public_target@lab")
        assert recovered["publication_state"] == "recovered"
        assert fresh is not None and fresh is not failed
        assert fresh.scope is not failed.scope
        assert fresh.module_path != failed.module_path
        assert fresh.load_error is None
        assert fresh.state == "active"
        assert failed.load_error is not None
        assert trace.index(("cleanup-done", failed.generation_id)) < trace.index(
            ("load", fresh.generation_id)
        )
        assert manager._live_root is root
        assert manager.generation("public_peer") is peer
        async with peer_context.runtime_scope():
            assert peer_context.require(ServiceKey("public.peer")) is peer_state
        assert peer.fiber is peer_identity[0]
        assert peer.fiber.context is peer_identity[1]
        assert peer_context.fiber.activation_token is peer_identity[2]
        assert tuple(peer.fiber.effects) == peer_effects
        assert peer_state["events"] == ["starting", "started"]
        assert peer_state["effect"] == 1
        assert peer_state["cleanup"] == 0
        assert not peer_context.fiber._fiber._in_flight_calls
    finally:
        release_cleanup.set()
        if operation is None:
            operation = manager._operation
        try:
            if operation is not None and not operation_retrieved:
                try:
                    await operation.task
                finally:
                    operation_retrieved = True
        finally:
            try:
                await manager.terminate_all()
            finally:
                await event_bus.aclose()


@pytest.mark.asyncio
async def test_selected_pre_fiber_failure_retains_cleanup_owner_until_explicit_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import and Scope cleanup failures retain the same real selected owner."""
    bad_source = (
        "import os\n"
        + _v3_source(
            "a3_bad",
            exports=(
                "if os.environ.get('A3_IMPORT_FAIL') == 'yes':\n"
                "    raise ImportError('a3 import blocked')\n"
            ),
        )
    )
    _write_checked_plugin(tmp_path / "plugins", "a3_bad", bad_source)
    _write_checked_plugin(
        tmp_path / "plugins", "a3_peer", _peer_source("a3_peer", "a3.peer"),
    )
    initialize_plugin_workspace(tmp_path / "workspace")

    monkeypatch.setenv("A3_IMPORT_FAIL", "no")
    first_bus = EventBus()
    first = PluginManager(
        [tmp_path / "plugins"], event_bus=first_bus,
        workspace=tmp_path / "workspace", installed_cache_root=tmp_path / "home/cache",
    )
    try:
        try:
            await first.load_all()
            selected = first._selection.read()
            assert selected is not None
        finally:
            await first.terminate_all()
    finally:
        await first_bus.aclose()

    monkeypatch.setenv("A3_IMPORT_FAIL", "yes")
    event_bus = EventBus()
    manager = PluginManager(
        [tmp_path / "plugins"], event_bus=event_bus,
        workspace=tmp_path / "workspace", installed_cache_root=tmp_path / "home/cache",
    )
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_fail = True
    cleanup_attempts = 0
    trace: list[tuple[str, str]] = []
    observed: Any = None
    operation: Any = None
    operation_retrieved = False
    load_task_retrieved = False
    original_load_live = manager._load_live_generation

    async def observe_load_live(generation: Any) -> None:
        nonlocal observed
        if generation.plugin_id == "a3_bad":
            trace.append(("load", generation.generation_id))
        if observed is None and generation.plugin_id == "a3_bad":
            observed = generation

            async def cleanup() -> None:
                nonlocal cleanup_attempts
                cleanup_attempts += 1
                cleanup_started.set()
                await release_cleanup.wait()
                if cleanup_fail:
                    raise OSError("a3 cleanup blocked")
                trace.append(("cleanup-done", generation.generation_id))

            generation.scope.defer("a3-controlled-cleanup", cleanup)
        await original_load_live(generation)

    monkeypatch.setattr(manager, "_load_live_generation", observe_load_live)
    load_task = asyncio.create_task(manager.load_all(), name="a3-selected-load")
    try:
        async with asyncio.timeout(10):
            await cleanup_started.wait()
        operation = manager._operation
        assert operation is not None
        assert operation.task is not load_task
        failed = observed
        assert failed is not None
        assert failed is manager.generation("a3_bad")
        assert failed.archive_ref in manager._selection_components(selected)
        assert failed.state == "failed"
        assert failed.fiber is None
        assert failed.load_error is not None
        assert str(failed.load_error) == "a3 import blocked"
        assert manager._active_generations["a3_bad"] is failed
        assert manager._draining_generations["a3_bad"] == [failed]
        assert failed.scope.closed is False
        assert failed.module_path in sys.modules
        assert failed.module_path in manager._fresh_importer._roots
        assert load_task.done() is False

        peer = manager.generation("a3_peer")
        assert peer is not None and peer.fiber is None

        from agent.plugin_composition.runtime_catalog import build_runtime_catalog

        root = manager._live_root
        assert root is not None
        catalog = build_runtime_catalog(
            root, manager._active_generations, manager._draining_generations,
        )
        item = next(
            item for item in cast(list[dict[str, object]], catalog["plugins"])
            if item["id"] == "a3_bad"
        )
        composition = cast(dict[str, object], item["composition"])
        assert item["archive_ref"] == failed.archive_ref
        assert item["state"] == "failed"
        assert item["load_error"] == "a3 import blocked"
        assert item["cleanup_pending"] is True
        assert composition["fibers"] == []
        assert composition["ready"] is False
        assert composition["incident_count"] == 0

        release_cleanup.set()
        try:
            await load_task
        finally:
            load_task_retrieved = True
        try:
            await operation.task
        finally:
            operation_retrieved = True
        assert cleanup_attempts == 1
        assert failed.scope.closed is False
        assert manager.generation("a3_bad") is failed
        assert manager._draining_generations["a3_bad"] == [failed]
        assert failed.module_path in sys.modules
        assert failed.module_path in manager._fresh_importer._roots

        peer = manager.generation("a3_peer")
        assert peer is not None and peer.fiber is not None
        peer_context = peer.fiber.context
        peer_identity = (peer.fiber, peer_context, peer_context.fiber.activation_token)
        peer_effects = tuple(peer.fiber.effects)
        async with peer_context.runtime_scope():
            peer_state = peer_context.require(ServiceKey("a3.peer"))
        assert isinstance(peer_state, dict)
        assert peer_state["events"] == ["starting", "started"]
        assert peer_state["effect"] == 1
        assert peer_state["cleanup"] == 0
        assert not peer_context.fiber._fiber._in_flight_calls

        with pytest.raises(OperationBusyError, match="上一次更新仍有资源 owner"):
            await manager.reconcile_changed()

        import_error = failed.load_error
        assert import_error is not None
        with pytest.raises(
            RuntimeError,
            match="generation scope cleanup 未完成.*a3 cleanup blocked",
        ) as cleanup_error:
            await manager.retry_runtime_recovery("a3_bad")
        assert "a3 cleanup blocked" in str(cleanup_error.value)
        assert any(
            item.resource == "a3-controlled-cleanup"
            and item.error == "a3 cleanup blocked"
            for item in manager._cleanup_failures
        )
        assert cleanup_attempts == 2
        assert manager.generation("a3_bad") is failed
        assert failed.load_error is import_error
        assert failed.scope.closed is False
        assert failed.module_path in sys.modules
        assert manager._draining_generations["a3_bad"] == [failed]
        async with peer_context.runtime_scope():
            assert peer_context.require(ServiceKey("a3.peer")) is peer_state
        assert peer.fiber is peer_identity[0]
        assert peer.fiber.context is peer_identity[1]
        assert peer_context.fiber.activation_token is peer_identity[2]
        assert tuple(peer.fiber.effects) == peer_effects

        cleanup_fail = False
        monkeypatch.setenv("A3_IMPORT_FAIL", "yes")
        with pytest.raises(ImportError, match="a3 import blocked"):
            await manager.retry_runtime_recovery("a3_bad")
        fresh_failed = manager.generation("a3_bad")
        assert fresh_failed is not None and fresh_failed is not failed
        assert failed.scope.closed
        assert failed.module_path not in sys.modules
        assert failed.module_path not in manager._fresh_importer._roots
        assert fresh_failed.fiber is None
        assert fresh_failed.load_error is not None
        fresh_error = fresh_failed.load_error
        assert fresh_failed.archive_ref == failed.archive_ref
        assert fresh_failed.archive_ref in manager._selection_components(selected)
        assert not manager._draining_generations.get("a3_bad")
        assert trace.index(("cleanup-done", failed.generation_id)) < trace.index(
            ("load", fresh_failed.generation_id)
        )
        assert failed.load_error is import_error

        async with peer_context.runtime_scope():
            assert peer_context.require(ServiceKey("a3.peer")) is peer_state
        assert peer.fiber is peer_identity[0]
        assert peer.fiber.context is peer_identity[1]
        assert peer_context.fiber.activation_token is peer_identity[2]
        assert tuple(peer.fiber.effects) == peer_effects
        assert peer_state["events"] == ["starting", "started"]
        assert peer_state["effect"] == 1
        assert peer_state["cleanup"] == 0
        assert not peer_context.fiber._fiber._in_flight_calls

        trace_before_reconcile = tuple(trace)
        repaired_source = _v3_source("a3_bad", version="2.0.0")
        repaired_entry = tmp_path / "plugins" / "a3_bad" / "plugin.py"
        repaired_tree = ast.parse(repaired_source, filename=str(repaired_entry))
        compile(repaired_tree, str(repaired_entry), "exec")
        repaired_entry.write_text(repaired_source, encoding="utf-8")
        reconciled = await manager.reconcile_changed()
        assert reconciled == [{
            "plugin_id": "a3_bad",
            "publication_state": "failed_selected",
            "error": "a3 import blocked",
        }]
        assert manager._selection.read() == selected
        assert fresh_failed is manager.generation("a3_bad")
        assert fresh_failed.archive_ref in manager._selection_components(selected)
        assert fresh_failed.load_error is fresh_error
        assert tuple(trace) == trace_before_reconcile
        async with peer_context.runtime_scope():
            assert peer_context.require(ServiceKey("a3.peer")) is peer_state

        monkeypatch.setenv("A3_IMPORT_FAIL", "no")
        await manager.retry_runtime_recovery("a3_bad")
        recovered = manager.generation("a3_bad")
        assert recovered is not None and recovered is not fresh_failed
        assert recovered.archive_ref == fresh_failed.archive_ref
        assert recovered.static_manifest is not None
        assert recovered.static_manifest.version == "1.0.0"
        assert recovered.state == "active"
        assert recovered.fiber is not None
        assert recovered.load_error is None
    finally:
        cleanup_fail = False
        release_cleanup.set()
        try:
            if not load_task_retrieved:
                try:
                    await load_task
                finally:
                    load_task_retrieved = True
            if operation is None:
                operation = manager._operation
            if operation is not None and not operation_retrieved:
                try:
                    await operation.task
                finally:
                    operation_retrieved = True
        finally:
            try:
                await manager.terminate_all()
            finally:
                await event_bus.aclose()


@pytest.mark.asyncio
async def test_live_cold_cancel_cleans_imported_generation_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later import cancellation cleans every earlier and current owner."""
    first_source = _v3_source("alpha_imported")
    cancel_source = _v3_source(
        "omega_cancelled_import",
        exports=(
            "import asyncio\n"
            "import os\n"
            "if os.environ.get('CANCEL_SELECTED') == 'yes':\n"
            "    raise asyncio.CancelledError('selected import cancelled')\n"
        ),
    )
    _write_checked_plugin(tmp_path / "plugins", "alpha_imported", first_source)
    _write_checked_plugin(tmp_path / "plugins", "omega_cancelled_import", cancel_source)
    initialize_plugin_workspace(tmp_path / "workspace")
    monkeypatch.setenv("CANCEL_SELECTED", "no")
    first = _manager(tmp_path)
    try:
        await first.load_all()
        selected = first._selection.read()
        assert selected is not None
        selected_expected: list[tuple[str, str, Path]] = []
        for archive_ref in first._selection_components(selected):
            record = first._archive.read_descriptor(archive_ref)
            code_dir = first._archive.open(cast(str, record["code"])).resolve()
            selected_expected.append(
                (cast(str, record["plugin_id"]), archive_ref, code_dir)
            )
        selected_plugins = tuple(item[0] for item in selected_expected)
        assert selected_plugins == ("alpha_imported", "omega_cancelled_import")
    finally:
        await first.terminate_all()

    monkeypatch.setenv("CANCEL_SELECTED", "yes")
    manager = _manager(tmp_path)
    import_events: list[dict[str, object]] = []
    original_import = manager._import_plugin

    def observe_import(module_name: str, plugin_root: Path) -> None:
        resolved_root = plugin_root.resolve()
        matches = tuple(
            generation
            for generation in manager._active_generations.values()
            if generation.module_path == module_name
            and generation.code_dir.resolve() == resolved_root
        )
        assert len(matches) == 1
        generation = matches[0]
        assert (
            generation.plugin_id,
            generation.archive_ref,
            generation.code_dir.resolve(),
        ) in selected_expected
        event: dict[str, object] = {
            "generation": generation,
            "scope": generation.scope,
            "archive_ref": generation.archive_ref,
            "plugin": generation.plugin_id,
            "module": module_name,
            "code_dir": resolved_root,
            "module_registered": False,
            "importer_registered": False,
            "error": None,
        }
        try:
            original_import(module_name, plugin_root)
        except BaseException as error:
            event["module_registered"] = module_name in sys.modules
            event["importer_registered"] = module_name in manager._fresh_importer._roots
            event["error"] = error
            import_events.append(event)
            raise
        event["module_registered"] = module_name in sys.modules
        event["importer_registered"] = module_name in manager._fresh_importer._roots
        import_events.append(event)

    monkeypatch.setattr(manager, "_import_plugin", observe_import)
    before_modules = set(sys.modules)
    try:
        with pytest.raises(asyncio.CancelledError, match="selected import cancelled") as cancelled:
            await manager.load_all()
        assert manager._selection.read() == selected
        assert cancelled.value is import_events[-1]["error"]
        assert [event["plugin"] for event in import_events] == list(selected_plugins)
        assert import_events[-1]["plugin"] == "omega_cancelled_import"
        assert [
            (event["plugin"], event["archive_ref"], event["code_dir"])
            for event in import_events
        ] == selected_expected
        assert import_events[0]["error"] is None
        assert isinstance(import_events[-1]["error"], asyncio.CancelledError)
        assert all(event["module_registered"] for event in import_events)
        assert all(event["importer_registered"] for event in import_events)
        for event in import_events:
            generation = cast(Any, event["generation"])
            assert generation.scope.closed
            assert event["module"] not in sys.modules
            assert event["module"] not in manager._fresh_importer._roots
        assert manager.live_root is None
        assert manager._active_generations == {}
        assert manager._draining_generations == {}
        assert manager._building_roots == {}
        assert not any(
            name.startswith("_akashic_archive_")
            for name in set(sys.modules) - before_modules
        )
    finally:
        await manager.terminate_all()
    assert manager._fresh_importer._roots == {}
    assert all(event["module"] not in sys.modules for event in import_events)


@pytest.mark.asyncio
async def test_selected_pre_fiber_cleanup_cancellation_waits_for_scope_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller cancellation cannot unload a retained failed scope before its gate."""
    bad_source = (
        "import os\n"
        + _v3_source(
            "a3_cancel_bad",
            exports=(
                "if os.environ.get('A3_CANCEL_IMPORT_FAIL') == 'yes':\n"
                "    raise ImportError('a3 cancel import blocked')\n"
            ),
        )
    )
    _write_checked_plugin(tmp_path / "plugins", "a3_cancel_bad", bad_source)
    initialize_plugin_workspace(tmp_path / "workspace")
    monkeypatch.setenv("A3_CANCEL_IMPORT_FAIL", "no")
    first_bus = EventBus()
    first = PluginManager(
        [tmp_path / "plugins"], event_bus=first_bus,
        workspace=tmp_path / "workspace", installed_cache_root=tmp_path / "home/cache",
    )
    try:
        try:
            await first.load_all()
            selected = first._selection.read()
            assert selected is not None
        finally:
            await first.terminate_all()
    finally:
        await first_bus.aclose()

    monkeypatch.setenv("A3_CANCEL_IMPORT_FAIL", "yes")
    event_bus = EventBus()
    manager = PluginManager(
        [tmp_path / "plugins"], event_bus=event_bus,
        workspace=tmp_path / "workspace", installed_cache_root=tmp_path / "home/cache",
    )
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    observed: Any = None
    operation: Any = None
    load_task_retrieved = False
    operation_retrieved = False
    original_load_live = manager._load_live_generation

    async def observe_load_live(generation: Any) -> None:
        nonlocal observed
        if generation.plugin_id == "a3_cancel_bad" and observed is None:
            observed = generation

            async def cleanup() -> None:
                cleanup_started.set()
                await release_cleanup.wait()

            generation.scope.defer("a3-cancel-controlled-cleanup", cleanup)
        await original_load_live(generation)

    monkeypatch.setattr(manager, "_load_live_generation", observe_load_live)
    load_task = asyncio.create_task(manager.load_all(), name="a3-cancel-selected-load")
    try:
        async with asyncio.timeout(10):
            await cleanup_started.wait()
        operation = manager._operation
        assert operation is not None
        assert operation.task is not load_task
        failed = observed
        assert failed is not None
        assert failed.archive_ref in manager._selection_components(selected)
        assert failed.load_error is not None
        assert str(failed.load_error) == "a3 cancel import blocked"
        assert failed.fiber is None
        assert failed.scope.closed is False
        assert failed.module_path in sys.modules
        assert failed.module_path in manager._fresh_importer._roots
        assert manager._active_generations["a3_cancel_bad"] is failed
        assert manager._draining_generations["a3_cancel_bad"] == [failed]
        load_error = failed.load_error
        assert load_error is not None

        load_task.cancel()
        async with asyncio.timeout(10):
            with pytest.raises(asyncio.CancelledError):
                await load_task
        load_task_retrieved = True
        assert operation.revoked
        assert operation.task.done() is False
        assert operation.task.cancelling() >= 1
        assert failed.module_path in sys.modules
        assert failed.module_path in manager._fresh_importer._roots
        assert manager._active_generations["a3_cancel_bad"] is failed
        assert manager._draining_generations["a3_cancel_bad"] == [failed]

        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await operation.task
        operation_retrieved = True
        assert failed.load_error is load_error
        assert failed.load_error.args == ("a3 cancel import blocked",)
        assert manager.live_root is None
        assert manager._active_generations == {}
        assert manager._draining_generations == {}
        assert manager._building_roots == {}
        assert failed.module_path not in sys.modules
        assert failed.module_path not in manager._fresh_importer._roots
    finally:
        release_cleanup.set()
        try:
            if not load_task_retrieved:
                try:
                    await load_task
                finally:
                    load_task_retrieved = True
            if operation is None:
                operation = manager._operation
            if operation is not None and not operation_retrieved:
                try:
                    await operation.task
                finally:
                    operation_retrieved = True
        finally:
            try:
                await manager.terminate_all()
            finally:
                await event_bus.aclose()


@pytest.mark.asyncio
async def test_local_retry_reloads_selected_archive_after_start_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit retry uses selected B and a fresh Scope after B fails to start."""
    monkeypatch.setenv("LOCAL_START_ALLOWED", "yes")
    plugin = _write_plugin(
        tmp_path / "plugins",
        "retry_local",
        _v3_source(
            "retry_local", version="1.0.0",
            exports="import os\n",
            body='    if os.environ["LOCAL_START_ALLOWED"] != "yes":\n'
                 '        raise RuntimeError("start blocked")\n',
        ),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        old = manager.generation("retry_local")
        assert old is not None
        (plugin / "plugin.py").write_text(
            _v3_source(
                "retry_local", version="2.0.0",
                exports="import os\n",
                body='    if os.environ["LOCAL_START_ALLOWED"] != "yes":\n'
                     '        raise RuntimeError("start blocked")\n',
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("LOCAL_START_ALLOWED", "no")
        with pytest.raises(RuntimeError, match="目标依赖未 ACTIVE"):
            await manager.reconcile_changed()
        selected_after_failure = manager._selection.read()
        assert selected_after_failure is not None
        failed = manager.generation("retry_local")
        assert failed is not None
        assert failed.state == "failed"
        assert failed.load_error is not None
        monkeypatch.setenv("LOCAL_START_ALLOWED", "yes")
        recovered = await manager.retry_runtime_recovery("retry_local")
        fresh = manager.generation("retry_local")
        assert recovered["publication_state"] == "recovered"
        assert manager._selection.read() == selected_after_failure
        assert fresh is not old
        assert fresh is not None and not fresh.scope.closed
        assert old.scope.closed
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["old_cleanup", "new_cleanup"])
async def test_local_retry_drains_the_real_failed_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """Both old-owner and new-owner cleanup failures remain explicit retry work."""
    monkeypatch.setenv("LOCAL_A_CLEANUP", "yes")
    monkeypatch.setenv("LOCAL_B_CLEANUP", "yes")
    monkeypatch.setenv("LOCAL_B_START", "yes")
    def source(version: str, cleanup_var: str, *, fail_start: bool) -> str:
        body = (
            "    await _setup(ctx)\n"
            if not fail_start else
            "    await _setup(ctx)\n"
            "    if os.environ[\"LOCAL_B_START\"] != \"yes\":\n"
            "        raise RuntimeError(\"start blocked\")\n"
        )
        return _v3_source(
            "cleanup_local", version=version,
            exports=(
                "import os\n"
                "async def _setup(ctx):\n"
                "    def cleanup():\n"
                f"        if os.environ[{cleanup_var!r}] != 'yes':\n"
                "            raise RuntimeError('cleanup blocked')\n"
                "    await ctx.effect(lambda: cleanup)\n"
            ),
            body=body,
        )
    plugin = _write_plugin(
        tmp_path / "plugins", "cleanup_local",
        source("1.0.0", "LOCAL_A_CLEANUP", fail_start=False),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        (plugin / "plugin.py").write_text(
            source(
                "2.0.0",
                "LOCAL_B_CLEANUP" if failure_stage == "new_cleanup" else "LOCAL_A_CLEANUP",
                fail_start=failure_stage == "new_cleanup",
            ),
            encoding="utf-8",
        )
        if failure_stage == "old_cleanup":
            monkeypatch.setenv("LOCAL_A_CLEANUP", "no")
        else:
            monkeypatch.setenv("LOCAL_B_CLEANUP", "no")
            monkeypatch.setenv("LOCAL_B_START", "no")
        with pytest.raises((RuntimeError, BaseExceptionGroup)):
            await manager.reconcile_changed()
        selected_after_failure = manager._selection.read()
        assert selected_after_failure is not None
        retained = manager._draining_generations.get("cleanup_local")
        assert retained and len(retained) == 1
        assert retained[0] is manager._active_generations.get("cleanup_local")
        assert retained[0].scope.closed is False
        assert retained[0].module_path in sys.modules
        status = next(
            item for item in cast(list[dict[str, object]], manager.plugin_status()["plugins"])
            if item["plugin_id"] == "cleanup_local"
        )
        assert status["cleanup_pending"] is True
        assert status["load_error"] is None
        root = manager._live_root
        assert root is not None
        from agent.plugin_composition.runtime_catalog import build_runtime_catalog
        catalog = build_runtime_catalog(
            root, manager._active_generations, manager._draining_generations,
        )
        view = next(
            item for item in cast(list[dict[str, object]], catalog["plugins"])
            if item["id"] == "cleanup_local"
        )
        assert view["cleanup_pending"] is True
        assert view["load_error"] is None
        if failure_stage == "old_cleanup":
            assert retained[0].instance.version == "1.0.0"
        else:
            assert retained[0].instance.version == "2.0.0"
        monkeypatch.setenv("LOCAL_A_CLEANUP", "yes")
        monkeypatch.setenv("LOCAL_B_CLEANUP", "yes")
        monkeypatch.setenv("LOCAL_B_START", "yes")
        await manager.retry_runtime_recovery("cleanup_local")
        fresh = manager.generation("cleanup_local")
        assert fresh is not None and fresh.instance.version == "2.0.0"
        assert manager._selection.read() == selected_after_failure
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_local_readiness_checks_captured_consumers_not_unrelated_optional_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider update checks a required child owned by its hard consumer."""
    monkeypatch.setenv("LOCAL_D_FAIL", "no")
    key = "from agent.plugin_composition import ServiceKey\nLOCAL_KEY = ServiceKey('local.changed')\n"
    changed = _write_plugin(
        tmp_path / "plugins", "changed", _v3_source(
            "changed", version="1.0.0", exports=key,
            body="    await ctx.provide(LOCAL_KEY, {'version': 'a'})\n",
        ),
    )
    _write_plugin(
        tmp_path / "plugins", "stable_host", _v3_source(
            "stable_host", exports=key,
            body=(
                "    async def injected(child):\n"
                "        child.require(LOCAL_KEY)\n"
                "    await ctx.mount(\n"
                "        injected, name='host-injected', inject=(LOCAL_KEY,),\n"
                "        required_for_readiness=False,\n"
                "    )\n"
                "    async def unrelated(child):\n"
                "        raise RuntimeError('unrelated child failed')\n"
                "    await ctx.mount(\n"
                "        unrelated, name='unrelated-child',\n"
                "        required_for_readiness=False,\n"
                "    )\n"
            ),
        ),
    )
    _write_plugin(
        tmp_path / "plugins", "consumer_parent", _v3_source(
            "consumer_parent",
            exports=key + "inject = (LOCAL_KEY,)\n",
            body=(
                "    ctx.require(LOCAL_KEY)\n"
                "    async def required_child(child):\n"
                "        import os\n"
                "        if os.environ['LOCAL_D_FAIL'] == 'yes':\n"
                "            raise RuntimeError('required child D failed')\n"
                "    await ctx.mount(required_child, name='required-D')\n"
            ),
        ),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        root = manager._live_root
        assert root is not None
        stable_host = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "stable_host"
        )
        stable_host_context = stable_host.context
        unrelated_child = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "unrelated-child"
        )
        consumer = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "consumer_parent"
        )
        old_consumer_context = consumer.context
        old_required_d = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "required-D"
        )
        (changed / "plugin.py").write_text(
            _v3_source(
                "changed", version="2.0.0", exports=key,
                body="    await ctx.provide(LOCAL_KEY, {'version': 'b'})\n",
            ), encoding="utf-8"
        )
        await manager.reconcile_changed()
        assert manager._live_root is root
        assert consumer.state == FiberState.ACTIVE
        assert consumer.context is not old_consumer_context
        new_required_d = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "required-D"
        )
        assert new_required_d is not old_required_d
        assert old_required_d not in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
        assert stable_host.context is stable_host_context
        assert stable_host.state == FiberState.ACTIVE
        assert unrelated_child.state == FiberState.FAILED
        assert unrelated_child in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setenv("LOCAL_D_FAIL", "yes")
        (changed / "plugin.py").write_text(
            _v3_source(
                "changed", version="3.0.0", exports=key,
                body="    await ctx.provide(LOCAL_KEY, {'version': 'c'})\n",
            ), encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="required-D"):
            await manager.reconcile_changed()
        assert manager._live_root is root
        assert stable_host.state == FiberState.ACTIVE
        assert stable_host.context is stable_host_context
        assert unrelated_child.state == FiberState.FAILED
        assert unrelated_child in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_local_retry_rechecks_failed_current_hard_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed consumer with an emptied dependency store is found by B's provider edge."""
    monkeypatch.setenv("LOCAL_CONSUMER_START", "yes")
    key = "from agent.plugin_composition import ServiceKey\nLOCAL_KEY = ServiceKey('local.retry.key')\n"
    provider = _write_plugin(
        tmp_path / "plugins", "retry_provider", _v3_source(
            "retry_provider", version="1.0.0", exports=key,
            body="    await ctx.provide(LOCAL_KEY, {'version': 'a'})\n",
        ),
    )
    _write_plugin(
        tmp_path / "plugins", "retry_consumer", _v3_source(
            "retry_consumer", exports=key + "inject = (LOCAL_KEY,)\n",
            body=(
                "    ctx.require(LOCAL_KEY)\n"
                "    import os\n"
                "    if os.environ['LOCAL_CONSUMER_START'] != 'yes':\n"
                "        raise RuntimeError('consumer start blocked')\n"
            ),
        ),
    )
    _write_plugin(
        tmp_path / "plugins", "retry_unrelated", _v3_source("retry_unrelated"),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        root = manager._live_root
        unrelated = manager.generation("retry_unrelated")
        assert root is not None and unrelated is not None
        (provider / "plugin.py").write_text(
            _v3_source(
                "retry_provider", version="2.0.0", exports=key,
                body="    await ctx.provide(LOCAL_KEY, {'version': 'b'})\n",
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("LOCAL_CONSUMER_START", "no")
        with pytest.raises(RuntimeError, match="retry_consumer"):
            await manager.reconcile_changed()
        selected = manager._selection.read()
        assert selected is not None
        with pytest.raises(RuntimeError, match="retry_consumer"):
            await manager.retry_runtime_recovery("retry_provider")
        assert manager._live_root is root
        assert manager.generation("retry_unrelated") is unrelated
        monkeypatch.setenv("LOCAL_CONSUMER_START", "yes")
        result = await manager.retry_runtime_recovery("retry_provider")
        assert result["publication_state"] == "recovered"
        assert manager._selection.read() == selected
        assert manager._live_root is root
        assert manager.generation("retry_unrelated") is unrelated
        consumer = manager._active_generations["retry_consumer"]
        assert consumer.fiber is not None and consumer.fiber.state == FiberState.ACTIVE
    finally:
        await manager.terminate_all()
