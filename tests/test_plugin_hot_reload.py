from __future__ import annotations

import asyncio
import importlib
import os
import py_compile
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.convertors import CONVERTOR_TYPES, StringConvertor

from agent.looping.core import AgentLoop
from agent.looping.session_lane import SessionLaneRegistry
from agent.plugins.artifacts import ArtifactPointer, read_pointer, write_pointers
from agent.plugins.dashboard_host import (
    DashboardBinding,
    _plugin_routes,
    _require_routes_available,
)
from agent.plugins.manager import PluginManager, _source_revision
from agent.plugins.manifest import write_plugin_manifest
from agent.plugins.skill_host import SkillSnapshot
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
)
from agent.plugins.watcher import PluginWatcher
from agent.skills import SkillsLoader
from agent.tools.registry import ToolRegistry
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
        "async def apply(ctx, config):\n"
        f"{body}"
    )


def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir


def _manager(
    tmp_path: Path,
    *,
    tools: ToolRegistry | None = None,
    workspace: Path | None = None,
) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=tools,
        workspace=workspace or tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


def _write_static_manifest(
    root: Path,
    *,
    name: str,
    version: str,
    entrypoint: str = "plugin.py",
    python_runtime: str | None = None,
) -> None:
    lines = [
        "schema_version = 1",
        f"name = {name!r}",
        f"version = {version!r}",
        "api_version = 3",
        f"entrypoint = {entrypoint!r}",
        "",
    ]
    if python_runtime is not None:
        lines.extend(
            [
                "[[python]]",
                'requirements = "requirements.txt"',
                f'runtime_root = {python_runtime!r}',
                "",
            ]
        )
    (root / "akashic.plugin.toml").write_text("\n".join(lines), encoding="utf-8")


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
    marker = "name = "
    name_line = next(line for line in source.splitlines() if line.startswith(marker))
    name = name_line.split("=", 1)[1].strip().strip("'\"")
    version_line = next(
        line for line in source.splitlines() if line.startswith("version = ")
    )
    version = version_line.split("=", 1)[1].strip().strip("'\"")
    _write_static_manifest(artifact, name=name, version=version)
    return plugin_base, artifact


def _write_installed_skill(plugin_root: Path, name: str, body: str) -> Path:
    skill_dir = plugin_root / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    return skill_dir


def test_skill_snapshot_cleanup_removes_readonly_image_copies() -> None:
    snapshot = SkillSnapshot()
    nested = snapshot.root / "selected" / "skill"
    nested.mkdir(parents=True)
    skill_file = nested / "SKILL.md"
    skill_file.write_text("# test\n", encoding="utf-8")
    skill_file.chmod(0o444)
    nested.chmod(0o555)

    snapshot.cleanup()

    assert not snapshot.root.exists()


@pytest.mark.asyncio
async def test_candidate_gate_publishes_unique_generation(tmp_path: Path):
    _write_plugin(tmp_path / "plugins", "candidate", _v3_source("candidate"))
    manager = _manager(tmp_path)

    await manager.load_all()

    generation = manager.generation("candidate")
    gate = manager.latest_gate("candidate")
    assert generation is not None
    assert gate is not None and gate.status == "passed"
    assert generation.module_path.startswith("akasic_plugin_plugins_candidate__g")
    assert generation.instance.module.__name__ == generation.module_path
    assert generation.instance.version == "1.0.0"
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_static_semantic_failure_never_prepares_candidate(tmp_path: Path):
    source = _v3_source(
        "bad_semantic",
        exports=(
            "from agent.plugins.generation import PluginSemanticCheck\n"
            "def static_semantic_checks():\n"
            "    return [PluginSemanticCheck('model', False, 'missing')]\n"
        ),
    )
    _write_plugin(tmp_path / "plugins", "bad_semantic", source)
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("bad_semantic")
    assert manager.loaded_count == 0
    assert manager.generation("bad_semantic") is None
    assert gate is not None and gate.status == "failed"
    assert any(
        check.check_id == "semantic_checks" and check.status == "failed"
        for check in gate.checks
    )


@pytest.mark.asyncio
async def test_import_failure_returns_failed_gate_without_generation(tmp_path: Path):
    _write_plugin(tmp_path / "plugins", "broken", "this is not python !!!\n")
    manager = _manager(tmp_path)

    with pytest.raises(RuntimeError, match="插件 broken 导入失败"):
        await manager.load_all()

    gate = manager.latest_gate("broken")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id == "import"
    assert manager.generation("broken") is None


@pytest.mark.asyncio
async def test_candidate_failure_is_bound_to_requested_plugin(tmp_path: Path):
    root = tmp_path / "plugins"
    _write_plugin(root, "first", _v3_source("first"))
    _write_plugin(root, "second", _v3_source("second"))
    manager = _manager(tmp_path)
    await manager.load_all()

    for name in ("first", "second"):
        (root / name / "plugin.py").write_text(
            f"this is not valid python for {name} !!!\n", encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match=f"插件 {name} 导入失败"):
            await manager.prepare_candidate(name)

    first = manager.candidate_status("first")
    second = manager.candidate_status("second")
    assert first["candidate_plugin_id"] == "first"
    assert second["candidate_plugin_id"] == "second"
    assert first["candidate_reload_tx_id"] != second["candidate_reload_tx_id"]
    assert first["candidate_state"] == second["candidate_state"] == "aborted"
    assert "import:" in str(first["candidate_error"])
    assert "import:" in str(second["candidate_error"])
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_same_source_gets_new_generation_namespace_after_restart(tmp_path: Path):
    _write_plugin(tmp_path / "plugins", "repeat", _v3_source("repeat"))
    manager = _manager(tmp_path)
    await manager.load_all()
    first = manager.generation("repeat")
    assert first is not None

    await manager.terminate_all()
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
    (config_dir / "config.local.toml").write_text("", encoding="utf-8")
    manager = _manager(tmp_path)

    await manager.load_all()
    assert manager.latest_gate("module_tree").status == "failed"  # type: ignore[union-attr]
    assert not any("plugins_module_tree__g" in name for name in sys.modules)

    (config_dir / "config.local.toml").write_text("required = 'ok'\n", encoding="utf-8")
    await manager.load_all()
    generation = manager.generation("module_tree")
    assert generation is not None
    assert f"{generation.module_path}.child" in sys.modules
    stable_child = importlib.import_module("akasic_plugin_plugins_module_tree.child")
    assert stable_child.value == 1

    await manager.terminate_all()
    assert not any("plugins_module_tree__g" in name for name in sys.modules)
    assert "akasic_plugin_plugins_module_tree.child" not in sys.modules


@pytest.mark.asyncio
async def test_source_revision_includes_helper_changes(tmp_path: Path):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "revision",
        _v3_source("revision", exports="from . import helper\n"),
    )
    helper = plugin_dir / "helper.py"
    helper.write_text("value = 1\n", encoding="utf-8")
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
    outside = tmp_path / "plugins" / "outside" / "skill"
    outside.mkdir(parents=True)
    (outside / "SKILL.md").write_text("# outside\n", encoding="utf-8")
    _write_plugin(
        tmp_path / "plugins",
        "escaped",
        _v3_source("escaped", exports="skill_roots = ('../outside',)\n"),
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("escaped")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id in {"declarations", "identity"}


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
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("linked_source")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id == "source_boundary"


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
async def test_skill_catalog_rejects_cross_plugin_duplicates(tmp_path: Path):
    first_dir = _write_plugin(
        tmp_path / "plugins",
        "first_skills",
        _v3_source("first_skills", exports="skill_roots = ('skills',)\n"),
    )
    first_skill = first_dir / "skills" / "shared"
    first_skill.mkdir(parents=True)
    (first_skill / "SKILL.md").write_text(
        "---\ndescription: first\n---\nfirst\n", encoding="utf-8"
    )
    manager = _manager(tmp_path, workspace=tmp_path / "workspace")
    await manager.load_all()
    first = manager.generation("first_skills")
    assert first is not None and first.skill_catalog is not None
    assert first.skill_catalog.normal.get("shared").source_id == "first_skills"  # type: ignore[union-attr]

    second_dir = _write_plugin(
        tmp_path / "plugins",
        "second_skills",
        _v3_source("second_skills", exports="skill_roots = ('skills',)\n"),
    )
    second_skill = second_dir / "skills" / "shared"
    second_skill.mkdir(parents=True)
    (second_skill / "SKILL.md").write_text(
        "---\ndescription: second\n---\nsecond\n", encoding="utf-8"
    )

    await manager.load_all()

    gate = manager.latest_gate("second_skills")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[-1].check_id == "skill_catalog"
    assert manager.generation("second_skills") is None
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_skill_catalog_freezes_generation_and_ignores_old_root_link(tmp_path: Path):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "skill_reload",
        _v3_source("skill_reload", exports="skill_roots = ('skills-a',)\n"),
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
    manager = _manager(tmp_path, workspace=workspace)
    await manager.load_all()
    active = manager.generation("skill_reload")
    assert active is not None and active.skill_catalog is not None
    active_record = active.skill_catalog.normal.get("shared")
    assert active_record is not None

    release_b_skill = plugin_dir / "skills-b" / "shared"
    release_b_skill.mkdir(parents=True)
    (release_b_skill / "SKILL.md").write_text(
        "---\ndescription: release b\n---\nbody b\n", encoding="utf-8"
    )
    (plugin_dir / "plugin.py").write_text(
        _v3_source("skill_reload", exports="skill_roots = ('skills-b',)\n"),
        encoding="utf-8",
    )

    prepared = await manager.prepare_candidate("skill_reload")

    assert prepared is not None and prepared.skill_catalog is not None
    prepared_record = prepared.skill_catalog.normal.get("shared")
    assert prepared_record is not None
    assert active_record.description == "release a"
    assert prepared_record.description == "release b"
    assert active_record.root_dir != prepared_record.root_dir
    await manager.discard_prepared("skill_reload")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_skill_catalog_cleanup_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "skill_cleanup",
        _v3_source("skill_cleanup", exports="skill_roots = ('skills',)\n"),
    )
    (tmp_path / "plugins" / "skill_cleanup" / "skills").mkdir()
    manager = _manager(tmp_path)
    await manager.load_all()
    generation = manager.generation("skill_cleanup")
    assert generation is not None and generation.skill_catalog is not None
    snapshot_root = generation.skill_catalog.snapshot_root
    real_rmtree = shutil.rmtree

    def fail_snapshot_cleanup(path: Path, *args: Any, **kwargs: Any) -> None:
        if Path(path) == snapshot_root and not args and not kwargs:
            raise OSError("snapshot cleanup failed")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fail_snapshot_cleanup)
    await manager.terminate_all()

    assert any(
        failure.resource == "skill_catalog" and failure.error == "snapshot cleanup failed"
        for failure in manager.cleanup_failures
    )


def _installed_snapshot_source(
    version: str,
    *,
    skills: bool = False,
) -> str:
    exports = "skill_roots = ('skills',)\n" if skills else ""
    return _v3_source("installed_snapshot", version=version, exports=exports)


def _installed_command_source(description: str, version: str) -> str:
    return (
        "from agent.plugin_composition import COMMANDS, CommandDefinition, CommandResult\n"
        "api_version = 3\n"
        "name = 'installed_commands'\n"
        f"version = {version!r}\n"
        "inject = (COMMANDS,)\n"
        "async def apply(ctx, config):\n"
        "    async def handler(invocation):\n"
        "        return CommandResult('success', invocation.raw_input or 'ok')\n"
        "    await ctx.require(COMMANDS).register(ctx, CommandDefinition(\n"
        f"        name='hello', description={description!r}, handler=handler))\n"
    )


@pytest.mark.asyncio
async def test_installed_candidate_requires_explicit_promote_or_discard(tmp_path: Path) -> None:
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
    write_plugin_manifest({"installed_snapshot@lab": True}, plugins_home=tmp_path / "home")
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
    assert stable_lease.snapshot.generations["installed_snapshot@lab"].instance.version == "release-a"
    assert latest_lease.snapshot.generations["installed_snapshot@lab"].instance.version == "release-b"
    await stable_lease.release()
    await latest_lease.release()

    discarded = await manager.drop_candidate("installed_snapshot@lab")
    assert discarded["publication_state"] == "discarded"
    assert read_pointer(plugin_base, "stable") == stable_pointer
    assert read_pointer(plugin_base, "latest") == stable_pointer
    assert not latest_root.samefile(stable_root)

    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
    promoted = await manager.switch_ready("installed_snapshot@lab")
    assert promoted["publication_state"] == "promoted"
    assert manager.generation("installed_snapshot@lab").instance.version == "release-b"  # type: ignore[union-attr]
    assert read_pointer(plugin_base, "stable") == latest_pointer

    write_pointers(plugin_base, stable=latest_pointer, latest=next_pointer)
    assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
    await manager.switch_ready("installed_snapshot@lab")
    assert manager.generation("installed_snapshot@lab").instance.version == "release-c"  # type: ignore[union-attr]
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_installed_candidate_promotion_syncs_stable_skill_projection(tmp_path: Path) -> None:
    plugin_base, stable_root = _write_installed_artifact(
        tmp_path, "1.0.0-aaaa", _installed_snapshot_source("release-a", skills=True)
    )
    _, candidate_root = _write_installed_artifact(
        tmp_path, "2.0.0-bbbb", _installed_snapshot_source("release-b", skills=True)
    )
    _write_installed_skill(stable_root, "stable-skill", "stable body\n")
    _write_installed_skill(candidate_root, "candidate-skill", "candidate body\n")
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    candidate_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    workspace = tmp_path / "workspace"
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    manager.sync_skill_links()
    loader = SkillsLoader(workspace, builtin_skills_dir=tmp_path / "builtin")
    stable_link = workspace / "skills" / "stable-skill"
    assert stable_link.resolve() == stable_root / "skills" / "stable-skill"
    assert loader.load_skill_body("stable-skill") == "stable body\n"

    write_pointers(plugin_base, stable=stable_pointer, latest=candidate_pointer)
    assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
    assert stable_link.resolve() == stable_root / "skills" / "stable-skill"
    assert not (workspace / "skills" / "candidate-skill").exists()
    await manager.drop_candidate("installed_snapshot@lab")

    write_pointers(plugin_base, stable=stable_pointer, latest=candidate_pointer)
    promoted = await manager.switch_ready("installed_snapshot@lab") if manager.ready_candidate else None
    if promoted is None:
        assert (await manager.reconcile_changed())[0]["publication_state"] == "latest_ready"
        promoted = await manager.switch_ready("installed_snapshot@lab")
    candidate_link = workspace / "skills" / "candidate-skill"
    assert promoted["publication_state"] == "promoted"
    assert not stable_link.exists()
    assert candidate_link.resolve() == candidate_root / "skills" / "candidate-skill"
    assert loader.load_skill_body("candidate-skill") == "candidate body\n"
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_skill_projection_conflict_fails_before_stable_promotion(tmp_path: Path) -> None:
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
    with pytest.raises(RuntimeError, match="用户文件或目录冲突"):
        await manager.switch_ready("installed_snapshot@lab")
    assert manager.current_snapshot is stable_snapshot
    assert manager.generation("installed_snapshot@lab") is stable_generation
    assert read_pointer(plugin_base, "stable") == stable_pointer
    assert personal.is_dir() and not personal.is_symlink()
    assert (personal / "SKILL.md").read_text(encoding="utf-8") == "user body\n"
    await manager.drop_candidate("installed_snapshot@lab")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_rejected_installed_candidate_restores_latest_to_stable(tmp_path: Path) -> None:
    plugin_base, _ = _write_installed_artifact(
        tmp_path, "1.0.0-aaaa", _installed_snapshot_source("release-a")
    )
    _, _ = _write_installed_artifact(
        tmp_path,
        "2.0.0-bbbb",
        _v3_source(
            "installed_snapshot",
            version="release-b",
            exports=(
                "from agent.plugins.generation import PluginSemanticCheck\n"
                "def static_semantic_checks():\n"
                "    return [PluginSemanticCheck('candidate', False, 'rejected')]\n"
            ),
        ),
    )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    write_plugin_manifest({"installed_snapshot@lab": True}, plugins_home=tmp_path / "home")
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    results = await manager.reconcile_changed()
    assert results[0]["prepared_generation"] is None
    assert manager.generation("installed_snapshot@lab").instance.version == "release-a"  # type: ignore[union-attr]
    assert read_pointer(plugin_base, "stable") == stable_pointer
    assert read_pointer(plugin_base, "latest") == stable_pointer
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("promoted_on_disk", [False, True])
async def test_startup_recovers_installed_candidate_from_durable_pointers(
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
    write_pointers(
        plugin_base,
        stable=latest_pointer if promoted_on_disk else stable_pointer,
        latest=latest_pointer,
    )
    write_plugin_manifest({"installed_snapshot@lab": True}, plugins_home=tmp_path / "home")
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
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

    assert manager.reload_journal.get(tx_id).phase == (
        "recovered" if promoted_on_disk else "aborted"
    )
    expected = "release-b" if promoted_on_disk else "release-a"
    assert manager.generation("installed_snapshot@lab").instance.version == expected  # type: ignore[union-attr]
    assert manager.ready_candidate is None
    if not promoted_on_disk:
        assert stable_root.exists()
        assert read_pointer(plugin_base, "latest") == stable_pointer
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_snapshot_admission_waits_while_current_is_quiesced(tmp_path: Path) -> None:
    _write_plugin(tmp_path / "plugins", "snapshot_admission", _v3_source("snapshot_admission"))
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
async def test_runtime_snapshot_lease_commit_and_abort(tmp_path: Path) -> None:
    _write_plugin(tmp_path / "plugins", "snapshot", _v3_source("snapshot"))
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot")
    prepared = await manager.prepare_candidate("snapshot")
    installed = manager.current_snapshot
    assert active is not None and prepared is not None and installed is not None
    compiler = RuntimeSnapshotCompiler()
    v1 = compiler.compile({"snapshot": active}, catalog_generation=active)
    next_snapshot = compiler.compile({"snapshot": prepared}, catalog_generation=prepared)
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
    next_snapshot = compiler.compile({"snapshot": prepared}, catalog_generation=prepared)
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
async def test_runtime_snapshot_latest_requires_explicit_selector_and_promotion(tmp_path: Path) -> None:
    _write_plugin(tmp_path / "plugins", "snapshot_selector", _v3_source("snapshot_selector"))
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot_selector")
    prepared = await manager.prepare_candidate("snapshot_selector")
    assert active is not None and prepared is not None
    compiler = RuntimeSnapshotCompiler()
    stable = compiler.compile({"snapshot_selector": active}, snapshot_revision="stable")
    latest = compiler.compile({"snapshot_selector": prepared}, snapshot_revision="latest")
    drained: list[str] = []

    async def on_drained(snapshot: RuntimeSnapshot) -> None:
        drained.append(snapshot.snapshot_id)

    store = RuntimeSnapshotStore(on_drained)
    store.install(stable)
    await store.commit_latest(store.begin_publish(latest))
    stable_lease = store.lease()
    latest_lease = store.lease(selector="latest")
    assert stable_lease.snapshot is stable
    assert latest_lease.snapshot is latest
    with pytest.raises(RuntimeError, match="等待 promote/discard"):
        store.begin_publish(compiler.compile({"snapshot_selector": prepared}, snapshot_revision="next"))
    store.pause_candidate_admission(latest)
    await latest_lease.release()
    await store.wait_for_no_leases(latest)
    promoted = await store.promote_latest()
    assert promoted.previous is stable
    assert store.stable is latest
    await stable_lease.release()
    await store.retry_drains()
    assert drained == [stable.snapshot_id]
    await store.close()
    await manager.discard_prepared("snapshot_selector")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_runtime_snapshot_discard_keeps_stable_and_waits_for_latest_lease(tmp_path: Path) -> None:
    _write_plugin(tmp_path / "plugins", "snapshot_discard", _v3_source("snapshot_discard"))
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot_discard")
    prepared = await manager.prepare_candidate("snapshot_discard")
    assert active is not None and prepared is not None
    compiler = RuntimeSnapshotCompiler()
    stable = compiler.compile({"snapshot_discard": active}, snapshot_revision="stable")
    latest = compiler.compile({"snapshot_discard": prepared}, snapshot_revision="latest")
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
    assert store.stable is stable
    assert store.latest is stable
    await stable_lease.release()
    await store.close()
    await manager.discard_prepared("snapshot_discard")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_passive_runtime_admission_holds_one_snapshot(tmp_path: Path) -> None:
    _write_plugin(tmp_path / "plugins", "passive_snapshot", _v3_source("passive_snapshot"))
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("passive_snapshot")
    prepared = await manager.prepare_candidate("passive_snapshot")
    assert active is not None and prepared is not None
    compiler = RuntimeSnapshotCompiler()
    v1 = compiler.compile({"passive_snapshot": active}, catalog_generation=active)
    next_snapshot = compiler.compile({"passive_snapshot": prepared}, catalog_generation=prepared)
    store = RuntimeSnapshotStore()
    store.install(v1)
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = store
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: list[str] = []

    async def process(_msg, **_kwargs):
        from agent.plugins.snapshot import get_current_runtime_snapshot

        snapshot = get_current_runtime_snapshot()
        assert snapshot is not None
        seen.append(snapshot.snapshot_id)
        entered.set()
        await release.wait()
        assert get_current_runtime_snapshot() is snapshot
        seen.append(snapshot.snapshot_id)
        return "done"

    loop._process = process
    message = cast(Any, SimpleNamespace(session_key="cli:snapshot"))
    running = asyncio.create_task(loop._process_with_runtime_admission(message))
    await entered.wait()
    await store.commit(store.begin_publish(next_snapshot))
    release.set()
    assert await running == "done"
    assert seen == [v1.snapshot_id, v1.snapshot_id]
    await loop._process_with_runtime_admission(message)
    assert seen[-2:] == [next_snapshot.snapshot_id, next_snapshot.snapshot_id]
    await store.close()
    await manager.discard_prepared("passive_snapshot")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_reconcile_changed_adds_and_removes_discovered_plugin(tmp_path: Path) -> None:
    plugins = tmp_path / "plugins"
    _write_plugin(plugins, "anchor", _v3_source("anchor"))
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
async def test_plugin_watcher_reloads_v3_source_without_signal(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "watched",
        _v3_source("watched", version="release-a"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    baseline_revision = await asyncio.to_thread(manager.watch_revision)
    watcher = PluginWatcher(manager, baseline_revision=baseline_revision, interval_seconds=0.01)
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
    watcher = PluginWatcher(cast(PluginManager, manager), baseline_revision="stable", interval_seconds=0.01)
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

    watcher = PluginWatcher(cast(PluginManager, Manager()), baseline_revision="stable", interval_seconds=0.01)
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
) -> None:
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "home"))
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_dashboard",
        _v3_source("snapshot_dashboard", exports="dashboard_module = 'dashboard.py'\n"),
    )

    def write_dashboard(version: str) -> None:
        (plugin_dir / "dashboard.py").write_text(
            "def register(app, context):\n"
            "    @app.get('/api/dashboard/snapshot-version')\n"
            f"    def version(): return {{'version': '{version}'}}\n"
            "    class Closeable:\n"
            "        def close(self):\n"
            f"            (context.data_root / 'dashboard-{version}-closed').write_text('closed')\n"
            "    return Closeable()\n",
            encoding="utf-8",
        )

    write_dashboard("release-a")
    manager = _manager(tmp_path)
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    assert old_snapshot is not None
    old_generation = old_snapshot.generations["snapshot_dashboard"]
    old_lease = manager.snapshot_store.lease()
    app = create_dashboard_app(
        tmp_path / "workspace",
        memory_admin=cast(Any, SimpleNamespace()),
        plugin_manager=manager,
    )
    client = TestClient(app)
    assert client.get("/api/dashboard/snapshot-version").json() == {"version": "release-a"}
    write_dashboard("release-b")
    assert await manager.prepare_candidate("snapshot_dashboard") is not None
    await manager.publish_prepared("snapshot_dashboard")
    assert client.get("/api/dashboard/snapshot-version").json() == {"version": "release-b"}
    old_binding = old_snapshot.dashboard_bindings[0]
    assert TestClient(old_binding.app).get("/api/dashboard/snapshot-version").json() == {"version": "release-a"}  # type: ignore[attr-defined]
    assert not (old_generation.data_dir / "dashboard-release-a-closed").exists()
    await old_lease.release()
    await manager.snapshot_store.retry_drains()
    assert (old_generation.data_dir / "dashboard-release-a-closed").exists()
    assert old_generation.scope.closed
    client.close()
    await manager.terminate_all()


def test_dashboard_rejects_custom_path_convertor(monkeypatch: pytest.MonkeyPatch) -> None:
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
def test_dashboard_treats_missing_methods_as_wildcard(wildcard_methods: set[str] | None) -> None:
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


@pytest.mark.asyncio
async def test_skill_body_stays_on_snapshot_generation(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "snapshot_skill"
    for release, body in (("a", "body a"), ("b", "body b")):
        skill_dir = plugin_dir / f"skills-{release}" / "snapshot-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\ndescription: snapshot skill {release}\n---\n{body}\n",
            encoding="utf-8",
        )
    plugin_file = plugin_dir / "plugin.py"
    plugin_file.write_text(
        _v3_source("snapshot_skill", exports="skill_roots = ('skills-a',)\n"),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    manager = _manager(tmp_path, workspace=workspace)
    await manager.load_all()
    workspace_skills = workspace / "skills"
    workspace_skills.mkdir()
    (workspace_skills / "snapshot-skill").symlink_to(
        plugin_dir / "skills-a" / "snapshot-skill", target_is_directory=True
    )
    plugin_file.write_text(
        _v3_source("snapshot_skill", version="1.0.1", exports="skill_roots = ('skills-b',)\n"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_skill")
    assert candidate is not None
    skills = SkillsLoader(workspace, runtime_catalog="normal")
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = manager.snapshot_store
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: list[str | None] = []

    async def process(_msg, **_kwargs):
        seen.append(skills.load_skill_body("snapshot-skill"))
        entered.set()
        await release.wait()
        seen.append(skills.load_skill_body("snapshot-skill"))
        return "done"

    loop._process = process
    message = cast(Any, SimpleNamespace(session_key="cli:snapshot-skill"))
    old_turn = asyncio.create_task(loop._process_with_runtime_admission(message))
    await entered.wait()
    await manager.publish_prepared("snapshot_skill")
    (plugin_dir / "skills-a" / "snapshot-skill" / "SKILL.md").write_text(
        "---\ndescription: mutated\n---\nmutated\n", encoding="utf-8"
    )
    release.set()
    assert await old_turn == "done"
    await loop._process_with_runtime_admission(message)
    assert seen[:2] == ["body a", "body a"]
    assert seen[2:] == ["body b", "body b"]
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_workspace_skill_updates_without_plugin_snapshot_reload(tmp_path: Path) -> None:
    _write_plugin(tmp_path / "plugins", "workspace_skill_snapshot", _v3_source("workspace_skill_snapshot"))
    workspace = tmp_path / "workspace"
    skill_dir = workspace / "skills" / "workspace-live"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\ndescription: workspace live\n---\nworkspace release a\n", encoding="utf-8"
    )
    manager = _manager(tmp_path, workspace=workspace)
    await manager.load_all()
    snapshot = manager.current_snapshot
    skills = SkillsLoader(workspace, runtime_catalog="normal")
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = manager.snapshot_store
    seen: list[str | None] = []

    async def process(_msg, **_kwargs):
        seen.append(skills.load_skill_body("workspace-live"))
        return "done"

    loop._process = process
    message = cast(Any, SimpleNamespace(session_key="cli:workspace-skill"))
    await loop._process_with_runtime_admission(message)
    skill_file.write_text(
        "---\ndescription: workspace live\n---\nworkspace release b\n", encoding="utf-8"
    )
    await loop._process_with_runtime_admission(message)
    assert manager.current_snapshot is snapshot
    assert seen == ["workspace release a", "workspace release b"]
    await manager.terminate_all()
