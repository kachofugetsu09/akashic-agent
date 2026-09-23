"""The offline input builder and online loader share one archive fact."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.config_input import load_config, save_config
from agent.plugin_composition.context import Fiber
from agent.plugins.archive import PluginArchive, encode_config
from agent.plugins.generation import PluginGeneration
from agent.plugins.input_preparation import PLUGIN_ARCHIVE_BINDING_API, prepare_plugin_input
from agent.plugins.manager import PluginManager
from agent.plugins.scope import PluginScope
from agent.plugins.selection import PluginSelection
from agent.plugins.static_manifest import (
    PluginSourceCompileError,
    load_static_plugin_manifest,
)
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


def _plugin(root: Path, name: str, *, body: str = "") -> Path:
    """Write a checked dynamic plugin source without importing it."""
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    entry = plugin_dir / "plugin.py"
    source = (
        f"name = {name!r}\nversion = '1.0.0'\napi_version = 3\n"
        f"{body}"
        "async def apply(ctx):\n    return None\n"
    )
    compile(ast.parse(source, filename=str(entry)), str(entry), "exec")
    entry.write_text(source, encoding="utf-8")
    return plugin_dir


def _mod(plugin_dir: Path) -> dict[str, str]:
    return {
        "name": plugin_dir.name,
        "plugin_root": str(plugin_dir),
        "module_path": str(plugin_dir / "plugin.py"),
        "manifest_digest": load_static_plugin_manifest(plugin_dir).identity_digest,
        "marketplace": "",
        "source_type": "builtin",
    }


@pytest.mark.asyncio
async def test_online_loader_uses_the_same_complete_input(tmp_path: Path) -> None:
    plugin_dir = _plugin(tmp_path / "plugins", "shared")
    helper = plugin_dir / "helper.py"
    helper_source = "VALUE = 7\n"
    compile(ast.parse(helper_source, filename=str(helper)), str(helper), "exec")
    helper.write_text(helper_source, encoding="utf-8")
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    data_dir = workspace / "plugin-data" / "shared-builtin"
    save_config(data_dir, {"label": "shared", "count": 2})
    archive = PluginArchive(workspace / "runtime" / "plugin-archives")
    owner = PluginManager(
        [tmp_path / "plugins"], event_bus=EventBus(), workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )
    mod = owner.discover()[0]
    prepared = prepare_plugin_input(mod, workspace=workspace, archive=archive)
    descriptor = archive.read_descriptor(prepared.archive_ref)
    descriptor_bytes = (archive.path / f"{prepared.archive_ref}.json").read_bytes()
    config, config_revision = load_config(data_dir)
    assert json.loads(descriptor_bytes) == {
        "version": 4,
        "code": prepared.code_ref,
        "python_environments": {},
        "plugin_id": "shared",
        "source_revision": prepared.source_revision,
        "config_revision": config_revision,
        "config": encode_config(config),
        "source_type": "builtin",
        "data_dir": "plugin-data/shared-builtin",
        "runtime": {
            "python_tag": sys.implementation.cache_tag,
            "binding_api": PLUGIN_ARCHIVE_BINDING_API,
        },
    }
    assert {path.name for path in prepared.code_dir.iterdir()} == {"plugin.py", "helper.py"}

    generation = await owner._load_one(mod, activate=False, stage_stable=True)
    assert generation is not None
    try:
        assert generation.archive_ref == prepared.archive_ref
        assert archive.read_descriptor(generation.archive_ref) == descriptor
        assert (archive.path / f"{generation.archive_ref}.json").read_bytes() == descriptor_bytes
        assert generation.plugin_id == prepared.plugin_id
        assert generation.source_revision == prepared.source_revision
        assert generation.config_revision == prepared.config_revision
        assert generation.config_projection == prepared.config == {"label": "shared", "count": 2}
        assert generation.static_manifest == prepared.static_manifest
        assert generation.plugin_dir == prepared.plugin_dir
        assert generation.code_dir == prepared.code_dir
        assert generation.data_dir == prepared.data_dir
        assert generation.source_type == prepared.source_type
    finally:
        await owner._dispose_generation(generation, state="discarded")
        await owner.terminate_all()


def test_prepare_compiles_without_running_or_creating_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = _plugin(
        tmp_path / "plugins", "silent",
        body="raise AssertionError('plugin source ran')\n",
    )
    workspace = tmp_path / "workspace"
    archive = PluginArchive(workspace / "runtime" / "plugin-archives")
    modules_before = set(sys.modules)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("runtime owner created during input preparation")

    for owner in (CompositionRoot, Fiber, PluginGeneration, PluginScope):
        monkeypatch.setattr(owner, "__init__", forbidden)
    prepared = prepare_plugin_input(_mod(plugin_dir), workspace=workspace, archive=archive)

    assert prepared.plugin_id == "silent"
    assert archive.read_descriptor(prepared.archive_ref)["code"] == prepared.code_ref
    assert not any(name.startswith("_akashic_input_") for name in set(sys.modules) - modules_before)


def test_bad_secondary_source_preserves_compile_cause_and_selection(
    tmp_path: Path,
) -> None:
    plugin_dir = _plugin(tmp_path / "plugins", "broken")
    # This source is intentionally invalid; the checked plugin.py above is valid.
    (plugin_dir / "helper.py").write_text("def invalid(:\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    selection = PluginSelection(workspace)
    selected_before = selection.read()
    archive = PluginArchive(workspace / "runtime" / "plugin-archives")

    with pytest.raises(PluginSourceCompileError) as caught:
        prepare_plugin_input(_mod(plugin_dir), workspace=workspace, archive=archive)

    assert isinstance(caught.value.__cause__, SyntaxError)
    assert "helper.py" in str(caught.value)
    assert list(archive.path.glob("*.json")) == []
    assert selection.read() == selected_before
