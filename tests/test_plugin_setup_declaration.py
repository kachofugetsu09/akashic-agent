from __future__ import annotations

import os
from pathlib import Path

from agent.plugins.source_resolver import ResolvedPluginSource
from agent.plugins.static_manifest import load_static_plugin_manifest
from bootstrap import setup_wizard


def _write_plugin(root: Path) -> None:
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "setup.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['AKASHIC_SETUP_CONFIG_PATH']).write_text(\n"
        "    os.environ['AKASHIC_PLUGIN_ID'], encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'fixture_setup'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n"
        "\n[setup]\nentrypoint = 'setup.py'\n",
        encoding="utf-8",
    )


def test_manifest_exposes_plugin_owned_setup_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "fixture_setup"
    root.mkdir()
    _write_plugin(root)

    manifest = load_static_plugin_manifest(root)

    assert manifest.setup is not None
    assert manifest.setup.entrypoint == "setup.py"


def test_setup_runner_passes_plugin_data_boundary(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "fixture_setup"
    root.mkdir()
    _write_plugin(root)
    manifest = load_static_plugin_manifest(root)
    source = ResolvedPluginSource(
        plugin_root=root,
        source_type="builtin",
        static_manifest=manifest,
    )
    discovery: dict[str, object] = {}

    def discover(*args, **kwargs):
        discovery.update(kwargs)
        return [source]

    monkeypatch.setattr(setup_wizard, "resolve_plugin_sources", discover)
    monkeypatch.setattr(setup_wizard, "plugins_root", lambda: tmp_path / "plugin-home")

    workspace = tmp_path / "workspace"
    setup_wizard._run_declared_plugin_setups(workspace)

    config = workspace / "plugin-data" / "fixture_setup-builtin" / "config.local.toml"
    assert config.read_text(encoding="utf-8") == "fixture_setup"
    assert os.environ.get("AKASHIC_SETUP_CONFIG_PATH") is None
    assert discovery["installed_cache_root"] == tmp_path / "plugin-home" / "cache"
