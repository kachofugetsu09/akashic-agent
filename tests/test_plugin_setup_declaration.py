from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from agent.plugins.python_environment import ENVIRONMENT_FILE, PythonEnvironments
from agent.plugins.install import install_git_plugin
from agent.plugins.manifest import set_plugin_enabled
from agent.plugins.source_resolver import ResolvedPluginSource
from agent.plugins.static_manifest import load_static_plugin_manifest
from bootstrap import setup_wizard


def _write_plugin(root: Path) -> None:
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "setup").mkdir()
    (root / "setup" / "requirements.txt").write_text("", encoding="utf-8")
    (root / "setup.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "import sys\n"
        "Path(os.environ['AKASHIC_SETUP_CONFIG_PATH']).write_text(\n"
        "    os.environ['AKASHIC_PLUGIN_ID'] + '\\n' + sys.prefix,\n"
        "    encoding='utf-8',\n"
        ")\n",
        encoding="utf-8",
    )
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'fixture_setup'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n"
        "\n[[python]]\nrequirements = 'setup/requirements.txt'\n"
        "\n[setup]\nentrypoint = 'setup.py'\n"
        "python_runtime = 'setup'\n",
        encoding="utf-8",
    )


def _commit_source(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=setup-test",
            "-c",
            "user.email=setup-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )


def test_manifest_exposes_plugin_owned_setup_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "fixture_setup"
    root.mkdir()
    _write_plugin(root)

    manifest = load_static_plugin_manifest(root)

    assert manifest.setup is not None
    assert manifest.setup.entrypoint == "setup.py"
    assert manifest.setup.python_runtime == "setup"


def test_manifest_rejects_setup_entrypoint_outside_artifact(tmp_path: Path) -> None:
    root = tmp_path / "fixture_setup"
    root.mkdir()
    _write_plugin(root)
    manifest_path = root / "akashic.plugin.toml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            "entrypoint = 'setup.py'", "entrypoint = '../setup.py'"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact 内的相对路径"):
        load_static_plugin_manifest(root)


def test_wizard_initializes_core_before_plugin_data_setup(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    init_workspace_module = importlib.import_module("bootstrap.init_workspace")
    monkeypatch.setattr(
        setup_wizard,
        "_validate_config",
        lambda config_path, workspace: events.append("validate"),
    )
    monkeypatch.setattr(
        init_workspace_module,
        "init_workspace",
        lambda **kwargs: events.append("init"),
    )
    monkeypatch.setattr(
        setup_wizard,
        "_run_declared_plugin_setups",
        lambda workspace: events.append("plugins"),
    )
    monkeypatch.setattr(
        setup_wizard,
        "_print_completion",
        lambda workspace: None,
    )

    setup_wizard.run_setup_wizard(
        tmp_path / "config.toml",
        tmp_path / "workspace",
    )

    assert events == ["validate", "init", "plugins"]


def test_setup_runner_passes_plugin_data_boundary(tmp_path: Path, monkeypatch) -> None:
    plugin_home = tmp_path / "plugin-home"
    root = plugin_home / "cache" / "lab" / "fixture_setup" / ".artifacts" / "v1"
    root.mkdir(parents=True)
    _write_plugin(root)
    manifest = load_static_plugin_manifest(root)
    workspace = tmp_path / "workspace"
    python_environments = PythonEnvironments(workspace)
    environment_ref = python_environments.prepare(root, manifest.python[0])
    (root / ENVIRONMENT_FILE).write_text(
        json.dumps({manifest.python[0].runtime_root: environment_ref}),
        encoding="utf-8",
    )
    source = ResolvedPluginSource(
        plugin_root=root,
        source_type="installed",
        marketplace="lab",
        plugin_name=manifest.name,
        static_manifest=manifest,
    )
    discovery: dict[str, object] = {}

    def discover(*args, **kwargs):
        discovery["args"] = args
        discovery.update(kwargs)
        return [source]

    monkeypatch.setattr(setup_wizard, "resolve_plugin_sources", discover)
    monkeypatch.setattr(setup_wizard, "plugins_root", lambda: plugin_home)

    setup_wizard._run_declared_plugin_setups(workspace)

    config = workspace / "plugin-data" / "fixture_setup-lab" / "config.local.toml"
    lines = config.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "fixture_setup@lab"
    record = python_environments.archive.read_descriptor(environment_ref)
    archived_code = python_environments.archive.open(record["input"]["code"])
    environment_root = python_environments.open(
        environment_ref,
        archived_code,
        manifest.python[0],
    )
    assert Path(lines[1]) == environment_root / "setup" / ".venv"
    assert os.environ.get("AKASHIC_SETUP_CONFIG_PATH") is None
    assert discovery["args"] == ((),)
    assert discovery["installed_cache_root"] == plugin_home / "cache"


def test_setup_runner_reads_formal_install_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "fixture_setup-source"
    source.mkdir()
    _write_plugin(source)
    _commit_source(source)
    plugin_home = tmp_path / "plugin-home"
    workspace = tmp_path / "workspace"
    _ = install_git_plugin(
        workspace=workspace,
        source=str(source),
        marketplace="lab",
        plugins_home=plugin_home,
    )
    monkeypatch.setattr(setup_wizard, "plugins_root", lambda: plugin_home)

    setup_wizard._run_declared_plugin_setups(workspace)

    config = workspace / "plugin-data" / "fixture_setup-lab" / "config.local.toml"
    assert config.read_text(encoding="utf-8").splitlines()[0] == "fixture_setup@lab"


def test_setup_runner_skips_disabled_installed_plugin(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "fixture_setup-source"
    source.mkdir()
    _write_plugin(source)
    _commit_source(source)
    plugin_home = tmp_path / "plugin-home"
    workspace = tmp_path / "workspace"
    _ = install_git_plugin(
        workspace=workspace,
        source=str(source),
        marketplace="lab",
        plugins_home=plugin_home,
    )
    _ = set_plugin_enabled(
        "fixture_setup@lab",
        enabled=False,
        plugins_home=plugin_home,
    )
    monkeypatch.setattr(setup_wizard, "plugins_root", lambda: plugin_home)

    def unexpected_setup(*args, **kwargs):
        raise AssertionError("disabled plugin setup must not start")

    monkeypatch.setattr(setup_wizard.subprocess, "run", unexpected_setup)
    setup_wizard._run_declared_plugin_setups(workspace)

    assert not (
        workspace / "plugin-data" / "fixture_setup-lab" / "config.local.toml"
    ).exists()


def test_setup_runner_rejects_checkout_plugin_source(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "checkout" / "plugins" / "fixture_setup"
    root.mkdir(parents=True)
    _write_plugin(root)
    manifest = load_static_plugin_manifest(root)
    source = ResolvedPluginSource(
        plugin_root=root,
        source_type="builtin",
        static_manifest=manifest,
    )
    monkeypatch.setattr(
        setup_wizard,
        "resolve_plugin_sources",
        lambda *args, **kwargs: [source],
    )
    monkeypatch.setattr(
        setup_wizard,
        "plugins_root",
        lambda: tmp_path / "plugin-home",
    )

    with pytest.raises(RuntimeError, match="正式安装 artifact"):
        setup_wizard._run_declared_plugin_setups(tmp_path / "workspace")
