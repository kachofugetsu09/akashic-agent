import hashlib
import importlib.util
import json
import os
import stat
import sys
import tomllib
from pathlib import Path

import pytest
import yoyo

from agent.migrations.context import bind_migration_context

_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = (
    _PROJECT_ROOT
    / "migrations"
    / "yoyo"
    / "20260825_02_select_akasha_embedding_plugin.py"
)


def _load_migration():
    """Load the migration callback without wrapping it in Yoyo."""

    spec = importlib.util.spec_from_file_location(
        "select_akasha_embedding_plugin_under_test",
        _MIGRATION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载迁移: {_MIGRATION_PATH}")
    original_step = yoyo.step
    yoyo.step = lambda callback: callback  # type: ignore[assignment]
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        yoyo.step = original_step
    return module


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.select_akasha_embedding_plugin(None)


def _legacy_config(*, engine: str | None = "akasha", extra: str = "") -> bytes:
    engine_line = "" if engine is None else f'engine = "{engine}"\n'
    return (
        "# Preserve operator-owned content.\n"
        "[memory]\n"
        "enabled = true\n"
        f"{engine_line}\n"
        "[memory.embedding]\n"
        'model = "embedding-model"\n\n'
        f"{extra}"
        "[custom]\n"
        'value = "protected"\n'
    ).encode("utf-8")


def _backup_root(workspace: Path) -> Path:
    roots = sorted((workspace / "backups/select-akasha-embedding-plugin").iterdir())
    assert len(roots) == 1
    return roots[0]


def test_exact_legacy_akasha_selection_moves_to_plugin_claim(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    original = _legacy_config()
    config.write_bytes(original)
    config.chmod(0o640)

    _run(module, config, workspace)
    _run(module, config, workspace)

    migrated = tomllib.loads(config.read_text(encoding="utf-8"))
    assert migrated["memory"] == {
        "enabled": True,
        "embedding": {"model": "embedding-model"},
    }
    assert migrated["agent"]["plugins"]["disabled_builtin"] == []
    assert migrated["custom"] == {"value": "protected"}
    assert stat.S_IMODE(config.stat().st_mode) == 0o640

    backup_root = _backup_root(workspace)
    manifest = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    backup = backup_root / manifest["source"]["backup"]
    assert backup.read_bytes() == original
    assert manifest["source"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert stat.S_IMODE(backup_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert stat.S_IMODE((backup_root / "manifest.json").stat().st_mode) == 0o600


def test_existing_plugin_exclusions_are_preserved(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_bytes(
        _legacy_config(
            extra=(
                "[agent.plugins]\n"
                'disabled_builtin = ["scheduler", "default_memory", "akasha"]\n\n'
            )
        )
    )

    _run(module, config, workspace)

    migrated = tomllib.loads(config.read_text(encoding="utf-8"))
    assert migrated["agent"]["plugins"]["disabled_builtin"] == [
        "scheduler",
    ]


@pytest.mark.parametrize(
    "engine",
    (
        None,
        "",
        "default",
        "akasha",
    ),
)
def test_enabled_legacy_memory_choices_select_akasha(
    tmp_path: Path,
    engine: str | None,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_bytes(_legacy_config(engine=engine))

    _run(module, config, workspace)

    migrated = tomllib.loads(config.read_text(encoding="utf-8"))
    assert "engine" not in migrated["memory"]
    assert migrated["agent"]["plugins"]["disabled_builtin"] == []


def test_disabled_legacy_memory_stays_disabled_without_replay_selection(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    legacy_database = workspace / "memory" / "memory2.db"
    legacy_database.parent.mkdir()
    legacy_database.write_bytes(b"retired-default-memory-archive")
    config = tmp_path / "config.toml"
    config.write_text(
        '[memory]\nenabled = false\nengine = "default"\n',
        encoding="utf-8",
    )

    _run(module, config, workspace)

    migrated = tomllib.loads(config.read_text(encoding="utf-8"))
    assert migrated["memory"] == {"enabled": False}
    assert migrated["agent"]["plugins"]["disabled_builtin"] == ["akasha", "wake"]
    assert legacy_database.read_bytes() == b"retired-default-memory-archive"


@pytest.mark.parametrize(
    "memory",
    (
        '[memory]\nenabled = true\nengine = "custom"\n',
        "[custom]\nvalue = 1\n",
    ),
)
def test_nonmatching_memory_choices_are_noop(tmp_path: Path, memory: str) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    original = memory.encode("utf-8")
    config.write_bytes(original)

    _run(module, config, workspace)

    assert config.read_bytes() == original
    assert not (workspace / "backups").exists()


@pytest.mark.parametrize(
    ("plugins", "message"),
    (
        (
            '[agent.plugins]\ndisabled_builtin = "default_memory"\n',
            "必须是合法字符串数组",
        ),
        (
            '[agent.plugins]\ndisabled_builtin = ["scheduler", "scheduler"]\n',
            "不允许重复插件名",
        ),
    ),
)
def test_conflicting_plugin_configuration_fails_before_write(
    tmp_path: Path,
    plugins: str,
    message: str,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    original = _legacy_config(extra=f"{plugins}\n")
    config.write_bytes(original)

    with pytest.raises((RuntimeError, ValueError), match=message):
        _run(module, config, workspace)

    assert config.read_bytes() == original
    assert not (workspace / "backups").exists()


def test_symlink_identity_survives_migration(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "config-source.toml"
    target.write_bytes(_legacy_config())
    config = tmp_path / "config.toml"
    config.symlink_to(target.name)
    original_link = os.readlink(config)

    _run(module, config, workspace)

    assert config.is_symlink()
    assert os.readlink(config) == original_link
    assert "engine" not in tomllib.loads(target.read_text(encoding="utf-8"))["memory"]


def test_failed_publication_restores_exact_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    original = _legacy_config()
    config.write_bytes(original)

    def fail_after_write(snapshot, rendered):
        module._write_atomic(snapshot.resolved_target, rendered, snapshot.mode)
        raise RuntimeError("forced publication failure")

    monkeypatch.setattr(module, "_publish_config", fail_after_write)
    with pytest.raises(RuntimeError, match="forced publication failure"):
        _run(module, config, workspace)

    assert config.read_bytes() == original
    backup_root = _backup_root(workspace)
    assert (backup_root / "config.toml.before").read_bytes() == original
