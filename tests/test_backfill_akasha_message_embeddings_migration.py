from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pytest

from agent.migrations.context import bind_migration_context
from tests.legacy_migration_loader import load_migration_namespace

_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = (
    _PROJECT_ROOT / "plugins/legacy_upgrade/legacy_upgrade_migrations/20260826_02_backfill_akasha_message_embeddings.py"
)


def _load_migration():
    return load_migration_namespace("20260826_02_backfill_akasha_message_embeddings")


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.backfill_akasha_history(None)


def test_enabled_memory_backfills_current_config_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    raw = b'[memory]\nenabled = true\n[memory.embedding]\nmodel = "embed"\n'
    config.write_bytes(raw)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "_backfill_enabled_history",
        lambda **kwargs: calls.append(kwargs),
    )

    _run(module, config, workspace)

    assert calls == [
        {
            "config_path": config,
            "migrated_config": raw,
            "workspace": workspace,
        }
    ]
    assert config.read_bytes() == raw


def test_disabled_memory_performs_no_backfill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_text("[memory]\nenabled = false\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_backfill_enabled_history",
        lambda **_kwargs: pytest.fail("disabled memory must not backfill"),
    )

    _run(module, config, workspace)

    assert not (workspace / "backups").exists()


def test_remaining_custom_selector_fails_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        '[memory]\nenabled = true\nengine = "custom"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_backfill_enabled_history",
        lambda **_kwargs: pytest.fail("custom selector must fail before backfill"),
    )

    with pytest.raises(ValueError, match="自定义选择器已移除"):
        _run(module, config, workspace)


@pytest.mark.parametrize(
    "payload",
    (
        b"[custom]\nvalue = 1\n",
        b"[memory]\nenabled = 'yes'\n",
    ),
)
def test_missing_or_invalid_memory_contract_is_explicit(
    tmp_path: Path,
    payload: bytes,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_bytes(payload)

    if b"[memory]" not in payload:
        _run(module, config, workspace)
    else:
        with pytest.raises(ValueError, match="memory.enabled 必须是 boolean"):
            _run(module, config, workspace)


def test_disabled_memory_does_not_import_akasha_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "plugins.legacy_upgrade.legacy_upgrade_migrations.support.akasha_embedding_backfill":
            raise AssertionError("disabled memory must not load Akasha backfill")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_text("[memory]\nenabled = false\n", encoding="utf-8")

    _run(module, config, workspace)
