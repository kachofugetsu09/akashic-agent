from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError
from yoyo import read_migrations

from agent.migrations.context import bind_migration_context
from plugins.reply.plugin import Config as ReplyConfig


_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION = "20260907_02_retire_legacy_agent_config.py"
_DEPENDENCY = "20260907_01_context_material_grants.py"
_DEFAULT_PROMPT = (
    "You are Akashic, a helpful AI assistant with access to tools. "
    "Always respond in the same language the user uses."
)


def _module(tmp_path: Path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (_DEPENDENCY, _MIGRATION):
        shutil.copy2(_PROJECT_ROOT / "migrations/yoyo" / name, catalog / name)
    migration = read_migrations(str(catalog))[-1]
    migration.load()
    return migration.module


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.retire_legacy_agent_config(None)


@pytest.mark.parametrize("dev_mode", ["false", "true"])
def test_migrates_zero_budget_and_preserves_veda(tmp_path: Path, dev_mode: str) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[agent]\n"
        f'system_prompt = "{_DEFAULT_PROMPT}"\n'
        "max_iterations = 0\n"
        f"dev_mode = {dev_mode}\n"
        "[agent.wiring]\n"
        'toolsets = ["meta_common"]\n',
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    veda = workspace / "memory/VEDA.md"
    veda.parent.mkdir(parents=True)
    veda.write_bytes(b"custom VEDA\n")
    module = _module(tmp_path)

    _run(module, config, workspace)

    assert tomllib.loads(config.read_text(encoding="utf-8")) == {}
    assert tomllib.loads(
        (workspace / "plugin-data/reply-builtin/config.local.toml").read_text(
            encoding="utf-8"
        )
    ) == {"max_steps": 0}
    assert veda.read_bytes() == b"custom VEDA\n"


def test_context_only_default_wiring_is_removed(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[agent.wiring]\ncontext = "default"\n', encoding="utf-8")
    workspace = tmp_path / "workspace"

    _run(_module(tmp_path), config, workspace)

    assert tomllib.loads(config.read_text(encoding="utf-8")) == {}
    assert not (workspace / "plugin-data/reply-builtin/config.local.toml").exists()


def test_old_budget_values_conflict_without_writing(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "max_iterations = 8\n[agent]\nmax_iterations = 7\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    before = config.read_bytes()

    with pytest.raises(RuntimeError, match="max_iterations.*冲突"):
        _run(_module(tmp_path), config, workspace)

    assert config.read_bytes() == before
    assert not (workspace / "backups").exists()


def test_hardlinked_config_targets_are_rejected_without_writing(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[agent]\nmax_iterations = 3\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    reply = workspace / "plugin-data/reply-builtin/config.local.toml"
    reply.parent.mkdir(parents=True)
    os.link(config, reply)
    before = config.read_bytes()

    with pytest.raises(RuntimeError, match="同一 inode"):
        _run(_module(tmp_path), config, workspace)

    assert config.read_bytes() == before
    assert reply.read_bytes() == before
    assert not (workspace / "backups").exists()


@pytest.mark.parametrize("value", [False, -1])
def test_reply_max_steps_rejects_bool_and_negative_values(value: object) -> None:
    with pytest.raises(ValidationError):
        ReplyConfig.model_validate({"max_steps": value})


def test_reply_max_steps_zero_is_unlimited() -> None:
    assert ReplyConfig.model_validate({"max_steps": 0}).max_steps == 0


def test_existing_reply_budget_must_match_and_repeat_is_idempotent(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[agent]\nmax_iterations = 7\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    reply = workspace / "plugin-data/reply-builtin/config.local.toml"
    reply.parent.mkdir(parents=True)
    reply.write_text("max_output_tokens = 123\nmax_steps = 7\n", encoding="utf-8")
    reply.chmod(0o640)
    module = _module(tmp_path)

    _run(module, config, workspace)
    config_after = config.read_bytes()
    reply_after = reply.read_bytes()
    _run(module, config, workspace)

    assert config.read_bytes() == config_after
    assert reply.read_bytes() == reply_after
    assert (reply.stat().st_mode & 0o777) == 0o640
    assert len(list((workspace / "backups/retire-legacy-agent-config").iterdir())) == 1


def test_existing_reply_budget_conflict_writes_nothing(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[agent]\nmax_iterations = 7\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    reply = workspace / "plugin-data/reply-builtin/config.local.toml"
    reply.parent.mkdir(parents=True)
    reply.write_text("max_steps = 8\n", encoding="utf-8")
    before_config = config.read_bytes()
    before_reply = reply.read_bytes()
    module = _module(tmp_path)

    with pytest.raises(RuntimeError, match="冲突"):
        _run(module, config, workspace)

    assert config.read_bytes() == before_config
    assert reply.read_bytes() == before_reply
    assert not (workspace / "backups").exists()


def test_custom_prompt_and_search_boolean_stop_before_any_write(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[agent]\nsystem_prompt = "custom\"\nmax_iterations = 7\n'
        "[agent.tools]\nsearch_enabled = true\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    before = config.read_bytes()
    module = _module(tmp_path)

    with pytest.raises(RuntimeError, match="自定义 system_prompt"):
        _run(module, config, workspace)
    assert config.read_bytes() == before
    assert not (workspace / "plugin-data").exists()

    config.write_text(
        "[agent]\nmax_iterations = 7\n[agent.tools]\nsearch_enabled = true\n",
        encoding="utf-8",
    )
    before = config.read_bytes()
    with pytest.raises(RuntimeError, match="reply.tools"):
        _run(module, config, workspace)
    assert config.read_bytes() == before
    assert not (workspace / "plugin-data").exists()


def test_publication_failure_restores_both_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[agent]\nmax_iterations = 3\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    original_config = config.read_bytes()
    module = _module(tmp_path)
    publish = module._publish
    calls = 0

    def fail_after_first(snapshot, payload, *, label):
        nonlocal calls
        publish(snapshot, payload, label=label)
        calls += 1
        if calls == 1:
            raise OSError("forced second publication failure")

    monkeypatch.setattr(module, "_publish", fail_after_first)
    with pytest.raises(OSError, match="forced second publication failure"):
        _run(module, config, workspace)

    assert config.read_bytes() == original_config
    assert not (workspace / "plugin-data/reply-builtin/config.local.toml").exists()
    assert len(list((workspace / "backups/retire-legacy-agent-config").iterdir())) == 1


@pytest.mark.parametrize("drift", ["bytes", "mode", "type", "absence"])
def test_source_drift_after_backup_is_preserved_and_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[agent]\nmax_iterations = 3\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    module = _module(tmp_path)
    original_backup = module._backup

    def backup_then_edit(snapshot, backup_root, name):
        result = original_backup(snapshot, backup_root, name)
        if name == "config.toml.before":
            if drift == "bytes":
                config.write_text("[agent]\nmax_iterations = 99\n", encoding="utf-8")
            elif drift == "mode":
                config.chmod(0o600)
            elif drift == "type":
                config.unlink()
                replacement = tmp_path / "replacement.toml"
                replacement.write_text("[agent]\nmax_iterations = 99\n", encoding="utf-8")
                config.symlink_to(replacement.name)
            else:
                config.unlink()
        return result

    monkeypatch.setattr(module, "_backup", backup_then_edit)
    with pytest.raises(RuntimeError, match="请从"):
        _run(module, config, workspace)

    if drift == "type":
        assert config.is_symlink()
    elif drift == "absence":
        assert not config.exists()
    else:
        assert config.exists()
    assert not (workspace / "plugin-data/reply-builtin/config.local.toml").exists()
    assert list((workspace / "backups/retire-legacy-agent-config").iterdir())


def test_operator_reply_creation_survives_failed_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[agent]\nmax_iterations = 3\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    reply = workspace / "plugin-data/reply-builtin/config.local.toml"
    module = _module(tmp_path)
    original_backup = module._backup

    def backup_then_create(snapshot, backup_root, name):
        result = original_backup(snapshot, backup_root, name)
        if name == "config.toml.before":
            reply.parent.mkdir(parents=True, exist_ok=True)
            reply.write_text("max_steps = 88\n", encoding="utf-8")
        return result

    monkeypatch.setattr(module, "_backup", backup_then_create)
    with pytest.raises(RuntimeError, match="恢复失败"):
        _run(module, config, workspace)

    assert reply.read_text(encoding="utf-8") == "max_steps = 88\n"


def test_reply_parent_symlink_drift_blocks_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[agent]\nmax_iterations = 3\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    reply = workspace / "plugin-data/reply-builtin/config.local.toml"
    outside = tmp_path / "outside"
    module = _module(tmp_path)
    original_backup = module._backup

    def backup_then_swap_parent(snapshot, backup_root, name):
        result = original_backup(snapshot, backup_root, name)
        if name == "config.toml.before":
            outside.mkdir()
            reply.parent.parent.mkdir(parents=True, exist_ok=True)
            reply.parent.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(module, "_backup", backup_then_swap_parent)
    with pytest.raises(RuntimeError, match="恢复失败"):
        _run(module, config, workspace)

    assert not (outside / "config.local.toml").exists()


def test_reply_parent_symlink_drift_blocks_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[agent]\nmax_iterations = 3\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    reply = workspace / "plugin-data/reply-builtin/config.local.toml"
    outside = tmp_path / "outside"
    module = _module(tmp_path)
    original_publish = module._publish

    def publish_then_swap_parent(snapshot, payload, *, label):
        original_publish(snapshot, payload, label=label)
        if label == "reply 插件配置":
            outside.mkdir()
            real_parent = reply.parent.with_name("reply-builtin-real")
            reply.parent.rename(real_parent)
            reply.parent.symlink_to(outside, target_is_directory=True)
            raise OSError("模拟后续发布失败")

    monkeypatch.setattr(module, "_publish", publish_then_swap_parent)
    with pytest.raises(RuntimeError, match="恢复失败"):
        _run(module, config, workspace)

    assert (reply.parent / "config.local.toml").exists() is False
    assert (
        workspace / "plugin-data/reply-builtin-real/config.local.toml"
    ).read_text(encoding="utf-8") == "max_steps = 3\n"
    assert not (outside / "config.local.toml").exists()


def test_operator_reply_edit_survives_failed_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[agent]\nmax_iterations = 3\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    reply = workspace / "plugin-data/reply-builtin/config.local.toml"
    module = _module(tmp_path)
    original_publish = module._publish

    def publish_then_edit(snapshot, payload, *, label):
        original_publish(snapshot, payload, label=label)
        if label == "reply 插件配置":
            reply.write_text("max_steps = 88\n", encoding="utf-8")
            raise OSError("模拟后续发布失败")

    monkeypatch.setattr(module, "_publish", publish_then_edit)
    with pytest.raises(RuntimeError, match="恢复失败"):
        _run(module, config, workspace)

    assert reply.read_text(encoding="utf-8") == "max_steps = 88\n"
    assert "max_iterations = 3" in config.read_text(encoding="utf-8")
