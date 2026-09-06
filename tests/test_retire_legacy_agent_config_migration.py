from __future__ import annotations

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


def test_migrates_zero_budget_and_preserves_veda(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[agent]\n"
        f'system_prompt = "{_DEFAULT_PROMPT}"\n'
        "max_iterations = 0\n"
        "dev_mode = false\n"
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
