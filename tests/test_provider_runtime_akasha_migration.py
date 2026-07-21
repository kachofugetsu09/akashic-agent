# pyright: reportPrivateUsage=false

from __future__ import annotations

import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

from agent.config import Config
from migrations.provider_runtimes_and_akasha.migration import (
    MigrationContext,
    _akasha_db_path,
    _akasha_has_marker,
    _apply,
    _config_assessment,
    _verify,
    _revert,
)
from plugins.akasha.store import AkashaStore
from session.store import SessionStore

_LEGACY_CONFIG = """\
[runtime]
workspace = "workspace"

[llm]
provider = "deepseek"

[llm.main]
model = "deepseek-v4-flash"
api_key = "main-secret"
base_url = "https://api.deepseek.com/v1"
enable_thinking = true
context_window = 128000

[llm.fast]
model = "qwen-flash"
api_key = "fast-secret"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[llm.vl]
model = "qwen-vl-plus"
api_key = "vl-secret"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
multimodal = true

[memory]
enabled = true

[memory.embedding]
model = "embed-model"
api_key = "embedding-secret"
base_url = "https://example.com/v1"
"""


def _context(tmp_path: Path) -> MigrationContext:
    return MigrationContext(
        config_path=tmp_path / "config.toml",
        workspace=tmp_path / "workspace",
        migration_commit="a" * 40,
        backup_dir=tmp_path / "backups" / "run",
    )


def test_legacy_config_maps_roles_and_preserves_secrets(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.config_path.write_text(_LEGACY_CONFIG, encoding="utf-8")

    _apply(context)
    _verify(context)

    parsed = tomllib.loads(context.config_path.read_text(encoding="utf-8"))
    llm = parsed["llm"]
    assert llm["main"] == "deepseek_main"
    assert llm["fast"] == "openai_fast"
    assert llm["vl"] == "openai_vl"
    assert "provider" not in llm
    assert llm["runtimes"]["deepseek_main"]["api_key"] == "main-secret"
    assert llm["runtimes"]["openai_fast"]["api_key"] == "fast-secret"
    assert llm["runtimes"]["openai_vl"]["input_modalities"] == ["text", "image"]
    assert context.config_path.with_name("config.toml").stat().st_mode & 0o777 == 0o600
    assert (context.backup_dir / "config.toml").exists()
    assert _config_assessment(context.config_path).state == "current"
    config = Config.load(context.config_path, workspace=context.workspace)
    assert config.model == "deepseek-v4-flash"
    assert config.light_model == "qwen-flash"
    assert config.vl_model == "qwen-vl-plus"


def test_mixed_legacy_and_named_runtime_is_blocked(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        _LEGACY_CONFIG + """\
[llm.runtimes.existing]
provider = "openai"
model = "existing"
""",
        encoding="utf-8",
    )

    assessment = _config_assessment(path)

    assert assessment.state == "blocked"
    assert "混杂" in assessment.reason


def test_root_level_legacy_model_fields_migrate_once(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.config_path.write_text(
        """\
provider = "openai"
model = "legacy-main"
api_key = "main-secret"
base_url = "https://example.com/v1"
light_model = "legacy-fast"
light_api_key = "fast-secret"
system_prompt = "legacy"
""",
        encoding="utf-8",
    )

    _apply(context)
    _verify(context)

    parsed = tomllib.loads(context.config_path.read_text(encoding="utf-8"))
    assert parsed["llm"]["main"] == "openai_main"
    assert parsed["llm"]["fast"] == "openai_fast"
    assert parsed["llm"]["runtimes"]["openai_main"]["model"] == "legacy-main"
    assert parsed["llm"]["runtimes"]["openai_fast"]["model"] == "legacy-fast"
    assert "model" not in parsed
    assert Config.load(
        context.config_path, workspace=context.workspace
    ).light_model == ("legacy-fast")


def test_explicit_revert_restores_exact_legacy_config(tmp_path: Path) -> None:
    context = _context(tmp_path)
    original = _LEGACY_CONFIG.encode("utf-8")
    context.config_path.write_bytes(original)

    _apply(context)
    _verify(context)
    _revert(context)

    assert context.config_path.read_bytes() == original
    assert _config_assessment(context.config_path).state == "legacy"


def test_akasha_rebuild_uses_staging_and_marks_completed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.config_path.write_text(_LEGACY_CONFIG, encoding="utf-8")
    context.workspace.mkdir(parents=True)
    session_store = SessionStore(context.workspace / "sessions.db")
    session_store.close()
    db_path = _akasha_db_path(context.workspace)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    AkashaStore(db_path).close()
    with closing(sqlite3.connect(context.workspace / "sessions.db")) as database:
        before_messages = database.execute(
            "SELECT id, session_key, seq, role, content FROM messages ORDER BY id"
        ).fetchall()

    _apply(context)
    _verify(context)

    assert _akasha_has_marker(db_path)
    assert (context.backup_dir / "akasha.db").exists()
    with closing(sqlite3.connect(context.workspace / "sessions.db")) as database:
        after_messages = database.execute(
            "SELECT id, session_key, seq, role, content FROM messages ORDER BY id"
        ).fetchall()
    assert after_messages == before_messages
