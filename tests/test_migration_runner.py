from __future__ import annotations

import os
import json
import hashlib
import shutil
import sqlite3
import stat
import tomllib
from pathlib import Path

import pytest
import toml

from agent.migrations.runner import MigrationRunner
from agent.model_runtime.auth.store import Credential, CredentialStore
from agent.model_runtime.store import ModelRegistryStore
from bootstrap.workspace_lock import WorkspaceInstanceLock

_PROJECT_ROOT = Path(__file__).parents[1]
_ORIGIN_ID = "20260802_01_yoyo_origin"
_AKASHA_V9_ID = "20260805_01_akasha_sparse_index_v9"
_COMPACTION_ID = "20260807_01_session_context_compaction_ledger"
_AUDIT_ID = "20260808_01_session_mutation_audits"
_PREPARE_ID = "20260808_02_session_compaction_prepares"
_CONFIG_ID = "20260808_03_remove_compaction_trigger"
_DIGEST_ID = "20260808_04_session_compaction_source_plan_digest"
_CURSOR_ID = "20260808_05_activate_session_compaction_cursor"
_RETIRE_ID = "20260808_06_retire_legacy_context_state"
_MODEL_REGISTRY_ID = "20260807_01_model_registry_database"
_EMBEDDING_REGISTRY_ID = "20260807_02_embedding_model_registry"
_MODEL_CAPABILITIES_ID = "20260808_01_restore_migrated_reasoning_efforts"
_OPENCODE_VARIANTS_ID = "20260808_02_correct_opencode_go_variants"
_AKASHA_V10_ID = "20260817_01_akasha_sparse_index_v10"
_TOOLSET_WIRING_ID = "20260823_01_retire_legacy_toolset_wiring"
_PROACTIVE_DELIVERY_TARGET_ID = "20260825_01_migrate_proactive_delivery_target"
_AKASHA_PLUGIN_SELECTION_ID = "20260825_02_select_akasha_embedding_plugin"
_TURN_EFFECTS_ID = "20260826_01_migrate_turn_effects"
_AKASHA_EMBEDDING_BACKFILL_ID = "20260826_02_backfill_akasha_message_embeddings"
_AKASHIC_CHANNEL_IDENTITY_ID = "20260826_03_unify_akashic_channel_identity"
_SESSION_TIMESTAMP_ID = "20260827_01_normalize_session_timestamps"
_MOBILE_CLIENT_ID_ID = "20260827_02_migrate_legacy_mobile_client_ids"
_EVENTMAIL_STATE_ID = "20260828_01_migrate_eventmail_state"
_WAKE_CONTENT_SCORES_ID = "20260828_02_add_wake_content_scores"
_PROGRAMMATIC_EFFECTS_ID = "20260829_01_backfill_plugin_programmatic_effects"
_EXPLICIT_PROGRAMMATIC_EFFECTS_ID = "20260829_02_backfill_explicit_programmatic_effects"
_RETIRE_CORE_MODEL_CONFIG_ID = "20260829_03_retire_core_model_config"
_CURRENT_IDS = (
    _ORIGIN_ID,
    _AKASHA_V9_ID,
    _MODEL_REGISTRY_ID,
    _COMPACTION_ID,
    _EMBEDDING_REGISTRY_ID,
    _MODEL_CAPABILITIES_ID,
    _AUDIT_ID,
    _OPENCODE_VARIANTS_ID,
    _PREPARE_ID,
    _DIGEST_ID,
    _CURSOR_ID,
    _CONFIG_ID,
    _RETIRE_ID,
    _AKASHA_V10_ID,
    _TOOLSET_WIRING_ID,
    _PROACTIVE_DELIVERY_TARGET_ID,
    _TURN_EFFECTS_ID,
    _AKASHA_PLUGIN_SELECTION_ID,
    _AKASHA_EMBEDDING_BACKFILL_ID,
    _AKASHIC_CHANNEL_IDENTITY_ID,
    _SESSION_TIMESTAMP_ID,
    _MOBILE_CLIENT_ID_ID,
    _EVENTMAIL_STATE_ID,
    _WAKE_CONTENT_SCORES_ID,
    _PROGRAMMATIC_EFFECTS_ID,
    _EXPLICIT_PROGRAMMATIC_EFFECTS_ID,
    _RETIRE_CORE_MODEL_CONFIG_ID,
)
_CURRENT_LEDGER_IDS = tuple(sorted(_CURRENT_IDS))


def _runner(root: Path, *, repo_root: Path = _PROJECT_ROOT) -> MigrationRunner:
    return MigrationRunner(
        repo_root=repo_root,
        config_path=root / "config.toml",
        workspace=root / "workspace",
    )


def _catalog(root: Path, migration_ids: tuple[str, ...]) -> Path:
    catalog = root / "migrations" / "yoyo"
    catalog.mkdir(parents=True)
    for migration_id in migration_ids:
        shutil.copy2(
            _PROJECT_ROOT / "migrations/yoyo" / f"{migration_id}.py",
            catalog / f"{migration_id}.py",
        )
    return root


def _applied_ids(ledger: Path) -> list[str]:
    connection = sqlite3.connect(ledger)
    try:
        rows = connection.execute(
            "SELECT migration_id FROM _yoyo_migration ORDER BY migration_id"
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def _create_sessions(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE sessions ("
            "key TEXT PRIMARY KEY, last_consolidated INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE messages (id TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO sessions VALUES ('chat', 4)")
        connection.execute("INSERT INTO messages VALUES ('m1', 'session-bytes')")
        connection.commit()
    finally:
        connection.close()


def test_origin_removes_legacy_state_without_touching_business_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    config = root / "config.toml"
    config.write_bytes(b"current = true\n")
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    memory = workspace / "memory/MEMORY.md"
    memory.parent.mkdir()
    memory.write_bytes(b"memory-bytes")

    cursor = root / "config.toml.migration-cursor"
    lock = root / "config.toml.migration-lock"
    backups = root / "config.toml.migration-backups"
    cursor.write_text("retired\n", encoding="utf-8")
    lock.write_text("123\n", encoding="utf-8")
    backups.mkdir()
    (backups / "old.bak").write_bytes(b"backup")

    outcome = _runner(root).run()

    assert outcome.state == "migrated"
    assert outcome.migrations == _CURRENT_IDS
    assert not cursor.exists()
    assert not lock.exists()
    assert not backups.exists()
    config_data = tomllib.loads(config.read_text(encoding="utf-8"))
    assert config_data["agent"]["context"]["compaction"] == {
        "keep_recent_tokens": 20_000,
    }
    migrated = sqlite3.connect(sessions)
    try:
        assert migrated.execute(
            "SELECT last_consolidated FROM sessions WHERE key = 'chat'"
        ).fetchone() == (0,)
        assert migrated.execute("SELECT body FROM messages").fetchall() == [
            ("session-bytes",)
        ]
        assert {
            row[0]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        } >= {
            "session_delete_audits",
            "session_source_mutation_audits",
        }
    finally:
        migrated.close()
    assert memory.read_bytes() == b"memory-bytes"
    assert _applied_ids(workspace / "migrations.sqlite3") == list(_CURRENT_LEDGER_IDS)

    migration_backups = sorted(
        (workspace / "backups/session-context-compaction-ledger").iterdir()
    )
    assert len(migration_backups) == 1
    manifest = json.loads((migration_backups[0] / "manifest.json").read_text())
    assert manifest["sqlite_integrity"] == "ok"
    archived = sqlite3.connect(migration_backups[0] / manifest["backup"])
    try:
        assert archived.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        archived.close()


def test_origin_is_a_noop_when_legacy_state_is_absent(tmp_path: Path) -> None:
    runner = _runner(tmp_path / "state")

    first = runner.run()
    second = runner.run()

    assert first.state == "migrated"
    assert first.migrations == _CURRENT_IDS
    assert second.state == "current"
    assert second.migrations == ()


def test_runner_supplies_yoyo_identity_without_os_username(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(key, raising=False)

    def get_user() -> str:
        username = os.environ.get("USER")
        if not username:
            raise OSError("No username set in the environment")
        return username

    monkeypatch.setattr("yoyo.backends.base.getpass.getuser", get_user)

    outcome = _runner(tmp_path / "state").run()

    assert outcome.migrations == _CURRENT_IDS
    assert "USER" not in os.environ


def test_origin_failure_is_not_marked_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    backups = root / "config.toml.migration-backups"
    backups.mkdir(parents=True)
    runner = _runner(root)

    def fail(_path: str | Path) -> None:
        raise PermissionError("forced cleanup failure")

    monkeypatch.setattr("shutil.rmtree", fail)
    with pytest.raises(RuntimeError, match="forced cleanup failure"):
        runner.run()

    assert backups.exists()
    assert _applied_ids(runner.ledger_path) == []

    monkeypatch.undo()
    assert runner.run().migrations == _CURRENT_IDS
    assert not backups.exists()


def test_workspace_lock_prevents_concurrent_migration(tmp_path: Path) -> None:
    root = tmp_path / "state"
    runner = _runner(root)
    lock = WorkspaceInstanceLock(runner.workspace)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="workspace 已由其他 runtime 占用"):
            runner.run()
    finally:
        lock.release()

    assert not runner.ledger_path.exists()


def test_catalog_ignores_archived_git_cursor_migrations(tmp_path: Path) -> None:
    runner = _runner(tmp_path / "state")

    outcome = runner.run()

    assert outcome.migrations == _CURRENT_IDS
    assert _applied_ids(runner.ledger_path) == list(_CURRENT_LEDGER_IDS)


def test_staged_catalog_upgrade_preserves_legacy_inputs_until_final_cutover(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    config = root / "config.toml"
    original_config = (
        "memory_window = 12\n"
        "[llm]\n"
        "effective_context_percent = 0.9\n"
        "[agent.context.compaction]\n"
        "trigger_percent = 0.74\n"
    ).encode()
    config.write_bytes(original_config)
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    recent = workspace / "memory/RECENT_CONTEXT.md"
    recent.parent.mkdir(parents=True)
    original_recent = b"legacy recent projection"
    recent.write_bytes(original_recent)

    additive_repo = _catalog(
        tmp_path / "additive-repo",
        (_ORIGIN_ID, _AKASHA_V9_ID, _COMPACTION_ID, _AUDIT_ID, _PREPARE_ID),
    )
    first = _runner(root, repo_root=additive_repo).run()
    assert first.migrations == (
        _ORIGIN_ID,
        _AKASHA_V9_ID,
        _COMPACTION_ID,
        _AUDIT_ID,
        _PREPARE_ID,
    )
    assert sessions.exists()
    connection = sqlite3.connect(sessions)
    try:
        assert connection.execute(
            "SELECT last_consolidated FROM sessions"
        ).fetchone() == (4,)
    finally:
        connection.close()
    assert config.read_bytes() == original_config
    assert recent.read_bytes() == original_recent

    current_without_model_cleanup = _catalog(
        tmp_path / "current-without-model-cleanup",
        _CURRENT_IDS[:-1],
    )
    second = _runner(root, repo_root=current_without_model_cleanup).run()
    assert _DIGEST_ID in second.migrations
    assert _CURSOR_ID in second.migrations
    assert _CONFIG_ID in second.migrations
    assert _RETIRE_ID in second.migrations
    connection = sqlite3.connect(sessions)
    try:
        assert connection.execute(
            "SELECT last_consolidated FROM sessions"
        ).fetchone() == (0,)
    finally:
        connection.close()
    final_config = tomllib.loads(config.read_text(encoding="utf-8"))
    assert "memory_window" not in final_config
    assert "effective_context_percent" not in final_config["llm"]
    assert final_config["agent"]["context"]["compaction"] == {
        "keep_recent_tokens": 20_000,
    }
    assert not recent.exists()


def test_toolset_wiring_migration_retires_only_the_exact_legacy_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    root.mkdir()
    config = root / "config.toml"
    config.write_text(
        "# Preserve user formatting and unrelated values.\n"
        "[agent.wiring]\n"
        'toolsets = ["meta_common", "spawn", "schedule"]\n'
        "\n"
        "[custom]\n"
        'value = "protected"\n',
        encoding="utf-8",
    )
    config.chmod(0o640)

    legacy_repo = _catalog(tmp_path / "legacy-repo", _CURRENT_IDS[:-13])
    assert _runner(root, repo_root=legacy_repo).run().migrations == _CURRENT_IDS[:-13]
    before = config.read_bytes()

    outcome = _runner(root).run()

    assert outcome.migrations == (
        _TOOLSET_WIRING_ID,
        _PROACTIVE_DELIVERY_TARGET_ID,
        _TURN_EFFECTS_ID,
        _AKASHA_PLUGIN_SELECTION_ID,
        _AKASHA_EMBEDDING_BACKFILL_ID,
        _AKASHIC_CHANNEL_IDENTITY_ID,
        _SESSION_TIMESTAMP_ID,
        _MOBILE_CLIENT_ID_ID,
        _EVENTMAIL_STATE_ID,
        _WAKE_CONTENT_SCORES_ID,
        _PROGRAMMATIC_EFFECTS_ID,
        _EXPLICIT_PROGRAMMATIC_EFFECTS_ID,
        _RETIRE_CORE_MODEL_CONFIG_ID,
    )
    migrated = tomllib.loads(config.read_text(encoding="utf-8"))
    assert migrated["agent"]["wiring"]["toolsets"] == ["meta_common"]
    assert migrated["custom"] == {"value": "protected"}
    assert stat.S_IMODE(config.stat().st_mode) == 0o640
    backups = sorted(
        (root / "workspace/backups/retire-legacy-toolset-wiring").iterdir()
    )
    assert len(backups) == 1
    manifest = json.loads((backups[0] / "manifest.json").read_text(encoding="utf-8"))
    backup = backups[0] / manifest["source"]["backup"]
    assert backup.read_bytes() == before
    assert manifest["source"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert stat.S_IMODE((backups[0] / "manifest.json").stat().st_mode) == 0o600

    assert _runner(root).run().state == "current"
    assert len(list(backups[0].parent.iterdir())) == 1


@pytest.mark.parametrize(
    "toolsets",
    (["meta_common"], ["meta_common", "spawn"], ["schedule"]),
)
def test_toolset_wiring_migration_leaves_nonlegacy_values_untouched(
    tmp_path: Path,
    toolsets: list[str],
) -> None:
    root = tmp_path / "state"
    root.mkdir()
    config = root / "config.toml"
    config.write_text(
        toml.dumps({"agent": {"wiring": {"toolsets": toolsets}}}),
        encoding="utf-8",
    )

    legacy_repo = _catalog(tmp_path / "legacy-repo", _CURRENT_IDS[:-13])
    _ = _runner(root, repo_root=legacy_repo).run()
    before = config.read_bytes()

    outcome = _runner(root).run()

    assert outcome.migrations == (
        _TOOLSET_WIRING_ID,
        _PROACTIVE_DELIVERY_TARGET_ID,
        _TURN_EFFECTS_ID,
        _AKASHA_PLUGIN_SELECTION_ID,
        _AKASHA_EMBEDDING_BACKFILL_ID,
        _AKASHIC_CHANNEL_IDENTITY_ID,
        _SESSION_TIMESTAMP_ID,
        _MOBILE_CLIENT_ID_ID,
        _EVENTMAIL_STATE_ID,
        _WAKE_CONTENT_SCORES_ID,
        _PROGRAMMATIC_EFFECTS_ID,
        _EXPLICIT_PROGRAMMATIC_EFFECTS_ID,
        _RETIRE_CORE_MODEL_CONFIG_ID,
    )
    assert config.read_bytes() == before
    assert not (root / "workspace/backups/retire-legacy-toolset-wiring").exists()


def test_toolset_wiring_migration_preserves_config_symlink_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    root.mkdir()
    source = root / "config-source.toml"
    source.write_text(
        "[agent.wiring]\n" 'toolsets = ["meta_common", "spawn", "schedule"]\n',
        encoding="utf-8",
    )
    config = root / "config.toml"
    config.symlink_to(source.name)

    legacy_repo = _catalog(tmp_path / "legacy-repo", _CURRENT_IDS[:-13])
    _ = _runner(root, repo_root=legacy_repo).run()

    outcome = _runner(root).run()

    assert outcome.migrations == (
        _TOOLSET_WIRING_ID,
        _PROACTIVE_DELIVERY_TARGET_ID,
        _TURN_EFFECTS_ID,
        _AKASHA_PLUGIN_SELECTION_ID,
        _AKASHA_EMBEDDING_BACKFILL_ID,
        _AKASHIC_CHANNEL_IDENTITY_ID,
        _SESSION_TIMESTAMP_ID,
        _MOBILE_CLIENT_ID_ID,
        _EVENTMAIL_STATE_ID,
        _WAKE_CONTENT_SCORES_ID,
        _PROGRAMMATIC_EFFECTS_ID,
        _EXPLICIT_PROGRAMMATIC_EFFECTS_ID,
        _RETIRE_CORE_MODEL_CONFIG_ID,
    )
    assert config.is_symlink()
    assert os.readlink(config) == source.name
    assert tomllib.loads(source.read_text(encoding="utf-8"))["agent"]["wiring"][
        "toolsets"
    ] == ["meta_common"]


def test_new_branch_migration_is_applied_even_after_sibling_ran(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    catalog = repo / "migrations/yoyo"
    catalog.mkdir(parents=True)
    root = tmp_path / "state"
    workspace_literal = repr(str(root / "workspace"))

    def write_migration(name: str, depends: str) -> None:
        (catalog / f"{name}.py").write_text(
            "from pathlib import Path\n"
            "from yoyo import step\n"
            f"__depends__ = {depends}\n"
            f"def apply(_connection):\n"
            f"    marker = Path({workspace_literal}) / 'order.log'\n"
            "    marker.parent.mkdir(parents=True, exist_ok=True)\n"
            f"    with marker.open('a', encoding='utf-8') as stream:\n"
            f"        stream.write({name!r} + '\\n')\n"
            "steps = [step(apply)]\n",
            encoding="utf-8",
        )

    write_migration("base", "set()")
    runner = _runner(root, repo_root=repo)
    assert runner.run().migrations == ("base",)

    write_migration("bob", "{'base'}")
    assert runner.run().migrations == ("bob",)

    write_migration("alice", "{'base'}")
    assert runner.run().migrations == ("alice",)
    assert (root / "workspace/order.log").read_text() == "base\nbob\nalice\n"


def test_embedding_backfill_runs_after_selection_is_already_recorded(
    tmp_path: Path,
) -> None:
    """Apply the appended backfill even when the prior release is in the ledger."""

    # 1. Recreate a workspace that already ran every migration through selection.
    root = tmp_path / "state"
    prior_repo = _catalog(tmp_path / "prior-repo", _CURRENT_IDS[:-9])
    first = _runner(root, repo_root=prior_repo).run()
    assert first.migrations == _CURRENT_IDS[:-9]
    assert _AKASHA_PLUGIN_SELECTION_ID in _applied_ids(
        root / "workspace/migrations.sqlite3"
    )

    # 2. Upgrade the catalog and prove the new ID remains independently pending.
    second = _runner(root).run()
    assert second.migrations == (
        _AKASHA_EMBEDDING_BACKFILL_ID,
        _AKASHIC_CHANNEL_IDENTITY_ID,
        _SESSION_TIMESTAMP_ID,
        _MOBILE_CLIENT_ID_ID,
        _EVENTMAIL_STATE_ID,
        _WAKE_CONTENT_SCORES_ID,
        _PROGRAMMATIC_EFFECTS_ID,
        _EXPLICIT_PROGRAMMATIC_EFFECTS_ID,
        _RETIRE_CORE_MODEL_CONFIG_ID,
    )


def test_ledger_supports_workspace_path_with_uri_characters(tmp_path: Path) -> None:
    root = tmp_path / "state with # and ?"

    outcome = _runner(root).run()

    assert outcome.migrations == _CURRENT_IDS
    assert (root / "workspace/migrations.sqlite3").is_file()


def test_model_registry_migration_moves_roles_without_touching_sessions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    config = root / "config.toml"
    config.write_text(
        """
[runtime]
workspace = "unused"

[llm]
main = "codex_main"
fast = "codex_fast"

[llm.runtimes.codex_main]
provider = "codex"
model = "gpt-main"
auth = "codex_default"
input_modalities = ["text"]

[llm.runtimes.codex_fast]
provider = "codex"
model = "gpt-fast"
auth = "codex_default"
input_modalities = ["text"]

[agent]
system_prompt = "test"
""",
        encoding="utf-8",
    )
    sessions = workspace / "sessions.db"
    sessions.write_bytes(b"protected-session-bytes")

    model_repo = _catalog(
        tmp_path / "model-registry-repo",
        (_ORIGIN_ID, _MODEL_REGISTRY_ID),
    )
    outcome = _runner(root, repo_root=model_repo).run()

    assert outcome.migrations == (_ORIGIN_ID, _MODEL_REGISTRY_ID)
    assert sessions.read_bytes() == b"protected-session-bytes"
    assert "runtimes" not in config.read_text(encoding="utf-8")
    snapshot = ModelRegistryStore.for_workspace(workspace).read_snapshot()
    assert snapshot is not None
    assert snapshot.roles["default"].runtime_id == "codex_main"
    assert snapshot.roles["fast"].runtime_id == "codex_fast"
    backups = list((workspace / "backups/model-registry-v1").glob("*/config.before"))
    assert len(backups) == 1
    assert b"gpt-main" in backups[0].read_bytes()


def test_opencode_variant_correction_repairs_existing_registry_without_identity_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "state"
    root.mkdir()
    config = root / "config.toml"
    config.write_text(
        """
[llm]
main = "opencode_go_main"

[llm.runtimes.opencode_go_main]
provider = "opencode-go"
model = "deepseek-v4-flash"
api_key = "migration-secret"
base_url = "https://opencode.ai/zen/go/v1"
reasoning_effort = "high"
context_window = 1000000
input_modalities = ["text"]
""",
        encoding="utf-8",
    )
    sessions = root / "workspace/sessions.db"
    sessions.parent.mkdir()
    sessions.write_bytes(b"protected-session-bytes")

    # 1. 重现已经被上一条迁移写入通用 LiteLLM effort 的 workspace
    legacy_repo = _catalog(
        tmp_path / "legacy-repo",
        (
            _ORIGIN_ID,
            _AKASHA_V9_ID,
            _MODEL_REGISTRY_ID,
            _EMBEDDING_REGISTRY_ID,
            _MODEL_CAPABILITIES_ID,
        ),
    )
    first = _runner(root, repo_root=legacy_repo).run()
    assert first.migrations == (
        _ORIGIN_ID,
        _AKASHA_V9_ID,
        _MODEL_REGISTRY_ID,
        _EMBEDDING_REGISTRY_ID,
        _MODEL_CAPABILITIES_ID,
    )
    store = ModelRegistryStore.for_workspace(root / "workspace")
    before = store.read_snapshot()
    assert before is not None
    assert before.runtimes["opencode_go_main"].supported_reasoning_efforts == (
        "low",
        "medium",
        "high",
    )
    config_after_first_migration = config.read_bytes()

    # 2. 只应用勘误并证明身份、凭据、角色与业务状态保持不变
    corrected_repo = _catalog(
        tmp_path / "corrected-repo",
        (
            _ORIGIN_ID,
            _AKASHA_V9_ID,
            _MODEL_REGISTRY_ID,
            _EMBEDDING_REGISTRY_ID,
            _MODEL_CAPABILITIES_ID,
            _OPENCODE_VARIANTS_ID,
        ),
    )
    corrected = _runner(root, repo_root=corrected_repo).run()
    assert corrected.migrations == (_OPENCODE_VARIANTS_ID,)
    after = store.read_snapshot()
    assert after is not None
    runtime = after.runtimes["opencode_go_main"]
    assert runtime.source_id == "source:opencode_go_main"
    assert runtime.reasoning_effort == "high"
    assert runtime.supported_reasoning_efforts == ("low", "high", "max")
    assert runtime.context_window == 1_000_000
    assert after.roles == before.roles
    assert after.revision == before.revision + 1
    assert config.read_bytes() == config_after_first_migration
    assert sessions.read_bytes() == b"protected-session-bytes"
    assert (
        CredentialStore.for_workspace(root / "workspace").api_key(
            "model_opencode_go_main"
        )
        == "migration-secret"
    )
    backups = list(
        (root / "workspace/backups/model-registry-opencode-variants-v1").glob(
            "*/registry.before.sqlite3"
        )
    )
    assert len(backups) == 1
    ModelRegistryStore(backups[0]).integrity_check()


def test_model_registry_migration_accepts_toml_rewritten_nested_tables(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    root.mkdir()
    config = root / "config.toml"
    config.write_text(
        toml.dumps(
            {
                "llm": {
                    "main": "plugin_gate",
                    "runtimes": {
                        "plugin_gate": {
                            "provider": "openai",
                            "model": "plugin-gate",
                            "api_key": "gate-not-used",
                        }
                    },
                },
                "agent": {"system_prompt": "plugin gate"},
                "app_server": {"listen": "/sandbox/akashic.sock"},
            }
        ),
        encoding="utf-8",
    )

    outcome = _runner(root).run()

    assert outcome.migrations == _CURRENT_IDS
    migrated = tomllib.loads(config.read_text(encoding="utf-8"))
    assert "llm" not in migrated
    assert migrated["agent"] == {
        "system_prompt": "plugin gate",
        "context": {"compaction": {"keep_recent_tokens": 20_000}},
    }
    assert migrated["app_server"] == {"listen": "/sandbox/akashic.sock"}


def test_model_registry_migration_moves_inline_key_to_credential_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "state"
    root.mkdir()
    config = root / "config.toml"
    config.write_text(
        """
[llm]
main = "deepseek_main"

[llm.runtimes.deepseek_main]
provider = "openai"
catalog_provider_id = "deepseek"
model = "deepseek-chat"
base_url = "https://api.deepseek.com/v1"
api_key = "secret-value"
input_modalities = ["text"]

[agent]
system_prompt = "test"
""",
        encoding="utf-8",
    )

    _ = _runner(root).run()

    assert "secret-value" not in config.read_text(encoding="utf-8")
    assert (
        CredentialStore.for_workspace(root / "workspace").api_key("model_deepseek_main")
        == "secret-value"
    )
    assert not CredentialStore().path.exists()
    snapshot = ModelRegistryStore.for_workspace(root / "workspace").read_snapshot()
    assert snapshot is not None
    assert snapshot.runtimes["deepseek_main"].auth_id == "model_deepseek_main"


def test_model_registry_migration_copies_referenced_legacy_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    legacy = CredentialStore()
    legacy.put(
        "deepseek_default",
        Credential(driver="api_key", access_token="legacy-secret"),
    )
    root = tmp_path / "state"
    root.mkdir()
    (root / "config.toml").write_text(
        """
[llm]
main = "deepseek_main"

[llm.runtimes.deepseek_main]
provider = "deepseek"
model = "deepseek-chat"
auth = "deepseek_default"
base_url = "https://api.deepseek.com/v1"
input_modalities = ["text"]
""",
        encoding="utf-8",
    )

    _ = _runner(root).run()

    assert (
        CredentialStore.for_workspace(root / "workspace").api_key("deepseek_default")
        == "legacy-secret"
    )
    assert legacy.api_key("deepseek_default") == "legacy-secret"


def test_model_registry_migration_failure_restores_inputs_and_retries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    root.mkdir()
    config = root / "config.toml"
    invalid = b"""
[llm]
main = "missing"

[llm.runtimes.broken]
provider = "deepseek"
model = "deepseek-chat"
api_key = "secret"
"""
    config.write_bytes(invalid)
    runner = _runner(root)

    with pytest.raises(RuntimeError, match="必须引用已配置 runtime"):
        runner.run()

    assert config.read_bytes() == invalid
    assert not (root / "workspace/model-registry.sqlite3").exists()
    assert _applied_ids(runner.ledger_path) == [_ORIGIN_ID, _AKASHA_V9_ID]

    config.write_text(
        """
[llm]
main = "deepseek_main"

[llm.runtimes.deepseek_main]
provider = "deepseek"
model = "deepseek-chat"
api_key = "secret"
""",
        encoding="utf-8",
    )
    outcome = runner.run()

    assert outcome.migrations == (
        _MODEL_REGISTRY_ID,
        _COMPACTION_ID,
        _EMBEDDING_REGISTRY_ID,
        _MODEL_CAPABILITIES_ID,
        _AUDIT_ID,
        _OPENCODE_VARIANTS_ID,
        _PREPARE_ID,
        _DIGEST_ID,
        _CURSOR_ID,
        _CONFIG_ID,
        _RETIRE_ID,
        _AKASHA_V10_ID,
        _TOOLSET_WIRING_ID,
        _PROACTIVE_DELIVERY_TARGET_ID,
        _TURN_EFFECTS_ID,
        _AKASHA_PLUGIN_SELECTION_ID,
        _AKASHA_EMBEDDING_BACKFILL_ID,
        _AKASHIC_CHANNEL_IDENTITY_ID,
        _SESSION_TIMESTAMP_ID,
        _MOBILE_CLIENT_ID_ID,
        _EVENTMAIL_STATE_ID,
        _WAKE_CONTENT_SCORES_ID,
        _PROGRAMMATIC_EFFECTS_ID,
        _EXPLICIT_PROGRAMMATIC_EFFECTS_ID,
        _RETIRE_CORE_MODEL_CONFIG_ID,
    )
    assert (
        CredentialStore.for_workspace(root / "workspace").api_key("model_deepseek_main")
        == "secret"
    )
