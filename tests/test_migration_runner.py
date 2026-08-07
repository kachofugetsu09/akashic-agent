from __future__ import annotations

import os
import shutil
import sqlite3
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
_MODEL_REGISTRY_ID = "20260807_01_model_registry_database"
_EMBEDDING_REGISTRY_ID = "20260807_02_embedding_model_registry"
_MODEL_CAPABILITIES_ID = "20260808_01_restore_migrated_reasoning_efforts"
_OPENCODE_VARIANTS_ID = "20260808_02_correct_opencode_go_variants"
_CURRENT_IDS = (
    _ORIGIN_ID,
    _AKASHA_V9_ID,
    _MODEL_REGISTRY_ID,
    _EMBEDDING_REGISTRY_ID,
    _MODEL_CAPABILITIES_ID,
    _OPENCODE_VARIANTS_ID,
)


def _runner(root: Path, *, repo_root: Path = _PROJECT_ROOT) -> MigrationRunner:
    return MigrationRunner(
        repo_root=repo_root,
        config_path=root / "config.toml",
        workspace=root / "workspace",
    )


def _applied_ids(ledger: Path) -> list[str]:
    connection = sqlite3.connect(ledger)
    try:
        rows = connection.execute(
            "SELECT migration_id FROM _yoyo_migration ORDER BY migration_id"
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def test_origin_removes_legacy_state_without_touching_business_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    config = root / "config.toml"
    config.write_bytes(b"current = true\n")
    sessions = workspace / "sessions.db"
    sessions.write_bytes(b"session-bytes")
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
    assert config.read_bytes() == b"current = true\n"
    assert sessions.read_bytes() == b"session-bytes"
    assert memory.read_bytes() == b"memory-bytes"
    assert _applied_ids(workspace / "migrations.sqlite3") == list(_CURRENT_IDS)


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
    assert _applied_ids(runner.ledger_path) == list(_CURRENT_IDS)


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

    outcome = _runner(root).run()

    assert outcome.migrations == _CURRENT_IDS
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
    legacy_repo = tmp_path / "legacy-repo"
    legacy_catalog = legacy_repo / "migrations/yoyo"
    legacy_catalog.mkdir(parents=True)
    for migration_id in (
        _ORIGIN_ID,
        _AKASHA_V9_ID,
        _MODEL_REGISTRY_ID,
        _EMBEDDING_REGISTRY_ID,
        _MODEL_CAPABILITIES_ID,
    ):
        shutil.copy2(
            _PROJECT_ROOT / "migrations/yoyo" / f"{migration_id}.py",
            legacy_catalog / f"{migration_id}.py",
        )
    first = _runner(root, repo_root=legacy_repo).run()
    assert first.migrations == _CURRENT_IDS[:-1]
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
    corrected = _runner(root).run()
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
    assert migrated["llm"] == {"registry": "workspace"}
    assert migrated["agent"] == {"system_prompt": "plugin gate"}
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
        _EMBEDDING_REGISTRY_ID,
        _MODEL_CAPABILITIES_ID,
        _OPENCODE_VARIANTS_ID,
    )
    assert (
        CredentialStore.for_workspace(root / "workspace").api_key("model_deepseek_main")
        == "secret"
    )
