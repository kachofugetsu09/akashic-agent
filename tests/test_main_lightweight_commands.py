from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

from agent.persona import read_default_veda

_PROJECT_ROOT = Path(__file__).parents[1]


def test_init_records_yoyo_origin_in_workspace_ledger(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"

    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "main.py"),
            "init",
            "--config",
            str(config_path),
            "--workspace",
            str(workspace),
        ],
        cwd=_PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    ledger = workspace / "migrations.sqlite3"
    connection = sqlite3.connect(ledger)
    try:
        applied = connection.execute(
            "SELECT migration_id FROM _yoyo_migration"
        ).fetchall()
    finally:
        connection.close()
    assert applied == [
        ("20260802_01_yoyo_origin",),
        ("20260805_01_akasha_sparse_index_v9",),
        ("20260807_01_model_registry_database",),
        ("20260807_01_session_context_compaction_ledger",),
        ("20260807_02_embedding_model_registry",),
        ("20260808_01_restore_migrated_reasoning_efforts",),
        ("20260808_01_session_mutation_audits",),
        ("20260808_02_correct_opencode_go_variants",),
        ("20260808_02_session_compaction_prepares",),
        ("20260808_04_session_compaction_source_plan_digest",),
        ("20260808_05_activate_session_compaction_cursor",),
        ("20260808_03_remove_compaction_trigger",),
        ("20260808_06_retire_legacy_context_state",),
        ("20260817_01_akasha_sparse_index_v10",),
        ("20260823_01_retire_legacy_toolset_wiring",),
        ("20260825_01_migrate_proactive_delivery_target",),
        ("20260826_01_migrate_turn_effects",),
        ("20260825_02_select_akasha_embedding_plugin",),
        ("20260826_02_backfill_akasha_message_embeddings",),
        ("20260826_03_unify_akashic_channel_identity",),
        ("20260827_01_normalize_session_timestamps",),
        ("20260827_02_migrate_legacy_mobile_client_ids",),
        ("20260828_01_migrate_eventmail_state",),
        ("20260828_02_add_wake_content_scores",),
        ("20260829_01_backfill_plugin_programmatic_effects",),
        ("20260829_02_backfill_explicit_programmatic_effects",),
        ("20260829_03_retire_core_model_config",),
        ("20260831_01_migrate_compaction_plugin_config",),
    ]
    assert not config_path.with_name("config.toml.migration-cursor").exists()


def test_veda_reset_runs_before_agent_runtime_and_preserves_original_bytes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    veda = workspace / "memory/VEDA.md"
    veda.parent.mkdir(parents=True)
    original = b"\xffbroken"
    veda.write_bytes(original)

    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "main.py"),
            "veda-reset",
            "--workspace",
            str(workspace),
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert veda.read_text(encoding="utf-8").strip() == read_default_veda()
    backups = list((workspace / "memory/veda-backups").glob("*/VEDA.md"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert "原内容 sha256=" in output
    assert "apscheduler" not in output


def test_veda_reset_reports_noop_without_creating_backup(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"

    first = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "main.py"),
            "veda-reset",
            "--workspace",
            str(workspace),
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "main.py"),
            "veda-reset",
            "--workspace",
            str(workspace),
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Veda 已是默认内容" in second.stdout
    assert not (workspace / "memory/veda-backups").exists()


def test_help_lists_veda_reset() -> None:
    result = subprocess.run(
        [sys.executable, str(_PROJECT_ROOT / "main.py"), "--help"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "veda-reset" in result.stdout
