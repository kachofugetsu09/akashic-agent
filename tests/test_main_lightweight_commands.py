from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

from agent.persona import read_default_veda


_PROJECT_ROOT = Path(__file__).parents[1]


def test_setup_main_does_not_import_agent_runtime(tmp_path: Path) -> None:
    """setup-main 应在完整 Agent runtime 依赖加载前完成分发。"""
    missing_config = tmp_path / "missing.toml"

    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "main.py"),
            "setup-main",
            "--config",
            str(missing_config),
            "--workspace",
            str(tmp_path / "workspace"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "配置文件不存在" in output
    assert "apscheduler" not in output


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
