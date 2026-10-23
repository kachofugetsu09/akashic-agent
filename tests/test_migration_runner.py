from __future__ import annotations
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]

def _applied_ids(ledger: Path) -> tuple[str, ...]:
    if not ledger.exists():
        return ()
    with closing(sqlite3.connect(ledger)) as connection:
        try:
            rows = connection.execute(
                "SELECT migration_id FROM _yoyo_migration ORDER BY rowid"
            ).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error).lower():
                raise
            return ()
    return tuple(str(row[0]) for row in rows)

def test_core_only_cli_restarts_after_creating_runtime_data(
    tmp_path: Path,
) -> None:
    """CLI startup must not discover checkout plugins implicitly."""

    config = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    environment = dict(os.environ)
    environment["AKASHIC_PLUGIN_HOME"] = str(tmp_path / "plugin-home")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(_PROJECT_ROOT / "sdk/python/src"), str(_PROJECT_ROOT))
    )
    arguments = ["--config", str(config), "--workspace", str(workspace)]
    commands = (
        ["init", *arguments],
        ["--inspect-modules", *arguments],
        ["--inspect-modules", *arguments],
    )
    for command in commands:
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "main.py"), *command],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=40,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert (workspace / "migrations.sqlite3").is_file()
    assert set(_applied_ids(workspace / "migrations.sqlite3")) == {
        "20260921_01_plugin_update_input_ref",
        "20260928_01_plugin_config_updates",
        "20261006_01_config_update_receipts",
        "20261011_01_plugin_contract_restart",
        "20261019_01_vision_split",
        "20261019_02_skills_split",
        "20261019_03_shell_split",
        "20261019_04_files_split",
        "20261019_05_working_directory_rename",
        "20261019_06_web_provider_split",
    }
    assert not (workspace / "sessions.db").exists()


def test_web_provider_upgrade_preserves_explicit_choices(tmp_path: Path) -> None:
    """An old Web tools install must pass preflight and retain search/fetch providers."""
    # 场景保留 EVIDENCE 目录供排查；放进 pytest 临时目录，由 pytest 轮换清理。
    environment = {**os.environ, "TMPDIR": str(tmp_path), "PYTHONPATH": os.pathsep.join(
        (str(_PROJECT_ROOT / "sdk/python/src"), str(_PROJECT_ROOT)))}
    result = subprocess.run(
        [sys.executable, str(_PROJECT_ROOT / "scripts/deployment_migration_scenario.py"), "--web-providers"],
        cwd=_PROJECT_ROOT, env=environment, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
