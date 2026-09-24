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
    }
