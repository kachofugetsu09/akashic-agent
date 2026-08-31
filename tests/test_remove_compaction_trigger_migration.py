from __future__ import annotations

import json
import hashlib
import stat
import tomllib
from pathlib import Path

from agent.migrations.runner import MigrationRunner


_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_ID = "20260808_03_remove_compaction_trigger"


def _runner(root: Path) -> MigrationRunner:
    return MigrationRunner(
        repo_root=_PROJECT_ROOT,
        config_path=root / "config.toml",
        workspace=root / "workspace",
    )


def _config_text(keep_recent_tokens: int) -> str:
    return (
        "[agent.context.compaction]\n"
        "trigger_percent = 0.74\n"
        f"keep_recent_tokens = {keep_recent_tokens}\n"
    )


def test_correction_removes_trigger_and_keeps_recoverable_backup(tmp_path: Path) -> None:
    root = tmp_path / "state"
    root.mkdir()
    config = root / "config.toml"
    original = _config_text(21_000).encode("utf-8")
    config.write_bytes(original)

    outcome = _runner(root).run()

    assert _MIGRATION_ID in outcome.migrations
    loaded = tomllib.loads(config.read_text(encoding="utf-8"))
    assert "agent" not in loaded
    plugin_config = (
        root
        / "workspace/plugin-data/compaction-builtin/config.local.toml"
    )
    assert tomllib.loads(plugin_config.read_text(encoding="utf-8")) == {
        "keep_recent_tokens": 21_000
    }
    backups = sorted((root / "workspace/backups/remove-compaction-trigger").iterdir())
    assert len(backups) == 1
    manifest = json.loads((backups[0] / "manifest.json").read_text(encoding="utf-8"))
    backup = backups[0] / manifest["source"]["backup"]
    assert backup.read_bytes() == original
    assert manifest["source"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert stat.S_IMODE((backups[0] / "manifest.json").stat().st_mode) == 0o600

    second = _runner(root).run()
    assert second.state == "current"
    assert len(list((root / "workspace/backups/remove-compaction-trigger").iterdir())) == 1
