from __future__ import annotations

# pyright: reportPrivateUsage=false

import json
from pathlib import Path

import pytest

from docker.debug.plugin_composition_migration_gate import (
    DEFAULT_LOCK,
    _database_files,
    _load_lock,
)


def test_checked_in_lock_binds_exact_migration_candidates() -> None:
    lock = _load_lock(DEFAULT_LOCK)

    assert lock.profile == "observe-status-ownership-v1"
    assert [(item.id, item.commit) for item in lock.plugins] == [
        ("observe", "9d53df0e1fef0c34f9b0a026d77ba943c08a5ace"),
        (
            "status_commands",
            "7de1d1385691d9891a814c62d1fd9650e2387640",
        ),
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"schema_version": 2}, "不支持的插件组合迁移锁版本"),
        ({"extra": True}, "插件组合迁移锁根结构无效"),
        ({"plugins": []}, "必须按 Observe、status_commands 排列"),
    ),
)
def test_migration_lock_rejects_schema_drift(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    raw = json.loads(DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw.update(mutation)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _ = _load_lock(path)


def test_database_files_detects_write_set_change(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    db_path.write_bytes(b"before")

    before = _database_files(db_path)
    db_path.write_bytes(b"after")
    after = _database_files(db_path)

    assert before != after
    assert set(before["sessions.db"]) == {"bytes", "sha256"}
