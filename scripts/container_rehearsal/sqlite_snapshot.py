"""Create verified SQLite backups and validate persisted media references."""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast

from scripts.container_rehearsal.model import SnapshotDriftError, is_relative_to

SQLITE_HEADER = b"SQLite format 3\x00"


def is_sqlite(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(SQLITE_HEADER)) == SQLITE_HEADER


def _integrity_rows(connection: sqlite3.Connection) -> list[tuple[str]]:
    return connection.execute("PRAGMA integrity_check").fetchall()


def copy_sqlite(source: Path, destination: Path) -> dict[str, Any]:
    """在线备份一个 SQLite，并验证源与副本完整性。"""

    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True, timeout=30)) as source_db:
        _ = source_db.execute("PRAGMA query_only = ON")
        source_integrity = _integrity_rows(source_db)
        if source_integrity != [("ok",)]:
            raise sqlite3.DatabaseError(
                f"源 SQLite integrity_check 失败: {source}: {source_integrity[:3]}"
            )
        with closing(sqlite3.connect(destination)) as destination_db:
            source_db.backup(destination_db, pages=256, sleep=0.05)
            destination_db.commit()
            target_integrity = _integrity_rows(destination_db)
            if target_integrity != [("ok",)]:
                raise sqlite3.DatabaseError(
                    f"副本 SQLite integrity_check 失败: {destination}: "
                    f"{target_integrity[:3]}"
                )
            page_count = int(destination_db.execute("PRAGMA page_count").fetchone()[0])
    shutil.copymode(source, destination, follow_symlinks=False)
    return {
        "source_integrity_check": "ok",
        "target_integrity_check": "ok",
        "page_count": page_count,
    }


def verify_session_media_references(
    source: Path, destination: Path, databases: list[dict[str, Any]]
) -> None:
    """验证 SessionDB 中明确属于 Workspace 的媒体路径已进入副本。"""

    session_database = destination / "sessions.db"
    record = next((item for item in databases if item["path"] == "sessions.db"), None)
    if record is None or not session_database.is_file():
        return
    checked = 0
    source_missing: set[str] = set()
    with closing(sqlite3.connect(session_database)) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        if not {"id", "extra"}.issubset(columns):
            record["workspace_media_references"] = "schema_not_applicable"
            return
        rows = connection.execute(
            "SELECT id, extra FROM messages WHERE extra LIKE ?", ('%"media"%',)
        ).fetchall()
    for message_id, raw_extra in rows:
        try:
            extra = json.loads(raw_extra or "{}")
        except (TypeError, ValueError) as exc:
            raise sqlite3.DatabaseError(
                f"SessionDB message extra JSON 损坏: {message_id}"
            ) from exc
        extra_dict = cast(dict[str, object], extra) if isinstance(extra, dict) else {}
        media: object = extra_dict.get("media")
        if media is None:
            continue
        media_items = cast(list[object], media) if isinstance(media, list) else []
        if not isinstance(media, list) or not all(
            isinstance(item, str) for item in media_items
        ):
            raise sqlite3.DatabaseError(
                f"SessionDB message media 不是字符串数组: {message_id}"
            )
        for item in cast(list[str], media_items):
            media_path = Path(item).expanduser()
            if not media_path.is_absolute():
                continue
            resolved = media_path.resolve(strict=False)
            if not is_relative_to(resolved, source):
                continue
            relative = resolved.relative_to(source)
            if not (source / relative).is_file():
                source_missing.add(relative.as_posix())
                continue
            if not (destination / relative).is_file():
                raise SnapshotDriftError(
                    f"SessionDB 引用的 Workspace 媒体未进入副本: {relative}"
                )
            checked += 1
    record["workspace_media_references"] = {
        "checked": checked,
        "preexisting_missing": sorted(source_missing),
        "status": "ok",
    }
