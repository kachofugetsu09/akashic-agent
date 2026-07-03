from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

SKILL_NAME = "default_memory:audit-dirty-memories"


def _audited_ids(drift_dir: Path) -> set[str]:
    db_file = drift_dir / "drift.db"
    if not db_file.exists():
        return set()
    conn = sqlite3.connect(db_file)
    try:
        exists = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'skill_journal'
            """
        ).fetchone()
        if exists is None:
            return set()
        rows = conn.execute(
            """
            SELECT key
            FROM skill_journal
            WHERE skill_name = ?
              AND entry_type = 'memory_audited'
              AND key != ''
            """,
            (SKILL_NAME,),
        ).fetchall()
    finally:
        conn.close()
    return {str(row[0]) for row in rows if str(row[0]).strip()}


def _sample(drift_dir: Path) -> dict[str, object]:
    workspace = drift_dir.parent
    memory_db = workspace / "memory" / "memory2.db"
    audited = _audited_ids(drift_dir)
    if not memory_db.exists():
        return {
            "found": False,
            "reason": "memory2.db not found",
            "audited_count": len(audited),
        }

    conn = sqlite3.connect(memory_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, memory_type, summary, source_ref, happened_at
            FROM memory_items
            WHERE status = 'active'
              AND source_ref IS NOT NULL
              AND TRIM(source_ref) != ''
              AND source_ref NOT LIKE '%@post_response'
            ORDER BY RANDOM()
            LIMIT 200
            """
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        memory_id = str(row["id"])
        if memory_id in audited:
            continue
        return {
            "found": True,
            "audited_count": len(audited),
            "journal_append_required": {
                "entry_type": "memory_audited",
                "key": memory_id,
                "payload": {
                    "result": "<clean|unverifiable_old_source|suspicious_reported>",
                    "source_ref": row["source_ref"] or "",
                    "summary": row["summary"] or "",
                },
            },
            "item": {
                "id": memory_id,
                "memory_type": row["memory_type"] or "",
                "summary": row["summary"] or "",
                "source_ref": row["source_ref"] or "",
                "happened_at": row["happened_at"] or "",
            },
        }
    return {"found": False, "audited_count": len(audited)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sample = sub.add_parser("sample")
    _ = sample.add_argument("--drift-dir", default=".")
    args = parser.parse_args()

    drift_dir = Path(str(args.drift_dir)).expanduser().resolve()
    payload = _sample(drift_dir)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
