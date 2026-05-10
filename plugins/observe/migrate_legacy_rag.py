from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import open_db


@dataclass(frozen=True)
class LegacyRagMigrationResult:
    migrated_events: int
    migrated_hits: int
    dropped_tables: tuple[str, ...]


def migrate_legacy_rag_tables(db_path: Path) -> LegacyRagMigrationResult:
    conn = open_db(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "rag_events"):
            return LegacyRagMigrationResult(0, 0, ())

        has_items = _table_exists(conn, "rag_items")
        events = conn.execute("SELECT * FROM rag_events ORDER BY id").fetchall()
        migrated_hits = 0

        with conn:
            for event in events:
                items = _load_items(conn, event["id"]) if has_items else []
                hits, injected_count = _build_hits(items)
                migrated_hits += len(hits)
                _ = conn.execute(
                    """
                    INSERT INTO rag_queries (
                        ts, caller, session_key, query, orig_query,
                        aux_queries, hits_json, injected_count, route_decision, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _get(event, "ts") or "",
                        _caller_from_source(_get(event, "source")),
                        _get(event, "session_key") or "",
                        _get(event, "query") or _get(event, "original_query") or "",
                        _orig_query(event),
                        _aux_queries(event),
                        json.dumps(hits, ensure_ascii=False) if hits else None,
                        injected_count,
                        _get(event, "route_decision"),
                        _get(event, "error"),
                    ),
                )

            if has_items:
                _ = conn.execute("DROP TABLE rag_items")
            _ = conn.execute("DROP TABLE rag_events")

        dropped = ("rag_items", "rag_events") if has_items else ("rag_events",)
        return LegacyRagMigrationResult(len(events), migrated_hits, dropped)
    finally:
        conn.close()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _load_items(conn: sqlite3.Connection, event_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM rag_items WHERE rag_event_id = ? ORDER BY id",
        (event_id,),
    ).fetchall()


def _get(row: sqlite3.Row, key: str) -> Any:
    if key not in row.keys():
        return None
    return row[key]


def _caller_from_source(source: Any) -> str:
    if source == "proactive":
        return "proactive"
    if source == "explicit":
        return "explicit"
    return "passive"


def _orig_query(event: sqlite3.Row) -> str | None:
    original_query = _get(event, "original_query")
    query = _get(event, "query")
    if not original_query or original_query == query:
        return None
    return str(original_query)


def _aux_queries(event: sqlite3.Row) -> str | None:
    hypothesis = _get(event, "hyde_hypothesis")
    if not hypothesis:
        return None
    return json.dumps([hypothesis], ensure_ascii=False)


def _build_hits(items: list[sqlite3.Row]) -> tuple[list[dict[str, Any]], int]:
    hits: list[dict[str, Any]] = []
    injected_count = 0
    for item in items:
        injected = bool(_get(item, "injected") or 0)
        if injected:
            injected_count += 1
        hits.append(
            {
                "id": _get(item, "item_id") or "",
                "type": _get(item, "memory_type") or "",
                "score": float(_get(item, "score") or 0),
                "summary": _get(item, "summary") or "",
                "injected": injected,
            }
        )
    return hits, injected_count


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("db_path", type=Path)
    args = parser.parse_args()

    result = migrate_legacy_rag_tables(args.db_path)
    print(
        "migrated_events=%d migrated_hits=%d dropped_tables=%s"
        % (result.migrated_events, result.migrated_hits, ",".join(result.dropped_tables))
    )


if __name__ == "__main__":
    main()
