from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Protocol, cast


class CompactionReceiptPort(Protocol):
    """Store immutable checkpoint saga receipts by source identity."""

    def read(self, source_ref: str) -> dict[str, object] | None: ...

    def write(
        self,
        source_ref: str,
        payload: dict[str, object],
    ) -> dict[str, object]: ...

    def list_all(self) -> tuple[dict[str, object], ...]: ...


class SqliteCompactionReceipts:
    """Use the retained consolidation ledger for compaction receipts only."""

    _KIND = "session_compaction_receipt"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with closing(sqlite3.connect(str(self._path))) as conn:
            _ = conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_writes (
                    source_ref TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT,
                    trailing_blank_line INTEGER NOT NULL DEFAULT 0,
                    done_at TEXT NOT NULL,
                    PRIMARY KEY (source_ref, kind)
                )""")
            columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(consolidation_writes)"
                ).fetchall()
            }
            if "payload" not in columns:
                _ = conn.execute(
                    "ALTER TABLE consolidation_writes ADD COLUMN payload TEXT"
                )
            if "trailing_blank_line" not in columns:
                _ = conn.execute(
                    "ALTER TABLE consolidation_writes ADD COLUMN "
                    "trailing_blank_line INTEGER NOT NULL DEFAULT 0"
                )

    def read(self, source_ref: str) -> dict[str, object] | None:
        source = source_ref.strip()
        if not source:
            raise ValueError("compaction receipt source_ref 不能为空")
        with (
            self._lock,
            closing(sqlite3.connect(str(self._path), timeout=30.0)) as conn,
        ):
            row = conn.execute(
                "SELECT payload FROM consolidation_writes "
                "WHERE source_ref=? AND kind=?",
                (source, self._KIND),
            ).fetchone()
        if row is None:
            return None
        raw = row[0]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"compaction receipt payload 缺失: {source}")
        value = cast(Any, json.loads(raw))
        if not isinstance(value, dict):
            raise ValueError(f"compaction receipt 必须是 JSON object: {source}")
        items = cast(dict[object, object], value)
        return {str(key): item for key, item in items.items()}

    def write(
        self,
        source_ref: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        source = source_ref.strip()
        if not source:
            raise ValueError("compaction receipt source_ref 不能为空")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with (
            self._lock,
            closing(sqlite3.connect(str(self._path), timeout=30.0)) as conn,
        ):
            _ = conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT payload FROM consolidation_writes "
                "WHERE source_ref=? AND kind=?",
                (source, self._KIND),
            ).fetchone()
            if row is not None:
                if row[0] != encoded:
                    raise ValueError(f"compaction receipt 内容冲突: {source}")
                conn.commit()
                return dict(payload)
            _ = conn.execute(
                "INSERT INTO consolidation_writes "
                "(source_ref, kind, payload, trailing_blank_line, done_at) "
                "VALUES (?, ?, ?, 0, datetime('now'))",
                (source, self._KIND, encoded),
            )
            conn.commit()
        return dict(payload)

    def list_all(self) -> tuple[dict[str, object], ...]:
        """Return detached receipts in durable creation order."""

        with (
            self._lock,
            closing(sqlite3.connect(str(self._path), timeout=30.0)) as conn,
        ):
            rows = conn.execute(
                "SELECT source_ref, payload FROM consolidation_writes "
                "WHERE kind=? ORDER BY done_at, source_ref",
                (self._KIND,),
            ).fetchall()
        receipts: list[dict[str, object]] = []
        for source_ref, raw in rows:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"compaction receipt payload 缺失: {source_ref}")
            value = cast(Any, json.loads(raw))
            if not isinstance(value, dict):
                raise ValueError(f"compaction receipt 必须是 JSON object: {source_ref}")
            raw_items = cast(dict[object, object], value)
            receipt: dict[str, object] = {
                str(key): item for key, item in raw_items.items()
            }
            if receipt.get("source_ref", source_ref) != source_ref:
                raise ValueError(f"compaction receipt source_ref 冲突: {source_ref}")
            receipts.append(receipt)
        return tuple(receipts)
