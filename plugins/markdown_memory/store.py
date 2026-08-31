from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast

from infra.persistence.json_store import atomic_write_text


DEFAULT_SELF_MD = """# Akashic 的自我认知

## 人格与形象
- 我是 Akashic，一个直接、温暖、主动参与思考的长期协作伙伴。
- 我优先给出结论，再补充必要细节；不把自己伪装成没有立场的工具。

## 我对当前用户的理解
- 我会从长期记忆中逐步形成对当前用户的理解，不在缺少证据时编造画像。

## 我们关系的定义
- 我与当前用户的关系以透明、尊重边界和持续协作为基础。
"""


def content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class MarkdownProfileStore:
    """Own the two Markdown profiles and their idempotent projection receipts."""

    def __init__(self, memory_path: Path, self_path: Path, receipts_path: Path) -> None:
        self.memory_path = memory_path
        self.self_path = self_path
        self.receipts_path = receipts_path
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not memory_path.exists():
            atomic_write_text(memory_path, "", domain="markdown_memory")
        if not self_path.exists():
            atomic_write_text(self_path, DEFAULT_SELF_MD, domain="markdown_memory")
        self._init_schema()

    def read_memory(self) -> str:
        return self.memory_path.read_text(encoding="utf-8")

    def read_self(self) -> str:
        return self.self_path.read_text(encoding="utf-8")

    def read_draft(self, source_ref: str) -> dict[str, object] | None:
        return self._read_receipt(source_ref, "markdown_profile_model_v1")

    def write_draft(
        self,
        source_ref: str,
        payload: dict[str, object],
        *,
        session_key: str,
        generation: int,
    ) -> dict[str, object]:
        stored = self._write_once(source_ref, "markdown_profile_model_v1", payload)
        _ = self._write_once(
            source_ref,
            "markdown_projection_order_v1",
            {"session_key": session_key, "generation": generation},
        )
        self._write_document_drafts(source_ref, stored)
        return stored

    def is_applied(self, source_ref: str) -> bool:
        return all(
            self._read_receipt(source_ref, f"markdown_{document}_applied_v1")
            is not None
            for document in ("memory", "self")
        )

    def pending_source_refs(self) -> tuple[str, ...]:
        """List drafts whose two document receipts have not both committed."""

        with closing(sqlite3.connect(str(self.receipts_path), timeout=30.0)) as conn:
            rows = conn.execute(
                "SELECT DISTINCT source_ref, done_at FROM consolidation_writes "
                "WHERE kind IN (?, ?) ORDER BY done_at, source_ref",
                ("markdown_memory_draft_v1", "markdown_self_draft_v1"),
            ).fetchall()
        pending: list[tuple[int, str, int, str]] = []
        for position, row in enumerate(rows):
            source_ref = str(row[0])
            if self.is_applied(source_ref):
                continue
            order = self._read_receipt(source_ref, "markdown_projection_order_v1")
            if order is None:
                raise RuntimeError(f"Markdown draft 缺少 durable generation: {source_ref}")
            session_key = order.get("session_key")
            generation = order.get("generation")
            if (
                not isinstance(session_key, str)
                or not session_key
                or not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation < 0
            ):
                raise ValueError(f"Markdown draft durable generation 无效: {source_ref}")
            pending.append((position, session_key, generation, source_ref))
        # Generation is authoritative within one Session. Between Sessions,
        # preserve the earliest original receipt that is currently eligible.
        queues: dict[str, list[tuple[int, str, int, str]]] = {}
        for item in pending:
            queues.setdefault(item[1], []).append(item)
        for queue in queues.values():
            queue.sort(key=lambda item: item[2])
        ordered: list[str] = []
        while queues:
            session_key, queue = min(
                queues.items(),
                key=lambda item: item[1][0][0],
            )
            ordered.append(queue.pop(0)[3])
            if not queue:
                del queues[session_key]
        return tuple(ordered)

    def read_backup(self, source_ref: str, document: str) -> str | None:
        """Return one immutable before-image for explicit recovery tooling."""

        if document not in {"memory", "self"}:
            raise ValueError(f"Markdown profile backup kind 无效: {document}")
        receipt = self._read_receipt(
            source_ref,
            f"markdown_profile_{document}_backup_v1",
        )
        if receipt is None:
            return None
        return _required_string(receipt, "content")

    def read_legacy_pending_migration(self) -> dict[str, object] | None:
        return self._read_receipt(
            "legacy-pending-migration-v1",
            "legacy_pending_source_v1",
        )

    def write_legacy_pending_migration(
        self,
        payload: dict[str, object],
    ) -> None:
        _ = self._write_once(
            "legacy-pending-migration-v1",
            "legacy_pending_source_v1",
            payload,
        )

    def mark_legacy_pending_retired(self, source_ref: str) -> None:
        _ = self._write_once(
            "legacy-pending-migration-v1",
            "legacy_pending_retired_v1",
            {"source_ref": source_ref},
        )

    def apply_draft(self, source_ref: str, payload: dict[str, object]) -> None:
        """Install independent document drafts and converge after any crash point."""

        self._write_document_drafts(source_ref, payload)
        self.apply_pending(source_ref)

    def apply_pending(self, source_ref: str) -> None:
        """Converge each document from its own immutable draft and receipt."""

        self._apply_document(source_ref, "memory", self.memory_path)
        self._apply_document(source_ref, "self", self.self_path)

    def _write_document_drafts(
        self,
        source_ref: str,
        payload: dict[str, object],
    ) -> None:
        for document in ("memory", "self"):
            content_key = document if document == "memory" else "self"
            before_key = f"{document}_before"
            before = _required_string(payload, before_key)
            after = _required_string(payload, content_key)
            _ = self._write_once(
                source_ref,
                f"markdown_{document}_draft_v1",
                {
                    "before": before,
                    "before_digest": content_digest(before),
                    "after": after,
                    "after_digest": content_digest(after),
                },
            )

    def _apply_document(self, source_ref: str, document: str, path: Path) -> None:
        applied_kind = f"markdown_{document}_applied_v1"
        if self._read_receipt(source_ref, applied_kind) is not None:
            return
        draft = self._read_receipt(source_ref, f"markdown_{document}_draft_v1")
        if draft is None:
            raise RuntimeError(f"Markdown {document} draft 缺失: {source_ref}")
        before = _required_string(draft, "before")
        after = _required_string(draft, "after")
        before_digest = _required_string(draft, "before_digest")
        after_digest = _required_string(draft, "after_digest")
        if content_digest(before) != before_digest or content_digest(after) != after_digest:
            raise ValueError(f"Markdown {document} draft digest 无效")
        current = path.read_text(encoding="utf-8")
        current_digest = content_digest(current)
        if current_digest not in {before_digest, after_digest}:
            raise RuntimeError(f"{path.name} 在 profile projection 期间发生并发变化")
        self._backup(document, before, source_ref)
        if current_digest != after_digest:
            atomic_write_text(path, after, domain="markdown_memory")
        _ = self._write_once(
            source_ref,
            applied_kind,
            {"digest": after_digest},
        )

    def _backup(self, document: str, content: str, source_ref: str) -> None:
        """Keep one immutable before-image inside the declared receipt file."""

        _ = self._write_once(
            source_ref,
            f"markdown_profile_{document}_backup_v1",
            {"content": content, "digest": content_digest(content)},
        )

    def _init_schema(self) -> None:
        self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(self.receipts_path))) as conn:
            _ = conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_writes (
                    source_ref TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT,
                    trailing_blank_line INTEGER NOT NULL DEFAULT 0,
                    done_at TEXT NOT NULL,
                    PRIMARY KEY (source_ref, kind)
                )""")

    def _read_receipt(self, source_ref: str, kind: str) -> dict[str, object] | None:
        with closing(sqlite3.connect(str(self.receipts_path), timeout=30.0)) as conn:
            row = conn.execute(
                "SELECT payload FROM consolidation_writes WHERE source_ref=? AND kind=?",
                (source_ref, kind),
            ).fetchone()
        if row is None:
            return None
        raw = row[0]
        if not isinstance(raw, str):
            raise ValueError(f"Markdown profile receipt payload 缺失: {source_ref}/{kind}")
        value = cast(Any, json.loads(raw))
        if not isinstance(value, dict):
            raise ValueError(f"Markdown profile receipt schema 无效: {source_ref}/{kind}")
        return {str(key): item for key, item in cast(dict[object, object], value).items()}

    def _write_once(
        self,
        source_ref: str,
        kind: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with closing(sqlite3.connect(str(self.receipts_path), timeout=30.0)) as conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT payload FROM consolidation_writes WHERE source_ref=? AND kind=?",
                (source_ref, kind),
            ).fetchone()
            if row is not None:
                if row[0] != encoded:
                    raise ValueError(f"Markdown profile receipt 内容冲突: {source_ref}/{kind}")
                conn.commit()
                return dict(payload)
            _ = conn.execute(
                "INSERT INTO consolidation_writes "
                "(source_ref, kind, payload, trailing_blank_line, done_at) "
                "VALUES (?, ?, ?, 0, datetime('now'))",
                (source_ref, kind, encoded),
            )
            conn.commit()
        return dict(payload)


def _required_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Markdown profile draft 缺少字符串字段: {field}")
    return value
