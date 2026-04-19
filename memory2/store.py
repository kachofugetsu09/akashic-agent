"""
Memory v2 SQLite 存储层
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import struct
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

try:
    import sqlite_vec

    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False

logger = logging.getLogger(__name__)

VEC_DIM = 1024  # 默认维度，MemoryStore2 构造时可覆盖

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id            TEXT PRIMARY KEY,
    memory_type   TEXT NOT NULL,
    summary       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    embedding     TEXT,
    reinforcement INTEGER NOT NULL DEFAULT 1,
    emotional_weight INTEGER NOT NULL DEFAULT 0,
    extra_json    TEXT,
    source_ref    TEXT,
    happened_at   TEXT,
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_items_hash
    ON memory_items (content_hash, memory_type);
CREATE TABLE IF NOT EXISTS consolidation_events (
    source_ref  TEXT PRIMARY KEY,
    item_id     TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_replacements (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    old_item_id       TEXT NOT NULL,
    old_memory_type   TEXT NOT NULL,
    old_summary       TEXT NOT NULL,
    old_source_ref    TEXT,
    old_happened_at   TEXT,
    old_extra_json    TEXT,
    new_item_id       TEXT NOT NULL,
    new_memory_type   TEXT NOT NULL,
    new_summary       TEXT NOT NULL,
    new_source_ref    TEXT,
    new_happened_at   TEXT,
    new_extra_json    TEXT,
    relation_type     TEXT NOT NULL DEFAULT 'supersede',
    source_ref        TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_memory_replacements_old_item
    ON memory_replacements (old_item_id, created_at);
CREATE INDEX IF NOT EXISTS ix_memory_replacements_new_item
    ON memory_replacements (new_item_id, created_at);
"""

# VEC_SCHEMA 在 MemoryStore2.__init__ 中按 vec_dim 动态生成


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(summary: str, memory_type: str) -> str:
    text = re.sub(r"\s+", " ", summary.lower().strip()) + memory_type
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _coerce_emotional_weight(value: object) -> int:
    try:
        return max(0, min(10, int(value or 0)))
    except (TypeError, ValueError):
        return 0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    a_norm = float(np.linalg.norm(va)) + 1e-9
    b_norm = float(np.linalg.norm(vb)) + 1e-9
    return float(va @ vb) / a_norm / b_norm


def _hotness_score(
    reinforcement: int,
    updated_at: datetime,
    now: datetime | None = None,
    half_life_days: float = 14.0,
    emotional_weight: int = 0,
) -> float:
    """计算热度分：频度 * 时间衰减，结果在 (0, 1) 区间。"""
    if now is None:
        now = datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    effective_half_life = max(
        half_life_days * (1.0 + 0.5 * _coerce_emotional_weight(emotional_weight) / 10.0),
        0.1,
    )
    freq    = 1.0 / (1.0 + math.exp(-math.log1p(max(0, reinforcement))))
    age_d   = max((now - updated_at).total_seconds() / 86400.0, 0.0)
    recency = math.exp(-math.log(2) / effective_half_life * age_d)
    return freq * recency


def _normalize_emb(emb: list[float]) -> list[float]:
    """L2 归一化，供 vec_items 存储用（L2 KNN on unit vectors ≡ cosine ranking）。"""
    v = np.array(emb, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return emb
    return (v / n).tolist()


def _emb_to_blob(emb: list[float]) -> bytes:
    """将归一化后的 embedding 打包为 float32 blob。"""
    normed = _normalize_emb(emb)
    return struct.pack(f"{len(normed)}f", *normed)


def _l2dist_to_cosine(distance: float) -> float:
    """将单位球上的 L2 距离转换回 cosine similarity。
    |a-b|² = 2(1 - cos) → cos = 1 - d²/2
    """
    return 1.0 - (distance * distance) / 2.0


class MemoryStore2:
    def __init__(self, db_path: str | Path, vec_dim: int = VEC_DIM) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self._closed = False
        self._db.executescript(SCHEMA)
        self._db.commit()

        cols = {r[1] for r in self._db.execute("PRAGMA table_info(memory_items)")}
        if "status" not in cols:
            self._db.execute(
                "ALTER TABLE memory_items ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
            )
            self._db.commit()
        if "emotional_weight" not in cols:
            self._db.execute(
                "ALTER TABLE memory_items ADD COLUMN emotional_weight INTEGER NOT NULL DEFAULT 0"
            )
            self._db.commit()
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_items_status ON memory_items (status)"
        )
        self._db.commit()

        # --- sqlite-vec 初始化 ---
        self._vec_dim = vec_dim
        self._vec_enabled = False
        self._vec_init_error: str | None = None
        self._vec_fallback_logged = False
        if _SQLITE_VEC_AVAILABLE:
            try:
                self._db.enable_load_extension(True)
                sqlite_vec.load(self._db)
                self._db.enable_load_extension(False)
                vec_schema = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
    embedding float[{self._vec_dim}]
);
"""
                self._db.executescript(vec_schema)
                self._db.commit()
                self._vec_enabled = True
                self._migrate_existing_to_vec()
                logger.info("sqlite-vec 已启用（dim=%d）", self._vec_dim)
            except Exception as exc:
                self._vec_init_error = str(exc)
                logger.warning("sqlite-vec 初始化失败（%s），回退到全表扫描", exc)
        else:
            self._vec_init_error = "sqlite_vec 未安装"
            logger.debug("sqlite-vec 未安装，使用全表扫描")

    # ------------------------------------------------------------------
    # vec_items 内部辅助
    # ------------------------------------------------------------------

    def _migrate_existing_to_vec(self) -> None:
        """启动时将 memory_items 中尚未同步到 vec_items 的 embedding 迁移过去。"""
        existing = {r[0] for r in self._db.execute("SELECT rowid FROM vec_items").fetchall()}
        rows = self._db.execute(
            "SELECT rowid, embedding FROM memory_items WHERE embedding IS NOT NULL"
        ).fetchall()
        migrated = 0
        for rowid, emb_json in rows:
            if rowid in existing:
                continue
            try:
                emb = json.loads(emb_json)
                if len(emb) != self._vec_dim:
                    continue
                self._db.execute(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    (rowid, _emb_to_blob(emb)),
                )
                migrated += 1
            except Exception as exc:
                logger.debug("vec migrate skip rowid %s: %s", rowid, exc)
        if migrated:
            self._db.commit()
            logger.info("sqlite-vec: 迁移了 %d 条历史 embedding", migrated)

    def _vec_insert(self, rowid: int, emb: list[float]) -> None:
        """向 vec_items 插入一条向量（幂等：先删再插）。维度不匹配时静默跳过。"""
        if not self._vec_enabled or len(emb) != self._vec_dim:
            return
        try:
            self._db.execute("DELETE FROM vec_items WHERE rowid=?", (rowid,))
            self._db.execute(
                "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                (rowid, _emb_to_blob(emb)),
            )
        except Exception as exc:
            logger.warning("vec_insert rowid=%s 失败: %s", rowid, exc)

    def _vec_delete(self, rowids: list[int]) -> None:
        """从 vec_items 批量删除。"""
        if not self._vec_enabled or not rowids:
            return
        try:
            self._db.executemany(
                "DELETE FROM vec_items WHERE rowid=?", [(r,) for r in rowids]
            )
        except Exception as exc:
            logger.warning("vec_delete 失败: %s", exc)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._db.close()
        finally:
            self._closed = True

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------

    def upsert_item(
        self,
        memory_type: str,
        summary: str,
        embedding: list[float] | None,
        source_ref: str | None = None,
        extra: dict[str, object] | None = None,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        """写入或强化一条记忆。返回 'new:id' 或 'reinforced:id'"""
        chash = _content_hash(summary, memory_type)
        emotional_weight = _coerce_emotional_weight(emotional_weight)
        existing = self._db.execute(
            "SELECT id, status FROM memory_items WHERE content_hash=? AND memory_type=?",
            (chash, memory_type),
        ).fetchone()
        if existing:
            row_id, status = existing
            if status == "superseded":
                self._db.execute(
                    "UPDATE memory_items SET status='active', reinforcement=reinforcement+1, updated_at=?, emotional_weight=MAX(emotional_weight, ?) WHERE id=?",
                    (_now_iso(), emotional_weight, row_id),
                )
            else:
                self._db.execute(
                    "UPDATE memory_items SET reinforcement=reinforcement+1, updated_at=?, emotional_weight=MAX(emotional_weight, ?) WHERE id=?",
                    (_now_iso(), emotional_weight, row_id),
                )
            self._db.commit()
            return f"reinforced:{row_id}"

        item_id = hashlib.md5(f"{chash}{time.time()}".encode()).hexdigest()[:12]
        cur = self._db.execute(
            """INSERT INTO memory_items
               (id, memory_type, summary, content_hash, embedding, emotional_weight,
                extra_json, source_ref, happened_at, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item_id,
                memory_type,
                summary,
                chash,
                json.dumps(embedding) if embedding is not None else None,
                emotional_weight,
                json.dumps(extra) if extra else None,
                source_ref,
                happened_at,
                _now_iso(),
                _now_iso(),
            ),
        )
        item_rowid = cur.lastrowid
        self._db.commit()

        if embedding is not None and item_rowid is not None:
            self._vec_insert(item_rowid, embedding)
            self._db.commit()

        return f"new:{item_id}"

    def upsert_consolidation_event(
        self,
        *,
        source_ref: str,
        summary: str,
        embedding: list[float] | None,
        extra: dict[str, object] | None = None,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        """原子写入 consolidation event：同一 source_ref 最多写一次。"""
        src = (source_ref or "").strip()
        text = (summary or "").strip()
        if not src or not text:
            return "skipped:empty"
        emotional_weight = _coerce_emotional_weight(emotional_weight)

        self._db.execute("BEGIN IMMEDIATE")
        new_item_rowid: int | None = None
        new_item_emb: list[float] | None = None
        try:
            already = self._db.execute(
                "SELECT item_id FROM consolidation_events WHERE source_ref=?",
                (src,),
            ).fetchone()
            if already is not None:
                self._db.execute("COMMIT")
                existing_id = already[0] or ""
                return f"skipped:{existing_id or src}"

            chash = _content_hash(text, "event")
            existing = self._db.execute(
                "SELECT id, status FROM memory_items WHERE content_hash=? AND memory_type=?",
                (chash, "event"),
            ).fetchone()

            if existing:
                row_id, status = existing
                if status == "superseded":
                    self._db.execute(
                        "UPDATE memory_items SET status='active', reinforcement=reinforcement+1, updated_at=?, emotional_weight=MAX(emotional_weight, ?) WHERE id=?",
                        (_now_iso(), emotional_weight, row_id),
                    )
                else:
                    self._db.execute(
                        "UPDATE memory_items SET reinforcement=reinforcement+1, updated_at=?, emotional_weight=MAX(emotional_weight, ?) WHERE id=?",
                        (_now_iso(), emotional_weight, row_id),
                    )
                item_id = row_id
                result = f"reinforced:{row_id}"
            else:
                item_id = hashlib.md5(f"{chash}{time.time()}".encode()).hexdigest()[:12]
                cur = self._db.execute(
                    """INSERT INTO memory_items
                       (id, memory_type, summary, content_hash, embedding, emotional_weight,
                        extra_json, source_ref, happened_at, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        item_id,
                        "event",
                        text,
                        chash,
                        json.dumps(embedding) if embedding is not None else None,
                        emotional_weight,
                        json.dumps(extra) if extra else None,
                        src,
                        happened_at,
                        _now_iso(),
                        _now_iso(),
                    ),
                )
                new_item_rowid = cur.lastrowid
                new_item_emb = embedding
                result = f"new:{item_id}"

            self._db.execute(
                "INSERT INTO consolidation_events(source_ref, item_id, created_at) VALUES (?, ?, ?)",
                (src, item_id, _now_iso()),
            )
            self._db.execute("COMMIT")

            if new_item_rowid is not None and new_item_emb is not None:
                self._vec_insert(new_item_rowid, new_item_emb)
                self._db.commit()

            return result
        except Exception:
            try:
                self._db.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def has_consolidation_source_ref(self, source_ref: str) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM consolidation_events WHERE source_ref=? LIMIT 1",
            ((source_ref or "").strip(),),
        ).fetchone()
        return row is not None

    def mark_superseded(self, item_id: str) -> None:
        """将指定条目标记为已退休。"""
        self._db.execute(
            "UPDATE memory_items SET status='superseded', updated_at=? WHERE id=?",
            (_now_iso(), item_id),
        )
        self._db.commit()

    def mark_superseded_batch(self, ids: list[str]) -> None:
        if not ids:
            return
        now = _now_iso()
        self._db.executemany(
            "UPDATE memory_items SET status='superseded', updated_at=? WHERE id=?",
            [(now, item_id) for item_id in ids],
        )
        self._db.commit()

    def get_items_by_ids(self, ids: list[str]) -> list[dict[str, object]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._db.execute(
            "SELECT id, memory_type, summary, extra_json, source_ref, happened_at, "
            "status, created_at, updated_at, emotional_weight "
            f"FROM memory_items WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id: dict[str, dict[str, object]] = {}
        for (
            row_id,
            memory_type,
            summary,
            extra_json,
            source_ref,
            happened_at,
            status,
            created_at,
            updated_at,
            emotional_weight,
        ) in rows:
            by_id[str(row_id)] = {
                "id": row_id,
                "memory_type": memory_type,
                "summary": summary,
                "extra_json": json.loads(extra_json) if extra_json else {},
                "source_ref": source_ref,
                "happened_at": happened_at,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at,
                "emotional_weight": emotional_weight,
            }
        return [by_id[item_id] for item_id in ids if item_id in by_id]

    def record_replacements(
        self,
        *,
        old_items: list[dict[str, object]],
        new_item: dict[str, object],
        source_ref: str | None = None,
        relation_type: str = "supersede",
    ) -> int:
        if not old_items or not new_item or not new_item.get("id"):
            return 0
        now = _now_iso()
        rows = []
        for old_item in old_items:
            if not old_item or not old_item.get("id"):
                continue
            rows.append(
                (
                    str(old_item.get("id")),
                    str(old_item.get("memory_type") or ""),
                    str(old_item.get("summary") or ""),
                    old_item.get("source_ref"),
                    old_item.get("happened_at"),
                    json.dumps(old_item.get("extra_json") or {}, ensure_ascii=False),
                    str(new_item.get("id")),
                    str(new_item.get("memory_type") or ""),
                    str(new_item.get("summary") or ""),
                    new_item.get("source_ref"),
                    new_item.get("happened_at"),
                    json.dumps(new_item.get("extra_json") or {}, ensure_ascii=False),
                    relation_type,
                    source_ref or new_item.get("source_ref"),
                    now,
                )
            )
        if not rows:
            return 0
        self._db.executemany(
            """INSERT INTO memory_replacements
               (old_item_id, old_memory_type, old_summary, old_source_ref, old_happened_at,
                old_extra_json, new_item_id, new_memory_type, new_summary, new_source_ref,
                new_happened_at, new_extra_json, relation_type, source_ref, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self._db.commit()
        return len(rows)

    def list_replacements(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT old_item_id, old_memory_type, old_summary, old_source_ref, "
            "old_happened_at, old_extra_json, new_item_id, new_memory_type, "
            "new_summary, new_source_ref, new_happened_at, new_extra_json, "
            "relation_type, source_ref, created_at "
            "FROM memory_replacements ORDER BY id ASC"
        ).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "old_item_id": row[0],
                    "old_memory_type": row[1],
                    "old_summary": row[2],
                    "old_source_ref": row[3],
                    "old_happened_at": row[4],
                    "old_extra_json": json.loads(row[5]) if row[5] else {},
                    "new_item_id": row[6],
                    "new_memory_type": row[7],
                    "new_summary": row[8],
                    "new_source_ref": row[9],
                    "new_happened_at": row[10],
                    "new_extra_json": json.loads(row[11]) if row[11] else {},
                    "relation_type": row[12],
                    "source_ref": row[13],
                    "created_at": row[14],
                }
            )
        return result

    def reinforce_items_batch(self, ids: list[str], emotional_weight: int = 0) -> None:
        if not ids:
            return
        now = _now_iso()
        emotional_weight = _coerce_emotional_weight(emotional_weight)
        self._db.executemany(
            "UPDATE memory_items SET reinforcement=reinforcement+1, updated_at=?, emotional_weight=MAX(emotional_weight, ?) WHERE id=?",
            [(now, emotional_weight, item_id) for item_id in ids],
        )
        self._db.commit()

    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------

    def list_items_for_dashboard(
        self,
        *,
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, object]], int]:
        with self._lock:
            safe_sort_by = sort_by if sort_by in {
                "updated_at",
                "created_at",
                "happened_at",
                "reinforcement",
                "emotional_weight",
                "memory_type",
            } else "updated_at"
            safe_sort_order = "asc" if sort_order == "asc" else "desc"
            safe_page = max(1, page)
            safe_page_size = max(1, min(page_size, 200))
            offset = (safe_page - 1) * safe_page_size

            where_parts = ["1=1"]
            params: list[object] = []

            if q:
                where_parts.append("(id LIKE ? OR summary LIKE ? OR COALESCE(source_ref, '') LIKE ?)")
                like = f"%{q}%"
                params.extend([like, like, like])
            if memory_type:
                where_parts.append("memory_type = ?")
                params.append(memory_type)
            if status:
                where_parts.append("status = ?")
                params.append(status)
            if source_ref:
                where_parts.append("COALESCE(source_ref, '') LIKE ?")
                params.append(f"%{source_ref}%")
            if scope_channel:
                where_parts.append(
                    "COALESCE(TRIM(json_extract(extra_json, '$.scope_channel')), '') = ?"
                )
                params.append(scope_channel.strip())
            if scope_chat_id:
                where_parts.append(
                    "COALESCE(TRIM(json_extract(extra_json, '$.scope_chat_id')), '') = ?"
                )
                params.append(scope_chat_id.strip())
            if has_embedding is True:
                where_parts.append("embedding IS NOT NULL")
            elif has_embedding is False:
                where_parts.append("embedding IS NULL")

            where_sql = " AND ".join(where_parts)
            total = int(
                self._db.execute(
                    f"SELECT COUNT(*) FROM memory_items WHERE {where_sql}",
                    tuple(params),
                ).fetchone()[0]
            )
            rows = self._db.execute(
                f"""
                SELECT id, memory_type, summary, source_ref, happened_at, status,
                       created_at, updated_at, reinforcement, emotional_weight,
                       extra_json, embedding IS NOT NULL
                FROM memory_items
                WHERE {where_sql}
                ORDER BY {safe_sort_by} {safe_sort_order}, id ASC
                LIMIT ? OFFSET ?
                """,
                tuple([*params, safe_page_size, offset]),
            ).fetchall()
            items: list[dict[str, object]] = []
            for row in rows:
                (
                    row_id,
                    row_memory_type,
                    summary,
                    row_source_ref,
                    happened_at,
                    row_status,
                    created_at,
                    updated_at,
                    reinforcement,
                    emotional_weight,
                    extra_json,
                    row_has_embedding,
                ) = row
                extra = json.loads(extra_json) if extra_json else {}
                items.append(
                    {
                        "id": row_id,
                        "memory_type": row_memory_type,
                        "summary": summary,
                        "source_ref": row_source_ref,
                        "happened_at": happened_at,
                        "status": row_status,
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "reinforcement": reinforcement,
                        "emotional_weight": emotional_weight,
                        "has_embedding": bool(row_has_embedding),
                        "scope_channel": extra.get("scope_channel", ""),
                        "scope_chat_id": extra.get("scope_chat_id", ""),
                    }
                )
            return items, total

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, memory_type, summary, content_hash, embedding, reinforcement, "
                "emotional_weight, extra_json, source_ref, happened_at, status, created_at, updated_at "
                "FROM memory_items WHERE id=?",
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        (
            row_id,
            memory_type,
            summary,
            content_hash,
            embedding_json,
            reinforcement,
            emotional_weight,
            extra_json,
            source_ref,
            happened_at,
            status,
            created_at,
            updated_at,
        ) = row
        embedding = json.loads(embedding_json) if embedding_json else None
        return {
            "id": row_id,
            "memory_type": memory_type,
            "summary": summary,
            "content_hash": content_hash,
            "reinforcement": reinforcement,
            "emotional_weight": emotional_weight,
            "extra_json": json.loads(extra_json) if extra_json else {},
            "source_ref": source_ref,
            "happened_at": happened_at,
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
            "has_embedding": embedding is not None,
            "embedding_dim": len(embedding) if embedding is not None else 0,
            "embedding": embedding if include_embedding else None,
        }

    def update_item_for_dashboard(
        self,
        item_id: str,
        *,
        status: str | None = None,
        extra_json: dict[str, object] | None = None,
        source_ref: str | None = None,
        happened_at: str | None = None,
        emotional_weight: int | None = None,
    ) -> dict[str, object] | None:
        with self._lock:
            updates: list[str] = []
            params: list[object] = []

            if status is not None:
                safe_status = status.strip()
                if safe_status not in {"active", "superseded"}:
                    raise ValueError("status 仅支持 active 或 superseded")
                updates.append("status=?")
                params.append(safe_status)
            if extra_json is not None:
                updates.append("extra_json=?")
                params.append(json.dumps(extra_json, ensure_ascii=False))
            if source_ref is not None:
                updates.append("source_ref=?")
                params.append(source_ref)
            if happened_at is not None:
                updates.append("happened_at=?")
                params.append(happened_at)
            if emotional_weight is not None:
                updates.append("emotional_weight=?")
                params.append(_coerce_emotional_weight(emotional_weight))
            if not updates:
                return self.get_item_for_dashboard(item_id)

            updates.append("updated_at=?")
            params.append(_now_iso())
            params.append(item_id)
            cur = self._db.execute(
                f"UPDATE memory_items SET {', '.join(updates)} WHERE id=?",
                params,
            )
            self._db.commit()
            if cur.rowcount <= 0:
                return None
        return self.get_item_for_dashboard(item_id)

    def delete_item(self, item_id: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT rowid FROM memory_items WHERE id=?",
                (item_id,),
            ).fetchone()
            if row is None:
                return False
            cur = self._db.execute(
                "DELETE FROM memory_items WHERE id=?",
                (item_id,),
            )
            self._vec_delete([row[0]])
            self._db.commit()
            return cur.rowcount > 0

    def delete_items_batch(self, ids: list[str]) -> int:
        if not ids:
            return 0
        with self._lock:
            placeholders = ",".join("?" for _ in ids)
            rowids = [
                r[0]
                for r in self._db.execute(
                    f"SELECT rowid FROM memory_items WHERE id IN ({placeholders})",
                    ids,
                ).fetchall()
            ]
            cur = self._db.execute(
                f"DELETE FROM memory_items WHERE id IN ({placeholders})",
                ids,
            )
            self._vec_delete(rowids)
            self._db.commit()
            return int(cur.rowcount or 0)

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        base = self.get_item_for_dashboard(item_id, include_embedding=True)
        if base is None:
            raise KeyError(item_id)
        embedding = base.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("memory 没有 embedding")

        results = self.vector_search(
            query_vec=embedding,
            top_k=max(1, top_k) + 1,
            memory_types=[memory_type] if memory_type else None,
            score_threshold=score_threshold,
            include_superseded=include_superseded,
        )
        filtered = [item for item in results if item.get("id") != item_id]
        return filtered[: max(1, top_k)]

    def get_all_with_embedding(self, include_superseded: bool = False) -> list[tuple]:
        """返回 [(id, memory_type, summary, embedding_list, extra_json_dict, happened_at, source_ref)]
        extra_json_dict 中注入 _reinforcement / _updated_at / _emotional_weight
        （_ 前缀，不污染用户字段）。
        """
        where = "" if include_superseded else "AND status='active'"
        rows = self._db.execute(
            "SELECT id, memory_type, summary, embedding, extra_json, happened_at, "
            "reinforcement, updated_at, source_ref, emotional_weight "
            f"FROM memory_items WHERE embedding IS NOT NULL {where}"
        ).fetchall()
        result = []
        for row_id, mtype, summary, emb_json, extra_json, happened_at, reinforcement, updated_at, source_ref, emotional_weight in rows:
            emb = json.loads(emb_json) if emb_json else None
            extra = json.loads(extra_json) if extra_json else {}
            extra["_reinforcement"] = reinforcement
            extra["_updated_at"] = updated_at
            extra["_emotional_weight"] = emotional_weight
            result.append((row_id, mtype, summary, emb, extra, happened_at, source_ref))
        return result

    def vector_search(
        self,
        query_vec: list[float],
        top_k: int = 8,
        memory_types: list[str] | None = None,
        score_threshold: float = 0.0,
        include_superseded: bool = False,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
        hotness_alpha: float = 0.0,
        hotness_half_life_days: float = 14.0,
    ) -> list[dict[str, object]]:
        """cosine similarity 检索，返回 top-k 结果。
        hotness_alpha > 0 时启用热度融合：final = (1-alpha)*semantic + alpha*hotness。
        """
        if self._vec_enabled:
            return self._vector_search_vec(
                query_vec,
                top_k=top_k,
                memory_types=memory_types,
                score_threshold=score_threshold,
                include_superseded=include_superseded,
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
                require_scope_match=require_scope_match,
                hotness_alpha=hotness_alpha,
                hotness_half_life_days=hotness_half_life_days,
            )
        if not self._vec_fallback_logged:
            reason = self._vec_init_error or "sqlite-vec 未启用"
            logger.warning("vector_search 已降级为全表扫描：%s", reason)
            self._vec_fallback_logged = True
        return self._vector_search_fullscan(
            query_vec,
            top_k=top_k,
            memory_types=memory_types,
            score_threshold=score_threshold,
            include_superseded=include_superseded,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            require_scope_match=require_scope_match,
            hotness_alpha=hotness_alpha,
            hotness_half_life_days=hotness_half_life_days,
        )

    def _vector_search_vec(
        self,
        query_vec: list[float],
        top_k: int = 8,
        memory_types: list[str] | None = None,
        score_threshold: float = 0.0,
        include_superseded: bool = False,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
        hotness_alpha: float = 0.0,
        hotness_half_life_days: float = 14.0,
    ) -> list[dict]:
        """sqlite-vec KNN 检索路径。维度不符时自动回退全表扫描。"""
        if len(query_vec) != self._vec_dim:
            logger.debug(
                "query dim %d ≠ vec_dim %d，回退全表扫描", len(query_vec), self._vec_dim
            )
            return self._vector_search_fullscan(
                query_vec,
                top_k=top_k,
                memory_types=memory_types,
                score_threshold=score_threshold,
                include_superseded=include_superseded,
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
                require_scope_match=require_scope_match,
                hotness_alpha=hotness_alpha,
                hotness_half_life_days=hotness_half_life_days,
            )
        blob = _emb_to_blob(query_vec)

        # KNN 多取一些候选，以补偿 score_threshold 截断的损耗
        fetch_k = max(top_k * 2, 20)

        params: list = [blob, fetch_k]

        status_filter = "" if include_superseded else "AND m.status = 'active'"

        # memory_type 推入 SQL 过滤，避免 Python 二次扫描
        if memory_types:
            placeholders = ",".join("?" * len(memory_types))
            type_filter = f"AND m.memory_type IN ({placeholders})"
            params.extend(memory_types)
        else:
            type_filter = ""

        # scope 推入 SQL，用 json_extract 读取 extra_json 字段
        if require_scope_match:
            s_channel = (scope_channel or "").strip()
            s_chat = (scope_chat_id or "").strip()
            scope_filter = (
                "AND COALESCE(TRIM(json_extract(m.extra_json, '$.scope_channel')), '') = ?"
                " AND COALESCE(TRIM(json_extract(m.extra_json, '$.scope_chat_id')), '') = ?"
            )
            params.extend([s_channel, s_chat])
        else:
            scope_filter = ""

        sql = f"""
            SELECT m.id, m.memory_type, m.summary, m.extra_json, m.happened_at,
                   m.reinforcement, m.updated_at, m.source_ref, m.emotional_weight,
                   v.distance
            FROM (
                SELECT rowid, distance
                FROM vec_items
                WHERE embedding MATCH ?
                  AND k = ?
            ) v
            JOIN memory_items m ON m.rowid = v.rowid
            WHERE 1=1 {status_filter} {type_filter} {scope_filter}
            ORDER BY v.distance ASC
        """
        rows = self._db.execute(sql, params).fetchall()

        now = datetime.now(timezone.utc)
        scored = []
        for row_id, mtype, summary, extra_json, happened_at, reinforcement, updated_at_str, source_ref, emotional_weight, distance in rows:
            # L2 distance on unit sphere → cosine similarity
            similarity = _l2dist_to_cosine(distance)
            if similarity < score_threshold:
                continue

            extra = json.loads(extra_json) if extra_json else {}
            extra["_reinforcement"] = reinforcement
            extra["_updated_at"] = updated_at_str
            extra["_emotional_weight"] = emotional_weight

            hotness = 0.0
            if hotness_alpha > 0 and updated_at_str:
                try:
                    updated_at = datetime.fromisoformat(updated_at_str)
                    hotness = _hotness_score(
                        reinforcement,
                        updated_at,
                        now,
                        hotness_half_life_days,
                        emotional_weight=emotional_weight,
                    )
                except (ValueError, TypeError):
                    pass

            final = (1.0 - hotness_alpha) * similarity + hotness_alpha * hotness
            scored.append(
                {
                    "id": row_id,
                    "memory_type": mtype,
                    "summary": summary,
                    "extra_json": extra,
                    "happened_at": happened_at,
                    "source_ref": source_ref,
                    "score": round(final, 4),
                    "_score_debug": {
                        "semantic": round(similarity, 4),
                        "hotness": round(hotness, 4),
                        "final": round(final, 4),
                    },
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _vector_search_fullscan(
        self,
        query_vec: list[float],
        top_k: int = 8,
        memory_types: list[str] | None = None,
        score_threshold: float = 0.0,
        include_superseded: bool = False,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
        hotness_alpha: float = 0.0,
        hotness_half_life_days: float = 14.0,
    ) -> list[dict]:
        """全表扫描回退路径（sqlite-vec 不可用时使用）。"""
        rows = self.get_all_with_embedding(include_superseded=include_superseded)
        if not rows:
            return []

        if memory_types:
            rows = [r for r in rows if r[1] in memory_types]

        if require_scope_match:
            s_channel = (scope_channel or "").strip()
            s_chat = (scope_chat_id or "").strip()
            rows = [
                r
                for r in rows
                if str((r[4] or {}).get("scope_channel", "")).strip() == s_channel
                and str((r[4] or {}).get("scope_chat_id", "")).strip() == s_chat
            ]

        if not rows:
            return []

        q = np.array(query_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q)) + 1e-9
        now = datetime.now(timezone.utc)

        scored = []
        for row_id, mtype, summary, emb, extra, happened_at, source_ref in rows:
            if emb is None:
                continue
            e = np.array(emb, dtype=np.float32)
            semantic = float(e @ q) / (float(np.linalg.norm(e)) + 1e-9) / q_norm
            if semantic < score_threshold:
                continue

            hotness = 0.0
            if hotness_alpha > 0:
                reinforcement = extra.get("_reinforcement", 1)
                updated_at_str = extra.get("_updated_at")
                emotional_weight = extra.get("_emotional_weight", 0)
                if updated_at_str:
                    try:
                        updated_at = datetime.fromisoformat(updated_at_str)
                        hotness = _hotness_score(
                            reinforcement,
                            updated_at,
                            now,
                            hotness_half_life_days,
                            emotional_weight=emotional_weight,
                        )
                    except (ValueError, TypeError):
                        pass

            final = (1.0 - hotness_alpha) * semantic + hotness_alpha * hotness

            scored.append(
                {
                    "id": row_id,
                    "memory_type": mtype,
                    "summary": summary,
                    "extra_json": extra,
                    "happened_at": happened_at,
                    "source_ref": source_ref,
                    "score": round(final, 4),
                    "_score_debug": {
                        "semantic": round(semantic, 4),
                        "hotness": round(hotness, 4),
                        "final": round(final, 4),
                    },
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def merge_item_raw(
        self,
        item_id: str,
        new_summary: str,
        new_hash: str,
        new_embedding: list[float],
        new_extra: dict[str, object] | None = None,
    ) -> None:
        """原子更新 merge 目标：summary + content_hash + embedding + reinforcement。
        new_extra 若提供则同步更新 extra_json。
        若 content_hash 冲突（极低概率），则 supersede 旧条目并由 upsert_item 写入新摘要。
        """
        try:
            if new_extra is not None:
                self._db.execute(
                    """UPDATE memory_items
                       SET summary=?, content_hash=?, embedding=?, extra_json=?,
                           reinforcement=reinforcement+1, updated_at=?
                       WHERE id=?""",
                    (
                        new_summary, new_hash, json.dumps(new_embedding),
                        json.dumps(new_extra), _now_iso(), item_id,
                    ),
                )
            else:
                self._db.execute(
                    """UPDATE memory_items
                       SET summary=?, content_hash=?, embedding=?,
                           reinforcement=reinforcement+1, updated_at=?
                       WHERE id=?""",
                    (new_summary, new_hash, json.dumps(new_embedding), _now_iso(), item_id),
                )
            self._db.commit()

            # 同步更新 vec_items（embedding 变了）
            if self._vec_enabled:
                row = self._db.execute(
                    "SELECT rowid FROM memory_items WHERE id=?", (item_id,)
                ).fetchone()
                if row:
                    self._vec_insert(row[0], new_embedding)
                    self._db.commit()

        except sqlite3.IntegrityError:
            # content_hash 撞上库中已有条目（极低概率）
            # 安全降级：supersede 旧条目，让 upsert_item 走 reinforce 路径
            logger.warning(
                "merge_item_raw: content_hash collision for item %s, "
                "superseding and falling back to upsert",
                item_id,
            )
            try:
                self._db.execute("ROLLBACK")
            except Exception:
                pass
            row = self._db.execute(
                "SELECT memory_type FROM memory_items WHERE id=?", (item_id,)
            ).fetchone()
            if row:
                self.mark_superseded(item_id)
                self.upsert_item(
                    memory_type=row[0],
                    summary=new_summary,
                    embedding=new_embedding,
                )

    def list_by_type(self, memory_type: str) -> list[dict[str, object]]:
        rows = self._db.execute(
            "SELECT id, memory_type, summary, extra_json, happened_at, reinforcement, emotional_weight "
            "FROM memory_items WHERE memory_type=?",
            (memory_type,),
        ).fetchall()
        result = []
        for row_id, mtype, summary, extra_json, happened_at, reinforcement, emotional_weight in rows:
            result.append(
                {
                    "id": row_id,
                    "memory_type": mtype,
                    "summary": summary,
                    "extra_json": json.loads(extra_json) if extra_json else {},
                    "happened_at": happened_at,
                    "reinforcement": reinforcement,
                    "emotional_weight": emotional_weight,
                }
            )
        return result

    def find_similar_recent_events(
        self,
        embedding: list[float],
        *,
        days_back: int = 7,
        threshold: float = 0.92,
        top_k: int = 3,
    ) -> list[str]:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max(1, int(days_back)))
        ).isoformat()
        rows = self._db.execute(
            "SELECT id, embedding FROM memory_items "
            "WHERE memory_type='event' AND status='active' "
            "AND embedding IS NOT NULL AND created_at >= ?",
            (cutoff,),
        ).fetchall()
        scored: list[tuple[str, float]] = []
        for row_id, emb_json in rows:
            if not emb_json:
                continue
            score = _cosine_similarity(embedding, json.loads(emb_json))
            if score >= float(threshold):
                scored.append((row_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [row_id for row_id, _score in scored[: max(1, int(top_k))]]

    def delete_by_source_ref(self, source_ref: str) -> int:
        """删除指定 source_ref 的所有条目，返回删除行数。"""
        rowids = [
            r[0]
            for r in self._db.execute(
                "SELECT rowid FROM memory_items WHERE source_ref=?", (source_ref,)
            ).fetchall()
        ]
        cur = self._db.execute(
            "DELETE FROM memory_items WHERE source_ref=?", (source_ref,)
        )
        self._vec_delete(rowids)
        self._db.commit()
        return cur.rowcount

    def has_item_by_source_ref(
        self,
        source_ref: str,
        memory_type: str | None = None,
    ) -> bool:
        """检查是否已存在指定 source_ref 的条目。"""
        if memory_type:
            row = self._db.execute(
                "SELECT 1 FROM memory_items WHERE source_ref=? AND memory_type=? LIMIT 1",
                (source_ref, memory_type),
            ).fetchone()
        else:
            row = self._db.execute(
                "SELECT 1 FROM memory_items WHERE source_ref=? LIMIT 1",
                (source_ref,),
            ).fetchone()
        return row is not None

    def keyword_match_procedures(self, action_tokens: list[str]) -> list[dict[str, object]]:
        """对 trigger_tags 做纯关键字匹配，无需向量检索。

        action_tokens 是从工具调用中提取的 token 列表，例如：
          ["shell", "pacman"]  / ["web_search"] / ["read_file", "yt-dlp-downloader"]

        只返回 scope=tool_triggered 且命中的 procedure 条目。
        """
        if not action_tokens:
            return []

        token_set = {t.lower() for t in action_tokens if t}
        action_text = " ".join(action_tokens).lower()

        rows = self._db.execute(
            "SELECT id, summary, extra_json FROM memory_items "
            "WHERE memory_type='procedure' AND status='active' AND extra_json IS NOT NULL"
        ).fetchall()

        matched: list[dict] = []
        for row_id, summary, extra_json_str in rows:
            try:
                extra = json.loads(extra_json_str) if extra_json_str else {}
            except Exception:
                continue
            tags = extra.get("trigger_tags") or {}
            if tags.get("scope") != "tool_triggered":
                continue

            # 过滤掉太短的 keyword（长度 < 3），避免 "i"、"-c" 之类造成误匹配
            keywords = [k for k in (tags.get("keywords") or []) if k and len(k) >= 3]

            if keywords:
                # 有 keyword 时：必须命中至少一个 keyword 才算匹配
                # keyword 是精确区分上下文的标志（如 "pacman"、"bilibili"），
                # 仅靠 tool name 不足以触发（避免 shell/read_file 过度泛化）
                hit = any(kw.lower() in action_text for kw in keywords)
            else:
                # 无 keyword：tool/skill 名精确匹配
                # tools 超过 4 个说明是泛规范（LLM 把全量工具都填进去了），降级为 global 跳过
                proc_tools = tags.get("tools") or []
                proc_skills = tags.get("skills") or []
                if len(proc_tools) > 4:
                    continue
                tag_token_set = {t.lower() for t in proc_tools}
                tag_token_set |= {s.lower() for s in proc_skills}
                hit = bool(token_set & tag_token_set)

            if hit:
                matched.append(
                    {
                        "id": row_id,
                        "memory_type": "procedure",
                        "summary": summary,
                        "extra_json": extra,
                        "intercept": bool(tags.get("intercept", False)),
                        "score": 1.0,
                    }
                )

        return matched

    def keyword_search_summary(
        self,
        terms: list[str],
        memory_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """对 summary 字段做 OR-LIKE 关键字检索，按命中词数降序排列。

        每条结果携带 keyword_score（命中词数 / 总词数），供 RRF 融合使用。
        """
        terms = [t for t in terms if t and len(t) >= 2]
        if not terms:
            return []

        type_filter = ""
        type_params: list[str] = []
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            type_filter = f" AND memory_type IN ({placeholders})"
            type_params = list(memory_types)

        or_conditions = " OR ".join("summary LIKE ?" for _ in terms)
        score_expr = " + ".join(
            f"(CASE WHEN summary LIKE ? THEN 1 ELSE 0 END)" for _ in terms
        )
        like_vals = [f"%{t}%" for t in terms]

        sql = (
            f"SELECT id, memory_type, summary, source_ref, happened_at, created_at, "
            f"reinforcement, ({score_expr}) AS kw_score "
            f"FROM memory_items "
            f"WHERE status='active' AND ({or_conditions}){type_filter} "
            f"ORDER BY kw_score DESC, reinforcement DESC "
            f"LIMIT ?"
        )
        params: Sequence[object] = tuple(like_vals + like_vals + type_params + [limit])
        rows = self._db.execute(sql, params).fetchall()
        results = []
        for row in rows:
            row_id, mtype, summary, source_ref, happened_at, created_at, reinforcement, kw_score = row
            results.append({
                "id": row_id,
                "memory_type": mtype,
                "summary": summary,
                "source_ref": source_ref or "",
                "happened_at": happened_at or created_at or "",
                "keyword_score": float(kw_score) / len(terms),
            })
        return results
