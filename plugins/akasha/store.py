from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from plugins.akasha.core import (
    AkashaNode, ActivationUpdate, EdgeUpdate, ActivationEventRow,
    SourceMessage, turn_key, serialize_f32, deserialize_f32, parse_ts_unix,
    advance_salience_state,
    bounded_add,
    causal_salience,
    effective_edge_weight,
    heterosynaptic_depression,
    initial_strength,
    normalize as _normalize,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS akasha_query_log (
    query_id              TEXT PRIMARY KEY,
    session_key           TEXT NOT NULL,
    seq                   INTEGER NOT NULL,
    query_text            TEXT NOT NULL,
    intent                TEXT NOT NULL,
    ts                    TEXT NOT NULL,
    seed_count            INTEGER NOT NULL DEFAULT 0,
    pool_count            INTEGER NOT NULL DEFAULT 0,
    activated_count       INTEGER NOT NULL DEFAULT 0,
    activation_threshold  REAL NOT NULL DEFAULT 0,
    dense_count           INTEGER NOT NULL DEFAULT 0,
    ripple_count          INTEGER NOT NULL DEFAULT 0,
    inject_chars          INTEGER NOT NULL DEFAULT 0,
    source_ref_count      INTEGER NOT NULL DEFAULT 0,
    activation_items      TEXT,
    dense_items           TEXT,
    ripple_items          TEXT,
    text_block_preview    TEXT
);
CREATE INDEX IF NOT EXISTS ix_akasha_query_log_session
    ON akasha_query_log (session_key, seq DESC);
CREATE INDEX IF NOT EXISTS ix_akasha_query_log_ts
    ON akasha_query_log (ts DESC);
CREATE TABLE IF NOT EXISTS akasha_nodes (
    key                TEXT PRIMARY KEY,
    anchor_id          TEXT NOT NULL,
    session_key        TEXT NOT NULL,
    turn_seq           INTEGER NOT NULL,
    first_ts_unix      REAL NOT NULL,
    salience           REAL NOT NULL,
    strength           REAL NOT NULL,
    resource           REAL NOT NULL,
    recall_count       INTEGER NOT NULL,
    last_activated_ts  REAL NOT NULL,
    last_strength_ts   REAL NOT NULL,
    last_resource_ts   REAL NOT NULL,
    embedding          BLOB NOT NULL,
    emb_count          INTEGER NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_akasha_nodes_session_seq
    ON akasha_nodes (session_key, turn_seq);
CREATE TABLE IF NOT EXISTS akasha_edges (
    src_key       TEXT NOT NULL,
    dst_key       TEXT NOT NULL,
    weight        REAL NOT NULL,
    co_count      INTEGER NOT NULL,
    last_used_ts  REAL NOT NULL,
    PRIMARY KEY (src_key, dst_key)
);
CREATE INDEX IF NOT EXISTS ix_akasha_edges_src
    ON akasha_edges (src_key);
CREATE INDEX IF NOT EXISTS ix_akasha_edges_dst
    ON akasha_edges (dst_key);
CREATE TABLE IF NOT EXISTS akasha_activation_events (
    seq           INTEGER NOT NULL,
    query_id      TEXT NOT NULL,
    activated_key TEXT NOT NULL,
    source        TEXT NOT NULL,
    score         REAL NOT NULL,
    direct_score  REAL NOT NULL,
    state_score   REAL NOT NULL,
    edge_score    REAL NOT NULL,
    long_score    REAL NOT NULL,
    resource      REAL NOT NULL,
    fan           INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_akasha_events_seq
    ON akasha_activation_events (seq);
CREATE TABLE IF NOT EXISTS akasha_embedding_cache (
    message_id   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    dim          INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (message_id, model)
);
CREATE INDEX IF NOT EXISTS ix_akasha_embedding_cache_hash
    ON akasha_embedding_cache (content_hash, model);
CREATE TABLE IF NOT EXISTS akasha_salience_state (
    key         TEXT PRIMARY KEY,
    vector_sum BLOB NOT NULL,
    count       INTEGER NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS akasha_migration_runs (
    id               TEXT PRIMARY KEY,
    source_db_path   TEXT NOT NULL,
    target_db_path   TEXT NOT NULL,
    embedding_model  TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    status           TEXT NOT NULL,
    message_count    INTEGER NOT NULL DEFAULT 0,
    activation_count INTEGER NOT NULL DEFAULT 0,
    cache_hit_count  INTEGER NOT NULL DEFAULT 0,
    cache_miss_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS akasha_source_session_snapshot (
    run_id            TEXT NOT NULL,
    session_key       TEXT NOT NULL,
    last_consolidated INTEGER NOT NULL,
    next_seq          INTEGER NOT NULL,
    max_seq           INTEGER NOT NULL,
    PRIMARY KEY (run_id, session_key)
);
"""

RESET_SQL = """
DROP TABLE IF EXISTS akasha_query_log;
DROP TABLE IF EXISTS akasha_activation_events;
DROP TABLE IF EXISTS akasha_edges;
DROP TABLE IF EXISTS akasha_nodes;
DROP TABLE IF EXISTS akasha_salience_state;
"""


@dataclass(frozen=True)
class SourceSessionSnapshot:
    session_key: str
    last_consolidated: int
    next_seq: int
    max_seq: int


# 计算 message 内容指纹。
def content_hash(content: str) -> str:
    # 1. cache 命中必须同时匹配 message_id、model 和原文内容。
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# 生成 UTC ISO 时间。
def _now_iso() -> str:
    # 1. sidecar 内部时间统一用 UTC。
    return datetime.now(timezone.utc).isoformat()


def _table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    rows = db.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


class AkashaStore:
    def __init__(self, db_path: str | Path) -> None:
        # 1. 初始化 sidecar 数据库和 schema。
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        self.ensure_schema()

    @property
    def db(self) -> sqlite3.Connection:
        return self._db

    # 关闭 SQLite 连接。
    def close(self) -> None:
        # 1. close 需要幂等，方便 runtime 多路径收尾。
        if self._closed:
            return
        self._db.close()
        self._closed = True

    # 确保 Akasha schema 存在。
    def ensure_schema(self) -> None:
        # 1. schema 可重复执行，用于启动和迁移前检查。
        with self._lock:
            if "run_id" not in _table_columns(self._db, "akasha_source_session_snapshot"):
                _ = self._db.execute("DROP TABLE IF EXISTS akasha_source_session_snapshot")
            if "message_count" not in _table_columns(self._db, "akasha_migration_runs"):
                _ = self._db.execute("DROP TABLE IF EXISTS akasha_migration_runs")
            _ = self._db.executescript(SCHEMA)
            self._db.commit()

    # 清空并重建 Akasha 状态表。
    def reset_schema(self) -> None:
        # 1. 只清 replay 产物，保留 embedding cache 和迁移记录。
        with self._lock:
            _ = self._db.executescript(RESET_SQL)
            _ = self._db.executescript(SCHEMA)
            self._db.commit()

    # 读取可复用的 message embedding。
    def get_cached_embedding(
        self,
        *,
        message: SourceMessage,
        model: str,
    ) -> list[float] | None:
        # 1. 同一 message id 只有在内容和模型都一致时复用。
        digest = content_hash(message.content)
        with self._lock:
            row = self._db.execute(
                """
                SELECT embedding
                FROM akasha_embedding_cache
                WHERE message_id = ? AND model = ? AND content_hash = ?
                """,
                (message.id, model, digest),
            ).fetchone()
        if row is None:
            return None
        return deserialize_f32(row["embedding"]).astype(float).tolist()

    # 写入或刷新 message embedding cache。
    def upsert_cached_embedding(
        self,
        *,
        message: SourceMessage,
        model: str,
        embedding: list[float],
    ) -> None:
        # 1. message 内容变更时覆盖同 model 的旧 embedding。
        vector = np.array(embedding, dtype=np.float32)
        now = _now_iso()
        with self._lock:
            _ = self._db.execute(
                """
                INSERT INTO akasha_embedding_cache
                    (message_id, content_hash, model, embedding, dim, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id, model) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    embedding = excluded.embedding,
                    dim = excluded.dim,
                    updated_at = excluded.updated_at
                """,
                (
                    message.id,
                    content_hash(message.content),
                    model,
                    serialize_f32(vector),
                    int(vector.size),
                    now,
                    now,
                ),
            )
            self._db.commit()

    # 读取指定模型的全部 message embedding cache。
    def list_cached_embeddings(self, *, model: str) -> list[tuple[str, np.ndarray]]:
        # 1. Dense 展示使用 message-level embedding，再映射回 turn。
        with self._lock:
            rows = self._db.execute(
                """
                SELECT message_id, embedding
                FROM akasha_embedding_cache
                WHERE model = ?
                """,
                (model,),
            ).fetchall()
        return [
            (str(row["message_id"]), deserialize_f32(row["embedding"]))
            for row in rows
        ]

    # 删除指定消息的 embedding cache。
    def delete_cached_embeddings(self, message_ids: list[str]) -> int:
        clean_ids = [str(item).strip() for item in message_ids if str(item).strip()]
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        with self._lock:
            cur = self._db.execute(
                f"DELETE FROM akasha_embedding_cache WHERE message_id IN ({placeholders})",
                clean_ids,
            )
            self._db.commit()
        return int(cur.rowcount or 0)

    # 开始记录一次 Akasha 迁移。
    def start_migration_run(
        self,
        *,
        source_db_path: Path,
        embedding_model: str,
    ) -> str:
        # 1. run 记录只保存路径和模型，不保存密钥。
        run_id = uuid.uuid4().hex
        now = _now_iso()
        with self._lock:
            _ = self._db.execute(
                """
                INSERT INTO akasha_migration_runs
                    (id, source_db_path, target_db_path, embedding_model, started_at, status)
                VALUES (?, ?, ?, ?, ?, 'running')
                """,
                (
                    run_id,
                    str(source_db_path),
                    str(self.db_path),
                    embedding_model,
                    now,
                ),
            )
            self._db.commit()
        return run_id

    # 完成一次 Akasha 迁移记录。
    def finish_migration_run(
        self,
        *,
        run_id: str,
        status: str,
        message_count: int,
        activation_count: int,
        cache_hit_count: int,
        cache_miss_count: int,
    ) -> None:
        # 1. 脚本失败时也写入 status，便于知道当前 sidecar 是否可信。
        with self._lock:
            _ = self._db.execute(
                """
                UPDATE akasha_migration_runs
                SET finished_at = ?,
                    status = ?,
                    message_count = ?,
                    activation_count = ?,
                    cache_hit_count = ?,
                    cache_miss_count = ?
                WHERE id = ?
                """,
                (
                    _now_iso(),
                    status,
                    message_count,
                    activation_count,
                    cache_hit_count,
                    cache_miss_count,
                    run_id,
                ),
            )
            self._db.commit()

    # 保存迁移开始时的 session 游标快照。
    def insert_session_snapshots(
        self,
        *,
        run_id: str,
        snapshots: list[SourceSessionSnapshot],
    ) -> None:
        # 1. 只记录旧系统游标，不修改 sessions.db。
        if not snapshots:
            return
        with self._lock:
            _ = self._db.executemany(
                """
                INSERT OR REPLACE INTO akasha_source_session_snapshot
                    (run_id, session_key, last_consolidated, next_seq, max_seq)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item.session_key,
                        item.last_consolidated,
                        item.next_seq,
                        item.max_seq,
                    )
                    for item in snapshots
                ],
            )
            self._db.commit()

    # 插入或更新一条 message 对应的 turn 节点。
    def upsert_message_node(
        self,
        message: SourceMessage,
        embedding: list[float],
    ) -> str:
        # 1. 按 cross CLI 规则把 user/assistant 映射到同一个 turn key。
        session_key, turn_seq, key = turn_key(
            message.session_key,
            message.seq,
            message.role,
        )
        vector = _normalize(np.array(embedding, dtype=np.float32))
        ts_unix = parse_ts_unix(message.ts)
        now = _now_iso()

        # 2. 新 turn 直接写入；已有 turn 用均值更新 embedding，并保留 user 作为 anchor。
        with self._lock:
            prior_sum, prior_count = self._load_salience_state_locked()
            salience = (
                causal_salience(vector, prior_sum, prior_count)
                if message.salience is None
                else min(1.0, max(0.0, float(message.salience)))
            )
            next_sum, next_count = advance_salience_state(prior_sum, prior_count, vector)
            row = self._db.execute(
                "SELECT * FROM akasha_nodes WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                # 编码即峰值：strength 按 salience 初始化为接近 cap 的值
                # 对应 Ebbinghaus 遗忘曲线起点 + ACT-R 初次激活高 base-level
                init_str = initial_strength(salience)
                # 新节点的 last_*_ts 设为 first_ts_unix（编码即"被激活"那一刻）
                _ = self._db.execute(
                    """
                    INSERT INTO akasha_nodes
                        (key, anchor_id, session_key, turn_seq, first_ts_unix,
                         salience, strength, resource, recall_count,
                         last_activated_ts, last_strength_ts, last_resource_ts,
                         embedding, emb_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, 0, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        key,
                        message.id,
                        session_key,
                        turn_seq,
                        ts_unix,
                        salience,
                        init_str,
                        ts_unix,
                        ts_unix,
                        ts_unix,
                        serialize_f32(vector),
                        now,
                        now,
                    ),
                )
                self._write_salience_state_locked(next_sum, next_count, now)
                self._db.commit()
                return key

            old_embedding = deserialize_f32(row["embedding"])
            old_count = max(1, int(row["emb_count"] or 1))
            merged = _normalize(old_embedding * old_count + vector)
            anchor_id = message.id if message.role == "user" else str(row["anchor_id"])
            _ = self._db.execute(
                """
                UPDATE akasha_nodes
                SET anchor_id = ?,
                    salience = ?,
                    embedding = ?,
                    emb_count = ?,
                    updated_at = ?
                WHERE key = ?
                """,
                (
                    anchor_id,
                    max(float(row["salience"] or 0.0), salience),
                    serialize_f32(merged),
                    old_count + 1,
                    now,
                    key,
                ),
            )
            self._write_salience_state_locked(next_sum, next_count, now)
            self._db.commit()
        return key

    def _load_salience_state_locked(self) -> tuple[np.ndarray | None, int]:
        row = self._db.execute(
            "SELECT vector_sum, count FROM akasha_salience_state WHERE key = 'global'"
        ).fetchone()
        if row is None:
            return None, 0
        vector_sum = deserialize_f32(row["vector_sum"])
        count = int(row["count"] or 0)
        return (vector_sum if vector_sum.size else None), count

    def _write_salience_state_locked(
        self,
        vector_sum: np.ndarray,
        count: int,
        now: str,
    ) -> None:
        _ = self._db.execute(
            """
            INSERT INTO akasha_salience_state (key, vector_sum, count, updated_at)
            VALUES ('global', ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                vector_sum = excluded.vector_sum,
                count = excluded.count,
                updated_at = excluded.updated_at
            """,
            (serialize_f32(vector_sum), count, now),
        )

    # 读取全部 turn 节点。
    def list_nodes(self) -> list[AkashaNode]:
        # 1. 查询阶段一次性读入节点，和 cross CLI 的内存图保持一致。
        with self._lock:
            rows = self._db.execute("SELECT * FROM akasha_nodes").fetchall()
        return [node for row in rows if (node := _row_to_node(row)) is not None]

    # 读取单个 turn 节点。
    def get_node(self, key: str) -> AkashaNode | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM akasha_nodes WHERE key = ?",
                (key,),
            ).fetchone()
        return _row_to_node(row) if row is not None else None

    # 读取全部共激活边。
    def load_edges(self) -> dict[tuple[str, str], float]:
        edges, _ = self.load_edges_with_meta()
        return edges

    # 读取全部共激活边和更新时间。
    def load_edges_with_meta(
        self,
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT src_key, dst_key, weight, last_used_ts FROM akasha_edges"
            ).fetchall()
        edges = {
            (str(row["src_key"]), str(row["dst_key"])): float(row["weight"] or 0.0)
            for row in rows
        }
        meta = {
            (str(row["src_key"]), str(row["dst_key"])): float(row["last_used_ts"] or 0.0)
            for row in rows
        }
        return edges, meta

    # 批量更新被激活节点的长期状态。
    def update_activation_batch(self, updates: list[ActivationUpdate]) -> None:
        # 1. 查询阶段只更新旧节点状态，不创建当前 turn 的边。
        if not updates:
            return
        now = _now_iso()
        with self._lock:
            _ = self._db.executemany(
                """
                UPDATE akasha_nodes
                SET strength = ?,
                    resource = ?,
                    recall_count = ?,
                    last_activated_ts = ?,
                    last_strength_ts = ?,
                    last_resource_ts = ?,
                    updated_at = ?
                WHERE key = ?
                """,
                [
                    (
                        item.strength,
                        item.resource,
                        item.recall_count,
                        item.ts,
                        item.ts,
                        item.ts,
                        now,
                        item.key,
                    )
                    for item in updates
                ],
            )
            self._db.commit()

    # 批量写入或增强共激活边。
    def upsert_edges(self, updates: list[EdgeUpdate]) -> None:
        # 1. after-turn 拿到真实 user message id 后，再建立当前 turn 和旧节点的边。
        if not updates:
            return
        with self._lock:
            for item in updates:
                if item.src_key == item.dst_key:
                    continue
                row = self._db.execute(
                    """
                    SELECT weight, co_count, last_used_ts
                    FROM akasha_edges
                    WHERE src_key = ? AND dst_key = ?
                    """,
                    (item.src_key, item.dst_key),
                ).fetchone()
                if row is None:
                    weight = 0.12 * item.strength
                    _ = self._db.execute(
                        "INSERT INTO akasha_edges VALUES (?, ?, ?, 1, ?)",
                        (item.src_key, item.dst_key, weight, item.ts),
                    )
                    continue
                old = effective_edge_weight(
                    float(row["weight"] or 0.0),
                    float(row["last_used_ts"] or 0.0),
                    item.ts,
                )
                new_weight = bounded_add(old, 0.12 * item.strength, 2.0)
                _ = self._db.execute(
                    """
                    UPDATE akasha_edges
                    SET weight = ?,
                        co_count = ?,
                        last_used_ts = ?
                    WHERE src_key = ? AND dst_key = ?
                    """,
                    (
                        new_weight,
                        int(row["co_count"] or 0) + 1,
                        item.ts,
                        item.src_key,
                        item.dst_key,
                    ),
                )

            # heterosynaptic：被强化节点的非活动出边按权重压抑（last_used_ts/co_count 不变）
            def _out_neighbors(src_key: str) -> dict[str, float]:
                rows = self._db.execute(
                    "SELECT dst_key, weight FROM akasha_edges WHERE src_key = ?",
                    (src_key,),
                ).fetchall()
                return {str(r["dst_key"]): float(r["weight"] or 0.0) for r in rows}

            for src_key, dst_key, new_w in heterosynaptic_depression(updates, _out_neighbors):
                _ = self._db.execute(
                    "UPDATE akasha_edges SET weight = ? WHERE src_key = ? AND dst_key = ?",
                    (new_w, src_key, dst_key),
                )
            self._db.commit()

    # 批量记录本轮激活诊断事件。
    def insert_activation_events(self, rows: list[ActivationEventRow]) -> None:
        # 1. 诊断事件只用于观察，不参与原始事实来源。
        if not rows:
            return
        with self._lock:
            _ = self._db.executemany(
                """
                INSERT INTO akasha_activation_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.seq,
                        row.query_id,
                        row.activated_key,
                        row.source,
                        row.score,
                        row.direct_score,
                        row.state_score,
                        row.edge_score,
                        row.long_score,
                        row.resource,
                        row.fan,
                    )
                    for row in rows
                ],
            )
            self._db.commit()

    # 读取 dashboard 需要的节点列表。
    def list_items_for_dashboard(
        self,
        *,
        q: str = "",
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, object]], int]:
        # 1. Akasha dashboard MVP 只展示节点状态，不展示原文。
        where = ""
        params: list[object] = []
        if q.strip():
            where = "WHERE key LIKE ? OR anchor_id LIKE ?"
            like = f"%{q.strip()}%"
            params.extend([like, like])
        safe_sort = sort_by if sort_by in {"updated_at", "first_ts_unix", "strength", "resource"} else "updated_at"
        safe_order = "ASC" if sort_order.lower() == "asc" else "DESC"
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 200))
        offset = (page - 1) * page_size
        with self._lock:
            count_row = self._db.execute(
                f"SELECT COUNT(1) AS c FROM akasha_nodes {where}",
                params,
            ).fetchone()
            rows = self._db.execute(
                f"""
                SELECT *
                FROM akasha_nodes
                {where}
                ORDER BY {safe_sort} {safe_order}
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            ).fetchall()
        total = int((count_row["c"] if count_row else 0) or 0)
        return [_node_row_to_dashboard(row) for row in rows], total

    # 按 id 读取 dashboard 节点详情。
    def get_item_for_dashboard(self, item_id: str) -> dict[str, object] | None:
        # 1. item_id 对应 turn key。
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM akasha_nodes WHERE key = ?",
                (item_id,),
            ).fetchone()
        return _node_row_to_dashboard(row) if row is not None else None

    # 物理删除一个 Akasha 节点和相关边。
    def delete_item(self, item_id: str) -> bool:
        # 1. 删除只影响 sidecar，不影响 sessions.db 原始消息。
        with self._lock:
            cur = self._db.execute("DELETE FROM akasha_nodes WHERE key = ?", (item_id,))
            _ = self._db.execute(
                "DELETE FROM akasha_edges WHERE src_key = ? OR dst_key = ?",
                (item_id, item_id),
            )
            self._db.commit()
        return cur.rowcount > 0

    # 批量物理删除 Akasha 节点。
    def delete_items_batch(self, ids: list[str]) -> int:
        # 1. 复用单删逻辑，保持边清理一致。
        count = 0
        for item_id in ids:
            if self.delete_item(item_id):
                count += 1
        return count

    # 删除指定 turn 的检索诊断状态。
    def delete_query_state_for_turns(self, turns: list[tuple[str, int]]) -> None:
        if not turns:
            return
        with self._lock:
            for session_key, seq in turns:
                _ = self._db.execute(
                    "DELETE FROM akasha_query_log WHERE session_key = ? AND seq = ?",
                    (session_key, seq),
                )
                _ = self._db.execute(
                    "DELETE FROM akasha_activation_events WHERE query_id = ?",
                    (f"{session_key}:{seq}",),
                )
            self._db.commit()

    # 写入一条检索诊断日志。
    def insert_query_log(
        self,
        *,
        query_id: str,
        session_key: str,
        seq: int,
        query_text: str,
        intent: str,
        ts: str,
        seed_count: int,
        pool_count: int,
        activated_count: int,
        activation_threshold: float,
        dense_count: int,
        ripple_count: int,
        inject_chars: int,
        source_ref_count: int,
        activation_items_json: str,
        dense_items_json: str,
        ripple_items_json: str,
        text_block_preview: str,
    ) -> None:
        # 1. 诊断日志是可验收状态，写失败要直接暴露。
        with self._lock:
            _ = self._db.execute(
                """
                INSERT OR REPLACE INTO akasha_query_log (
                    query_id, session_key, seq, query_text, intent, ts,
                    seed_count, pool_count, activated_count, activation_threshold,
                    dense_count, ripple_count, inject_chars, source_ref_count,
                    activation_items, dense_items, ripple_items, text_block_preview
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id, session_key, seq, query_text, intent, ts,
                    seed_count, pool_count, activated_count, activation_threshold,
                    dense_count, ripple_count, inject_chars, source_ref_count,
                    activation_items_json, dense_items_json, ripple_items_json,
                    text_block_preview,
                ),
            )
            self._db.commit()

    # 读取检索诊断日志列表（仅轻量字段）。
    def list_query_logs(
        self,
        *,
        session_key: str = "",
        q: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, object]], int]:
        # 1. 过滤 session_key 和 query_text 关键字。
        clauses: list[str] = []
        params: list[object] = []
        if session_key.strip():
            clauses.append("session_key = ?")
            params.append(session_key.strip())
        if q.strip():
            clauses.append("query_text LIKE ?")
            params.append(f"%{q.strip()}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        page = max(1, page)
        page_size = max(1, min(page_size, 200))
        offset = (page - 1) * page_size
        with self._lock:
            count_row = self._db.execute(
                f"SELECT COUNT(1) FROM akasha_query_log {where}", params
            ).fetchone()
            rows = self._db.execute(
                f"""
                SELECT query_id, session_key, seq, query_text, intent, ts,
                       seed_count, pool_count, activated_count, activation_threshold,
                       dense_count, ripple_count, inject_chars, source_ref_count
                FROM akasha_query_log {where}
                ORDER BY ts DESC, seq DESC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            ).fetchall()
        total = int((count_row[0] if count_row else 0) or 0)
        items: list[dict[str, object]] = [
            {
                "query_id": str(row["query_id"]),
                "session_key": str(row["session_key"]),
                "seq": int(row["seq"]),
                "query_text": str(row["query_text"]),
                "intent": str(row["intent"]),
                "ts": str(row["ts"]),
                "seed_count": int(row["seed_count"] or 0),
                "pool_count": int(row["pool_count"] or 0),
                "activated_count": int(row["activated_count"] or 0),
                "activation_threshold": float(row["activation_threshold"] or 0.0),
                "dense_count": int(row["dense_count"] or 0),
                "ripple_count": int(row["ripple_count"] or 0),
                "inject_chars": int(row["inject_chars"] or 0),
                "source_ref_count": int(row["source_ref_count"] or 0),
            }
            for row in rows
        ]
        return items, total

    # 按 query_id 读取完整检索诊断记录。
    def get_query_log(self, query_id: str) -> dict[str, object] | None:
        # 1. 完整记录包含 JSON 列，由调用方反序列化。
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM akasha_query_log WHERE query_id = ?",
                (query_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "query_id": str(row["query_id"]),
            "session_key": str(row["session_key"]),
            "seq": int(row["seq"]),
            "query_text": str(row["query_text"]),
            "intent": str(row["intent"]),
            "ts": str(row["ts"]),
            "seed_count": int(row["seed_count"] or 0),
            "pool_count": int(row["pool_count"] or 0),
            "activated_count": int(row["activated_count"] or 0),
            "activation_threshold": float(row["activation_threshold"] or 0.0),
            "dense_count": int(row["dense_count"] or 0),
            "ripple_count": int(row["ripple_count"] or 0),
            "inject_chars": int(row["inject_chars"] or 0),
            "source_ref_count": int(row["source_ref_count"] or 0),
            "activation_items_json": str(row["activation_items"] or "[]"),
            "dense_items_json": str(row["dense_items"] or "[]"),
            "ripple_items_json": str(row["ripple_items"] or "[]"),
            "text_block_preview": str(row["text_block_preview"] or ""),
        }


# 把 SQLite row 转成 AkashaNode。
def _row_to_node(row: sqlite3.Row) -> AkashaNode | None:
    # 1. embedding 损坏时跳过该节点，避免一次坏数据打断整轮检索。
    embedding = deserialize_f32(row["embedding"])
    if embedding.size == 0:
        return None
    return AkashaNode(
        key=str(row["key"]),
        anchor_id=str(row["anchor_id"]),
        session_key=str(row["session_key"]),
        turn_seq=int(row["turn_seq"]),
        first_ts_unix=float(row["first_ts_unix"] or 0.0),
        salience=float(row["salience"] or 0.0),
        strength=float(row["strength"] or 0.0),
        resource=float(row["resource"] or 1.0),
        recall_count=int(row["recall_count"] or 0),
        last_activated_ts=float(row["last_activated_ts"] or 0.0),
        last_strength_ts=float(row["last_strength_ts"] or 0.0),
        last_resource_ts=float(row["last_resource_ts"] or 0.0),
        embedding=embedding,
        emb_count=int(row["emb_count"] or 1),
    )


# 把节点 row 转成 dashboard 通用 item。
def _node_row_to_dashboard(row: sqlite3.Row) -> dict[str, object]:
    # 1. dashboard 使用 MemoryAdminApi 的通用字段名。
    return {
        "id": str(row["key"]),
        "memory_type": "turn",
        "summary": str(row["key"]),
        "source_ref": str(row["anchor_id"]),
        "status": "active",
        "created_at": str(row["created_at"] or ""),
        "updated_at": str(row["updated_at"] or ""),
        "happened_at": str(row["first_ts_unix"] or ""),
        "extra_json": {
            "session_key": str(row["session_key"]),
            "turn_seq": int(row["turn_seq"]),
            "strength": float(row["strength"] or 0.0),
            "resource": float(row["resource"] or 1.0),
            "recall_count": int(row["recall_count"] or 0),
            "emb_count": int(row["emb_count"] or 1),
        },
        "has_embedding": True,
        "embedding_dim": len(deserialize_f32(row["embedding"])),
    }
