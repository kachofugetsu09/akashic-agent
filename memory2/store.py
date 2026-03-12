"""
Memory v2 SQLite 存储层
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id            TEXT PRIMARY KEY,
    memory_type   TEXT NOT NULL,
    summary       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    embedding     TEXT,
    reinforcement INTEGER NOT NULL DEFAULT 1,
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
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(summary: str, memory_type: str) -> str:
    text = re.sub(r"\s+", " ", summary.lower().strip()) + memory_type
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class MemoryStore2:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()

        cols = {r[1] for r in self._db.execute("PRAGMA table_info(memory_items)")}
        if "status" not in cols:
            self._db.execute(
                "ALTER TABLE memory_items ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
            )
            self._db.commit()
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_items_status ON memory_items (status)"
        )
        self._db.commit()

    def close(self) -> None:
        db = getattr(self, "_db", None)
        if db is None:
            return
        try:
            db.close()
        finally:
            self._db = None

    def __del__(self) -> None:
        self.close()

    def upsert_item(
        self,
        memory_type: str,
        summary: str,
        embedding: list[float] | None,
        source_ref: str | None = None,
        extra: dict | None = None,
        happened_at: str | None = None,
    ) -> str:
        """写入或强化一条记忆。返回 'new:id' 或 'reinforced:id'"""
        chash = _content_hash(summary, memory_type)
        existing = self._db.execute(
            "SELECT id, status FROM memory_items WHERE content_hash=? AND memory_type=?",
            (chash, memory_type),
        ).fetchone()
        if existing:
            row_id, status = existing
            if status == "superseded":
                self._db.execute(
                    "UPDATE memory_items SET status='active', reinforcement=reinforcement+1, updated_at=? WHERE id=?",
                    (_now_iso(), row_id),
                )
            else:
                self._db.execute(
                    "UPDATE memory_items SET reinforcement=reinforcement+1, updated_at=? WHERE id=?",
                    (_now_iso(), row_id),
                )
            self._db.commit()
            return f"reinforced:{row_id}"

        item_id = hashlib.md5(f"{chash}{time.time()}".encode()).hexdigest()[:12]
        self._db.execute(
            """INSERT INTO memory_items
               (id, memory_type, summary, content_hash, embedding, extra_json,
                source_ref, happened_at, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                item_id,
                memory_type,
                summary,
                chash,
                json.dumps(embedding) if embedding is not None else None,
                json.dumps(extra) if extra else None,
                source_ref,
                happened_at,
                _now_iso(),
                _now_iso(),
            ),
        )
        self._db.commit()
        return f"new:{item_id}"

    def upsert_consolidation_event(
        self,
        *,
        source_ref: str,
        summary: str,
        embedding: list[float] | None,
        extra: dict | None = None,
        happened_at: str | None = None,
    ) -> str:
        """原子写入 consolidation event：同一 source_ref 最多写一次。"""
        src = (source_ref or "").strip()
        text = (summary or "").strip()
        if not src or not text:
            return "skipped:empty"

        self._db.execute("BEGIN IMMEDIATE")
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
                        "UPDATE memory_items SET status='active', reinforcement=reinforcement+1, updated_at=? WHERE id=?",
                        (_now_iso(), row_id),
                    )
                else:
                    self._db.execute(
                        "UPDATE memory_items SET reinforcement=reinforcement+1, updated_at=? WHERE id=?",
                        (_now_iso(), row_id),
                    )
                item_id = row_id
                result = f"reinforced:{row_id}"
            else:
                item_id = hashlib.md5(f"{chash}{time.time()}".encode()).hexdigest()[:12]
                self._db.execute(
                    """INSERT INTO memory_items
                       (id, memory_type, summary, content_hash, embedding, extra_json,
                        source_ref, happened_at, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        item_id,
                        "event",
                        text,
                        chash,
                        json.dumps(embedding) if embedding is not None else None,
                        json.dumps(extra) if extra else None,
                        src,
                        happened_at,
                        _now_iso(),
                        _now_iso(),
                    ),
                )
                result = f"new:{item_id}"

            self._db.execute(
                "INSERT INTO consolidation_events(source_ref, item_id, created_at) VALUES (?, ?, ?)",
                (src, item_id, _now_iso()),
            )
            self._db.execute("COMMIT")
            return result
        except Exception:
            try:
                self._db.execute("ROLLBACK")
            except Exception:
                pass
            raise

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

    def get_all_with_embedding(self, include_superseded: bool = False) -> list[tuple]:
        """返回 [(id, memory_type, summary, embedding_list, extra_json_dict, happened_at)]"""
        where = "" if include_superseded else "AND status='active'"
        rows = self._db.execute(
            "SELECT id, memory_type, summary, embedding, extra_json, happened_at "
            f"FROM memory_items WHERE embedding IS NOT NULL {where}"
        ).fetchall()
        result = []
        for row_id, mtype, summary, emb_json, extra_json, happened_at in rows:
            emb = json.loads(emb_json) if emb_json else None
            extra = json.loads(extra_json) if extra_json else {}
            result.append((row_id, mtype, summary, emb, extra, happened_at))
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
    ) -> list[dict]:
        """cosine similarity 检索，返回 top-k 结果"""
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

        scored = []
        for row_id, mtype, summary, emb, extra, happened_at in rows:
            if emb is None:
                continue
            e = np.array(emb, dtype=np.float32)
            score = float(e @ q) / (float(np.linalg.norm(e)) + 1e-9) / q_norm
            if score < score_threshold:
                continue
            scored.append(
                {
                    "id": row_id,
                    "memory_type": mtype,
                    "summary": summary,
                    "extra_json": extra,
                    "happened_at": happened_at,
                    "score": round(score, 4),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def list_by_type(self, memory_type: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, memory_type, summary, extra_json, happened_at, reinforcement "
            "FROM memory_items WHERE memory_type=?",
            (memory_type,),
        ).fetchall()
        result = []
        for row_id, mtype, summary, extra_json, happened_at, reinforcement in rows:
            result.append(
                {
                    "id": row_id,
                    "memory_type": mtype,
                    "summary": summary,
                    "extra_json": json.loads(extra_json) if extra_json else {},
                    "happened_at": happened_at,
                    "reinforcement": reinforcement,
                }
            )
        return result

    def delete_by_source_ref(self, source_ref: str) -> int:
        """删除指定 source_ref 的所有条目，返回删除行数。"""
        cur = self._db.execute(
            "DELETE FROM memory_items WHERE source_ref=?", (source_ref,)
        )
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

    def keyword_match_procedures(self, action_tokens: list[str]) -> list[dict]:
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
                        "score": 1.0,
                    }
                )

        return matched

    def close(self) -> None:
        self._db.close()
