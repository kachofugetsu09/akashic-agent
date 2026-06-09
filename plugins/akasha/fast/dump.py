"""dump_to_db —— 把内存 MemoryStore 一次性批量落进已开的 AkashaStore（sqlite），复用 embedding_cache。

列与序列化严格对齐 store.py 的慢路写法：
  * embedding / salience vector_sum 用 core.serialize_f32
  * created_at/updated_at 用 _now_iso（与慢路 store._now_iso 同语义）
  * 各表 VALUES 顺序对齐 store.py 的 INSERT
reset_schema 只清 5 张图表(nodes/edges/salience_state/query_log/activation_events)，
保留 embedding_cache / migration_runs / source_session_snapshot。复用调用方已开的连接（单连接）。
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from plugins.akasha.core import serialize_f32
from plugins.akasha.store import AkashaStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_to_db(store: AkashaStore, mem) -> dict[str, int]:
    """把 mem(MemoryStore/CapturingMemoryStore)的全图 + 诊断批量写入 store。"""
    now = _now_iso()
    ql = list(getattr(mem, "query_logs", []))
    ev = list(getattr(mem, "activation_events", []))
    with store._lock:  # pyright: ignore[reportPrivateUsage]
        store.reset_schema()
        db = store._db  # pyright: ignore[reportPrivateUsage]

        node_rows = [
            (
                n.key, n.anchor_id, n.session_key, n.turn_seq, n.first_ts_unix,
                n.salience, n.strength, n.resource, n.recall_count,
                n.last_activated_ts, n.last_strength_ts, n.last_resource_ts,
                serialize_f32(np.asarray(n.embedding, dtype=np.float32)), n.emb_count, now, now,
            )
            for n in mem._nodes.values()  # pyright: ignore[reportPrivateUsage]
        ]
        db.executemany(
            """INSERT INTO akasha_nodes
               (key, anchor_id, session_key, turn_seq, first_ts_unix,
                salience, strength, resource, recall_count,
                last_activated_ts, last_strength_ts, last_resource_ts,
                embedding, emb_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            node_rows,
        )

        edges = mem._edges  # pyright: ignore[reportPrivateUsage]
        meta = mem._meta  # pyright: ignore[reportPrivateUsage]
        cocount = mem._cocount  # pyright: ignore[reportPrivateUsage]
        edge_rows = [
            (s, d, w, int(cocount.get((s, d), 1)), float(meta.get((s, d), 0.0)))
            for (s, d), w in edges.items()
        ]
        db.executemany("INSERT INTO akasha_edges VALUES (?, ?, ?, ?, ?)", edge_rows)

        csum = mem._csum  # pyright: ignore[reportPrivateUsage]
        if csum is not None:
            db.execute(
                "INSERT INTO akasha_salience_state (key, vector_sum, count, updated_at) "
                "VALUES ('global', ?, ?, ?)",
                (serialize_f32(np.asarray(csum, dtype=np.float32)),
                 int(mem._ccount), now),  # pyright: ignore[reportPrivateUsage]
            )

        if ql:
            db.executemany(
                """INSERT OR REPLACE INTO akasha_query_log (
                       query_id, session_key, seq, query_text, intent, ts,
                       seed_count, pool_count, activated_count, activation_threshold,
                       dense_count, ripple_count, inject_chars, source_ref_count,
                       activation_items, dense_items, ripple_items, text_block_preview
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (q["query_id"], q["session_key"], q["seq"], q["query_text"], q["intent"], q["ts"],
                     q["seed_count"], q["pool_count"], q["activated_count"], q["activation_threshold"],
                     q["dense_count"], q["ripple_count"], q["inject_chars"], q["source_ref_count"],
                     q["activation_items_json"], q["dense_items_json"], q["ripple_items_json"],
                     q["text_block_preview"])
                    for q in ql
                ],
            )

        if ev:
            db.executemany(
                "INSERT INTO akasha_activation_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (r.seq, r.query_id, r.activated_key, r.source, r.score, r.direct_score,
                     r.state_score, r.edge_score, r.long_score, r.resource, r.fan)
                    for r in ev
                ],
            )

        db.commit()

    return {
        "nodes": len(mem._nodes),  # pyright: ignore[reportPrivateUsage]
        "edges": len(edges),
        "query_logs": len(ql),
        "activation_events": len(ev),
    }
