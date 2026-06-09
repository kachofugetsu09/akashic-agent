"""Akasha 快速重建默认路径回归。"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

try:
    import plugins.akasha.core as core
    import plugins.akasha.replay as replay
    from plugins.akasha.config import AkashaConfig
    from plugins.akasha.fast import fast_dense, graph_fast
    from plugins.akasha.fast.dump import dump_to_db
    from plugins.akasha.fast.mem_store import CapturingMemoryStore
    from plugins.akasha.replay import AkashaReplayRuntime, ReplayMessage
    from plugins.akasha.store import AkashaStore
except Exception:  # pragma: no cover - host 依赖缺失时跳过
    pytest.skip("akasha host 依赖缺失", allow_module_level=True)


T0 = 1_700_000_000.0

_TURN_PAIRS = [
    ("u0", 0, "alpha 引流条", [1.0, 0.0, 0.0, 0.0], "a0", "收到 alpha", [0.96, 0.04, 0.0, 0.0]),
    ("u1", 2, "beta 鱼石脂", [0.9, 0.1, 0.0, 0.0], "a1", "收到 beta", [0.88, 0.12, 0.0, 0.0]),
    ("u2", 4, "gamma 换药", [0.8, 0.2, 0.1, 0.0], "a2", "收到 gamma", [0.79, 0.21, 0.09, 0.0]),
    ("u3", 6, "alpha 又问引流条", [0.95, 0.05, 0.0, 0.0], "a3", "继续 alpha", [0.94, 0.06, 0.0, 0.0]),
    ("u4", 8, "beta 又问鱼石脂", [0.85, 0.15, 0.0, 0.02], "a4", "继续 beta", [0.84, 0.16, 0.0, 0.02]),
    ("u5", 10, "delta 旁支", [0.0, 0.0, 1.0, 0.0], "a5", "收到 delta", [0.0, 0.0, 0.95, 0.05]),
]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _source_message(
    message_id: str,
    seq: int,
    role: str,
    content: str,
) -> core.SourceMessage:
    return core.SourceMessage(message_id, "s", seq, role, content, _iso(T0 + seq))


def _turn_messages() -> list[list[tuple[core.SourceMessage, list[float]]]]:
    turns: list[list[tuple[core.SourceMessage, list[float]]]] = []
    for user_id, seq, user_text, user_vec, assistant_id, assistant_text, assistant_vec in _TURN_PAIRS:
        turns.append([
            (_source_message(user_id, seq, "user", user_text), user_vec),
            (_source_message(assistant_id, seq + 1, "assistant", assistant_text), assistant_vec),
        ])
    return turns


def _init_sessions(path: Path) -> None:
    with closing(sqlite3.connect(str(path))) as db:
        db.execute(
            "CREATE TABLE messages (id TEXT PRIMARY KEY, session_key TEXT, seq INTEGER, "
            "role TEXT, content TEXT, ts TEXT)"
        )
        db.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(content)")
        rowid = 1
        for turn in _turn_messages():
            for message, _ in turn:
                db.execute(
                    "INSERT INTO messages(rowid, id, session_key, seq, role, content, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (rowid, message.id, message.session_key, message.seq, message.role, message.content, message.ts),
                )
                db.execute("INSERT INTO messages_fts(rowid, content) VALUES (?, ?)", (rowid, message.content))
                rowid += 1
        db.commit()


def _core_config(config: AkashaConfig) -> core.CoreConfig:
    return core.CoreConfig(
        dense_top_k=config.dense_top_k,
        dense_seed_threshold=config.dense_seed_threshold,
        activation_threshold=config.activation_threshold,
        cross_boost=config.cross_boost,
        nearby_time_seconds=config.nearby_time_seconds,
        nearby_dense_threshold=config.nearby_dense_threshold,
        soft_recall_threshold=config.soft_recall_threshold,
        soft_recall_direct_floor=config.soft_recall_direct_floor,
        activate_limit=config.activate_limit,
    )


def _message_embeddings() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    embeddings: dict[str, np.ndarray] = {}
    turn_keys: dict[str, str] = {}
    for turn in _turn_messages():
        for message, vector in turn:
            embeddings[message.id] = np.array(vector, dtype=np.float32)
            turn_keys[message.id] = core.turn_key(message.session_key, message.seq, message.role)[2]
    return embeddings, turn_keys


def _replay(store, sessions_db: Path, embeddings, turn_keys) -> None:
    with closing(sqlite3.connect(str(sessions_db))) as source_db:
        runtime = AkashaReplayRuntime(
            store=store,
            config=AkashaConfig(),
            source_db_path=sessions_db,
            source_cursor=source_db.cursor(),
            message_embeddings=embeddings,
            message_turn_keys=turn_keys,
        )
        for turn in _turn_messages():
            runtime.replay_turn([
                ReplayMessage(message=message, embedding=list(map(float, vector)))
                for message, vector in turn
            ])


def _snapshot(db_path: Path):
    db = sqlite3.connect(str(db_path))
    nodes = {
        row[0]: (round(row[1], 6), round(row[2], 6), int(row[3]))
        for row in db.execute("SELECT key, strength, resource, recall_count FROM akasha_nodes")
    }
    edges = {
        (row[0], row[1]): (round(row[2], 6), int(row[3]), round(row[4], 6))
        for row in db.execute("SELECT src_key, dst_key, weight, co_count, last_used_ts FROM akasha_edges")
    }
    db.close()
    return nodes, edges


def _build_fast(tmp_path: Path, tag: str) -> Path:
    sessions = tmp_path / f"sessions_{tag}.db"
    _init_sessions(sessions)
    db_path = tmp_path / f"fast_{tag}.db"
    store = AkashaStore(db_path)
    mem = CapturingMemoryStore()
    graph_fast.install(mem)
    fast_dense.install()
    embeddings, turn_keys = _message_embeddings()
    try:
        _replay(mem, sessions, embeddings, turn_keys)
        dump_to_db(store, mem)
    finally:
        graph_fast.uninstall()
        fast_dense.uninstall()
        store.close()
    return db_path


def _build_online_path(tmp_path: Path) -> Path:
    sessions = tmp_path / "online_sessions.db"
    _init_sessions(sessions)
    db_path = tmp_path / "online.db"
    store = AkashaStore(db_path)
    config = AkashaConfig()
    core_config = _core_config(config)
    nodes: dict[str, core.AkashaNode] = {}
    edges: dict[tuple[str, str], float] = {}
    edges_meta: dict[tuple[str, str], float] = {}
    edges_by_src: dict[str, dict[str, float]] = {}
    fan: dict[str, int] = {}
    message_embeddings: dict[str, np.ndarray] = {}
    message_turn_keys: dict[str, str] = {}
    message_index = core.build_dense_message_index({})
    with closing(sqlite3.connect(str(sessions))) as source_db:
        source_cursor = source_db.cursor()
        try:
            for turn in _turn_messages():
                user_message, user_vector = turn[0]
                query_vec = np.array(user_vector, dtype=np.float32)
                now_ts = core.parse_ts_unix(user_message.ts)
                snapshot = core.AkashaActivationSnapshot(
                    nodes=dict(nodes),
                    edges=dict(edges),
                    edges_meta=dict(edges_meta),
                    fan=dict(fan),
                    edges_by_src={key: dict(value) for key, value in edges_by_src.items()},
                    message_embeddings=dict(message_embeddings),
                    message_turn_keys=dict(message_turn_keys),
                    message_index=message_index,
                )
                activation_items: list[core.AkashaCandidate] = []
                if snapshot.nodes:
                    graph_seed_keys = core.graph_seed_keys_from_snapshot(
                        query_vec,
                        snapshot,
                        limit=config.dense_top_k,
                    )
                    activation_items, _, _ = core.compute_candidates_from_snapshot(
                        user_message.content,
                        query_vec,
                        snapshot,
                        now_ts,
                        config=core_config,
                        source_cursor=source_cursor,
                        soft_recall=False,
                        return_limit=config.activate_limit,
                        graph_seed_keys=graph_seed_keys,
                    )
                updates = core.activation_updates(activation_items, snapshot.nodes, now_ts)
                store.update_activation_batch(updates)
                for item in updates:
                    node = nodes.get(item.key)
                    if node is None:
                        continue
                    nodes[item.key] = replace(
                        node,
                        strength=item.strength,
                        resource=item.resource,
                        recall_count=item.recall_count,
                        last_activated_ts=item.ts,
                        last_strength_ts=item.ts,
                        last_resource_ts=item.ts,
                    )
                current_key = ""
                for message, vector in turn:
                    current_key = store.upsert_message_node(message, vector)
                    node = store.get_node(current_key)
                    assert node is not None
                    nodes[current_key] = node
                    message_embeddings[message.id] = np.array(vector, dtype=np.float32)
                    message_turn_keys[message.id] = core.turn_key(
                        message.session_key,
                        message.seq,
                        message.role,
                    )[2]
                    message_index = core.build_dense_message_index(message_embeddings)
                if current_key and activation_items:
                    edge_updates = core.activation_edge_updates(current_key, activation_items, now_ts)
                    store.upsert_edges(edge_updates)
                    for item in edge_updates:
                        if item.src_key == item.dst_key:
                            continue
                        edge_key = (item.src_key, item.dst_key)
                        old = edges.get(edge_key)
                        if old is None:
                            weight = 0.12 * item.strength
                        else:
                            decayed = core.effective_edge_weight(
                                old,
                                edges_meta.get(edge_key, 0.0),
                                item.ts,
                            )
                            weight = core.bounded_add(decayed, 0.12 * item.strength, 2.0)
                        edges[edge_key] = weight
                        edges_meta[edge_key] = item.ts
                        edges_by_src.setdefault(item.src_key, {})[item.dst_key] = weight
                    fan = core.fan_counts(edges)
        finally:
            store.close()
    return db_path


def test_fast_matches_online_turn_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("plugins.akasha.core.get_jieba_keywords", lambda _: "")
    online = _snapshot(_build_online_path(tmp_path))
    fast = _snapshot(_build_fast(tmp_path, "online"))
    assert fast == online


def test_fast_rebuild_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("plugins.akasha.core.get_jieba_keywords", lambda _: "")
    a = _snapshot(_build_fast(tmp_path, "a"))
    b = _snapshot(_build_fast(tmp_path, "b"))
    assert a == b


def test_fast_rebuild_restores_global_patches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("plugins.akasha.core.get_jieba_keywords", lambda _: "")
    originals = (
        core.graph_expand_candidates,
        core.has_user_turn,
        core.dense_message_candidates,
        replay.edges_by_src,
        replay.fan_counts,
        replay.dense_message_candidates,
    )
    _build_fast(tmp_path, "restore")
    assert (
        core.graph_expand_candidates,
        core.has_user_turn,
        core.dense_message_candidates,
        replay.edges_by_src,
        replay.fan_counts,
        replay.dense_message_candidates,
    ) == originals
