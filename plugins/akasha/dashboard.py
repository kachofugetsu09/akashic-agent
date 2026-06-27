from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException

from plugins.akasha.config import AkashaConfig, load_akasha_config, resolve_akasha_db_path
from plugins.akasha.graph_snapshot import (
    EdgeRow,
    GraphSnapshotConfig,
    as_float,
    as_int,
    build_snapshot_to_file,
    chunks,
    clean_text,
    clip,
    community_legend,
    default_snapshot_path,
    dense_sim,
    graph_from_edges,
    layout_graph,
    load_snapshot,
    node_radius,
    normalize_positions,
    normalized_embedding,
    read_graph_signature,
)
from plugins.akasha.store import AkashaStore


def plugin_enabled(app: FastAPI) -> bool:
    return _active_memory_engine(app) == "akasha"


class AkashaInspectorReader:
    def __init__(self, store: AkashaStore) -> None:
        self._store = store

    def get_overview(self) -> dict[str, Any]:
        items, total = self._store.list_query_logs(page=1, page_size=1)
        latest = items[0]["ts"] if items else None
        return {"available": True, "total": total, "latest_at": latest}

    def list_turns(
        self,
        *,
        session_key: str = "",
        q: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        return self._store.list_query_logs(
            session_key=session_key,
            q=q,
            page=page,
            page_size=page_size,
        )

    def get_turn(self, query_id: str) -> dict[str, Any] | None:
        raw = self._store.get_query_log(query_id)
        if raw is None:
            return None
        result = dict(raw)
        for json_key, out_key in [
            ("activation_items_json", "activation_items"),
            ("dense_items_json", "dense_items"),
            ("ripple_items_json", "ripple_items"),
        ]:
            raw_json = result.pop(json_key, "[]")
            try:
                parsed = json.loads(str(raw_json))
            except Exception:
                parsed = []
            result[out_key] = parsed if isinstance(parsed, list) else []
        return cast(dict[str, Any], result)


class AkashaGraphReader:
    def __init__(
        self,
        store: AkashaStore,
        *,
        akasha_db_path: Path,
        sessions_db_path: Path,
        snapshot_path: Path,
    ) -> None:
        self._store = store
        self._akasha_db_path = akasha_db_path
        self._sessions_db_path = sessions_db_path
        self._snapshot_path = snapshot_path
        self._lock = threading.RLock()
        self._rebuild_lock = threading.RLock()
        self._rebuild_running = False
        self._rebuild_thread: threading.Thread | None = None

    def get_global_graph(self) -> dict[str, Any]:
        snapshot = load_snapshot(self._snapshot_path)
        if snapshot is not None:
            status = self._snapshot_status(snapshot)
            if bool(status["stale"]):
                self.ensure_rebuild_started()
            snapshot.setdefault("meta", {}).update(status)
            return snapshot
        self.ensure_rebuild_started()
        return {
            "nodes": [],
            "edges": [],
            "legend": [],
            "meta": {
                "missing": True,
                "rebuilding": self._rebuild_running,
                "snapshot_path": str(self._snapshot_path),
            },
        }

    def rebuild_global_graph(self) -> dict[str, Any]:
        thread_to_wait: threading.Thread | None = None
        with self._rebuild_lock:
            if self._rebuild_running and self._rebuild_thread is not threading.current_thread():
                thread_to_wait = self._rebuild_thread
        if thread_to_wait is not None:
            thread_to_wait.join()
            snapshot = load_snapshot(self._snapshot_path)
            if snapshot is not None:
                return snapshot
        with self._lock:
            return build_snapshot_to_file(
                akasha_db_path=self._akasha_db_path,
                sessions_db_path=self._sessions_db_path,
                snapshot_path=self._snapshot_path,
                config=GraphSnapshotConfig(),
            )

    def ensure_rebuild_started(self) -> None:
        with self._rebuild_lock:
            if self._rebuild_running:
                return
            thread = threading.Thread(target=self._rebuild_in_background, daemon=True)
            self._rebuild_running = True
            self._rebuild_thread = thread
        thread.start()

    def _rebuild_in_background(self) -> None:
        try:
            _ = self.rebuild_global_graph()
        finally:
            with self._rebuild_lock:
                self._rebuild_running = False
                self._rebuild_thread = None

    def _snapshot_status(self, snapshot: dict[str, Any]) -> dict[str, object]:
        current = read_graph_signature(self._akasha_db_path)
        raw_meta = snapshot.get("meta", {})
        meta = cast(dict[str, object], raw_meta) if isinstance(raw_meta, dict) else {}
        old_signature = meta.get("signature")
        stale = bool(old_signature != current.as_dict())
        return {
            "missing": False,
            "stale": stale,
            "rebuilding": self._rebuild_running,
            "current_signature": current.as_dict(),
        }

    def get_query_graph(self, query_id: str) -> dict[str, Any]:
        raw = self._store.get_query_log(query_id)
        if raw is None:
            raise KeyError(query_id)
        activation_items = _json_items(raw.get("activation_items_json"))
        dense_items = _json_items(raw.get("dense_items_json"))
        ripple_items = _json_items(raw.get("ripple_items_json"))
        node_meta = _query_node_meta(activation_items, dense_items, ripple_items)
        focus_keys = set(node_meta)
        for item in [*activation_items, *dense_items, *ripple_items]:
            for key_name in ("seed_key", "bridge_key"):
                value = str(item.get(key_name) or "")
                if value:
                    focus_keys.add(value)
                    _ = node_meta.setdefault(value, {"kind": "neighbor"})
        edge_rows = self._load_edges_touching(focus_keys, limit=1200)
        all_keys = set(focus_keys)
        for row in edge_rows:
            all_keys.add(row.src)
            all_keys.add(row.dst)
        payload = self._build_query_payload(all_keys, edge_rows, node_meta=node_meta)
        payload["query"] = {
            "query_id": raw.get("query_id"),
            "session_key": raw.get("session_key"),
            "seq": raw.get("seq"),
            "query_text": raw.get("query_text"),
            "intent": raw.get("intent"),
            "ts": raw.get("ts"),
            "seed_count": raw.get("seed_count"),
            "pool_count": raw.get("pool_count"),
            "activated_count": raw.get("activated_count"),
            "activation_threshold": raw.get("activation_threshold"),
            "dense_count": raw.get("dense_count"),
            "ripple_count": raw.get("ripple_count"),
        }
        return payload

    def _build_query_payload(
        self,
        keys: set[str],
        edge_rows: list[EdgeRow],
        *,
        node_meta: dict[str, dict[str, object]],
    ) -> dict[str, Any]:
        if not keys:
            return {"nodes": [], "edges": [], "legend": []}
        nodes_by_key = self._load_nodes(keys)
        if not nodes_by_key:
            return {"nodes": [], "edges": [], "legend": []}
        graph = graph_from_edges([
            row
            for row in edge_rows
            if row.src in nodes_by_key and row.dst in nodes_by_key
        ])
        for key in keys:
            if key in nodes_by_key:
                cast(Any, graph).add_node(key)
        pos, node_to_comm, comms = layout_graph(graph)
        texts = self._load_texts(nodes_by_key)
        colors, legend = community_legend(graph, comms, node_to_comm, nodes_by_key, texts)
        node_list = [str(node) for node in graph.nodes()]
        node_id = {key: index for index, key in enumerate(node_list)}
        saliences = [as_float(nodes_by_key[key]["salience"]) for key in node_list]
        min_sal = min(saliences) if saliences else 0.0
        max_sal = max(saliences) if saliences else 0.0
        coords = normalize_positions(pos)
        payload_nodes: list[dict[str, object]] = []
        for key in node_list:
            row = nodes_by_key[key]
            meta = node_meta.get(key, {})
            comm = node_to_comm.get(key, 0)
            salience = as_float(row["salience"])
            x, y = coords.get(key, (500.0, 500.0))
            payload_nodes.append({
                "id": key,
                "anchor_id": row["anchor_id"],
                "session_key": row["session_key"],
                "turn_seq": row["turn_seq"],
                "x": x,
                "y": y,
                "r": node_radius(salience, min_sal, max_sal),
                "c": colors.get(comm, "#7a7f8a"),
                "g": comm,
                "t": clip(texts.get(key, ""), 120),
                "salience": salience,
                "strength": as_float(row["strength"]),
                "resource": as_float(row["resource"]),
                "recall_count": as_int(row["recall_count"]),
                "kind": str(meta.get("kind") or "global"),
                "score": meta.get("score"),
                "source": meta.get("source"),
                "path_type": meta.get("path_type"),
                "direct": meta.get("direct"),
                "state": meta.get("state"),
                "edge": meta.get("edge"),
                "ripple": meta.get("ripple"),
            })
        payload_edges: list[dict[str, object]] = []
        for src, dst, data in cast(Any, graph).edges(data=True):
            payload_edges.append({
                "s": node_id[str(src)],
                "t": node_id[str(dst)],
                "w": round(float(data.get("weight", 0.0)), 4),
                "cc": int(data.get("cc", 0)),
                "sim": dense_sim(nodes_by_key[str(src)].get("embedding"), nodes_by_key[str(dst)].get("embedding")),
            })
        return {"nodes": payload_nodes, "edges": payload_edges, "legend": legend}

    def _load_nodes(self, keys: set[str]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        key_list = sorted(keys)
        with self._lock:
            for part in chunks(key_list, 800):
                placeholders = ",".join("?" for _ in part)
                rows = self._store.db.execute(
                    f"""
                    SELECT key, anchor_id, session_key, turn_seq, salience, strength,
                           resource, recall_count, embedding
                    FROM akasha_nodes
                    WHERE key IN ({placeholders})
                    """,
                    part,
                ).fetchall()
                for row in rows:
                    result[str(row["key"])] = {
                        "id": str(row["key"]),
                        "anchor_id": str(row["anchor_id"]),
                        "session_key": str(row["session_key"]),
                        "turn_seq": int(row["turn_seq"] or 0),
                        "salience": float(row["salience"] or 0.0),
                        "strength": float(row["strength"] or 0.0),
                        "resource": float(row["resource"] or 0.0),
                        "recall_count": int(row["recall_count"] or 0),
                        "embedding": normalized_embedding(row["embedding"]),
                    }
        return result

    def _load_edges_touching(self, keys: set[str], *, limit: int) -> list[EdgeRow]:
        if not keys:
            return []
        rows: list[sqlite3.Row] = []
        key_list = sorted(keys)
        with self._lock:
            for part in chunks(key_list, 400):
                placeholders = ",".join("?" for _ in part)
                rows.extend(self._store.db.execute(
                    f"""
                    SELECT src_key, dst_key, weight, co_count
                    FROM akasha_edges
                    WHERE src_key IN ({placeholders}) OR dst_key IN ({placeholders})
                    ORDER BY co_count DESC, weight DESC
                    LIMIT ?
                    """,
                    [*part, *part, limit],
                ).fetchall())
        return _merge_undirected_edges(rows)[:limit]

    def _load_texts(self, nodes_by_key: dict[str, dict[str, object]]) -> dict[str, str]:
        anchor_to_key = {
            str(row["anchor_id"]): key
            for key, row in nodes_by_key.items()
            if str(row.get("anchor_id") or "")
        }
        result = {key: "" for key in nodes_by_key}
        if not anchor_to_key or not self._sessions_db_path.exists():
            return result
        with sqlite3.connect(str(self._sessions_db_path)) as db:
            for part in chunks(sorted(anchor_to_key), 800):
                placeholders = ",".join("?" for _ in part)
                rows = db.execute(
                    f"SELECT id, content FROM messages WHERE id IN ({placeholders})",
                    part,
                ).fetchall()
                for msg_id, content in rows:
                    result[anchor_to_key[str(msg_id)]] = clean_text(str(content or ""))
        return result


def register(app: FastAPI, plugin_dir: Path, workspace: Path) -> list[object]:
    _ = plugin_dir
    akasha_config = _load_akasha_config(workspace)
    if akasha_config is None:
        return []

    akasha_db_path = resolve_akasha_db_path(workspace=workspace, akasha_config=akasha_config)
    store = AkashaStore(akasha_db_path)
    reader = AkashaInspectorReader(store)
    graph_reader = AkashaGraphReader(
        store,
        akasha_db_path=akasha_db_path,
        sessions_db_path=workspace / "sessions.db",
        snapshot_path=default_snapshot_path(workspace),
    )

    @app.get("/api/dashboard/akasha-inspector/overview")
    def get_akasha_inspector_overview() -> dict[str, Any]:
        return reader.get_overview()

    @app.get("/api/dashboard/akasha-inspector/turns")
    def list_akasha_inspector_turns(
        session_key: str = "",
        q: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        items, total = reader.list_turns(
            session_key=session_key,
            q=q,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/akasha-inspector/turns/{query_id:path}")
    def get_akasha_inspector_turn(query_id: str) -> dict[str, Any]:
        item = reader.get_turn(query_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Akasha 检索记录不存在")
        return item

    @app.get("/api/dashboard/akasha-graph/global")
    def get_akasha_graph_global() -> dict[str, Any]:
        return graph_reader.get_global_graph()

    @app.post("/api/dashboard/akasha-graph/rebuild")
    def rebuild_akasha_graph_global() -> dict[str, Any]:
        return graph_reader.rebuild_global_graph()

    @app.get("/api/dashboard/akasha-graph/query/{query_id:path}")
    def get_akasha_graph_query(query_id: str) -> dict[str, Any]:
        try:
            return graph_reader.get_query_graph(query_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Akasha 检索记录不存在") from None

    return [store]


def _active_memory_engine(app: FastAPI) -> str:
    memory_admin = getattr(app.state, "memory_admin", None)
    describe = getattr(memory_admin, "describe", None)
    if not callable(describe):
        return ""
    return str(getattr(describe(), "name", ""))


def _load_akasha_config(workspace: Path) -> AkashaConfig | None:
    _ = workspace
    try:
        plugin_dir = Path(__file__).resolve().parent
        return load_akasha_config(plugin_dir=plugin_dir)
    except Exception:
        return AkashaConfig()


def _json_items(value: object) -> list[dict[str, object]]:
    try:
        loaded = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [
        cast(dict[str, object], item)
        for item in cast(list[object], loaded)
        if isinstance(item, dict)
    ]


def _query_node_meta(
    activation_items: list[dict[str, object]],
    dense_items: list[dict[str, object]],
    ripple_items: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for kind, items in [("activated", activation_items), ("dense", dense_items), ("ripple", ripple_items)]:
        for item in items:
            key = str(item.get("key") or "")
            if not key:
                continue
            current = result.get(key, {})
            if _kind_rank(kind) >= _kind_rank(str(current.get("kind") or "")):
                result[key] = {
                    "kind": kind,
                    "score": item.get("score"),
                    "source": item.get("source") or item.get("lane"),
                    "path_type": item.get("path_type"),
                    "direct": item.get("direct"),
                    "state": item.get("state"),
                    "edge": item.get("edge"),
                    "ripple": item.get("ripple"),
                }
    return result


def _kind_rank(kind: str) -> int:
    return {"neighbor": 0, "ripple": 1, "dense": 2, "activated": 3}.get(kind, 0)


def _merge_undirected_edges(rows: list[sqlite3.Row]) -> list[EdgeRow]:
    merged: dict[tuple[str, str], EdgeRow] = {}
    for row in rows:
        src = str(row["src_key"])
        dst = str(row["dst_key"])
        if src == dst:
            continue
        key = (src, dst) if src < dst else (dst, src)
        weight = as_float(row["weight"])
        co_count = as_int(row["co_count"])
        old = merged.get(key)
        if old is None:
            merged[key] = EdgeRow(key[0], key[1], weight, co_count)
        else:
            merged[key] = EdgeRow(
                key[0],
                key[1],
                max(old.weight, weight),
                max(old.co_count, co_count),
            )
    return sorted(merged.values(), key=lambda item: (item.co_count, item.weight), reverse=True)
