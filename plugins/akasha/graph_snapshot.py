from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

BIG_COMMUNITY_SIZE = 8
LAYOUT_EDGE_LIMIT = 5000


@dataclass(frozen=True)
class GraphSnapshotConfig:
    min_co_count: int = 1
    layout_min_co_count: int = 2
    layout_edge_limit: int = LAYOUT_EDGE_LIMIT


@dataclass(frozen=True)
class GraphSnapshotSignature:
    node_count: int
    edge_count: int
    max_node_updated_at: str
    max_edge_last_used_ts: float

    def as_dict(self) -> dict[str, object]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "max_node_updated_at": self.max_node_updated_at,
            "max_edge_last_used_ts": self.max_edge_last_used_ts,
        }


def default_snapshot_path(workspace: Path) -> Path:
    return workspace / "memory" / "akasha_graph_snapshot.json"


def load_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def read_graph_signature(akasha_db_path: Path) -> GraphSnapshotSignature:
    with sqlite3.connect(str(akasha_db_path)) as db:
        node_row = db.execute(
            "SELECT COUNT(*) AS count, COALESCE(MAX(updated_at), '') AS max_updated_at FROM akasha_nodes"
        ).fetchone()
        edge_row = db.execute(
            "SELECT COUNT(*) AS count, COALESCE(MAX(last_used_ts), 0) AS max_last_used_ts FROM akasha_edges"
        ).fetchone()
    return GraphSnapshotSignature(
        node_count=int(node_row[0] or 0),
        edge_count=int(edge_row[0] or 0),
        max_node_updated_at=str(node_row[1] or ""),
        max_edge_last_used_ts=float(edge_row[1] or 0.0),
    )


def build_snapshot_to_file(
    *,
    akasha_db_path: Path,
    sessions_db_path: Path,
    snapshot_path: Path,
    config: GraphSnapshotConfig | None = None,
) -> dict[str, Any]:
    import uuid
    prev_snapshot = load_snapshot(snapshot_path)
    payload = build_snapshot(
        akasha_db_path=akasha_db_path,
        sessions_db_path=sessions_db_path,
        config=config or GraphSnapshotConfig(),
        prev_snapshot=prev_snapshot,
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = snapshot_path.with_name(snapshot_path.name + f".{uuid.uuid4().hex}.tmp")
    _ = tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    _ = tmp_path.replace(snapshot_path)
    return payload


def build_snapshot(
    *,
    akasha_db_path: Path,
    sessions_db_path: Path,
    config: GraphSnapshotConfig,
    prev_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    signature = read_graph_signature(akasha_db_path)
    with sqlite3.connect(str(akasha_db_path)) as akasha_db:
        akasha_db.row_factory = sqlite3.Row
        all_edges = _load_merged_edges(akasha_db, min_co_count=config.min_co_count)
        keys = {edge.src for edge in all_edges}
        keys.update(edge.dst for edge in all_edges)
        nodes_by_key = _load_nodes(akasha_db, keys)

    all_edges = [
        edge
        for edge in all_edges
        if edge.src in nodes_by_key and edge.dst in nodes_by_key
    ]
    keys = {edge.src for edge in all_edges}
    keys.update(edge.dst for edge in all_edges)
    nodes_by_key = {key: nodes_by_key[key] for key in keys if key in nodes_by_key}
    texts = _load_texts(sessions_db_path, nodes_by_key)

    layout_edges = _layout_edges(
        all_edges,
        layout_min_co_count=config.layout_min_co_count,
        layout_edge_limit=config.layout_edge_limit,
    )
    layout_graph = _graph_from_edges(layout_edges)

    incremental = _incremental_layout_from_snapshot(prev_snapshot, nodes_by_key, all_edges)
    if incremental is None:
        pos, node_to_comm, comms = _layout_graph(layout_graph)
        _place_missing_nodes(pos, node_to_comm, comms, nodes_by_key, all_edges)
        coords = _normalize_positions(pos)
        layout_mode = "full"
    else:
        coords, node_to_comm, comms = incremental
        layout_mode = "incremental"

    full_graph = _graph_from_edges(all_edges)
    colors, legend = _community_legend(
        full_graph,
        comms,
        node_to_comm,
        nodes_by_key,
        texts,
    )
    payload_nodes = _payload_nodes(nodes_by_key, texts, coords, node_to_comm, colors)
    node_id = {str(node["id"]): index for index, node in enumerate(payload_nodes)}
    payload_edges = _payload_edges(all_edges, node_id, nodes_by_key)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "nodes": payload_nodes,
        "edges": payload_edges,
        "legend": legend,
        "meta": {
            "version": f"{signature.node_count}:{signature.edge_count}:{signature.max_node_updated_at}:{signature.max_edge_last_used_ts}",
            "signature": signature.as_dict(),
            "generated_at_unix": time.time(),
            "elapsed_ms": elapsed_ms,
            "node_count": len(payload_nodes),
            "edge_count": len(payload_edges),
            "layout_edge_count": len(layout_edges),
            "min_co_count": config.min_co_count,
            "layout_min_co_count": config.layout_min_co_count,
            "layout_edge_limit": config.layout_edge_limit,
            "layout_mode": layout_mode,
        },
    }


@dataclass(frozen=True)
class EdgeRow:
    src: str
    dst: str
    weight: float
    co_count: int


def _load_merged_edges(db: sqlite3.Connection, *, min_co_count: int) -> list[EdgeRow]:
    rows = db.execute(
        """
        SELECT src_key, dst_key, weight, co_count
        FROM akasha_edges
        WHERE co_count >= ?
        """,
        (min_co_count,),
    ).fetchall()
    merged: dict[tuple[str, str], EdgeRow] = {}
    for row in rows:
        src = str(row["src_key"])
        dst = str(row["dst_key"])
        if src == dst:
            continue
        key = (src, dst) if src < dst else (dst, src)
        weight = float(row["weight"] or 0.0)
        co_count = int(row["co_count"] or 0)
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


def _load_nodes(
    db: sqlite3.Connection,
    keys: set[str],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    key_list = sorted(keys)
    for part in _chunks(key_list, 800):
        placeholders = ",".join("?" for _ in part)
        rows = db.execute(
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
                "embedding": _normalized_embedding(row["embedding"]),
            }
    return result


def _layout_edges(
    edges: list[EdgeRow],
    *,
    layout_min_co_count: int,
    layout_edge_limit: int,
) -> list[EdgeRow]:
    strong = [edge for edge in edges if edge.co_count >= layout_min_co_count]
    if not strong:
        strong = edges
    return strong[:max(1, layout_edge_limit)]


def _graph_from_edges(edges: list[EdgeRow]) -> Any:
    graph = cast(Any, nx.Graph())
    for edge in edges:
        graph.add_edge(edge.src, edge.dst, weight=edge.weight, cc=edge.co_count)
    return graph


def _layout_graph(
    graph: Any,
) -> tuple[dict[str, tuple[float, float]], dict[str, int], list[set[str]]]:
    if graph.number_of_nodes() == 0:
        return {}, {}, []
    if graph.number_of_edges() == 0:
        nodes = [str(node) for node in graph.nodes()]
        return _spiral_positions(nodes), {key: index for index, key in enumerate(nodes)}, [{key} for key in nodes]

    comms = [
        set(str(item) for item in comm)
        for comm in nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    ]
    comms.sort(key=len, reverse=True)
    node_to_comm = {node: index for index, comm in enumerate(comms) for node in comm}
    pos: dict[str, tuple[float, float]] = {}
    island_centers = _island_centers([len(comm) for comm in comms])
    island_radii = [0.6 + 1.4 * math.sqrt(len(comm)) for comm in comms]
    for index, comm in enumerate(comms):
        sub = graph.subgraph(comm)
        if len(comm) == 1:
            local: dict[str, tuple[float, float]] = {next(iter(comm)): (0.0, 0.0)}
        else:
            local = {
                str(node): (float(xy[0]), float(xy[1]))
                for node, xy in nx.spring_layout(
                    cast(Any, sub),
                    k=1.1 / math.sqrt(len(comm)),
                    iterations=60,
                    weight="weight",
                    seed=0,
                ).items()
            }
        cx, cy = island_centers[index]
        radius = island_radii[index]
        for node in comm:
            lx, ly = local[node]
            pos[node] = (cx + lx * radius, cy + ly * radius)
    return pos, node_to_comm, comms


def _incremental_layout_from_snapshot(
    snapshot: dict[str, Any] | None,
    nodes_by_key: dict[str, dict[str, object]],
    edges: list[EdgeRow],
) -> tuple[dict[str, tuple[float, float]], dict[str, int], list[set[str]]] | None:
    if not snapshot:
        return None
    raw_nodes = snapshot.get("nodes")
    if not isinstance(raw_nodes, list):
        return None

    coords: dict[str, tuple[float, float]] = {}
    node_to_comm: dict[str, int] = {}
    for raw_node_item in cast(list[object], raw_nodes):
        if not isinstance(raw_node_item, dict):
            continue
        raw_node = cast(dict[str, object], raw_node_item)
        key = str(raw_node.get("id") or "")
        if key not in nodes_by_key:
            continue
        x = _as_float(raw_node.get("x"))
        y = _as_float(raw_node.get("y"))
        coords[key] = (x, y)
        node_to_comm[key] = _as_int(raw_node.get("g"))
    if not coords:
        return None

    comms = _communities_from_map(node_to_comm, nodes_by_key)
    _place_incremental_nodes(coords, node_to_comm, comms, nodes_by_key, edges)
    return coords, node_to_comm, comms


def _communities_from_map(
    node_to_comm: dict[str, int],
    nodes_by_key: dict[str, dict[str, object]],
) -> list[set[str]]:
    max_comm = max(node_to_comm.values(), default=-1)
    comms: list[set[str]] = [set() for _ in range(max_comm + 1)]
    for key in nodes_by_key:
        comm = node_to_comm.get(key)
        if comm is None:
            continue
        if comm >= len(comms):
            for _ in range(comm - len(comms) + 1):
                comms.append(set())
        comms[comm].add(key)
    return comms


def _place_incremental_nodes(
    coords: dict[str, tuple[float, float]],
    node_to_comm: dict[str, int],
    comms: list[set[str]],
    nodes_by_key: dict[str, dict[str, object]],
    edges: list[EdgeRow],
) -> None:
    missing = [key for key in nodes_by_key if key not in coords]
    if not missing:
        return
    neighbors: dict[str, list[tuple[str, EdgeRow]]] = {}
    for edge in edges:
        neighbors.setdefault(edge.src, []).append((edge.dst, edge))
        neighbors.setdefault(edge.dst, []).append((edge.src, edge))

    next_comm = len(comms)
    for index, key in enumerate(missing):
        parent = _best_positioned_neighbor(neighbors.get(key, []), coords)
        if parent is None:
            x, y = _normalized_spiral_point(index, len(missing))
            comm_id = next_comm
            next_comm += 1
            comms.append({key})
        else:
            parent_key, rank = parent
            base_x, base_y = coords[parent_key]
            angle = (rank * 2.399963229728653) + index * 0.17
            radius = 9.0 + 2.2 * math.sqrt(index % 23)
            x = min(1000.0, max(0.0, base_x + math.cos(angle) * radius))
            y = min(1000.0, max(0.0, base_y + math.sin(angle) * radius))
            comm_id = node_to_comm.get(parent_key, next_comm)
            if comm_id == next_comm:
                next_comm += 1
                comms.append(set())
            comms[comm_id].add(key)
        coords[key] = (round(x, 1), round(y, 1))
        node_to_comm[key] = comm_id


def _place_missing_nodes(
    pos: dict[str, tuple[float, float]],
    node_to_comm: dict[str, int],
    comms: list[set[str]],
    nodes_by_key: dict[str, dict[str, object]],
    edges: list[EdgeRow],
) -> None:
    missing = [key for key in nodes_by_key if key not in pos]
    if not missing:
        return
    neighbors: dict[str, list[tuple[str, EdgeRow]]] = {}
    for edge in edges:
        neighbors.setdefault(edge.src, []).append((edge.dst, edge))
        neighbors.setdefault(edge.dst, []).append((edge.src, edge))
    next_comm = len(comms)
    for index, key in enumerate(missing):
        parent = _best_positioned_neighbor(neighbors.get(key, []), pos)
        if parent is None:
            x, y = _loose_spiral_point(index)
            comm_id = next_comm
            next_comm += 1
            comms.append({key})
        else:
            parent_key, rank = parent
            base_x, base_y = pos[parent_key]
            angle = (rank * 2.399963229728653) + index * 0.17
            radius = 0.55 + 0.08 * math.sqrt(index % 17)
            x = base_x + math.cos(angle) * radius
            y = base_y + math.sin(angle) * radius
            comm_id = node_to_comm.get(parent_key, next_comm)
            if comm_id == next_comm:
                next_comm += 1
                comms.append(set())
            comms[comm_id].add(key)
        pos[key] = (x, y)
        node_to_comm[key] = comm_id


def _best_positioned_neighbor(
    candidates: list[tuple[str, EdgeRow]],
    pos: dict[str, tuple[float, float]],
) -> tuple[str, int] | None:
    ranked = sorted(candidates, key=lambda item: (item[1].co_count, item[1].weight), reverse=True)
    for rank, (key, _) in enumerate(ranked):
        if key in pos:
            return key, rank
    return None


def _payload_nodes(
    nodes_by_key: dict[str, dict[str, object]],
    texts: dict[str, str],
    coords: dict[str, tuple[float, float]],
    node_to_comm: dict[str, int],
    colors: dict[int, str],
) -> list[dict[str, object]]:
    node_list = sorted(nodes_by_key, key=lambda key: (node_to_comm.get(key, 0), key))
    saliences = [_as_float(nodes_by_key[key]["salience"]) for key in node_list]
    min_sal = min(saliences) if saliences else 0.0
    max_sal = max(saliences) if saliences else 0.0
    result: list[dict[str, object]] = []
    for key in node_list:
        row = nodes_by_key[key]
        comm = node_to_comm.get(key, 0)
        salience = _as_float(row["salience"])
        x, y = coords.get(key, (500.0, 500.0))
        result.append({
            "id": key,
            "anchor_id": row["anchor_id"],
            "session_key": row["session_key"],
            "turn_seq": row["turn_seq"],
            "x": x,
            "y": y,
            "r": _node_radius(salience, min_sal, max_sal),
            "c": colors.get(comm, "#7a7f8a"),
            "g": comm,
            "t": _clip(texts.get(key, ""), 120),
            "salience": salience,
            "strength": _as_float(row["strength"]),
            "resource": _as_float(row["resource"]),
            "recall_count": _as_int(row["recall_count"]),
        })
    return result


def _payload_edges(
    edges: list[EdgeRow],
    node_id: dict[str, int],
    nodes_by_key: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for edge in edges:
        if edge.src not in node_id or edge.dst not in node_id:
            continue
        result.append({
            "s": node_id[edge.src],
            "t": node_id[edge.dst],
            "w": round(edge.weight, 4),
            "cc": edge.co_count,
            "sim": _dense_sim(
                nodes_by_key[edge.src].get("embedding"),
                nodes_by_key[edge.dst].get("embedding"),
            ),
        })
    return result


def _load_texts(
    sessions_db_path: Path,
    nodes_by_key: dict[str, dict[str, object]],
) -> dict[str, str]:
    anchor_to_key = {
        str(row["anchor_id"]): key
        for key, row in nodes_by_key.items()
        if str(row.get("anchor_id") or "")
    }
    result = {key: "" for key in nodes_by_key}
    if not anchor_to_key or not sessions_db_path.exists():
        return result
    with sqlite3.connect(str(sessions_db_path)) as db:
        for part in _chunks(sorted(anchor_to_key), 800):
            placeholders = ",".join("?" for _ in part)
            rows = db.execute(
                f"SELECT id, content FROM messages WHERE id IN ({placeholders})",
                part,
            ).fetchall()
            for msg_id, content in rows:
                result[anchor_to_key[str(msg_id)]] = _clean_text(str(content or ""))
    return result


def _community_legend(
    graph: Any,
    comms: list[set[str]],
    node_to_comm: dict[str, int],
    nodes_by_key: dict[str, dict[str, object]],
    texts: dict[str, str],
) -> tuple[dict[int, str], list[dict[str, object]]]:
    big_ids = [index for index, comm in enumerate(comms) if len(comm) >= BIG_COMMUNITY_SIZE]
    colors = {comm_id: _community_color(rank) for rank, comm_id in enumerate(big_ids)}
    legend: list[dict[str, object]] = []
    for comm_id in big_ids[:30]:
        comm = comms[comm_id]
        scored: list[tuple[float, str]] = []
        for node in comm:
            text = texts.get(node, "")
            if len(text) < 5 or len(text) > 140 or "http" in text:
                continue
            internal = sum(
                float(data.get("weight", 1.0))
                for neighbor, data in graph[node].items()
                if str(neighbor) in comm
            )
            total = float(graph.degree(node, weight="weight") or 0.0)
            purity = internal / (total + 1e-9)
            salience = max(_as_float(nodes_by_key[node]["salience"]), 0.1)
            scored.append((internal * purity * salience, _clip(text, 24)))
        scored.sort(key=lambda item: item[0], reverse=True)
        legend.append({
            "c": colors[comm_id],
            "size": len(comm),
            "label": " · ".join(text for _, text in scored[:3]) or f"社区{comm_id}",
        })
    all_colors = {
        node_to_comm.get(str(node), 0): colors.get(node_to_comm.get(str(node), 0), "#7a7f8a")
        for node in graph.nodes()
    }
    return all_colors, legend


def _spiral_positions(nodes: list[str]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(nodes):
        x, y = _loose_spiral_point(index)
        result[node] = (x, y)
    return result


def _loose_spiral_point(index: int) -> tuple[float, float]:
    golden = math.pi * (3 - math.sqrt(5))
    radius = math.sqrt(index + 1)
    angle = index * golden
    return radius * math.cos(angle), radius * math.sin(angle)


def _normalized_spiral_point(index: int, total: int) -> tuple[float, float]:
    golden = math.pi * (3 - math.sqrt(5))
    radius = min(460.0, 30.0 + 430.0 * math.sqrt((index + 1) / max(1, total)))
    angle = index * golden
    return 500.0 + radius * math.cos(angle), 500.0 + radius * math.sin(angle)


def _island_centers(sizes: list[int]) -> dict[int, tuple[float, float]]:
    golden = math.pi * (3 - math.sqrt(5))
    centers: dict[int, tuple[float, float]] = {}
    area_cum = 0.0
    for index, size in enumerate(sizes):
        island_radius = 0.6 + 1.4 * math.sqrt(size)
        area = (island_radius * 1.5) ** 2
        area_cum += area / 2
        radius = math.sqrt(area_cum) * 1.9
        angle = index * golden
        centers[index] = (radius * math.cos(angle), radius * math.sin(angle))
        area_cum += area / 2
    return centers


def _normalize_positions(pos: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    if not pos:
        return {}
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return {
        key: (
            round((xy[0] - x0) / (x1 - x0 + 1e-9) * 1000, 1),
            round((xy[1] - y0) / (y1 - y0 + 1e-9) * 1000, 1),
        )
        for key, xy in pos.items()
    }


def _community_color(index: int) -> str:
    hue = int((index * 137.508 + (index // 12) * 17) % 360)
    saturation = [72, 62, 82, 68][index % 4]
    lightness = [58, 66, 52, 72, 60, 48][(index // 4) % 6]
    return f"hsl({hue},{saturation}%,{lightness}%)"


def _node_radius(salience: float, min_sal: float, max_sal: float) -> float:
    return round(2.5 + 6.5 * ((salience - min_sal) / (max_sal - min_sal + 1e-9)), 1)


def _as_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _normalized_embedding(value: object) -> NDArray[np.float32] | None:
    if not isinstance(value, bytes | bytearray | memoryview):
        return None
    try:
        emb = np.frombuffer(cast(Any, value), dtype=np.float32).copy()
    except Exception:
        return None
    norm = float(np.linalg.norm(emb))
    if norm <= 0:
        return None
    return emb / norm


def _dense_sim(left: object, right: object) -> float:
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        return 0.0
    left_arr = cast(Any, left)
    right_arr = cast(Any, right)
    if left_arr.shape != right_arr.shape:
        return 0.0
    return round(float(np.dot(left_arr, right_arr)), 3)


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _clip(value: str, limit: int) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


as_float = _as_float
as_int = _as_int
chunks = _chunks
clean_text = _clean_text
clip = _clip
community_legend = _community_legend
dense_sim = _dense_sim
graph_from_edges = _graph_from_edges
layout_graph = _layout_graph
node_radius = _node_radius
normalize_positions = _normalize_positions
normalized_embedding = _normalized_embedding
