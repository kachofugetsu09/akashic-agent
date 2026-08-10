"""Read Akasha V2 retrieval evidence without mutating learned state."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import numpy as np

from agent.plugins.manifest import builtin_plugin_data_dir

from .config import load_akasha_config, resolve_workspace_path

_LOCAL_TZ = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class InspectorPaths:
    """Locate the two read-only Akasha sidecars."""

    memory: Path
    index: Path


@dataclass(frozen=True)
class _DenseSnapshot:
    """Keep normalized turn vectors and their display metadata in memory."""

    signature: tuple[int, int]
    turns: tuple[dict[str, object], ...]
    user: np.ndarray
    assistant: np.ndarray
    user_present: np.ndarray
    assistant_present: np.ndarray


def resolve_inspector_paths(workspace: Path) -> InspectorPaths:
    """Resolve current V2 sidecars from the active workspace config."""

    config_path = (
        builtin_plugin_data_dir("akasha", workspace)
        / "config.local.toml"
    )
    config = load_akasha_config(config_path)
    return InspectorPaths(
        memory=resolve_workspace_path(workspace, config.db_path),
        index=resolve_workspace_path(workspace, config.index_path),
    )


class AkashaInspectorReader:
    """Expose historical cue, activation, completion, and prompt evidence."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        config_path = (
            builtin_plugin_data_dir("akasha", workspace)
            / "config.local.toml"
        )
        self.config = load_akasha_config(config_path)
        self.paths = InspectorPaths(
            memory=resolve_workspace_path(
                workspace,
                self.config.db_path,
            ),
            index=resolve_workspace_path(
                workspace,
                self.config.index_path,
            ),
        )
        self._dense_lock = threading.RLock()
        self._dense_snapshot: _DenseSnapshot | None = None

    def get_overview(self) -> dict[str, object]:
        """Return the number and time range of committed retrieval events."""

        if not self.paths.memory.exists() and not self.paths.index.exists():
            return {
                "available": True,
                "total": 0,
                "latest_at": None,
                "earliest_at": None,
            }
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    MAX(turn.started_at) AS latest_at,
                    MIN(turn.started_at) AS earliest_at
                FROM memory_events AS event
                JOIN turn_nodes AS turn
                  ON turn.node_id = event.current_turn_node_id
                """
            ).fetchone()
        return {
            "available": True,
            "total": int(row["total"]),
            "latest_at": row["latest_at"],
            "earliest_at": row["earliest_at"],
        }

    def list_turns(
        self,
        *,
        session_key: str = "",
        q: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, object]], int]:
        """List committed retrievals in reverse causal order."""

        # 1. Build filters only from validated HTTP or mobile inputs.
        where: list[str] = []
        values: list[object] = []
        if session_key:
            where.append("turn.session_key = ?")
            values.append(session_key)
        if q:
            where.append(
                "instr(lower(turn.session_key || char(10) || "
                "sparse_turn.user_text || char(10) || "
                "sparse_turn.assistant_text), lower(?)) > 0"
            )
            values.append(q)
        predicate = f"WHERE {' AND '.join(where)}" if where else ""

        # 2. Count and page one stable read-only snapshot.
        with closing(self._connect()) as connection:
            total = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM memory_events AS event
                    JOIN turn_nodes AS turn
                      ON turn.node_id = event.current_turn_node_id
                    JOIN sparse.sparse_turns AS sparse_turn
                      ON sparse_turn.turn_id = turn.turn_id
                    {predicate}
                    """,
                    values,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT
                    turn.turn_id AS query_id,
                    turn.session_key,
                    turn.user_seq AS seq,
                    turn.started_at AS ts,
                    sparse_turn.user_text AS query_text,
                    event.seed_support AS seed_count,
                    COALESCE(
                        activation.completion_support,
                        (
                            SELECT COUNT(*)
                            FROM recall_items AS item
                            WHERE item.query_turn_node_id = turn.node_id
                        )
                    ) AS activation_count,
                    COALESCE(
                        activation.graph_only_support,
                        (
                            SELECT COUNT(*)
                            FROM recall_items AS item
                            WHERE item.query_turn_node_id = turn.node_id
                              AND item.is_pattern_only = 1
                        )
                    ) AS graph_only_count,
                    COALESCE(activation.pushes, event.pushes) AS pushes,
                    COALESCE(activation.residual_l1, event.residual_l1)
                        AS residual_l1,
                    COALESCE(recall.active_basin_count, 0)
                        AS basin_count,
                    recall.query_turn_node_id IS NOT NULL
                        AS recall_capture_available,
                    (
                        SELECT COUNT(*)
                        FROM recall_items AS item
                        WHERE item.query_turn_node_id = turn.node_id
                    ) AS completion_count
                FROM memory_events AS event
                JOIN turn_nodes AS turn
                  ON turn.node_id = event.current_turn_node_id
                JOIN sparse.sparse_turns AS sparse_turn
                  ON sparse_turn.turn_id = turn.turn_id
                LEFT JOIN activation_runs AS activation
                  ON activation.query_turn_node_id = turn.node_id
                LEFT JOIN recall_runs AS recall
                  ON recall.query_turn_node_id = turn.node_id
                {predicate}
                ORDER BY turn.node_id DESC
                LIMIT ? OFFSET ?
                """,
                [*values, page_size, (page - 1) * page_size],
            ).fetchall()
        items = [dict(row) for row in rows]
        for item in items:
            item["recall_capture_available"] = bool(
                item["recall_capture_available"]
            )
        return items, total

    def get_turn(self, query_id: str) -> dict[str, object] | None:
        """Return one retrieval with seeds, paths, prompt lanes, and learning."""

        # 1. Read the persisted causal evidence for this query turn.
        with closing(self._connect()) as connection:
            run = self._load_run(connection, query_id)
            if run is None:
                return None
            node_id = cast(int, run["node_id"])
            seeds = self._load_seeds(connection, node_id)
            activations = self._load_activations(connection, node_id)
            completions = self._load_completions(connection, node_id)
        tool_left, tool_right = _tool_recall_lanes(
            run.pop("tool_chain_json")
        )

        # 2. Reconstruct the host-visible lanes from frozen prior turns.
        dense = self._dense_items(node_id)
        dense_ids = {str(item["query_id"]) for item in dense}
        right = [
            item
            for item in completions
            if str(item["query_id"]) not in dense_ids
        ][: self.config.context_recall_limit]
        dense = _sort_by_time(dense)
        right = _sort_by_time(right)
        text_block = _render_context_block(
            dense,
            right,
            str(run["ts"]),
        )
        if len(text_block) > self.config.inject_max_chars:
            omitted = len(text_block) - self.config.inject_max_chars
            text_block = (
                text_block[: self.config.inject_max_chars].rstrip()
                + f"\n...[Akasha 已截断 {omitted} 字]"
            )
        return {
            **run,
            "seeds": seeds,
            "activation_items": activations,
            "left": dense,
            "right": right,
            "left_count": len(dense),
            "right_count": len(right),
            "tool_left": tool_left,
            "tool_right": tool_right,
            "tool_left_count": len(tool_left),
            "tool_right_count": len(tool_right),
            "inject_chars": len(text_block),
            "text_block_preview": text_block,
        }

    def latest_for_session(
        self,
        session_key: str,
    ) -> dict[str, object] | None:
        """Return the latest committed retrieval for one session."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT turn.turn_id
                FROM memory_events AS event
                JOIN turn_nodes AS turn
                  ON turn.node_id = event.current_turn_node_id
                WHERE turn.session_key = ?
                ORDER BY turn.node_id DESC
                LIMIT 1
                """,
                (session_key,),
            ).fetchone()
        return None if row is None else self.get_turn(str(row["turn_id"]))

    def for_assistant_message(
        self,
        session_key: str,
        message_id: str,
    ) -> dict[str, object] | None:
        """Resolve one persisted assistant message to its retrieval event."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT turn.turn_id
                FROM memory_events AS event
                JOIN turn_nodes AS turn
                  ON turn.node_id = event.current_turn_node_id
                WHERE turn.session_key = ?
                  AND turn.assistant_message_id = ?
                """,
                (session_key, message_id),
            ).fetchone()
        return None if row is None else self.get_turn(str(row["turn_id"]))

    def _connect(self) -> sqlite3.Connection:
        """Open both sidecars read-only for one coherent inspection query."""

        connection = sqlite3.connect(
            f"file:{self.paths.memory}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        _ = connection.execute(
            "ATTACH DATABASE ? AS sparse",
            (f"file:{self.paths.index}?mode=ro",),
        )
        _ = connection.execute(
            "ATTACH DATABASE ? AS sessions",
            (f"file:{self.workspace / 'sessions.db'}?mode=ro",),
        )
        _ = connection.execute("PRAGMA query_only = ON")
        return connection

    @staticmethod
    def _load_run(
        connection: sqlite3.Connection,
        query_id: str,
    ) -> dict[str, object] | None:
        row = connection.execute(
            """
            SELECT
                turn.node_id,
                turn.turn_id AS query_id,
                turn.session_key,
                turn.user_seq AS seq,
                turn.user_message_id,
                turn.assistant_message_id,
                turn.started_at AS ts,
                sparse_turn.user_text AS query_text,
                sparse_turn.assistant_text,
                assistant.tool_chain AS tool_chain_json,
                event.time_prior,
                event.continuation,
                COALESCE(activation.pushes, event.pushes) AS pushes,
                COALESCE(activation.residual_l1, event.residual_l1)
                    AS residual_l1,
                event.seed_support AS seed_count,
                activation.query_turn_node_id IS NOT NULL
                    AS activation_capture_available,
                recall.query_turn_node_id IS NOT NULL
                    AS recall_capture_available,
                COALESCE(
                    activation.completion_support,
                    (
                        SELECT COUNT(*)
                        FROM recall_items AS item
                        WHERE item.query_turn_node_id = turn.node_id
                    )
                ) AS activation_count,
                activation.completion_effective_support,
                activation.completion_mass,
                COALESCE(
                    activation.graph_only_support,
                    (
                        SELECT COUNT(*)
                        FROM recall_items AS item
                        WHERE item.query_turn_node_id = turn.node_id
                          AND item.is_pattern_only = 1
                    )
                ) AS graph_only_count,
                activation.graph_only_effective_support,
                activation.graph_only_mass,
                COALESCE(recall.active_basin_count, 0)
                    AS basin_count,
                COALESCE(recall.sharp_completion_count, 0)
                    AS sharp_completion_count,
                COALESCE(recall.basin_direct_count, 0)
                    AS basin_direct_count,
                COALESCE(recall.basin_completion_count, 0)
                    AS basin_completion_count,
                COALESCE(recall.relative_tail_count, 0)
                    AS relative_tail_count,
                event.surprise,
                event.modification_threshold,
                event.observed_mass,
                event.recurrent_mass,
                event.reactivated_mass,
                event.potentiated_mass,
                event.inhibited_mass
            FROM memory_events AS event
            JOIN turn_nodes AS turn
              ON turn.node_id = event.current_turn_node_id
            JOIN sparse.sparse_turns AS sparse_turn
              ON sparse_turn.turn_id = turn.turn_id
            LEFT JOIN sessions.messages AS assistant
              ON assistant.id = turn.assistant_message_id
            LEFT JOIN activation_runs AS activation
              ON activation.query_turn_node_id = turn.node_id
            LEFT JOIN recall_runs AS recall
              ON recall.query_turn_node_id = turn.node_id
            WHERE turn.turn_id = ?
            """,
            (query_id,),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["activation_capture_available"] = bool(
            item["activation_capture_available"]
        )
        item["recall_capture_available"] = bool(
            item["recall_capture_available"]
        )
        return item

    @staticmethod
    def _load_seeds(
        connection: sqlite3.Connection,
        node_id: int,
    ) -> list[dict[str, object]]:
        rows = connection.execute(
            """
            SELECT
                candidate.turn_id AS query_id,
                candidate.session_key,
                candidate.started_at AS ts,
                sparse_turn.user_text,
                sparse_turn.assistant_text,
                seed.value,
                seed.channels_json
            FROM memory_events AS event
            JOIN event_seeds AS seed
              ON seed.event_id = event.event_id
            JOIN turn_nodes AS candidate
              ON candidate.node_id = seed.candidate_turn_node_id
            JOIN sparse.sparse_turns AS sparse_turn
              ON sparse_turn.turn_id = candidate.turn_id
            WHERE event.current_turn_node_id = ?
            ORDER BY seed.value DESC, candidate.node_id
            """,
            (node_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "assistant_preview": _assistant_preview(
                    str(row["assistant_text"])
                ),
                "channels": _json_list(
                    row["channels_json"],
                    "event_seeds.channels_json",
                ),
            }
            for row in rows
        ]

    @staticmethod
    def _load_activations(
        connection: sqlite3.Connection,
        node_id: int,
    ) -> list[dict[str, object]]:
        rows = connection.execute(
            """
            SELECT
                candidate.turn_id AS query_id,
                candidate.session_key,
                candidate.started_at AS ts,
                sparse_turn.user_text,
                sparse_turn.assistant_text,
                item.rank,
                item.seed_score,
                item.direct_mass,
                item.settled_mass,
                item.completion_mass,
                item.is_graph_only,
                item.first_relation,
                item.dominant_path_json,
                item.relation_path_json
            FROM activation_items AS item
            JOIN turn_nodes AS candidate
              ON candidate.node_id = item.candidate_turn_node_id
            JOIN sparse.sparse_turns AS sparse_turn
              ON sparse_turn.turn_id = candidate.turn_id
            WHERE item.query_turn_node_id = ?
            ORDER BY item.rank
            """,
            (node_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "assistant_preview": _assistant_preview(
                    str(row["assistant_text"])
                ),
                "graph_only": bool(row["is_graph_only"]),
                "dominant_path": _json_list(
                    row["dominant_path_json"],
                    "activation_items.dominant_path_json",
                ),
                "relation_path": _json_list(
                    row["relation_path_json"],
                    "activation_items.relation_path_json",
                ),
            }
            for row in rows
        ]

    @staticmethod
    def _load_completions(
        connection: sqlite3.Connection,
        node_id: int,
    ) -> list[dict[str, object]]:
        rows = connection.execute(
            """
            SELECT
                candidate.turn_id AS query_id,
                candidate.session_key,
                candidate.user_message_id,
                candidate.assistant_message_id,
                candidate.started_at AS ts,
                sparse_turn.user_text,
                sparse_turn.assistant_text,
                item.rank,
                item.score,
                item.sources_json,
                item.basin_ids_json,
                item.is_pattern_only
            FROM recall_items AS item
            JOIN turn_nodes AS candidate
              ON candidate.node_id = item.candidate_turn_node_id
            JOIN sparse.sparse_turns AS sparse_turn
              ON sparse_turn.turn_id = candidate.turn_id
            WHERE item.query_turn_node_id = ?
            ORDER BY item.rank
            """,
            (node_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "assistant_preview": _assistant_preview(
                    str(row["assistant_text"])
                ),
                "sources": _json_list(
                    row["sources_json"],
                    "recall_items.sources_json",
                ),
                "basin_ids": _json_list(
                    row["basin_ids_json"],
                    "recall_items.basin_ids_json",
                ),
                "pattern_only": bool(row["is_pattern_only"]),
            }
            for row in rows
        ]

    def _dense_items(self, query_node_id: int) -> list[dict[str, object]]:
        """Compute the exact prior-only dense top five with vectorized dots."""

        snapshot = self._load_dense_snapshot()
        if query_node_id <= 0 or query_node_id >= len(snapshot.turns):
            return []
        if not snapshot.user_present[query_node_id]:
            return []

        # 1. Score both message fields for every causally prior turn.
        query = snapshot.user[query_node_id]
        user_scores = snapshot.user[:query_node_id] @ query
        assistant_scores = snapshot.assistant[:query_node_id] @ query
        user_scores[~snapshot.user_present[:query_node_id]] = -np.inf
        assistant_scores[
            ~snapshot.assistant_present[:query_node_id]
        ] = -np.inf
        scores = np.maximum(user_scores, assistant_scores)

        # 2. Apply the engine's score-desc, node-id-asc tie break.
        nodes = np.arange(query_node_id)
        ranked = np.lexsort((nodes, -scores))
        selected = [
            int(node)
            for node in ranked
            if math.isfinite(float(scores[node]))
        ][:5]
        return [
            {
                **snapshot.turns[node],
                "score": float(scores[node]),
                "sources": ["direct_dense"],
                "lane": "dense",
            }
            for node in selected
        ]

    def _load_dense_snapshot(self) -> _DenseSnapshot:
        """Reload the vector matrix only after an atomic index replacement."""

        signature = _file_signature(self.paths.index)
        with self._dense_lock:
            current = self._dense_snapshot
            if current is not None and current.signature == signature:
                return current
            loaded = self._read_dense_snapshot(signature)
            self._dense_snapshot = loaded
            return loaded

    def _read_dense_snapshot(
        self,
        signature: tuple[int, int],
    ) -> _DenseSnapshot:
        """Read and normalize all indexed turn vectors once."""

        # 1. Read stable graph order and both indexed message vectors.
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    turn.node_id,
                    turn.turn_id AS query_id,
                    turn.session_key,
                    turn.user_message_id,
                    turn.assistant_message_id,
                    turn.started_at AS ts,
                    sparse_turn.user_text,
                    sparse_turn.assistant_text,
                    user_dense.embedding AS user_embedding,
                    user_dense.dim AS user_dim,
                    assistant_dense.embedding AS assistant_embedding,
                    assistant_dense.dim AS assistant_dim
                FROM turn_nodes AS turn
                JOIN sparse.sparse_turns AS sparse_turn
                  ON sparse_turn.turn_id = turn.turn_id
                LEFT JOIN sparse.turn_dense AS user_dense
                  ON user_dense.turn_id = turn.turn_id
                 AND user_dense.field = 'user'
                LEFT JOIN sparse.turn_dense AS assistant_dense
                  ON assistant_dense.turn_id = turn.turn_id
                 AND assistant_dense.field = 'assistant'
                ORDER BY turn.node_id
                """
            ).fetchall()
        _require_contiguous_nodes(rows)
        dimension = _dense_dimension(rows)

        # 2. Materialize normalized matrices and immutable display rows.
        user, user_present = _dense_matrix(
            rows,
            blob_key="user_embedding",
            dim_key="user_dim",
            dimension=dimension,
        )
        assistant, assistant_present = _dense_matrix(
            rows,
            blob_key="assistant_embedding",
            dim_key="assistant_dim",
            dimension=dimension,
        )
        turns: tuple[dict[str, object], ...] = tuple(
            {
                "query_id": row["query_id"],
                "session_key": row["session_key"],
                "user_message_id": row["user_message_id"],
                "assistant_message_id": row["assistant_message_id"],
                "ts": row["ts"],
                "user_text": row["user_text"],
                "assistant_text": row["assistant_text"],
                "assistant_preview": _assistant_preview(
                    str(row["assistant_text"])
                ),
            }
            for row in rows
        )
        return _DenseSnapshot(
            signature=signature,
            turns=turns,
            user=user,
            assistant=assistant,
            user_present=user_present,
            assistant_present=assistant_present,
        )


def mobile_summary(item: dict[str, object]) -> dict[str, object]:
    """Project one inspector detail into the mobile read-only contract."""

    return {
        "query_id": item["query_id"],
        "query_text": item["query_text"],
        "query_preview": _clip(str(item["query_text"]), 180),
        "ts": item["ts"],
        "seed_count": item["seed_count"],
        "activation_capture_available": item[
            "activation_capture_available"
        ],
        "recall_capture_available": item[
            "recall_capture_available"
        ],
        "activation_count": item["activation_count"],
        "left_count": item["left_count"],
        "right_count": item["right_count"],
        "pushes": item["pushes"],
        "residual_l1": item["residual_l1"],
        "left": item["left"],
        "right": item["right"],
        "tool_left_count": item["tool_left_count"],
        "tool_right_count": item["tool_right_count"],
        "tool_left": item["tool_left"],
        "tool_right": item["tool_right"],
    }


def _tool_recall_lanes(
    raw_tool_chain: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Project persisted recall_memory results without changing graph state."""

    # 1. Decode only successful recall_memory tool results.
    if raw_tool_chain is None:
        return [], []
    chain = json.loads(str(raw_tool_chain))
    if not isinstance(chain, list):
        raise ValueError("assistant tool_chain must encode a JSON array")
    items: list[dict[str, object]] = []
    for step in chain:
        if not isinstance(step, dict):
            raise ValueError("assistant tool_chain step must be an object")
        calls = step.get("calls", [])
        if not isinstance(calls, list):
            raise ValueError("assistant tool_chain calls must be an array")
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("assistant tool call must be an object")
            if call.get("name") != "recall_memory":
                continue
            if call.get("status") != "success":
                continue
            result = call.get("result")
            if not isinstance(result, str):
                raise ValueError("recall_memory result must be JSON text")
            payload = json.loads(result)
            if not isinstance(payload, dict):
                raise ValueError("recall_memory result must encode an object")
            recalled = payload.get("items")
            if not isinstance(recalled, list):
                raise ValueError("recall_memory items must be an array")
            items.extend(_tool_recall_item(item) for item in recalled)

    # 2. Keep the first stable occurrence in each visible lane.
    left = _dedupe_tool_lane(items, "dense")
    right = _dedupe_tool_lane(items, "completion")
    return left, right


def _tool_recall_item(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError("recall_memory item must be an object")
    signals = raw.get("signals")
    if not isinstance(signals, dict):
        raise ValueError("recall_memory item signals must be an object")
    lane = signals.get("lane")
    if lane not in {"dense", "completion"}:
        raise ValueError(f"recall_memory item lane is invalid: {lane}")
    item_id = raw.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("recall_memory item id must be non-empty")
    sources = signals.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("recall_memory item sources must be an array")
    return {
        "query_id": item_id,
        "session_key": raw.get("source_ref", ""),
        "ts": signals.get("started_at", ""),
        "user_text": signals.get("user_text", ""),
        "assistant_preview": signals.get("assistant_preview", ""),
        "score": raw.get("score"),
        "sources": sources,
        "lane": lane,
    }


def _dedupe_tool_lane(
    items: list[dict[str, object]],
    lane: str,
) -> list[dict[str, object]]:
    seen: set[str] = set()
    result: list[dict[str, object]] = []
    for item in items:
        if item["lane"] != lane:
            continue
        item_id = str(item["query_id"])
        if item_id in seen:
            continue
        seen.add(item_id)
        result.append(item)
    return result


def _dense_matrix(
    rows: list[sqlite3.Row],
    *,
    blob_key: str,
    dim_key: str,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(rows), dimension), dtype=np.float32)
    present = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        blob = row[blob_key]
        if blob is None:
            continue
        if int(row[dim_key]) != dimension:
            raise ValueError(f"Akasha inspector dense dimension mismatch: {blob_key}")
        vector = np.frombuffer(blob, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.size != dimension or not math.isfinite(norm) or norm == 0.0:
            raise ValueError(f"Akasha inspector dense vector is invalid: {blob_key}")
        matrix[index] = vector / norm
        present[index] = True
    return matrix, present


def _dense_dimension(rows: list[sqlite3.Row]) -> int:
    dimensions = {
        int(row[key])
        for row in rows
        for key in ("user_dim", "assistant_dim")
        if row[key] is not None
    }
    if len(dimensions) != 1:
        raise ValueError(
            f"Akasha inspector expected one dense dimension: {sorted(dimensions)}"
        )
    return dimensions.pop()


def _require_contiguous_nodes(rows: list[sqlite3.Row]) -> None:
    nodes = [int(row["node_id"]) for row in rows]
    if nodes != list(range(len(rows))):
        raise ValueError("Akasha inspector turn node IDs are not contiguous")


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _json_list(value: object, field: str) -> list[object]:
    if not isinstance(value, str):
        raise ValueError(f"{field} is not JSON text")
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError(f"{field} is not a JSON list")
    return cast(list[object], loaded)


def _assistant_preview(text: str) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= 50 else normalized[:50] + "..."


def _clip(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[:limit] + "..."


def _sort_by_time(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(
        items,
        key=lambda item: (
            _parse_time(str(item["ts"])),
            str(item["query_id"]).encode("utf-8"),
        ),
        reverse=True,
    )


def _render_context_block(
    dense: list[dict[str, object]],
    completion: list[dict[str, object]],
    timestamp: str,
) -> str:
    parts = [
        f"# Akasha memory now={_parse_time(timestamp).astimezone(_LOCAL_TZ):%m-%d}"
    ]
    if dense:
        parts.append(_render_lane("## 左脑记忆：精确回忆", dense))
    if completion:
        parts.append(
            _render_lane(
                "## 右脑联想：潜意识第一反应",
                completion,
            )
        )
    return "\n\n".join(parts) if len(parts) > 1 else ""


def _render_lane(
    title: str,
    items: list[dict[str, object]],
) -> str:
    lines = [title]
    for item in items:
        refs = [
            str(item["user_message_id"]),
            str(item["assistant_message_id"]),
        ]
        timestamp = _parse_time(str(item["ts"])).astimezone(_LOCAL_TZ)
        lines.append(
            f"- user={json.dumps(item['user_text'], ensure_ascii=False)} "
            f"assistant={json.dumps(item['assistant_preview'], ensure_ascii=False)} "
            f"t={timestamp:%m-%d} "
            f"source_ref={json.dumps(refs, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (
        parsed.replace(tzinfo=_LOCAL_TZ)
        if parsed.tzinfo is None
        else parsed
    )
