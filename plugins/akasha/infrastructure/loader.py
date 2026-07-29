"""Load a sparse turn index into one deterministic causal stream."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from ..domain.model import Turn, TurnFeedback
from .sparse_index.schema import INDEX_VERSION


def load_turns(index_path: Path, max_turns: int | None = None) -> list[Turn]:
    """Validate and load committed turns in a total UTC order."""

    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        _validate_index(connection)
        rows = connection.execute(
            """
            SELECT turn_id, session_key, user_seq,
                   user_message_id, assistant_message_id,
                   started_at, committed_at, user_text, assistant_text,
                   remember_targets_json, forget_targets_json,
                   remember_boost
            FROM sparse_turns
            ORDER BY started_at, session_key, user_seq, turn_id
            """
        ).fetchall()
        rows.sort(key=_turn_sort_key)
        if max_turns is not None:
            if max_turns <= 0:
                raise ValueError("max_turns must be positive")
            rows = rows[:max_turns]
        selected = {row["turn_id"] for row in rows}
        dense = _load_dense(connection, selected)
        terms = _load_terms(connection, selected)
    finally:
        connection.close()
    return _materialize_turns(rows, dense, terms)


def _validate_index(connection: sqlite3.Connection) -> None:
    """Validate the read-only sparse-index trust boundary."""

    required = {"metadata", "sparse_turns", "turn_dense", "turn_terms"}
    actual = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
    }
    missing = required - actual
    if missing:
        raise ValueError(f"sparse index is missing tables: {sorted(missing)}")
    row = connection.execute(
        "SELECT value FROM metadata WHERE key='index_version'"
    ).fetchone()
    if row is None or str(row[0]) != INDEX_VERSION:
        actual_version = None if row is None else str(row[0])
        raise ValueError(
            f"unsupported sparse index version: {actual_version}"
        )


def _load_dense(
    connection: sqlite3.Connection,
    selected: set[str],
) -> dict[tuple[str, str], np.ndarray]:
    dense: dict[tuple[str, str], np.ndarray] = {}
    rows = connection.execute(
        """
        SELECT turn_id, field, embedding, dim
        FROM turn_dense
        ORDER BY turn_id, field
        """
    )
    for row in rows:
        if row["turn_id"] not in selected:
            continue
        vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        if vector.size != row["dim"] or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid embedding: {row['turn_id']}:{row['field']}")
        dense[(row["turn_id"], row["field"])] = _unit(vector)
    return dense


def _load_terms(
    connection: sqlite3.Connection,
    selected: set[str],
) -> dict[tuple[str, str], tuple[tuple[str, int], ...]]:
    grouped: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    rows = connection.execute(
        """
        SELECT turn_id, field, term, tf
        FROM turn_terms
        ORDER BY turn_id, field, term
        """
    )
    for row in rows:
        if row["turn_id"] in selected:
            grouped[(row["turn_id"], row["field"])].append(
                (row["term"], row["tf"])
            )
    return {key: tuple(values) for key, values in grouped.items()}


def _materialize_turns(
    rows: list[sqlite3.Row],
    dense: dict[tuple[str, str], np.ndarray],
    terms: dict[tuple[str, str], tuple[tuple[str, int], ...]],
) -> list[Turn]:
    """Attach validated features and derive global causal gaps."""

    turns: list[Turn] = []
    node_by_turn = {
        str(row["turn_id"]): node_id
        for node_id, row in enumerate(rows)
    }
    previous: datetime | None = None
    for node_id, row in enumerate(rows):
        started = _as_utc(row["started_at"])
        gap = None if previous is None else (started - previous).total_seconds()
        if gap is not None and gap < 0.0:
            raise ValueError("causal turn order produced a negative gap")
        turn_id = row["turn_id"]
        user_dense = dense.get((turn_id, "user"))
        assistant_dense = dense.get((turn_id, "assistant"))
        remember_nodes = _feedback_nodes(
            row["remember_targets_json"],
            node_by_turn,
            turn_id,
            "remember",
        )
        forget_nodes = _feedback_nodes(
            row["forget_targets_json"],
            node_by_turn,
            turn_id,
            "forget",
        )
        turns.append(
            Turn(
                node_id=node_id,
                turn_id=turn_id,
                session_key=row["session_key"],
                user_seq=row["user_seq"],
                user_message_id=row["user_message_id"],
                assistant_message_id=row["assistant_message_id"],
                started_at=row["started_at"],
                committed_at=row["committed_at"],
                user_text=row["user_text"],
                assistant_text=row["assistant_text"],
                user_dense=user_dense,
                assistant_dense=assistant_dense,
                user_terms=terms.get((turn_id, "user"), ()),
                assistant_terms=terms.get((turn_id, "assistant"), ()),
                inter_gap_seconds=gap,
                feedback=TurnFeedback(
                    remember_nodes=remember_nodes,
                    forget_nodes=forget_nodes,
                    remember_boost=float(row["remember_boost"]),
                ),
            )
        )
        previous = started
    return turns


def _feedback_nodes(
    raw: str,
    node_by_turn: dict[str, int],
    carrier_turn_id: str,
    action: str,
) -> tuple[int, ...]:
    targets = json.loads(raw)
    if (
        not isinstance(targets, list)
        or any(not isinstance(target, str) for target in targets)
    ):
        raise ValueError(
            f"invalid {action} targets at turn {carrier_turn_id}"
        )
    try:
        return tuple(node_by_turn[target] for target in targets)
    except KeyError as error:
        raise ValueError(
            f"missing {action} target at turn {carrier_turn_id}: {error.args[0]}"
        ) from error


def _turn_sort_key(row: sqlite3.Row) -> tuple[datetime, bytes, int, bytes]:
    return (
        _as_utc(row["started_at"]),
        row["session_key"].encode("utf-8"),
        row["user_seq"],
        row["turn_id"].encode("utf-8"),
    )


def _as_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return parsed.astimezone(timezone.utc)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("dense vector must be finite and non-zero")
    return vector / norm
