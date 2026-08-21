"""Write deterministic memory state and completion traces to SQLite."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.features import effective_support
from ..domain.graph import (
    MEMBERSHIP,
    TEMPORAL_BACKWARD,
    TEMPORAL_FORWARD,
    DynamicMemoryGraph,
    HubRecord,
)
from ..domain.model import (
    Capture,
    ContextState,
    MemoryConfig,
    PlasticityResult,
    SeedEvidence,
    Turn,
)
from ..domain.readout import (
    PatternCompletion,
    RecallCapture,
    RecallItem,
)


def write_memory_database(
    output_path: Path,
    *,
    turns: list[Turn],
    graph: DynamicMemoryGraph,
    events: list[PlasticityResult],
    evidence: list[SeedEvidence],
    captures: list[Capture],
    context: ContextState,
    burst_members: dict[str, list[int]],
    config: MemoryConfig,
    metadata: dict[str, str],
    recalls: list[RecallCapture] | tuple[RecallCapture, ...] = (),
) -> str:
    """Write a fresh deterministic database and return its SHA-256."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        _initialize(connection)
        _write_metadata(connection, metadata, turns, graph, config)
        _write_turns(connection, turns)
        _write_feedback_events(connection, turns)
        _write_graph(connection, graph)
        _write_events(connection, events, evidence)
        _write_captures(connection, turns, graph, captures, config.restart)
        _write_recalls(connection, recalls)
        _write_context(connection, context)
        _write_burst_members(connection, burst_members)
        connection.commit()
        _verify(connection)
        connection.execute("VACUUM")
    finally:
        connection.close()
    os.replace(temporary, output_path)
    return sha256_file(output_path)


def load_memory_state(
    memory_path: Path,
    *,
    turns: list[Turn],
    config: MemoryConfig,
    source_index_sha256: str | None,
    source_index_state_sha256: str | None = None,
) -> tuple[
    DynamicMemoryGraph,
    list[PlasticityResult],
    list[SeedEvidence],
    ContextState,
    list[RecallCapture],
    dict[str, list[int]],
]:
    """Restore a validated graph snapshot without replaying historical turns."""

    # 1. Validate schema, source identity, config, and turn bindings.
    connection = sqlite3.connect(
        f"file:{memory_path}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        _verify(connection)
        _validate_snapshot_identity(
            connection,
            turns,
            config,
            source_index_sha256,
            source_index_state_sha256,
        )

        # 2. Restore graph arrays and learned clocks exactly.
        graph = _load_graph(
            connection,
            _graph_turn_capacity(connection),
            len(turns),
            config,
        )
        events = _load_events(connection)
        evidence = _load_evidence(connection, len(events))
        context = _load_context(connection, turns)
        recalls = _load_recalls(connection)
        burst_members = _load_burst_members(connection)
        return (
            graph,
            events,
            evidence,
            context,
            recalls,
            burst_members,
        )
    finally:
        connection.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_state_sha256(path: Path) -> str:
    """Hash canonical learned state independently of SQLite file layout."""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        digest = hashlib.sha256()
        for table in _LOGICAL_STATE_TABLES:
            columns = [
                row[1]
                for row in connection.execute(
                    f'PRAGMA table_info("{table}")'
                )
            ]
            if not columns:
                raise ValueError(f"logical state table is missing: {table}")
            order = ", ".join(f'"{column}"' for column in columns)
            rows = connection.execute(
                f'SELECT * FROM "{table}" ORDER BY {order}'
            )
            digest.update(table.encode("utf-8") + b"\0")
            for row in rows:
                digest.update(
                    canonical_json(
                        [_logical_value(value) for value in row]
                    ).encode("utf-8")
                )
                digest.update(b"\n")
        return digest.hexdigest()
    finally:
        connection.close()


def _logical_value(value: object) -> object:
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    return value


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _initialize(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA page_size = 4096")
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_SCHEMA)


def _write_metadata(
    connection: sqlite3.Connection,
    metadata: dict[str, str],
    turns: list[Turn],
    graph: DynamicMemoryGraph,
    config: MemoryConfig,
) -> None:
    values = {
        **metadata,
        "config_json": canonical_json(asdict(config)),
        "engine": "single_state_empirical_recurrence_survival_v9_feedback",
        "graph_turn_capacity": str(graph.turn_count),
        "hub_count": str(len(graph.hubs)),
        "relation_count": str(len(graph.source)),
        "session_count": str(len({turn.session_key for turn in turns})),
        "turn_count": str(len(turns)),
    }
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        sorted(values.items()),
    )


def _write_turns(connection: sqlite3.Connection, turns: list[Turn]) -> None:
    connection.executemany(
        """
        INSERT INTO turn_nodes
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                turn.node_id,
                turn.turn_id,
                turn.session_key,
                turn.user_seq,
                turn.user_message_id,
                turn.assistant_message_id,
                turn.started_at,
                turn.committed_at,
                turn.inter_gap_seconds,
            )
            for turn in turns
        ],
    )


def _write_feedback_events(
    connection: sqlite3.Connection,
    turns: list[Turn],
) -> None:
    """Persist the canonical marker inputs that produced graph control."""

    rows = []
    for turn in turns:
        rows.extend(
            (
                turn.node_id,
                "remember",
                target,
                turn.feedback.remember_boost,
            )
            for target in turn.feedback.remember_nodes
        )
        rows.extend(
            (turn.node_id, "forget", target, 1.0)
            for target in turn.feedback.forget_nodes
        )
    connection.executemany(
        """
        INSERT INTO feedback_events(
            event_id, action, target_turn_node_id, boost
        ) VALUES (?, ?, ?, ?)
        """,
        rows,
    )


def _write_events(
    connection: sqlite3.Connection,
    events: list[PlasticityResult],
    evidence: list[SeedEvidence],
) -> None:
    connection.executemany(
        """
        INSERT INTO memory_events
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                event_id,
                event_id,
                item.hub_node_id,
                seed.time_prior,
                seed.continuation,
                seed.surprise,
                len(seed.seed),
                item.pushes,
                item.residual_l1,
                item.threshold,
                item.observed_mass,
                item.recurrent_mass,
                item.reactivated_mass,
                item.potentiated_mass,
                item.inhibited_mass,
                canonical_json(item.integrated),
            )
            for event_id, (item, seed) in enumerate(zip(events, evidence))
        ],
    )
    rows: list[tuple[int, int, float, str]] = []
    support_rows: list[tuple[int, str, int]] = []
    channel_rows: list[tuple[int, str]] = []
    for event_id, seed in enumerate(evidence):
        for node_id, value in seed.seed:
            channels = sorted(
                name for name, members in seed.channels.items() if node_id in members
            )
            rows.append((event_id, node_id, value, canonical_json(channels)))
        support_rows.extend(
            (event_id, name, node_id)
            for name, members in sorted(seed.channels.items())
            for node_id in sorted(members)
        )
        channel_rows.extend(
            (event_id, name)
            for name in sorted(seed.channels)
        )
    connection.executemany("INSERT INTO event_seeds VALUES (?, ?, ?, ?)", rows)
    connection.executemany(
        "INSERT INTO event_channel_support VALUES (?, ?, ?)",
        support_rows,
    )
    connection.executemany(
        "INSERT INTO event_channels VALUES (?, ?)",
        channel_rows,
    )


def _write_graph(
    connection: sqlite3.Connection,
    graph: DynamicMemoryGraph,
) -> None:
    connection.executemany(
        "INSERT INTO hub_nodes VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                hub.node_id,
                hub.created_event,
                hub.current_turn_id,
                hub.threshold,
                hub.innovation_mass,
                len(hub.member_edge_ids),
            )
            for hub in graph.hubs
        ],
    )
    membership_rows = []
    temporal_rows = []
    for edge_id in range(len(graph.source)):
        common = (
            edge_id,
            graph.source[edge_id],
            graph.target[edge_id],
            graph.weight[edge_id],
            graph.effective_weight(edge_id),
            graph.last_updated[edge_id],
            graph.created_event[edge_id],
            graph.observed_credit[edge_id],
            graph.recurrent_credit[edge_id],
            graph.support_credit[edge_id],
            graph.independent_credit[edge_id],
            graph.last_support_seconds[edge_id],
            graph.resource[edge_id],
            graph.plasticity_threshold[edge_id],
            graph.last_stimulated_seconds[edge_id],
        )
        if graph.kind[edge_id] == MEMBERSHIP:
            membership_rows.append(common)
        else:
            temporal_rows.append(
                (
                    edge_id,
                    graph.edge_relation_name(edge_id),
                    *common[1:],
                )
            )
    connection.executemany(
        "INSERT INTO hub_memberships VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        membership_rows,
    )
    connection.executemany(
        "INSERT INTO temporal_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        temporal_rows,
    )
    connection.execute(
        """
        INSERT INTO plasticity_clock
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            graph.elapsed_seconds,
            graph.short_log_gap,
            graph.long_log_gap,
            graph.resource_tau_seconds,
            graph.threshold_tau_seconds,
            graph.retention_tau_seconds,
            graph.short_recurrence_log_gap,
            graph.long_recurrence_log_gap,
            graph.short_recurrence_tau_seconds,
            graph.long_recurrence_tau_seconds,
            graph.short_gap_count,
            graph.long_gap_count,
            graph.short_recurrence_weight,
            graph.long_recurrence_weight,
            graph.short_recurrence_log_m2,
            graph.long_recurrence_log_m2,
            graph.recurrence_log_mean,
            graph.recurrence_log_m2,
            graph.recurrence_weight,
        ),
    )
    connection.executemany(
        "INSERT INTO external_seed_state VALUES (?, ?)",
        sorted(graph.last_external_seed_seconds.items()),
    )


def _write_captures(
    connection: sqlite3.Connection,
    turns: list[Turn],
    graph: DynamicMemoryGraph,
    captures: list[Capture],
    restart: float,
) -> None:
    for capture in sorted(captures, key=lambda item: item.query_node_id):
        _write_capture(connection, turns, graph, capture, restart)


def _write_recalls(
    connection: sqlite3.Connection,
    recalls: list[RecallCapture] | tuple[RecallCapture, ...],
) -> None:
    for capture in sorted(recalls, key=lambda item: item.query_node_id):
        completion = capture.completion
        connection.execute(
            "INSERT INTO recall_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                capture.query_node_id,
                completion.active_basin_count,
                completion.sharp_completion_count,
                completion.basin_direct_count,
                completion.basin_completion_count,
                completion.relative_tail_count,
                completion.pushes,
                completion.residual_l1,
            ),
        )
        connection.executemany(
            "INSERT INTO recall_items VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    capture.query_node_id,
                    item.node_id,
                    rank,
                    item.score,
                    canonical_json(item.sources),
                    canonical_json(item.basin_ids),
                    int("sharp_completion" not in item.sources),
                )
                for rank, item in enumerate(completion.items, start=1)
            ],
        )


def _write_capture(
    connection: sqlite3.Connection,
    turns: list[Turn],
    graph: DynamicMemoryGraph,
    capture: Capture,
    restart: float,
) -> None:
    seed = dict(capture.evidence.seed)
    reserve = capture.diffusion.reserve[: len(turns)]
    completion = reserve.copy()
    for node_id, value in seed.items():
        completion[node_id] = max(0.0, completion[node_id] - restart * value)
    active = np.flatnonzero(completion > 0.0)
    graph_only = active[
        np.fromiter(
            (int(node) not in seed for node in active),
            dtype=bool,
            count=active.size,
        )
    ]
    graph_only_completion = completion[graph_only]
    ranked = sorted(active, key=lambda node: (-completion[node], int(node)))
    connection.execute(
        """
        INSERT INTO activation_runs
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            capture.query_node_id,
            capture.evidence.time_prior,
            capture.evidence.continuation,
            capture.diffusion.pushes,
            capture.diffusion.residual_l1,
            len(seed),
            int(active.size),
            effective_support(completion),
            float(np.sum(completion)),
            int(graph_only.size),
            effective_support(graph_only_completion),
            float(np.sum(graph_only_completion)),
        ),
    )
    rows = []
    for rank, raw_node in enumerate(ranked, start=1):
        node_id = int(raw_node)
        path, relations = _dominant_path(graph, capture, node_id)
        rows.append(
            (
                capture.query_node_id,
                node_id,
                rank,
                seed.get(node_id, 0.0),
                restart * seed.get(node_id, 0.0),
                float(reserve[node_id]),
                float(completion[node_id]),
                int(node_id not in seed),
                relations[0] if relations else None,
                canonical_json(path),
                canonical_json(relations),
            )
        )
    connection.executemany(
        "INSERT INTO activation_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _dominant_path(
    graph: DynamicMemoryGraph,
    capture: Capture,
    start: int,
) -> tuple[list[int], list[str]]:
    parents = capture.diffusion.parent_node
    edges = capture.diffusion.parent_edge
    if parents is None or edges is None:
        return [start], []
    path = [start]
    relations: list[str] = []
    seen = {start}
    node = start
    for _ in range(32):
        parent = int(parents[node])
        edge_id = int(edges[node])
        if parent < 0 or parent in seen:
            break
        if edge_id >= 0:
            relations.append(graph.edge_relation_name(edge_id))
        path.append(parent)
        seen.add(parent)
        node = parent
    return path, relations


def _write_context(connection: sqlite3.Connection, context: ContextState) -> None:
    dense = None if context.dense is None else context.dense.astype(np.float64).tobytes()
    connection.execute(
        "INSERT INTO context_state VALUES (1, ?, ?, ?)",
        (
            canonical_json(context.members),
            dense,
            canonical_json(context.terms),
        ),
    )


def _write_burst_members(
    connection: sqlite3.Connection,
    burst_members: dict[str, list[int]],
) -> None:
    connection.executemany(
        "INSERT INTO burst_context_members VALUES (?, ?, ?)",
        [
            (session, position, node_id)
            for session, members in sorted(burst_members.items())
            for position, node_id in enumerate(members)
        ],
    )


def _verify(connection: sqlite3.Connection) -> None:
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise sqlite3.IntegrityError(f"foreign key violations: {violations[:3]}")


def _validate_snapshot_identity(
    connection: sqlite3.Connection,
    turns: list[Turn],
    config: MemoryConfig,
    source_index_sha256: str | None,
    source_index_state_sha256: str | None,
) -> None:
    metadata = dict(
        connection.execute(
            "SELECT key, value FROM metadata ORDER BY key"
        )
    )
    expected = {
        "config_json": canonical_json(asdict(config)),
        "turn_count": str(len(turns)),
    }
    if source_index_sha256 is not None:
        expected["source_index_sha256"] = source_index_sha256
    if source_index_state_sha256 is not None:
        expected["source_index_state_sha256"] = source_index_state_sha256
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"memory snapshot identity mismatch: {mismatches}")
    rows = connection.execute(
        """
        SELECT node_id, turn_id, user_message_id, assistant_message_id
        FROM turn_nodes
        ORDER BY node_id
        """
    ).fetchall()
    actual = tuple(
        (
            row["node_id"],
            row["turn_id"],
            row["user_message_id"],
            row["assistant_message_id"],
        )
        for row in rows
    )
    wanted = tuple(
        (
            turn.node_id,
            turn.turn_id,
            turn.user_message_id,
            turn.assistant_message_id,
        )
        for turn in turns
    )
    if actual != wanted:
        raise ValueError("memory snapshot turn bindings differ from source index")
    _validate_feedback_bindings(connection, turns)


def _validate_feedback_bindings(
    connection: sqlite3.Connection,
    turns: list[Turn],
) -> None:
    actual = tuple(
        tuple(row)
        for row in connection.execute(
            """
            SELECT event_id, action, target_turn_node_id, boost
            FROM feedback_events
            ORDER BY event_id, action, target_turn_node_id
            """
        )
    )
    wanted_rows = []
    for turn in turns:
        wanted_rows.extend(
            (
                turn.node_id,
                "remember",
                target,
                turn.feedback.remember_boost,
            )
            for target in turn.feedback.remember_nodes
        )
        wanted_rows.extend(
            (turn.node_id, "forget", target, 1.0)
            for target in turn.feedback.forget_nodes
        )
    wanted = tuple(
        sorted(
            wanted_rows,
            key=lambda row: (row[0], row[1], row[2]),
        )
    )
    if actual != wanted:
        raise ValueError(
            "memory snapshot feedback bindings differ from source index"
        )


def memory_turn_count(memory_path: Path) -> int:
    """Read the persisted turn count needed for crash recovery."""

    connection = sqlite3.connect(
        f"file:{memory_path}?mode=ro",
        uri=True,
    )
    try:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='turn_count'"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("memory snapshot is missing turn_count metadata")
    count = int(row[0])
    if count <= 0:
        raise ValueError("memory snapshot turn_count must be positive")
    return count


def memory_has_source_index_state(memory_path: Path) -> bool:
    """Report whether a snapshot uses the logical sparse-index identity."""

    connection = sqlite3.connect(
        f"file:{memory_path}?mode=ro",
        uri=True,
    )
    try:
        row = connection.execute(
            "SELECT 1 FROM metadata WHERE key='source_index_state_sha256'"
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def _load_graph(
    connection: sqlite3.Connection,
    turn_capacity: int,
    event_count: int,
    config: MemoryConfig,
) -> DynamicMemoryGraph:
    if turn_capacity < event_count:
        raise ValueError("persisted graph capacity is smaller than turns")
    graph = DynamicMemoryGraph(turn_capacity, config)
    rows = _edge_rows(connection)
    for expected_edge, row in enumerate(rows):
        if row["edge_id"] != expected_edge:
            raise ValueError("memory edge IDs must be contiguous")
        edge_id = graph._add_edge(  # noqa: SLF001
            source=row["source_node_id"],
            target=row["target_node_id"],
            kind=row["kind"],
            bidirectional=bool(row["bidirectional"]),
            event=row["created_event"],
            initial_weight=row["weight"],
            observed_credit=row["observed_credit"],
            recurrent_credit=row["recurrent_credit"],
            support_credit=row["support_credit"],
            independent_credit=row["independent_credit"],
        )
        _restore_edge_state(graph, edge_id, row)
        if row["kind"] != MEMBERSHIP:
            graph.temporal_lookup[
                (
                    row["kind"],
                    row["source_node_id"],
                    row["target_node_id"],
                )
            ] = edge_id
    _restore_hubs(connection, graph)
    _restore_clock(connection, graph, event_count)
    return graph


def _graph_turn_capacity(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key='graph_turn_capacity'"
    ).fetchone()
    if row is None:
        raise ValueError("memory snapshot is missing graph capacity")
    return int(row[0])


def _edge_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in connection.execute(
        "SELECT * FROM hub_memberships ORDER BY edge_id"
    ):
        item = dict(row)
        item["source_node_id"] = item.pop("turn_node_id")
        item["target_node_id"] = item.pop("hub_node_id")
        item["kind"] = MEMBERSHIP
        item["bidirectional"] = 1
        rows.append(item)
    relation_kind = {
        "temporal_forward": TEMPORAL_FORWARD,
        "temporal_backward": TEMPORAL_BACKWARD,
    }
    for row in connection.execute(
        "SELECT * FROM temporal_edges ORDER BY edge_id"
    ):
        item = dict(row)
        relation = item.pop("relation_type")
        if relation not in relation_kind:
            raise ValueError(f"unknown temporal relation: {relation}")
        item["kind"] = relation_kind[relation]
        item["bidirectional"] = 0
        rows.append(item)
    return sorted(rows, key=lambda item: int(item["edge_id"]))


def _restore_edge_state(
    graph: DynamicMemoryGraph,
    edge_id: int,
    row: dict[str, Any],
) -> None:
    graph.weight[edge_id] = float(row["weight"])
    graph.last_updated[edge_id] = int(row["last_updated_event"])
    graph.created_event[edge_id] = int(row["created_event"])
    graph.observed_credit[edge_id] = float(row["observed_credit"])
    graph.recurrent_credit[edge_id] = float(row["recurrent_credit"])
    graph.support_credit[edge_id] = float(row["support_credit"])
    graph.independent_credit[edge_id] = float(row["independent_credit"])
    graph.last_support_seconds[edge_id] = float(
        row["last_support_seconds"]
    )
    graph.resource[edge_id] = float(row["resource"])
    graph.plasticity_threshold[edge_id] = float(
        row["plasticity_threshold"]
    )
    graph.last_stimulated_seconds[edge_id] = float(
        row["last_stimulated_seconds"]
    )


def _restore_hubs(
    connection: sqlite3.Connection,
    graph: DynamicMemoryGraph,
) -> None:
    memberships: dict[int, list[int]] = {}
    for edge_id, target in enumerate(graph.target):
        if graph.kind[edge_id] == MEMBERSHIP:
            memberships.setdefault(target, []).append(edge_id)
    for row in connection.execute(
        "SELECT * FROM hub_nodes ORDER BY node_id"
    ):
        edge_ids = tuple(memberships.get(row["node_id"], ()))
        if len(edge_ids) != row["member_count"]:
            raise ValueError(
                f"hub member count mismatch: {row['node_id']}"
            )
        graph.hub_members[row["node_id"]] = list(edge_ids)
        graph.hubs.append(
            HubRecord(
                node_id=row["node_id"],
                created_event=row["created_event"],
                current_turn_id=row["current_turn_node_id"],
                threshold=row["threshold"],
                innovation_mass=row["innovation_mass"],
                member_edge_ids=edge_ids,
            )
        )


def _restore_clock(
    connection: sqlite3.Connection,
    graph: DynamicMemoryGraph,
    turn_count: int,
) -> None:
    row = connection.execute(
        "SELECT * FROM plasticity_clock WHERE singleton=1"
    ).fetchone()
    if row is None:
        raise ValueError("memory snapshot is missing plasticity clock")
    graph.elapsed_seconds = row["elapsed_seconds"]
    graph.short_log_gap = row["short_log_gap"]
    graph.long_log_gap = row["long_log_gap"]
    graph.short_gap_count = row["short_gap_count"]
    graph.long_gap_count = row["long_gap_count"]
    graph.short_recurrence_log_gap = row[
        "short_recurrence_log_gap"
    ]
    graph.long_recurrence_log_gap = row[
        "long_recurrence_log_gap"
    ]
    graph.short_recurrence_log_m2 = row[
        "short_recurrence_log_m2"
    ]
    graph.long_recurrence_log_m2 = row[
        "long_recurrence_log_m2"
    ]
    graph.short_recurrence_weight = row["short_recurrence_weight"]
    graph.long_recurrence_weight = row["long_recurrence_weight"]
    graph.recurrence_log_mean = row["recurrence_log_mean"]
    graph.recurrence_log_m2 = row["recurrence_log_m2"]
    graph.recurrence_weight = row["recurrence_weight"]
    graph.last_external_seed_seconds = dict(
        connection.execute(
            "SELECT turn_node_id, last_seed_seconds "
            "FROM external_seed_state ORDER BY turn_node_id"
        )
    )
    graph.current_event = turn_count - 1
    graph.current_external = {}


def _load_events(
    connection: sqlite3.Connection,
) -> list[PlasticityResult]:
    return [
        PlasticityResult(
            hub_node_id=row["hub_node_id"],
            threshold=row["modification_threshold"],
            integrated=tuple(
                (int(node), float(value))
                for node, value in json.loads(row["integrated_json"])
            ),
            inhibited_mass=row["inhibited_mass"],
            potentiated_mass=row["potentiated_mass"],
            observed_mass=row["observed_mass"],
            recurrent_mass=row["recurrent_mass"],
            reactivated_mass=row["reactivated_mass"],
            pushes=row["pushes"],
            residual_l1=row["residual_l1"],
        )
        for row in connection.execute(
            "SELECT * FROM memory_events ORDER BY event_id"
        )
    ]


def _load_recalls(
    connection: sqlite3.Connection,
) -> list[RecallCapture]:
    items: dict[int, list[RecallItem]] = {}
    for row in connection.execute(
        "SELECT * FROM recall_items "
        "ORDER BY query_turn_node_id, rank"
    ):
        items.setdefault(row["query_turn_node_id"], []).append(
            RecallItem(
                node_id=row["candidate_turn_node_id"],
                score=row["score"],
                sources=tuple(json.loads(row["sources_json"])),
                basin_ids=tuple(json.loads(row["basin_ids_json"])),
            )
        )
    return [
        RecallCapture(
            query_node_id=row["query_turn_node_id"],
            completion=PatternCompletion(
                items=tuple(items.get(row["query_turn_node_id"], ())),
                active_basin_count=row["active_basin_count"],
                sharp_completion_count=row["sharp_completion_count"],
                basin_direct_count=row["basin_direct_count"],
                basin_completion_count=row["basin_completion_count"],
                relative_tail_count=row["relative_tail_count"],
                pushes=row["pushes"],
                residual_l1=row["residual_l1"],
            ),
        )
        for row in connection.execute(
            "SELECT * FROM recall_runs ORDER BY query_turn_node_id"
        )
    ]


def _load_burst_members(
    connection: sqlite3.Connection,
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for row in connection.execute(
        "SELECT * FROM burst_context_members "
        "ORDER BY session_key, position"
    ):
        result.setdefault(row["session_key"], []).append(
            row["turn_node_id"]
        )
    return result


def _load_evidence(
    connection: sqlite3.Connection,
    event_count: int,
) -> list[SeedEvidence]:
    seeds: dict[int, list[tuple[int, float]]] = {}
    channels: dict[int, dict[str, set[int]]] = {}
    for row in connection.execute(
        "SELECT event_id, channel FROM event_channels "
        "ORDER BY event_id, channel"
    ):
        channels.setdefault(row["event_id"], {})[row["channel"]] = set()
    for row in connection.execute(
        "SELECT * FROM event_seeds ORDER BY event_id, candidate_turn_node_id"
    ):
        event_id = row["event_id"]
        node_id = row["candidate_turn_node_id"]
        seeds.setdefault(event_id, []).append((node_id, row["value"]))
    for row in connection.execute(
        """
        SELECT event_id, channel, candidate_turn_node_id
        FROM event_channel_support
        ORDER BY event_id, channel, candidate_turn_node_id
        """
    ):
        channels.setdefault(row["event_id"], {}).setdefault(
            row["channel"],
            set(),
        ).add(row["candidate_turn_node_id"])
    rows = connection.execute(
        """
        SELECT event_id, time_prior, continuation, surprise
        FROM memory_events
        ORDER BY event_id
        """
    ).fetchall()
    if len(rows) != event_count:
        raise ValueError("memory evidence count differs from events")
    return [
        SeedEvidence(
            seed=tuple(seeds.get(row["event_id"], ())),
            channels={
                name: frozenset(nodes)
                for name, nodes in sorted(
                    channels.get(row["event_id"], {}).items()
                )
            },
            time_prior=row["time_prior"],
            continuation=row["continuation"],
            surprise=row["surprise"],
        )
        for row in rows
    ]


def _load_context(
    connection: sqlite3.Connection,
    turns: list[Turn],
) -> ContextState:
    row = connection.execute(
        "SELECT * FROM context_state WHERE singleton=1"
    ).fetchone()
    if row is None:
        raise ValueError("memory snapshot is missing context state")
    dense = None
    if row["dense"] is not None:
        dense = np.frombuffer(row["dense"], dtype=np.float64).copy()
        expected = next(
            vector.size
            for turn in turns
            for vector in (turn.user_dense, turn.assistant_dense)
            if vector is not None
        )
        if dense.size != expected or not np.all(np.isfinite(dense)):
            raise ValueError("memory context dense vector is invalid")
    return ContextState(
        members=tuple(
            (int(node), float(value))
            for node, value in json.loads(row["members_json"])
        ),
        dense=dense,
        terms=tuple(
            (str(term), float(value))
            for term, value in json.loads(row["terms_json"])
        ),
    )


_LOGICAL_STATE_TABLES = (
    "turn_nodes",
    "feedback_events",
    "hub_nodes",
    "hub_memberships",
    "temporal_edges",
    "plasticity_clock",
    "external_seed_state",
    "memory_events",
    "event_seeds",
    "event_channel_support",
    "event_channels",
    "burst_context_members",
    "context_state",
)


_SCHEMA = """
PRAGMA application_id = 1095452754;
PRAGMA user_version = 2;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE turn_nodes (
    node_id INTEGER PRIMARY KEY,
    turn_id TEXT NOT NULL UNIQUE,
    session_key TEXT NOT NULL,
    user_seq INTEGER NOT NULL,
    user_message_id TEXT NOT NULL UNIQUE,
    assistant_message_id TEXT NOT NULL UNIQUE,
    started_at TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    inter_gap_seconds REAL
);

CREATE TABLE feedback_events (
    event_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    action TEXT NOT NULL CHECK(action IN ('remember', 'forget')),
    target_turn_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    boost REAL NOT NULL CHECK(boost >= 1.0 AND boost <= 3.0),
    PRIMARY KEY (event_id, action, target_turn_node_id)
) WITHOUT ROWID;

CREATE TABLE hub_nodes (
    node_id INTEGER PRIMARY KEY,
    created_event INTEGER NOT NULL,
    current_turn_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    threshold REAL NOT NULL,
    innovation_mass REAL NOT NULL,
    member_count INTEGER NOT NULL
);

CREATE TABLE hub_memberships (
    edge_id INTEGER PRIMARY KEY,
    turn_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    hub_node_id INTEGER NOT NULL REFERENCES hub_nodes(node_id),
    weight REAL NOT NULL CHECK(weight >= 0.0),
    effective_weight REAL NOT NULL CHECK(effective_weight >= 0.0),
    last_updated_event INTEGER NOT NULL,
    created_event INTEGER NOT NULL,
    observed_credit REAL NOT NULL,
    recurrent_credit REAL NOT NULL,
    support_credit REAL NOT NULL,
    independent_credit REAL NOT NULL,
    last_support_seconds REAL NOT NULL,
    resource REAL NOT NULL CHECK(resource >= 0.0 AND resource <= 1.0),
    plasticity_threshold REAL NOT NULL
        CHECK(plasticity_threshold >= 0.0 AND plasticity_threshold <= 1.0),
    last_stimulated_seconds REAL NOT NULL
);

CREATE INDEX hub_memberships_turn ON hub_memberships(turn_node_id, edge_id);

CREATE TABLE temporal_edges (
    edge_id INTEGER PRIMARY KEY,
    relation_type TEXT NOT NULL,
    source_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    target_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    weight REAL NOT NULL CHECK(weight >= 0.0),
    effective_weight REAL NOT NULL CHECK(effective_weight >= 0.0),
    last_updated_event INTEGER NOT NULL,
    created_event INTEGER NOT NULL,
    observed_credit REAL NOT NULL,
    recurrent_credit REAL NOT NULL,
    support_credit REAL NOT NULL,
    independent_credit REAL NOT NULL,
    last_support_seconds REAL NOT NULL,
    resource REAL NOT NULL CHECK(resource >= 0.0 AND resource <= 1.0),
    plasticity_threshold REAL NOT NULL
        CHECK(plasticity_threshold >= 0.0 AND plasticity_threshold <= 1.0),
    last_stimulated_seconds REAL NOT NULL
);

CREATE INDEX temporal_edges_source
    ON temporal_edges(source_node_id, relation_type, edge_id);

CREATE TABLE plasticity_clock (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    elapsed_seconds REAL NOT NULL,
    short_log_gap REAL,
    long_log_gap REAL,
    resource_tau_seconds REAL NOT NULL,
    threshold_tau_seconds REAL NOT NULL,
    retention_tau_seconds REAL NOT NULL,
    short_recurrence_log_gap REAL,
    long_recurrence_log_gap REAL,
    short_recurrence_tau_seconds REAL NOT NULL,
    long_recurrence_tau_seconds REAL NOT NULL,
    short_gap_count INTEGER NOT NULL,
    long_gap_count INTEGER NOT NULL,
    short_recurrence_weight REAL NOT NULL,
    long_recurrence_weight REAL NOT NULL,
    short_recurrence_log_m2 REAL NOT NULL,
    long_recurrence_log_m2 REAL NOT NULL,
    recurrence_log_mean REAL,
    recurrence_log_m2 REAL NOT NULL,
    recurrence_weight REAL NOT NULL
);

CREATE TABLE external_seed_state (
    turn_node_id INTEGER PRIMARY KEY REFERENCES turn_nodes(node_id),
    last_seed_seconds REAL NOT NULL
) WITHOUT ROWID;

CREATE TABLE memory_events (
    event_id INTEGER PRIMARY KEY,
    current_turn_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    hub_node_id INTEGER REFERENCES hub_nodes(node_id),
    time_prior REAL NOT NULL,
    continuation REAL NOT NULL,
    surprise REAL NOT NULL CHECK(surprise >= 0.0 AND surprise <= 1.0),
    seed_support INTEGER NOT NULL,
    pushes INTEGER NOT NULL,
    residual_l1 REAL NOT NULL,
    modification_threshold REAL NOT NULL,
    observed_mass REAL NOT NULL,
    recurrent_mass REAL NOT NULL,
    reactivated_mass REAL NOT NULL,
    potentiated_mass REAL NOT NULL,
    inhibited_mass REAL NOT NULL,
    integrated_json TEXT NOT NULL
);

CREATE TABLE event_seeds (
    event_id INTEGER NOT NULL REFERENCES memory_events(event_id),
    candidate_turn_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    value REAL NOT NULL,
    channels_json TEXT NOT NULL,
    PRIMARY KEY (event_id, candidate_turn_node_id)
) WITHOUT ROWID;

CREATE TABLE event_channel_support (
    event_id INTEGER NOT NULL REFERENCES memory_events(event_id),
    channel TEXT NOT NULL,
    candidate_turn_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    PRIMARY KEY (event_id, channel, candidate_turn_node_id)
) WITHOUT ROWID;

CREATE TABLE event_channels (
    event_id INTEGER NOT NULL REFERENCES memory_events(event_id),
    channel TEXT NOT NULL,
    PRIMARY KEY (event_id, channel)
) WITHOUT ROWID;

CREATE TABLE activation_runs (
    query_turn_node_id INTEGER PRIMARY KEY REFERENCES turn_nodes(node_id),
    time_prior REAL NOT NULL,
    continuation REAL NOT NULL,
    pushes INTEGER NOT NULL,
    residual_l1 REAL NOT NULL,
    seed_support INTEGER NOT NULL,
    completion_support INTEGER NOT NULL,
    completion_effective_support REAL NOT NULL,
    completion_mass REAL NOT NULL,
    graph_only_support INTEGER NOT NULL,
    graph_only_effective_support REAL NOT NULL,
    graph_only_mass REAL NOT NULL
);

CREATE TABLE activation_items (
    query_turn_node_id INTEGER NOT NULL REFERENCES activation_runs(query_turn_node_id),
    candidate_turn_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    rank INTEGER NOT NULL,
    seed_score REAL NOT NULL,
    direct_mass REAL NOT NULL,
    settled_mass REAL NOT NULL,
    completion_mass REAL NOT NULL,
    is_graph_only INTEGER NOT NULL CHECK(is_graph_only IN (0, 1)),
    first_relation TEXT,
    dominant_path_json TEXT NOT NULL,
    relation_path_json TEXT NOT NULL,
    PRIMARY KEY (query_turn_node_id, candidate_turn_node_id)
) WITHOUT ROWID;

CREATE INDEX activation_items_rank
    ON activation_items(query_turn_node_id, rank);

CREATE TABLE recall_runs (
    query_turn_node_id INTEGER PRIMARY KEY REFERENCES turn_nodes(node_id),
    active_basin_count INTEGER NOT NULL,
    sharp_completion_count INTEGER NOT NULL,
    basin_direct_count INTEGER NOT NULL,
    basin_completion_count INTEGER NOT NULL,
    relative_tail_count INTEGER NOT NULL,
    pushes INTEGER NOT NULL,
    residual_l1 REAL NOT NULL
);

CREATE TABLE recall_items (
    query_turn_node_id INTEGER NOT NULL REFERENCES recall_runs(query_turn_node_id),
    candidate_turn_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    sources_json TEXT NOT NULL,
    basin_ids_json TEXT NOT NULL,
    is_pattern_only INTEGER NOT NULL CHECK(is_pattern_only IN (0, 1)),
    PRIMARY KEY (query_turn_node_id, candidate_turn_node_id)
) WITHOUT ROWID;

CREATE INDEX recall_items_rank
    ON recall_items(query_turn_node_id, rank);

CREATE TABLE burst_context_members (
    session_key TEXT NOT NULL,
    position INTEGER NOT NULL,
    turn_node_id INTEGER NOT NULL REFERENCES turn_nodes(node_id),
    PRIMARY KEY (session_key, position),
    UNIQUE (session_key, turn_node_id)
) WITHOUT ROWID;

CREATE TABLE context_state (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    members_json TEXT NOT NULL,
    dense BLOB,
    terms_json TEXT NOT NULL
);
"""
