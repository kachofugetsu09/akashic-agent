"""Build and incrementally maintain a threshold-free sparse turn index."""

from __future__ import annotations

import json
import hashlib
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import numpy as np

from session.memory_policy import excludes_memory

from .encoding import LexicalState, lexical_identity, tokenize
from .model import CanonicalTurn, SessionState, SparseFeature, TimeStats
from .schema import INDEX_VERSION, SCHEMA

LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
INTERRUPTED_ASSISTANT_MARKER = "[interrupted]"
TurnPair = tuple[tuple[sqlite3.Row, ...], sqlite3.Row]


class AppendOnlyViolation(RuntimeError):
    """Report that an incremental source contains new historical turns."""


@dataclass(frozen=True)
class BuildConfig:
    """Configure causal BM25 while leaving feature cardinality unconstrained."""

    embedding_model: str = "text-embedding-v4"
    embedding_dimension: int | None = None
    bm25_k1: float = 1.2
    bm25_b: float = 0.75


@dataclass(frozen=True)
class EmbeddingIssue:
    """Describe one source message that lacks a valid frozen embedding."""

    message_id: str
    session_key: str
    seq: int
    role: str
    reason: str


@dataclass(frozen=True)
class EmbeddingAudit:
    """Summarize the frozen embedding boundary for eligible dialogue turns."""

    eligible_turns: int
    excluded_interrupted_turns: int
    excluded_memory_turns: int
    eligible_messages: int
    valid_messages: int
    dimension: int | None
    issues: tuple[EmbeddingIssue, ...]

    @property
    def complete(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class BuildResult:
    """Summarize an index build without hiding skipped or missing turns."""

    discovered_turns: int
    excluded_interrupted_turns: int
    excluded_memory_turns: int
    indexed_turns: int
    skipped_existing_turns: int
    turns_missing_embeddings: int
    dense_pointers: int
    lexical_dimensions: int
    time_observations: int


def build_sparse_index(
    source_path: Path,
    output_path: Path,
    config: BuildConfig = BuildConfig(),
) -> BuildResult:
    """Replay committed turns in causal order and persist lossless sparse evidence."""

    # 1. Open both trust boundaries and validate their schemas.
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = sqlite3.connect(output_path)
    output.row_factory = sqlite3.Row
    try:
        output.executescript(SCHEMA)
        _validate_source(source)
        _validate_index_version(output)

        # 2. Restore prior online state and identify append-only work.
        turns, missing, excluded_interrupted, excluded_memory = _load_canonical_turns(
            source,
            config,
        )
        existing = _load_existing_turns(output)
        new_turns = _select_new_turns(turns, existing)
        lexical = _load_lexical_states(output)
        time_stats = _load_time_stats(output)
        stream_state = _load_stream_state(output)

        # 3. Encode each turn before committing it to online statistics.
        with output:
            for turn in new_turns:
                encoded = _encode_turn(turn, lexical, time_stats, stream_state, config)
                _persist_turn(output, turn, encoded)
                _commit_stream_state(stream_state, turn)
            _persist_online_state(output, lexical, time_stats, stream_state)
            _set_metadata(output, "index_version", INDEX_VERSION)
            _set_metadata(output, "embedding_model", config.embedding_model)
            if config.embedding_dimension is not None:
                _set_metadata(
                    output,
                    "embedding_dimension",
                    str(config.embedding_dimension),
                )
            _set_metadata(
                output,
                "turns_missing_embeddings",
                str(missing),
            )
            _set_metadata(
                output,
                "turns_excluded_interrupted",
                str(excluded_interrupted),
            )
            _set_metadata(
                output,
                "turns_excluded_memory",
                str(excluded_memory),
            )
            for key, value in lexical_identity().items():
                _set_metadata(output, key, value)
        return _build_result(
            output,
            turns,
            missing,
            excluded_interrupted,
            excluded_memory,
            new_turns,
            existing,
        )
    finally:
        source.close()
        output.close()


def audit_source_embeddings(
    source_path: Path,
    config: BuildConfig,
) -> EmbeddingAudit:
    """Audit eligible messages without mutating the sessions database."""

    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        _validate_source(source)
        messages = _source_messages(source)
        pairs, excluded_interrupted, excluded_memory = _eligible_pairs(messages)
        rows = source.execute(
            """
            SELECT message_id, content_hash, embedding, dim
            FROM message_embeddings
            WHERE model = ?
            ORDER BY message_id
            """,
            (config.embedding_model,),
        ).fetchall()
        embeddings = {str(row["message_id"]): row for row in rows}
        issues, valid_dimensions, required = _embedding_issues(
            pairs,
            embeddings,
            config.embedding_dimension,
        )
    finally:
        source.close()
    dimension = (
        next(iter(valid_dimensions))
        if len(valid_dimensions) == 1
        else config.embedding_dimension
    )
    return EmbeddingAudit(
        eligible_turns=len(pairs),
        excluded_interrupted_turns=excluded_interrupted,
        excluded_memory_turns=excluded_memory,
        eligible_messages=required,
        valid_messages=required - len(issues),
        dimension=dimension,
        issues=tuple(issues),
    )


@dataclass(frozen=True)
class EncodedTurn:
    """Collect all current-turn evidence before it is persisted."""

    features: list[SparseFeature]
    term_fields: dict[str, Counter[str]]
    start_gap_seconds: float | None
    log_start_gap: float | None
    response_delta_seconds: float | None
    idle_gap_seconds: float | None
    log_idle_gap: float | None
    overlap_seconds: float | None
    log_overlap: float | None
    persisted_message_span_seconds: float
    log_persisted_message_span: float
    channel: str
    previous_turn_id: str | None
    session_turn_index: int
    local_hour: float
    hour_sin: float
    hour_cos: float
    weekday: int
    weekday_sin: float
    weekday_cos: float


def _validate_source(connection: sqlite3.Connection) -> None:
    required = {"sessions", "messages", "message_embeddings"}
    actual = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = required - actual
    if missing:
        raise ValueError(f"sessions database is missing required tables: {sorted(missing)}")


def _validate_index_version(connection: sqlite3.Connection) -> None:
    row = connection.execute("SELECT value FROM metadata WHERE key='index_version'").fetchone()
    if row is not None and row["value"] != INDEX_VERSION:
        raise ValueError(f"unsupported sparse index version: {row['value']}")


def _load_canonical_turns(
    connection: sqlite3.Connection,
    config: BuildConfig,
) -> tuple[list[CanonicalTurn], int, int, int]:
    """Reconstruct every dialogue turn and count incomplete dense cache rows."""

    # 1. Read source rows and the selected embedding space.
    messages = _source_messages(connection)
    pairs, excluded_interrupted, excluded_memory = _eligible_pairs(messages)
    audit = _audit_rows(
        connection,
        pairs,
        config,
        excluded_interrupted,
        excluded_memory,
    )
    invalid = {issue.message_id for issue in audit.issues}
    embeddings = {}
    for row in connection.execute(
        """
        SELECT message_id, embedding, dim
        FROM message_embeddings
        WHERE model = ?
        ORDER BY message_id
        """,
        (config.embedding_model,),
    ):
        message_id = str(row["message_id"])
        if message_id not in invalid:
            embeddings[message_id] = _decode_embedding(
                row["embedding"],
                row["dim"],
            )

    # 2. Preserve incomplete dense turns for lexical and temporal evidence.
    turns = [
        _make_turn(users, assistant, embeddings)
        for users, assistant in pairs
    ]
    missing_embeddings = sum(
        any(
            _message_text(user).strip() and user["id"] not in embeddings
            for user in users
        )
        or assistant["id"] not in embeddings
        for users, assistant in pairs
    )

    # 3. Establish one deterministic global causal order.
    turns.sort(
        key=lambda turn: (
            _parse_time(turn.committed_at),
            turn.session_key.encode("utf-8"),
            turn.user_seq,
            turn.turn_id.encode("utf-8"),
        )
    )
    turns = _resolve_feedback_targets(turns)
    return turns, missing_embeddings, excluded_interrupted, excluded_memory


def _source_messages(
    connection: sqlite3.Connection,
) -> list[sqlite3.Row]:
    """读取全部消息并带出 session metadata；孤儿消息 fail-loud 而不是静默消失。"""

    # 1. 先核对每条消息都有对应 session 行，损坏数据不得伪装成空结果。
    session_keys = {
        str(row["key"])
        for row in connection.execute("SELECT key FROM sessions")
    }
    orphan_keys = {
        str(row["session_key"])
        for row in connection.execute("SELECT DISTINCT session_key FROM messages")
    } - session_keys
    if orphan_keys:
        preview = ", ".join(sorted(orphan_keys)[:5])
        raise ValueError(
            f"messages 存在无 session 记录的孤儿 session_key: {preview}"
            + (f" 等 {len(orphan_keys)} 个" if len(orphan_keys) > 5 else "")
        )
    return connection.execute(
        """
        SELECT m.session_key, m.seq, m.id, m.role, m.content, m.extra, m.ts,
               s.metadata AS session_metadata
        FROM messages AS m
        JOIN sessions AS s ON s.key = m.session_key
        ORDER BY m.session_key, m.seq
        """
    ).fetchall()


def _eligible_pairs(
    messages: list[sqlite3.Row],
) -> tuple[list[TurnPair], int, int]:
    """按显式 turn 元数据重建新格式，并保留严格 legacy 相邻配对。"""

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for message in messages:
        grouped[str(message["session_key"])].append(message)
    pairs = []
    excluded_interrupted = 0
    excluded_memory = 0
    for session_messages in grouped.values():
        explicit: dict[str, list[sqlite3.Row]] = {}
        explicit_order: list[str] = []
        for message in session_messages:
            turn_id = _message_extra(message).get("control_turn_id")
            if turn_id is None:
                continue
            if not isinstance(turn_id, str) or not turn_id:
                raise ValueError(f"control_turn_id 必须是非空字符串: {message['id']}")
            if turn_id not in explicit:
                explicit[turn_id] = []
                explicit_order.append(turn_id)
            explicit[turn_id].append(message)

        # 1. 新格式只信任显式 ID、ordinal 和 terminal 标志。
        for turn_id in explicit_order:
            turn_messages = explicit[turn_id]
            users = [row for row in turn_messages if row["role"] == "user"]
            if not users and all(
                row["role"] == "assistant"
                and _message_extra(row).get("proactive") is True
                for row in turn_messages
            ):
                continue
            assistants = [
                row
                for row in turn_messages
                if row["role"] == "assistant"
                and _message_extra(row).get("turn_terminal") is True
            ]
            ordinals = [_message_extra(row).get("turn_input_ordinal") for row in users]
            if (
                not users
                or len(assistants) != 1
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in ordinals
                )
            ):
                raise ValueError(f"同 turn transcript 结构无效: {turn_id}")
            typed_ordinals = cast(list[int], ordinals)
            if sorted(typed_ordinals) != list(range(len(users))):
                raise ValueError(f"同 turn input ordinal 不连续: {turn_id}")
            users.sort(
                key=lambda row: cast(
                    int,
                    _message_extra(row)["turn_input_ordinal"],
                )
            )
            assistant = assistants[0]
            if _message_extra(assistant).get("turn_input_count") != len(users):
                raise ValueError(f"同 turn input count 不匹配: {turn_id}")
            if any(int(user["seq"]) >= int(assistant["seq"]) for user in users):
                raise ValueError(f"同 turn assistant 顺序无效: {turn_id}")
            first = users[0]
            if _excluded_session(str(first["session_key"]), first):
                excluded_memory += 1
                continue
            if _interrupted_pair(first, assistant):
                excluded_interrupted += 1
                continue
            if any(_skip_post_memory(row) for row in (*users, assistant)):
                continue
            if not any(_message_text(row) for row in (*users, assistant)):
                continue
            pairs.append((tuple(users), assistant))

        # 2. 无显式 turn metadata 的旧消息继续使用严格相邻配对。
        for user, assistant in zip(
            session_messages,
            session_messages[1:],
        ):
            if (
                _message_extra(user).get("control_turn_id") is not None
                or _message_extra(assistant).get("control_turn_id") is not None
            ):
                continue
            if user["role"] != "user" or assistant["role"] != "assistant":
                continue
            if _excluded_session(str(user["session_key"]), user):
                excluded_memory += 1
                continue
            if _interrupted_pair(user, assistant):
                excluded_interrupted += 1
                continue
            if _skip_post_memory(user) or _skip_post_memory(assistant):
                continue
            if not _message_text(user) and not _message_text(assistant):
                continue
            pairs.append(((user,), assistant))
    return pairs, excluded_interrupted, excluded_memory


def _excluded_session(session_key: str, user: sqlite3.Row) -> bool:
    return excludes_memory(session_key, _session_metadata(user))


def _session_metadata(message: sqlite3.Row) -> dict[str, object]:
    raw = message["session_metadata"]
    if not raw:
        return {}
    payload = json.loads(str(raw))
    if not isinstance(payload, dict):
        raise ValueError(
            f"session metadata must be an object: {message['session_key']}"
        )
    return payload


def _skip_post_memory(message: sqlite3.Row) -> bool:
    return bool(_message_extra(message).get("skip_post_memory"))


def _message_extra(message: sqlite3.Row) -> dict[str, object]:
    raw = message["extra"]
    if not raw:
        return {}
    payload = json.loads(str(raw))
    if not isinstance(payload, dict):
        raise ValueError(
            f"message extra must be an object: {message['id']}"
        )
    return payload


def _message_text(message: sqlite3.Row) -> str:
    return str(message["content"] or "")


def _interrupted_pair(
    user: sqlite3.Row,
    assistant: sqlite3.Row,
) -> bool:
    return (
        user["role"] == "user"
        and assistant["role"] == "assistant"
        and _message_text(assistant) == INTERRUPTED_ASSISTANT_MARKER
    )


def _audit_rows(
    connection: sqlite3.Connection,
    pairs: list[TurnPair],
    config: BuildConfig,
    excluded_interrupted: int,
    excluded_memory: int,
) -> EmbeddingAudit:
    rows = connection.execute(
        """
        SELECT message_id, content_hash, embedding, dim
        FROM message_embeddings
        WHERE model = ?
        ORDER BY message_id
        """,
        (config.embedding_model,),
    ).fetchall()
    embeddings = {str(row["message_id"]): row for row in rows}
    issues, dimensions, required = _embedding_issues(
        pairs,
        embeddings,
        config.embedding_dimension,
    )
    dimension = (
        next(iter(dimensions))
        if len(dimensions) == 1
        else config.embedding_dimension
    )
    return EmbeddingAudit(
        eligible_turns=len(pairs),
        excluded_interrupted_turns=excluded_interrupted,
        excluded_memory_turns=excluded_memory,
        eligible_messages=required,
        valid_messages=required - len(issues),
        dimension=dimension,
        issues=tuple(issues),
    )


def _embedding_issues(
    pairs: list[TurnPair],
    embeddings: dict[str, sqlite3.Row],
    expected_dimension: int | None,
) -> tuple[list[EmbeddingIssue], set[int], int]:
    """Validate identity, content, dimensions, and finite float payloads."""

    issues = []
    dimensions: set[int] = set()
    required = 0
    for message in _pair_messages(pairs):
        content = _message_text(message)
        if not content.strip():
            continue
        required += 1
        issue, dimension = _embedding_issue(
            message,
            embeddings.get(str(message["id"])),
            expected_dimension,
        )
        if issue is not None:
            issues.append(issue)
        elif dimension is not None:
            dimensions.add(dimension)
    if expected_dimension is None and len(dimensions) > 1:
        valid_dimension = max(
            dimensions,
            key=lambda value: sum(
                int(row["dim"]) == value
                for row in embeddings.values()
            ),
        )
        invalid_ids = {issue.message_id for issue in issues}
        issues.extend(
            _dimension_mismatches(
                pairs,
                embeddings,
                valid_dimension,
                invalid_ids,
            )
        )
        dimensions = {valid_dimension}
    issues.sort(
        key=lambda issue: (
            issue.session_key.encode("utf-8"),
            issue.seq,
            issue.message_id.encode("utf-8"),
        )
    )
    return issues, dimensions, required


def _pair_messages(pairs: list[TurnPair]) -> Iterator[sqlite3.Row]:
    """按 transcript 顺序展开全部用户消息和最终助手消息。"""

    for users, assistant in pairs:
        yield from users
        yield assistant


def _embedding_issue(
    message: sqlite3.Row,
    row: sqlite3.Row | None,
    expected_dimension: int | None,
) -> tuple[EmbeddingIssue | None, int | None]:
    identity = (
        str(message["id"]),
        str(message["session_key"]),
        int(message["seq"]),
        str(message["role"]),
    )
    if row is None:
        return EmbeddingIssue(*identity, "missing"), None
    content_hash = hashlib.sha256(
        _message_text(message).encode("utf-8")
    ).hexdigest()
    if str(row["content_hash"]) != content_hash:
        return EmbeddingIssue(*identity, "content_hash_mismatch"), None
    dimension = int(row["dim"])
    if expected_dimension is not None and dimension != expected_dimension:
        return EmbeddingIssue(*identity, "dimension_mismatch"), None
    vector = np.frombuffer(row["embedding"], dtype=np.float32)
    if vector.size != dimension or not np.all(np.isfinite(vector)):
        return EmbeddingIssue(*identity, "invalid_vector"), None
    if float(np.linalg.norm(vector)) == 0.0:
        return EmbeddingIssue(*identity, "zero_vector"), None
    return None, dimension


def _dimension_mismatches(
    pairs: list[TurnPair],
    embeddings: dict[str, sqlite3.Row],
    expected_dimension: int,
    invalid_ids: set[str],
) -> list[EmbeddingIssue]:
    issues = []
    for message in _pair_messages(pairs):
        if not _message_text(message).strip():
            continue
        message_id = str(message["id"])
        if message_id in invalid_ids:
            continue
        row = embeddings.get(message_id)
        if row is None or int(row["dim"]) == expected_dimension:
            continue
        issues.append(
            EmbeddingIssue(
                message_id,
                str(message["session_key"]),
                int(message["seq"]),
                str(message["role"]),
                "dimension_mismatch",
            )
        )
    return issues


def _make_turn(
    users: tuple[sqlite3.Row, ...],
    assistant: sqlite3.Row,
    embeddings: dict[str, np.ndarray],
) -> CanonicalTurn:
    user = users[0]
    turn_id = f"{user['id']}::{assistant['id']}"
    remember_targets, remember_boost = _feedback_marker(
        user,
        key="akasha_reinforce",
        action="remember",
        carrier_turn_id=turn_id,
        target_required=False,
    )
    forget_targets, _ = _feedback_marker(
        user,
        key="akasha_forget",
        action="forget",
        carrier_turn_id=turn_id,
        target_required=True,
    )
    user_vectors = [
        embeddings[str(row["id"])]
        for row in users
        if _message_text(row).strip() and str(row["id"]) in embeddings
    ]
    required_user_vectors = sum(bool(_message_text(row).strip()) for row in users)
    user_embedding = embeddings.get(str(user["id"])) if len(users) == 1 else None
    if (
        len(users) > 1
        and user_vectors
        and len(user_vectors) == required_user_vectors
    ):
        mean = np.mean(np.stack(user_vectors), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0.0:
            user_embedding = mean / norm
    return CanonicalTurn(
        turn_id=turn_id,
        session_key=user["session_key"],
        user_seq=user["seq"],
        user_message_id=user["id"],
        assistant_message_id=assistant["id"],
        started_at=user["ts"],
        committed_at=assistant["ts"],
        user_text="\n\n".join(_message_text(row) for row in users),
        assistant_text=assistant["content"] or "",
        user_embedding=user_embedding,
        assistant_embedding=embeddings.get(assistant["id"]),
        remember_target_turn_ids=remember_targets,
        forget_target_turn_ids=forget_targets,
        remember_boost=remember_boost,
    )


def _feedback_marker(
    message: sqlite3.Row,
    *,
    key: str,
    action: str,
    carrier_turn_id: str,
    target_required: bool,
) -> tuple[tuple[str, ...], float]:
    """Validate one persisted marker and retain only graph-relevant fields."""

    # 1. Distinguish absence from malformed marker data at the source boundary.
    raw = _message_extra(message).get(key)
    if raw is None:
        return (), 1.0
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be an object: {message['id']}")
    schema_version = raw.get("schema_version")
    if schema_version not in (None, 1):
        raise ValueError(
            f"{key} has unsupported schema_version: {message['id']}"
        )
    marker_action = raw.get("action")
    if marker_action is not None and marker_action != action:
        raise ValueError(f"{key} action mismatch: {message['id']}")

    # 2. Canonicalize stable turn targets; legacy reinforce marks its carrier.
    target_value = raw.get("target_turn_ids")
    if target_value is None:
        if target_required:
            raise ValueError(
                f"{key} requires target_turn_ids: {message['id']}"
            )
        targets = (carrier_turn_id,)
    else:
        if (
            not isinstance(target_value, list)
            or not target_value
            or any(
                not isinstance(value, str) or not value.strip()
                for value in target_value
            )
        ):
            raise ValueError(
                f"{key} target_turn_ids must be non-empty strings: "
                f"{message['id']}"
            )
        targets = tuple(
            dict.fromkeys(value.strip() for value in target_value)
        )

    # 3. Bound reinforcement gain before it can affect graph plasticity.
    if action == "forget":
        return targets, 1.0
    boost_value = raw.get("boost", 3.0)
    if isinstance(boost_value, bool) or not isinstance(
        boost_value,
        (int, float),
    ):
        raise ValueError(f"{key} boost must be numeric: {message['id']}")
    boost = float(boost_value)
    if not math.isfinite(boost) or not 1.0 <= boost <= 3.0:
        raise ValueError(f"{key} boost must be in [1, 3]: {message['id']}")
    return targets, boost


def _resolve_feedback_targets(
    turns: list[CanonicalTurn],
) -> list[CanonicalTurn]:
    """Resolve stable target turn IDs after establishing causal node order."""

    positions = {turn.turn_id: index for index, turn in enumerate(turns)}
    resolved = []
    for event, turn in enumerate(turns):
        remember = _resolve_marker_targets(
            turn.remember_target_turn_ids,
            carrier=turn,
            event=event,
            positions=positions,
            allow_current=True,
            action="remember",
        )
        forget = _resolve_marker_targets(
            turn.forget_target_turn_ids,
            carrier=turn,
            event=event,
            positions=positions,
            allow_current=False,
            action="forget",
        )
        overlap = set(remember) & set(forget)
        if overlap:
            raise ValueError(
                f"feedback actions overlap at turn {turn.turn_id}: "
                f"{sorted(overlap)}"
            )
        resolved.append(
            replace(
                turn,
                remember_target_turn_ids=remember,
                forget_target_turn_ids=forget,
            )
        )
    return resolved


def _resolve_marker_targets(
    targets: tuple[str, ...],
    *,
    carrier: CanonicalTurn,
    event: int,
    positions: dict[str, int],
    allow_current: bool,
    action: str,
) -> tuple[str, ...]:
    canonical = tuple(
        carrier.turn_id if target == "current_turn" else target
        for target in targets
    )
    missing = sorted(set(canonical) - positions.keys())
    if missing:
        raise ValueError(
            f"{action} target turns are not indexed at {carrier.turn_id}: "
            f"{missing}"
        )
    future = sorted(
        target
        for target in set(canonical)
        if positions[target] > event
        or (positions[target] == event and not allow_current)
    )
    if future:
        raise ValueError(
            f"{action} target turns are not causally available at "
            f"{carrier.turn_id}: {future}"
        )
    return tuple(sorted(set(canonical), key=positions.__getitem__))


def _decode_embedding(blob: bytes, dimension: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32).copy()
    if vector.size != dimension or not np.all(np.isfinite(vector)):
        raise ValueError(f"invalid embedding: expected {dimension} finite values, got {vector.size}")
    return vector


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE)
    return parsed.astimezone(timezone.utc)


def _local_time(value: str) -> datetime:
    return _parse_time(value).astimezone(LOCAL_TIMEZONE)


def _load_existing_turns(connection: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    return {
        row["turn_id"]: row
        for row in connection.execute(
            "SELECT turn_id, session_key, user_seq, assistant_message_id, "
            "started_at, committed_at, source_digest FROM sparse_turns"
        )
    }


def _select_new_turns(
    turns: list[CanonicalTurn],
    existing: dict[str, sqlite3.Row],
) -> list[CanonicalTurn]:
    """Accept only unchanged existing turns followed by causal append-only work."""

    source_turn_ids = {turn.turn_id for turn in turns}
    removed = sorted(set(existing) - source_turn_ids)
    if removed:
        raise AppendOnlyViolation(
            "source removed indexed turns; rebuild is required, "
            f"first={removed[0]}"
        )
    for turn in turns:
        row = existing.get(turn.turn_id)
        if row is None:
            continue
        if (
            row["assistant_message_id"] != turn.assistant_message_id
            or row["committed_at"] != turn.committed_at
            or row["source_digest"] != _turn_digest(turn)
        ):
            raise AppendOnlyViolation(f"indexed turn changed in source: {turn.turn_id}")
    if not existing:
        return turns
    last_indexed = max(
        (
            _parse_time(row["committed_at"]),
            row["session_key"].encode("utf-8"),
            row["user_seq"],
            row["turn_id"].encode("utf-8"),
        )
        for row in existing.values()
    )
    new_turns = [turn for turn in turns if turn.turn_id not in existing]
    historical = [
        turn.turn_id
        for turn in new_turns
        if (
            _parse_time(turn.committed_at),
            turn.session_key.encode("utf-8"),
            turn.user_seq,
            turn.turn_id.encode("utf-8"),
        )
        < last_indexed
    ]
    if historical:
        raise AppendOnlyViolation(
            f"source added {len(historical)} historical turns; rebuild is required, first={historical[0]}"
        )
    return new_turns


def _load_lexical_states(connection: sqlite3.Connection) -> dict[str, LexicalState]:
    states = {field: LexicalState(field) for field in ("user", "assistant")}
    for row in connection.execute("SELECT field, doc_count, total_length FROM lexical_corpora"):
        states[row["field"]].doc_count = row["doc_count"]
        states[row["field"]].total_length = row["total_length"]
    for row in connection.execute("SELECT field, term, df FROM lexical_stats"):
        states[row["field"]].document_frequency[row["term"]] = row["df"]
    return states


def _load_time_stats(connection: sqlite3.Connection) -> dict[str, TimeStats]:
    return {
        row["channel"]: TimeStats(
            channel=row["channel"],
            idle_gap_count=row["idle_gap_count"],
            mean_log_idle_gap=row["mean_log_idle_gap"],
            m2_log_idle_gap=row["m2_log_idle_gap"],
        )
        for row in connection.execute("SELECT * FROM time_stats")
    }


def _load_stream_state(connection: sqlite3.Connection) -> dict[str, SessionState]:
    return {
        row["session_key"]: SessionState(
            last_started_at=row["last_started_at"],
            last_committed_at=row["last_committed_at"],
            last_turn_id=row["last_turn_id"],
            turn_count=row["turn_count"],
        )
        for row in connection.execute("SELECT * FROM stream_state")
    }


def _encode_turn(
    turn: CanonicalTurn,
    lexical: dict[str, LexicalState],
    time_stats: dict[str, TimeStats],
    stream_state: dict[str, SessionState],
    config: BuildConfig,
) -> EncodedTurn:
    """Produce identity, lexical, and temporal evidence before state updates."""

    # 1. Score every observed term against prior-only corpus statistics.
    features: list[SparseFeature] = []
    term_fields = {"user": tokenize(turn.user_text), "assistant": tokenize(turn.assistant_text)}
    for field, terms in term_fields.items():
        features += lexical[field].encode(terms, config.bm25_k1, config.bm25_b)

    # 2. Record time without quantizing it into a fixed bucket.
    temporal = _encode_time(turn, time_stats, stream_state)
    features += temporal.features
    for field, terms in term_fields.items():
        lexical[field].update(terms)
    return EncodedTurn(
        features=features,
        term_fields=term_fields,
        start_gap_seconds=temporal.start_gap_seconds,
        log_start_gap=temporal.log_start_gap,
        response_delta_seconds=temporal.response_delta_seconds,
        idle_gap_seconds=temporal.idle_gap_seconds,
        log_idle_gap=temporal.log_idle_gap,
        overlap_seconds=temporal.overlap_seconds,
        log_overlap=temporal.log_overlap,
        persisted_message_span_seconds=temporal.persisted_message_span_seconds,
        log_persisted_message_span=temporal.log_persisted_message_span,
        channel=temporal.channel,
        previous_turn_id=temporal.previous_turn_id,
        session_turn_index=temporal.session_turn_index,
        local_hour=temporal.local_hour,
        hour_sin=temporal.hour_sin,
        hour_cos=temporal.hour_cos,
        weekday=temporal.weekday,
        weekday_sin=temporal.weekday_sin,
        weekday_cos=temporal.weekday_cos,
    )


@dataclass(frozen=True)
class EncodedTime:
    """Carry a raw time observation plus its prior-only calibration."""

    features: list[SparseFeature]
    start_gap_seconds: float | None
    log_start_gap: float | None
    response_delta_seconds: float | None
    idle_gap_seconds: float | None
    log_idle_gap: float | None
    overlap_seconds: float | None
    log_overlap: float | None
    persisted_message_span_seconds: float
    log_persisted_message_span: float
    channel: str
    previous_turn_id: str | None
    session_turn_index: int
    local_hour: float
    hour_sin: float
    hour_cos: float
    weekday: int
    weekday_sin: float
    weekday_cos: float


def _encode_time(
    turn: CanonicalTurn,
    time_stats: dict[str, TimeStats],
    stream_state: dict[str, SessionState],
) -> EncodedTime:
    """Store exact turn timing and update inter-turn moments after calibration."""

    channel = turn.session_key.split(":", 1)[0]
    previous = stream_state.get(turn.session_key)
    started = _parse_time(turn.started_at)
    committed = _parse_time(turn.committed_at)
    persisted_span = (committed - started).total_seconds()
    if persisted_span < 0:
        raise AppendOnlyViolation(f"negative persisted message span at turn {turn.turn_id}")
    start_gap = None if previous is None else (started - _parse_time(previous.last_started_at)).total_seconds()
    response_delta = (
        None
        if previous is None
        else (started - _parse_time(previous.last_committed_at)).total_seconds()
    )
    if start_gap is not None and (start_gap < 0 or response_delta is None):
        raise AppendOnlyViolation(f"negative session gap at turn {turn.turn_id}")
    log_start_gap = None if start_gap is None else math.log1p(start_gap)
    idle_gap = None if response_delta is None else max(response_delta, 0.0)
    overlap = None if response_delta is None else max(-response_delta, 0.0)
    log_idle_gap = None if idle_gap is None else math.log1p(idle_gap)
    log_overlap = None if overlap is None else math.log1p(overlap)
    stats = time_stats.setdefault(channel, TimeStats(channel, 0.0, 0.0, 0))
    variance = (
        stats.m2_log_idle_gap / (stats.idle_gap_count - 1)
        if stats.idle_gap_count > 1
        else None
    )
    local = _local_time(turn.started_at)
    evidence = {
        "response_delta_seconds": response_delta,
        "idle_gap_seconds": idle_gap,
        "overlap_seconds": overlap,
        "prior_count": stats.idle_gap_count,
        "prior_mean_log_idle_gap": stats.mean_log_idle_gap,
        "prior_variance_log_idle_gap": variance,
        "persisted_message_span_seconds": persisted_span,
        "start_gap_seconds": start_gap,
    }
    features = [
        SparseFeature(
            family="time_channel",
            feature_id=channel,
            value=1.0,
            rank=1,
            evidence_json=json.dumps(evidence, sort_keys=True),
        )
    ]
    if log_overlap is not None and overlap is not None and overlap > 0.0:
        features.append(
            SparseFeature(
                family="time_overlap",
                feature_id=channel,
                value=log_overlap,
                rank=1,
                evidence_json=json.dumps(evidence, sort_keys=True),
            )
        )
    if log_idle_gap is not None:
        _update_time_stats(stats, log_idle_gap)
    hour_angle = 2.0 * math.pi * (local.hour + local.minute / 60.0 + local.second / 3600.0) / 24.0
    weekday_angle = 2.0 * math.pi * local.weekday() / 7.0
    return EncodedTime(
        features=features,
        start_gap_seconds=start_gap,
        log_start_gap=log_start_gap,
        response_delta_seconds=response_delta,
        idle_gap_seconds=idle_gap,
        log_idle_gap=log_idle_gap,
        overlap_seconds=overlap,
        log_overlap=log_overlap,
        persisted_message_span_seconds=persisted_span,
        log_persisted_message_span=math.log1p(persisted_span),
        channel=channel,
        previous_turn_id=None if previous is None else previous.last_turn_id,
        session_turn_index=0 if previous is None else previous.turn_count,
        local_hour=local.hour + local.minute / 60.0 + local.second / 3600.0,
        hour_sin=math.sin(hour_angle),
        hour_cos=math.cos(hour_angle),
        weekday=local.weekday(),
        weekday_sin=math.sin(weekday_angle),
        weekday_cos=math.cos(weekday_angle),
    )


def _update_time_stats(stats: TimeStats, log_idle_gap: float) -> None:
    stats.idle_gap_count += 1
    delta = log_idle_gap - stats.mean_log_idle_gap
    stats.mean_log_idle_gap += delta / stats.idle_gap_count
    stats.m2_log_idle_gap += delta * (log_idle_gap - stats.mean_log_idle_gap)


def _persist_turn(
    connection: sqlite3.Connection,
    turn: CanonicalTurn,
    encoded: EncodedTurn,
) -> None:
    """Persist one immutable turn and all raw channel observations."""

    _persist_sparse_payload(connection, turn, encoded)
    _persist_channel_observations(connection, turn, encoded)


def _persist_sparse_payload(
    connection: sqlite3.Connection,
    turn: CanonicalTurn,
    encoded: EncodedTurn,
) -> None:
    """Persist canonical text plus every non-zero sparse dimension."""

    connection.execute(
        """
        INSERT INTO sparse_turns(
            turn_id, session_key, user_seq,
            user_message_id, assistant_message_id,
            started_at, committed_at, user_text, assistant_text,
            remember_targets_json, forget_targets_json,
            remember_boost, source_digest
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            turn.turn_id,
            turn.session_key,
            turn.user_seq,
            turn.user_message_id,
            turn.assistant_message_id,
            turn.started_at,
            turn.committed_at,
            turn.user_text,
            turn.assistant_text,
            json.dumps(
                turn.remember_target_turn_ids,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            json.dumps(
                turn.forget_target_turn_ids,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            turn.remember_boost,
            _turn_digest(turn),
        ),
    )
    connection.executemany(
        "INSERT INTO sparse_features VALUES (?, ?, ?, ?, ?, ?)",
        [
            (turn.turn_id, feature.family, feature.feature_id, feature.value, feature.rank, feature.evidence_json)
            for feature in encoded.features
        ],
    )
    connection.executemany(
        "INSERT INTO turn_terms VALUES (?, ?, ?, ?)",
        [
            (turn.turn_id, field, term, tf)
            for field, terms in encoded.term_fields.items()
            for term, tf in terms.items()
        ],
    )


def _turn_digest(turn: CanonicalTurn) -> str:
    digest = hashlib.sha256()
    for value in (
        turn.turn_id,
        turn.session_key,
        str(turn.user_seq),
        turn.user_message_id,
        turn.assistant_message_id,
        turn.started_at,
        turn.committed_at,
        turn.user_text,
        turn.assistant_text,
        json.dumps(
            turn.remember_target_turn_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        json.dumps(
            turn.forget_target_turn_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        turn.remember_boost.hex(),
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for vector in (turn.user_embedding, turn.assistant_embedding):
        if vector is None:
            digest.update(b"\0")
        else:
            raw = vector.astype(np.float32).tobytes()
            digest.update(b"\1")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _persist_channel_observations(
    connection: sqlite3.Connection,
    turn: CanonicalTurn,
    encoded: EncodedTurn,
) -> None:
    """Preserve dense and continuous-time evidence without clustering."""

    dense_rows = [
        (
            turn.turn_id,
            field,
            source_id,
            embedding.astype(np.float32).tobytes(),
            embedding.size,
        )
        for field, source_id, embedding in (
            ("user", turn.user_message_id, turn.user_embedding),
            ("assistant", turn.assistant_message_id, turn.assistant_embedding),
        )
        if embedding is not None
    ]
    connection.executemany(
        "INSERT INTO turn_dense VALUES (?, ?, ?, ?, ?)",
        dense_rows,
    )
    connection.execute(
        "INSERT INTO time_observations VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            turn.turn_id,
            encoded.channel,
            encoded.previous_turn_id,
            encoded.session_turn_index,
            encoded.start_gap_seconds,
            encoded.log_start_gap,
            encoded.response_delta_seconds,
            encoded.idle_gap_seconds,
            encoded.log_idle_gap,
            encoded.overlap_seconds,
            encoded.log_overlap,
            encoded.persisted_message_span_seconds,
            encoded.log_persisted_message_span,
            encoded.local_hour,
            encoded.hour_sin,
            encoded.hour_cos,
            encoded.weekday,
            encoded.weekday_sin,
            encoded.weekday_cos,
        ),
    )


def _commit_stream_state(
    stream_state: dict[str, SessionState],
    turn: CanonicalTurn,
) -> None:
    previous = stream_state.get(turn.session_key)
    stream_state[turn.session_key] = SessionState(
        last_started_at=turn.started_at,
        last_committed_at=turn.committed_at,
        last_turn_id=turn.turn_id,
        turn_count=1 if previous is None else previous.turn_count + 1,
    )


def _persist_online_state(
    connection: sqlite3.Connection,
    lexical: dict[str, LexicalState],
    time_stats: dict[str, TimeStats],
    stream_state: dict[str, SessionState],
) -> None:
    """Replace only compact sufficient statistics and causal cursors."""

    # 1. Replace lexical and time sufficient statistics.
    connection.execute("DELETE FROM lexical_corpora")
    connection.execute("DELETE FROM lexical_stats")
    connection.executemany(
        "INSERT INTO lexical_corpora VALUES (?, ?, ?)",
        [(state.field, state.doc_count, state.total_length) for state in lexical.values()],
    )
    connection.executemany(
        "INSERT INTO lexical_stats VALUES (?, ?, ?)",
        [(state.field, term, df) for state in lexical.values() for term, df in state.document_frequency.items()],
    )
    connection.execute("DELETE FROM time_stats")
    connection.executemany(
        "INSERT INTO time_stats VALUES (?, ?, ?, ?)",
        [
            (
                stats.channel,
                stats.idle_gap_count,
                stats.mean_log_idle_gap,
                stats.m2_log_idle_gap,
            )
            for stats in time_stats.values()
        ],
    )

    # 2. Replace per-session causal cursors.
    connection.execute("DELETE FROM stream_state")
    connection.executemany(
        "INSERT INTO stream_state VALUES (?, ?, ?, ?, ?)",
        [
            (
                session_key,
                state.last_started_at,
                state.last_committed_at,
                state.last_turn_id,
                state.turn_count,
            )
            for session_key, state in stream_state.items()
        ],
    )


def _build_result(
    output: sqlite3.Connection,
    turns: list[CanonicalTurn],
    missing: int,
    excluded_interrupted: int,
    excluded_memory: int,
    new_turns: list[CanonicalTurn],
    existing: dict[str, sqlite3.Row],
) -> BuildResult:
    lexical_dimensions = output.execute(
        "SELECT COUNT(*) FROM lexical_stats"
    ).fetchone()[0]
    return BuildResult(
        discovered_turns=len(turns),
        excluded_interrupted_turns=excluded_interrupted,
        excluded_memory_turns=excluded_memory,
        indexed_turns=len(new_turns),
        skipped_existing_turns=len(existing),
        turns_missing_embeddings=missing,
        dense_pointers=output.execute("SELECT COUNT(*) FROM turn_dense").fetchone()[0],
        lexical_dimensions=lexical_dimensions,
        time_observations=output.execute("SELECT COUNT(*) FROM time_observations").fetchone()[0],
    )


def _set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
