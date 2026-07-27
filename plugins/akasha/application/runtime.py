"""Maintain one online MemoryCycle backed by deterministic SQLite snapshots."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from ..domain.features import BurstAwareFeaturePool
from ..domain.model import MemoryConfig, Turn
from ..infrastructure.loader import load_turns
from ..infrastructure.lease import WriterLease
from ..infrastructure.persistence import (
    load_memory_state,
    memory_turn_count,
    write_memory_database,
)
from ..infrastructure.sparse_index import BuildConfig, build_sparse_index
from ..infrastructure.sparse_index.encoding import tokenize
from .cycle import CycleCommit, MemoryCycle, RetrievalTicket
from .rebuild import deterministic_metadata


@dataclass(frozen=True)
class OnlineCommit:
    """Return the committed turn and its shared-cycle learning result."""

    turn: Turn
    cycle: CycleCommit


class OnlineMemoryRuntime:
    """Serve online retrieval and persist the same state produced by replay."""

    def __init__(
        self,
        *,
        sessions_path: Path,
        index_path: Path,
        memory_path: Path,
        embedding_model: str,
        embedding_dimension: int | None,
        config: MemoryConfig,
    ) -> None:
        self.sessions_path = sessions_path
        self.index_path = index_path
        self.memory_path = memory_path
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.config = config
        self._writer_lease = WriterLease(memory_path)
        self.cycle = self._restore_or_replay()

    def close(self) -> None:
        self._writer_lease.close()

    def query_turn(
        self,
        *,
        text: str,
        dense: np.ndarray,
        session_key: str,
        timestamp: datetime,
        capture_paths: bool = True,
    ) -> tuple[Turn, RetrievalTicket]:
        """Encode an ephemeral user cue and retrieve without mutating state."""

        # 1. Validate and normalize the external embedding boundary.
        vector = _unit_dense(dense)
        started = _as_utc(timestamp)
        gap = _global_gap(self.cycle.turns, started)
        terms = tuple(
            sorted(
                tokenize(text).items(),
                key=lambda item: item[0].encode("utf-8"),
            )
        )

        # 2. Build the next causal turn shell and run the shared cycle.
        event = self.cycle.state_version
        turn = Turn(
            node_id=event,
            turn_id=f"pending:{session_key}:{started.isoformat()}",
            session_key=session_key,
            user_seq=-1,
            user_message_id=f"pending:user:{event}",
            assistant_message_id=f"pending:assistant:{event}",
            started_at=started.isoformat(),
            committed_at=started.isoformat(),
            user_text=text,
            assistant_text="",
            user_dense=vector,
            assistant_dense=None,
            user_terms=terms,
            assistant_terms=(),
            inter_gap_seconds=gap,
        )
        return turn, self.cycle.retrieve(
            turn,
            capture_paths=capture_paths,
        )

    def commit_from_source(
        self,
        *,
        user_message_id: str,
        assistant_message_id: str,
        ticket: RetrievalTicket | None,
    ) -> OnlineCommit:
        """Append canonical source turns and atomically publish their state."""

        # 1. Increment the causal feature index from sessions.db.
        build_sparse_index(
            self.sessions_path,
            self.index_path,
            BuildConfig(
                embedding_model=self.embedding_model,
                embedding_dimension=self.embedding_dimension,
            ),
        )
        turns = load_turns(self.index_path)
        if len(turns) <= self.cycle.state_version:
            raise ValueError("TurnCommitted did not append a new sparse turn")

        # 2. Apply missing source turns to an isolated candidate state.
        candidate = copy.deepcopy(self.cycle)
        last_commit: CycleCommit | None = None
        last_turn: Turn | None = None
        for turn in turns[self.cycle.state_version :]:
            selected = _matching_ticket(
                turn,
                user_message_id,
                assistant_message_id,
                ticket,
            )
            last_commit = candidate.commit(turn, selected)
            last_turn = turn
        if last_commit is None or last_turn is None:
            raise RuntimeError("online commit produced no memory event")
        if (
            last_turn.user_message_id != user_message_id
            or last_turn.assistant_message_id != assistant_message_id
        ):
            raise ValueError(
                "TurnCommitted is not the latest canonical sparse turn"
            )

        # 3. Publish one complete SQLite snapshot before adopting memory state.
        if candidate.context is None:
            raise RuntimeError("committed memory state has no context")
        write_memory_database(
            self.memory_path,
            turns=candidate.turns,
            graph=candidate.graph,
            events=candidate.events,
            evidence=candidate.evidence,
            captures=[],
            context=candidate.context,
            burst_members=candidate.burst_members,
            config=self.config,
            metadata=deterministic_metadata(self.index_path),
            recalls=candidate.recalls,
        )
        self.cycle = candidate
        return OnlineCommit(last_turn, last_commit)

    def _restore_or_replay(self) -> MemoryCycle:
        """Restore a snapshot and causally catch up any indexed source turns."""

        # 1. Bring the derived sparse index to the sessions source boundary.
        result = build_sparse_index(
            self.sessions_path,
            self.index_path,
            BuildConfig(
                embedding_model=self.embedding_model,
                embedding_dimension=self.embedding_dimension,
            ),
        )
        turns = load_turns(self.index_path) if result.discovered_turns else []
        if not self.memory_path.exists():
            cycle = MemoryCycle(
                self.config,
                turn_capacity=len(turns),
                feature_pool=(
                    BurstAwareFeaturePool(turns)
                    if turns
                    else None
                ),
            )
            return self._catch_up(cycle, turns)

        # 2. Restore the persisted prefix and replay only crash-window suffixes.
        persisted = memory_turn_count(self.memory_path)
        if persisted > len(turns):
            raise ValueError(
                "memory snapshot contains more turns than sessions source"
            )
        prefix = turns[:persisted]
        (
            graph,
            events,
            evidence,
            context,
            recalls,
            burst_members,
        ) = load_memory_state(
            self.memory_path,
            turns=prefix,
            config=self.config,
            source_index_sha256=None,
        )
        cycle = MemoryCycle.restore(
            config=self.config,
            turns=prefix,
            graph=graph,
            context=context,
            events=events,
            evidence=evidence,
            recalls=recalls,
            burst_members=burst_members,
        )
        cycle.feature_pool = (
            BurstAwareFeaturePool(turns)
            if turns
            else None
        )
        return self._catch_up(cycle, turns[persisted:])

    def _catch_up(
        self,
        cycle: MemoryCycle,
        turns: list[Turn],
    ) -> MemoryCycle:
        """Replay missing source turns and persist the recovered snapshot."""

        for turn in turns:
            cycle.commit(
                turn,
                cycle.retrieve(
                    turn,
                    include_completion=False,
                    isolate_graph=False,
                ),
            )
        cycle.feature_pool = None
        if not turns:
            return cycle
        if cycle.context is None:
            raise RuntimeError("recovered memory state has no context")
        write_memory_database(
            self.memory_path,
            turns=cycle.turns,
            graph=cycle.graph,
            events=cycle.events,
            evidence=cycle.evidence,
            captures=[],
            context=cycle.context,
            burst_members=cycle.burst_members,
            config=self.config,
            metadata=deterministic_metadata(self.index_path),
            recalls=cycle.recalls,
        )
        return cycle


def _matching_ticket(
    turn: Turn,
    user_message_id: str,
    assistant_message_id: str,
    ticket: RetrievalTicket | None,
) -> RetrievalTicket | None:
    if (
        ticket is None
        or turn.user_message_id != user_message_id
        or turn.assistant_message_id != assistant_message_id
        or ticket.cue_text != turn.user_text
        or _as_utc(datetime.fromisoformat(ticket.cue_started_at))
        != _as_utc(datetime.fromisoformat(turn.started_at))
    ):
        return None
    return replace(ticket, turn_id=turn.turn_id)


def _unit_dense(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError("query embedding must be one finite vector")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("query embedding must be non-zero")
    return value / norm


def _as_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return timestamp.astimezone(timezone.utc)


def _global_gap(turns: list[Turn], started: datetime) -> float | None:
    if not turns:
        return None
    previous = datetime.fromisoformat(turns[-1].started_at)
    gap = (started - previous.astimezone(timezone.utc)).total_seconds()
    if gap < 0.0:
        raise ValueError("online turn would violate global causal order")
    return gap
