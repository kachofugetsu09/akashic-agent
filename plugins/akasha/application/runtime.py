"""Maintain one online MemoryCycle backed by deterministic SQLite snapshots."""

from __future__ import annotations

import logging
import math
import os
import tempfile
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from ..domain.features import BurstAwareFeaturePool
from ..domain.model import MemoryConfig, Turn
from ..infrastructure.loader import load_turn_suffix, load_turns
from ..infrastructure.lease import WriterLease
from ..infrastructure.persistence import (
    load_memory_state,
    memory_turn_count,
    write_memory_database,
)
from ..infrastructure.sparse_index import (
    AppendOnlyViolation,
    BuildConfig,
    build_sparse_index,
)
from ..infrastructure.sparse_index.encoding import tokenize
from .cycle import CycleCommit, MemoryCycle, RetrievalTicket
from .rebuild import deterministic_metadata, rebuild_memory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnlineCommit:
    """Return the committed turn and its shared-cycle learning result."""

    turn: Turn
    cycle: CycleCommit


@dataclass(frozen=True)
class StagedOnlineCommit:
    """Hold one durable sparse-index suffix until graph publication."""

    base_version: int
    turns: tuple[Turn, ...]
    user_message_id: str
    assistant_message_id: str
    ticket: RetrievalTicket | None


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
        self._state_lock = threading.RLock()
        self.cycle = self._restore_or_replay()

    def close(self) -> None:
        self._writer_lease.close()

    def rebuild_from_source(self) -> None:
        """从 canonical sessions source 全量替换派生索引与图快照。"""

        with self._state_lock:
            self.cycle = self._fresh_rebuild_from_source()

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
        terms = tuple(
            sorted(
                tokenize(text).items(),
                key=lambda item: item[0].encode("utf-8"),
            )
        )

        # 2. Preview the mutable graph under its owning runtime lock.
        with self._state_lock:
            gap = _global_gap(self.cycle.turns, started)
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

        staged = self.stage_from_source(
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            ticket=ticket,
        )
        return self.publish_staged(staged)

    def stage_from_source(
        self,
        *,
        user_message_id: str,
        assistant_message_id: str,
        ticket: RetrievalTicket | None,
    ) -> StagedOnlineCommit:
        """Persist the causal sparse suffix without publishing graph state."""

        # 1. Increment the durable causal feature index from sessions.db.
        build_sparse_index(
            self.sessions_path,
            self.index_path,
            BuildConfig(
                embedding_model=self.embedding_model,
                embedding_dimension=self.embedding_dimension,
            ),
        )
        suffix = tuple(
            load_turn_suffix(
                self.index_path,
                self.cycle.state_version,
            )
        )
        if not suffix:
            raise ValueError("TurnCommitted did not append a new sparse turn")
        latest = suffix[-1]
        if (
            latest.user_message_id != user_message_id
            or latest.assistant_message_id != assistant_message_id
        ):
            raise ValueError(
                "TurnCommitted is not the latest canonical sparse turn"
            )
        return StagedOnlineCommit(
            base_version=self.cycle.state_version,
            turns=suffix,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            ticket=ticket,
        )

    def publish_staged(
        self,
        staged: StagedOnlineCommit,
    ) -> OnlineCommit:
        """Apply one staged suffix and atomically publish its graph snapshot."""

        with self._state_lock:
            return self._publish_staged_locked(staged)

    def _publish_staged_locked(
        self,
        staged: StagedOnlineCommit,
    ) -> OnlineCommit:
        """Publish one staged suffix while holding the graph ownership lock."""

        # 1. Reject overlapping writers before mutating the published cache.
        if self.cycle.state_version != staged.base_version:
            raise RuntimeError(
                "staged Akasha commit no longer matches published state"
            )
        last_commit: CycleCommit | None = None
        last_turn: Turn | None = None
        try:
            for turn in staged.turns:
                selected = _matching_ticket(
                    turn,
                    staged.user_message_id,
                    staged.assistant_message_id,
                    staged.ticket,
                )
                last_commit = self.cycle.commit(turn, selected)
                last_turn = turn
            if last_commit is None or last_turn is None:
                raise RuntimeError("online commit produced no memory event")
            if (
                last_turn.user_message_id != staged.user_message_id
                or last_turn.assistant_message_id
                != staged.assistant_message_id
            ):
                raise ValueError(
                    "TurnCommitted is not the latest canonical sparse turn"
                )

            # 2. Publish durable state before exposing the completed transaction.
            if self.cycle.context is None:
                raise RuntimeError("committed memory state has no context")
            write_memory_database(
                self.memory_path,
                turns=self.cycle.turns,
                graph=self.cycle.graph,
                events=self.cycle.events,
                evidence=self.cycle.evidence,
                captures=[],
                context=self.cycle.context,
                burst_members=self.cycle.burst_members,
                config=self.config,
                metadata=deterministic_metadata(self.index_path),
                recalls=self.cycle.recalls,
            )
        except Exception:
            self.cycle = self._restore_persisted_cycle()
            raise
        return OnlineCommit(last_turn, last_commit)

    def _restore_or_replay(self) -> MemoryCycle:
        """Restore a snapshot and causally catch up any indexed source turns."""

        # 1. Bring the derived sparse index to the sessions source boundary.
        try:
            result = build_sparse_index(
                self.sessions_path,
                self.index_path,
                self._build_config(),
            )
        except AppendOnlyViolation as exc:
            logger.warning(
                "Akasha sparse source changed; rebuilding derived state: %s",
                exc,
            )
            return self._fresh_rebuild_from_source()
        turns = load_turns(self.index_path) if result.discovered_turns else []
        if not self.memory_path.exists():
            cycle = MemoryCycle(
                self.config,
                turn_capacity=len(turns),
                feature_pool=(
                    BurstAwareFeaturePool(turns, appendable=True)
                    if turns
                    else None
                ),
            )
            return self._catch_up(cycle, turns)

        # 2. Restore the persisted prefix and replay only crash-window suffixes.
        try:
            cycle, suffix = self._load_persisted_prefix(turns)
        except ValueError as exc:
            logger.warning(
                "Akasha memory snapshot no longer matches source; rebuilding: %s",
                exc,
            )
            return self._fresh_rebuild_from_source()
        return self._catch_up(cycle, suffix)

    def _fresh_rebuild_from_source(self) -> MemoryCycle:
        """先生成完整候选，再按 index→memory 顺序发布可恢复派生状态。"""

        # 1. 在目标文件同一文件系统生成并验证完整候选。
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="akasha-index-rebuild-",
            dir=self.index_path.parent,
        ) as index_dir, tempfile.TemporaryDirectory(
            prefix="akasha-memory-rebuild-",
            dir=self.memory_path.parent,
        ) as memory_dir:
            candidate_index = Path(index_dir) / self.index_path.name
            candidate_memory = Path(memory_dir) / self.memory_path.name
            result = build_sparse_index(
                self.sessions_path,
                candidate_index,
                self._build_config(),
            )
            if result.discovered_turns:
                rebuild_memory(
                    candidate_index,
                    candidate_memory,
                    config=self.config,
                    target_sequences=(),
                    target_session="",
                )

            # 2. 先发布 index；其后的崩溃窗口会在重启时重建不匹配 memory。
            os.replace(candidate_index, self.index_path)
            if result.discovered_turns:
                os.replace(candidate_memory, self.memory_path)
            elif self.memory_path.exists():
                self.memory_path.unlink()

        # 3. 从刚发布的确定性状态恢复唯一在线 cycle。
        turns = load_turns(self.index_path) if result.discovered_turns else []
        if not turns:
            return MemoryCycle(self.config)
        cycle, suffix = self._load_persisted_prefix(turns)
        if suffix:
            raise RuntimeError("fresh Akasha rebuild left an unpublished suffix")
        return cycle

    def _build_config(self) -> BuildConfig:
        return BuildConfig(
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
        )

    def _restore_persisted_cycle(self) -> MemoryCycle:
        """Reload exactly the durable prefix after an in-memory write failure."""

        if not self.memory_path.exists():
            return MemoryCycle(self.config)
        turns = load_turns(self.index_path)
        cycle, _ = self._load_persisted_prefix(turns)
        return cycle

    def _load_persisted_prefix(
        self,
        turns: list[Turn],
    ) -> tuple[MemoryCycle, list[Turn]]:
        """Restore the durable prefix and return its unprocessed source suffix."""

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
            BurstAwareFeaturePool(turns, appendable=True)
            if turns
            else None
        )
        return cycle, turns[persisted:]

    def _catch_up(
        self,
        cycle: MemoryCycle,
        turns: list[Turn],
    ) -> MemoryCycle:
        """Replay missing source turns and persist the recovered snapshot."""

        for turn in turns:
            cycle.commit(
                turn,
                cycle.retrieve(turn),
            )
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
