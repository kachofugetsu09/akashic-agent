"""Data objects used by the sparse index builder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanonicalTurn:
    """Represent one persisted user message and its immediate assistant reply."""

    turn_id: str
    session_key: str
    user_seq: int
    user_message_id: str
    assistant_message_id: str
    started_at: str
    committed_at: str
    user_text: str
    assistant_text: str
    user_embedding: np.ndarray | None
    assistant_embedding: np.ndarray | None
    remember_target_turn_ids: tuple[str, ...]
    forget_target_turn_ids: tuple[str, ...]
    remember_boost: float


@dataclass(frozen=True)
class SparseFeature:
    """Store one non-zero, explainable feature of a turn."""

    family: str
    feature_id: str
    value: float
    rank: int
    evidence_json: str


@dataclass
class TimeStats:
    """Maintain threshold-free sufficient statistics for one channel's gaps."""

    channel: str
    mean_log_inter_gap: float
    m2_log_inter_gap: float
    inter_gap_count: int


@dataclass(frozen=True)
class SessionState:
    """Track the immediately preceding completed turn of one session."""

    last_started_at: str
    last_committed_at: str
    last_turn_id: str
    turn_count: int
