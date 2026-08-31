"""Shared immutable records for deterministic memory replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MemoryConfig:
    """Configure one deterministic memory dynamics contract."""

    restart: float = 0.25
    tolerance: float = 1e-7
    learning_rate: float = 0.5
    activation_power: float = 2.0
    recurrent_budget: float = 1.0
    reverse_temporal_ratio: float = 0.25
    forgetting_enabled: bool = True

    def validate(self) -> None:
        """Reject invalid dynamics before the input database is read."""

        values = asdict(self)
        positive = (
            "tolerance",
            "learning_rate",
            "activation_power",
            "recurrent_budget",
        )
        for name in positive:
            if values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.restart <= 1.0:
            raise ValueError("restart must be in (0, 1]")
        if self.learning_rate > 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.activation_power < 1.0:
            raise ValueError("activation_power must be at least 1")
        if not 0.0 < self.reverse_temporal_ratio < 1.0:
            raise ValueError("reverse_temporal_ratio must be in (0, 1)")


@dataclass(frozen=True)
class TurnFeedback:
    """Carry canonical Message-level feedback into one causal commit."""

    remember_nodes: tuple[int, ...] = ()
    forget_nodes: tuple[int, ...] = ()
    remember_boost: float = 1.0


@dataclass(frozen=True)
class Turn:
    """Hold one committed user/assistant turn in global causal order."""

    node_id: int
    turn_id: str
    session_key: str
    user_seq: int
    user_message_id: str
    assistant_message_id: str
    started_at: str
    committed_at: str
    user_text: str
    assistant_text: str
    user_dense: np.ndarray | None
    assistant_dense: np.ndarray | None
    user_terms: tuple[tuple[str, int], ...]
    assistant_terms: tuple[tuple[str, int], ...]
    inter_gap_seconds: float | None
    feedback: TurnFeedback = TurnFeedback()


@dataclass(frozen=True)
class ContextState:
    """Carry the completed previous event as content features."""

    members: tuple[tuple[int, float], ...]
    dense: np.ndarray | None
    terms: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class SeedEvidence:
    """Expose one causal seed and its source-channel supports."""

    seed: tuple[tuple[int, float], ...]
    channels: dict[str, frozenset[int]]
    time_prior: float
    continuation: float
    surprise: float


@dataclass(frozen=True)
class DiffusionResult:
    """Return a fixed-point lower bound and optional dominant paths."""

    reserve: np.ndarray
    active_nodes: np.ndarray
    pushes: int
    residual_l1: float
    parent_node: np.ndarray | None
    parent_edge: np.ndarray | None


@dataclass(frozen=True)
class PlasticityResult:
    """Describe one event's graph mutation and compact next context."""

    hub_node_id: int | None
    threshold: float
    integrated: tuple[tuple[int, float], ...]
    inhibited_mass: float
    potentiated_mass: float
    observed_mass: float
    recurrent_mass: float
    reactivated_mass: float
    pushes: int = 0
    residual_l1: float = 0.0


@dataclass(frozen=True)
class Capture:
    """Persist one target query's seed and completion activation."""

    query_node_id: int
    evidence: SeedEvidence
    diffusion: DiffusionResult
