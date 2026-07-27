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


def remap_diffusion_growth(
    diffusion: DiffusionResult,
    old_turn_count: int,
    new_turn_count: int,
) -> DiffusionResult:
    """Remap persisted diffusion coordinates after turn capacity grows."""

    # 1. Build the old-to-new coordinate map with stable turn IDs.
    if new_turn_count <= old_turn_count:
        raise ValueError("diffusion growth requires a larger turn count")
    offset = new_turn_count - old_turn_count
    old_size = old_turn_count * 2
    if diffusion.reserve.size != old_size:
        raise ValueError("diffusion size does not match old graph capacity")
    mapping = np.arange(old_size, dtype=np.int32)
    mapping[old_turn_count:] += offset

    # 2. Move mass and optional dominant-path coordinates.
    reserve = np.zeros(new_turn_count * 2, dtype=np.float64)
    reserve[mapping] = diffusion.reserve
    active_nodes = mapping[diffusion.active_nodes]
    parent_node = _remap_parent_nodes(
        diffusion.parent_node,
        mapping,
        old_turn_count,
        offset,
        new_turn_count,
    )
    parent_edge = _remap_parent_edges(
        diffusion.parent_edge,
        mapping,
        new_turn_count,
    )
    return DiffusionResult(
        reserve=reserve,
        active_nodes=active_nodes,
        pushes=diffusion.pushes,
        residual_l1=diffusion.residual_l1,
        parent_node=parent_node,
        parent_edge=parent_edge,
    )


def _remap_parent_nodes(
    parents: np.ndarray | None,
    mapping: np.ndarray,
    old_turn_count: int,
    offset: int,
    new_turn_count: int,
) -> np.ndarray | None:
    if parents is None:
        return None
    result = np.full(new_turn_count * 2, -1, dtype=np.int32)
    values = parents.copy()
    hub_parents = values >= old_turn_count
    values[hub_parents] += offset
    result[mapping] = values
    return result


def _remap_parent_edges(
    edges: np.ndarray | None,
    mapping: np.ndarray,
    new_turn_count: int,
) -> np.ndarray | None:
    if edges is None:
        return None
    result = np.full(new_turn_count * 2, -1, dtype=np.int32)
    result[mapping] = edges
    return result
