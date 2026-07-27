"""Run the single read-before-write memory cycle used online and in replay."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np

from ..domain.diffusion import residual_push
from ..domain.features import (
    BurstAwareFeaturePool,
    BurstDecision,
)
from ..domain.graph import DynamicMemoryGraph
from ..domain.readout import (
    PatternCompletion,
    RecallCapture,
    read_pattern_completion,
)
from ..domain.model import (
    ContextState,
    DiffusionResult,
    MemoryConfig,
    PlasticityResult,
    SeedEvidence,
    Turn,
)


@dataclass(frozen=True)
class RetrievalTicket:
    """Carry one versioned retrieval into the matching atomic write."""

    state_version: int
    previous_turn_capacity: int
    turn_id: str
    cue_text: str
    cue_started_at: str
    completion_evaluated: bool
    burst_decision: BurstDecision
    evidence: SeedEvidence
    diffusion: DiffusionResult
    completion: PatternCompletion
    prepared_graph: DynamicMemoryGraph


@dataclass(frozen=True)
class CycleCommit:
    """Return the committed event and whether stale retrieval was recomputed."""

    event: PlasticityResult
    evidence: SeedEvidence
    diffusion: DiffusionResult
    retrieval_recomputed: bool


class MemoryCycle:
    """Own one causal memory state and apply identical online/replay events."""

    def __init__(
        self,
        config: MemoryConfig = MemoryConfig(),
        *,
        turn_capacity: int = 0,
        feature_pool: BurstAwareFeaturePool | None = None,
    ) -> None:
        config.validate()
        if turn_capacity < 0:
            raise ValueError("turn capacity cannot be negative")
        if feature_pool is not None and len(feature_pool.turns) != turn_capacity:
            raise ValueError("feature pool must span the complete turn capacity")
        self.config = config
        self.turns: list[Turn] = []
        self.graph = DynamicMemoryGraph(turn_capacity, config)
        self.feature_pool = feature_pool
        self.context: ContextState | None = None
        self.events: list[PlasticityResult] = []
        self.evidence: list[SeedEvidence] = []
        self.recalls: list[RecallCapture] = []
        self.burst_members: dict[str, list[int]] = {}

    @classmethod
    def restore(
        cls,
        *,
        config: MemoryConfig,
        turns: list[Turn],
        graph: DynamicMemoryGraph,
        context: ContextState,
        events: list[PlasticityResult],
        evidence: list[SeedEvidence],
        recalls: list[RecallCapture],
        burst_members: dict[str, list[int]],
    ) -> MemoryCycle:
        """Restore one validated persisted cycle without historical replay."""

        # 1. Enforce the single-version state invariant.
        count = len(turns)
        if graph.turn_count < count:
            raise ValueError("restored graph capacity is smaller than turns")
        if len(events) != count or len(evidence) != count:
            raise ValueError("restored event history differs from turns")
        if graph.current_event != count - 1:
            raise ValueError("restored graph event clock differs from turns")

        # 2. Adopt already validated domain state.
        cycle = cls(config, turn_capacity=graph.turn_count)
        cycle.turns = list(turns)
        cycle.graph = graph
        cycle.context = context
        cycle.events = list(events)
        cycle.evidence = list(evidence)
        cycle.recalls = list(recalls)
        cycle.burst_members = {
            session: list(members)
            for session, members in burst_members.items()
        }
        return cycle

    @property
    def state_version(self) -> int:
        return len(self.turns)

    def retrieve(
        self,
        turn: Turn,
        *,
        capture_paths: bool = False,
        include_completion: bool = True,
        isolate_graph: bool = True,
    ) -> RetrievalTicket:
        """Read the current state and return a non-mutating retrieval ticket."""

        # 1. Validate the ephemeral turn against the next causal position.
        event = self.state_version
        if turn.node_id != event:
            raise ValueError(
                f"query turn node_id must be {event}, got {turn.node_id}"
            )

        # 2. Build causal evidence from history plus the current user cue.
        pool = self._retrieval_pool(turn)
        decision = self._burst_decision(
            pool,
            turn,
            capture_paths,
        )
        evidence = decision.evidence

        # 3. Advance an isolated graph copy and settle its local fixed point.
        previous_turn_capacity = self.graph.turn_count
        prepared = (
            copy.deepcopy(self.graph)
            if isolate_graph
            else self.graph
        )
        if event >= prepared.turn_count:
            prepared.grow_turn_capacity(event + 1)
        prepared.prepare_retrieval(
            event,
            turn.inter_gap_seconds,
            evidence,
        )
        diffusion = residual_push(
            prepared,
            evidence.seed,
            event,
            restart=self.config.restart,
            tolerance=self.config.tolerance,
            capture_paths=capture_paths,
        )
        completion = (
            _empty_completion(diffusion)
            if event == 0 or not include_completion
            else read_pattern_completion(
                graph=prepared,
                turns=self.turns,
                query=turn,
                context=decision.context,
                evidence=evidence,
                diffusion=diffusion,
                historical_surprise=_historical_surprise(
                    self.evidence
                ),
                config=self.config,
                visible_nodes=decision.visible_nodes,
                burst_continued=decision.continued,
            )
        )
        return RetrievalTicket(
            state_version=event,
            previous_turn_capacity=previous_turn_capacity,
            turn_id=turn.turn_id,
            cue_text=turn.user_text,
            cue_started_at=turn.started_at,
            completion_evaluated=include_completion,
            burst_decision=decision,
            evidence=evidence,
            diffusion=diffusion,
            completion=completion,
            prepared_graph=prepared,
        )

    def commit(
        self,
        turn: Turn,
        ticket: RetrievalTicket | None,
    ) -> CycleCommit:
        """Commit one turn and its retrieval-induced plasticity atomically."""

        # 1. Recompute a missing or stale ticket on the latest graph state.
        recomputed = (
            ticket is None
            or ticket.state_version != self.state_version
        )
        causal_turn = (
            replace(turn, node_id=self.state_version)
            if recomputed
            else turn
        )
        if recomputed:
            selected = self.retrieve(
                causal_turn,
                include_completion=False,
            )
        else:
            if ticket is None:
                raise RuntimeError("fresh retrieval ticket cannot be absent")
            selected = ticket
        if selected.turn_id != turn.turn_id:
            raise ValueError(
                "retrieval ticket turn identity does not match committed turn"
            )

        # 2. Remap historical event references to the prepared graph capacity.
        offset = (
            selected.prepared_graph.turn_count
            - selected.previous_turn_capacity
        )
        if offset < 0:
            raise RuntimeError("committed memory cycle cannot shrink capacity")
        if offset:
            self.events = [
                replace(
                    event,
                    hub_node_id=(
                        None
                        if event.hub_node_id is None
                        else event.hub_node_id + offset
                    ),
                )
                for event in self.events
            ]

        # 3. Adopt the prepared state and learn the settled activity once.
        self.graph = selected.prepared_graph
        plasticity = replace(
            self.graph.learn(
                self.state_version,
                selected.evidence,
                selected.diffusion,
            ),
            pushes=selected.diffusion.pushes,
            residual_l1=selected.diffusion.residual_l1,
        )
        self.turns.append(causal_turn)

        # 4. Advance only the committed turn's stream-local visible burst.
        members = self.burst_members.setdefault(
            causal_turn.session_key,
            [],
        )
        if selected.burst_decision.continued:
            members.append(causal_turn.node_id)
        else:
            members[:] = [causal_turn.node_id]
        pool = self.feature_pool or BurstAwareFeaturePool(self.turns)
        self.context = pool.build_context(
            tuple((node, 1.0) for node in members)
        )
        self.events.append(plasticity)
        self.evidence.append(selected.evidence)
        if selected.completion_evaluated:
            self.recalls.append(
                RecallCapture(
                    query_node_id=causal_turn.node_id,
                    completion=selected.completion,
                )
            )
        return CycleCommit(
            event=plasticity,
            evidence=selected.evidence,
            diffusion=selected.diffusion,
            retrieval_recomputed=recomputed,
        )

    def _retrieval_pool(
        self,
        turn: Turn,
    ) -> BurstAwareFeaturePool:
        if self.feature_pool is None:
            return BurstAwareFeaturePool([*self.turns, turn])
        expected = self.feature_pool.turns[self.state_version]
        if expected.turn_id != turn.turn_id:
            raise ValueError("feature pool turn differs from causal query")
        return self.feature_pool

    def _burst_decision(
        self,
        pool: BurstAwareFeaturePool,
        turn: Turn,
        capture_channels: bool,
    ) -> BurstDecision:
        if self.state_version == 0:
            return BurstDecision(
                evidence=SeedEvidence((), {}, 0.5, 0.5, 1.0),
                base_continuation=0.5,
                context_dependence=0.5,
                context_mass=0.0,
                continued=False,
                visible_nodes=(),
                context=ContextState((), None, ()),
            )
        visible = tuple(self.burst_members.get(turn.session_key, ()))
        context = (
            pool.build_context(
                tuple((node, 1.0) for node in visible)
            )
            if visible
            else ContextState((), None, ())
        )
        return pool.infer_burst_seed(
            self.state_version,
            context,
            visible,
            capture_channels,
        )


def _empty_completion(
    diffusion: DiffusionResult,
) -> PatternCompletion:
    return PatternCompletion(
        items=(),
        active_basin_count=0,
        sharp_completion_count=0,
        basin_direct_count=0,
        basin_completion_count=0,
        relative_tail_count=0,
        pushes=diffusion.pushes,
        residual_l1=diffusion.residual_l1,
    )


def _historical_surprise(
    evidence: list[SeedEvidence],
) -> np.ndarray:
    return np.asarray(
        [item.surprise for item in evidence],
        dtype=np.float64,
    )
