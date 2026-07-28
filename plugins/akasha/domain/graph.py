"""Deterministic single-state associative memory graph."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import DiffusionResult, MemoryConfig, PlasticityResult, SeedEvidence

MEMBERSHIP = 0
TEMPORAL_FORWARD = 1
TEMPORAL_BACKWARD = 2


@dataclass(frozen=True)
class HubRecord:
    """Describe one event-factorized engram."""

    node_id: int
    created_event: int
    current_turn_id: int
    threshold: float
    innovation_mass: float
    member_edge_ids: tuple[int, ...]


class DynamicMemoryGraph:
    """Learn one bounded relation weight from every settled retrieval event."""

    def __init__(self, turn_count: int, config: MemoryConfig) -> None:
        config.validate()
        self.turn_count = turn_count
        self.max_nodes = turn_count * 2
        self.config = config
        self.adjacency: list[list[int]] = [[] for _ in range(self.max_nodes)]
        self.source: list[int] = []
        self.target: list[int] = []
        self.kind: list[int] = []
        self.bidirectional: list[bool] = []
        self.weight: list[float] = []
        self.last_updated: list[int] = []
        self.created_event: list[int] = []
        self.observed_credit: list[float] = []
        self.recurrent_credit: list[float] = []
        self.resource: list[float] = []
        self.plasticity_threshold: list[float] = []
        self.last_stimulated_seconds: list[float] = []
        self.last_support_seconds: list[float] = []
        self.support_credit: list[float] = []
        self.independent_credit: list[float] = []
        self.hubs: list[HubRecord] = []
        self.hub_members: dict[int, list[int]] = {}
        self.temporal_lookup: dict[tuple[int, int, int], int] = {}
        self.elapsed_seconds = 0.0
        self.short_log_gap: float | None = None
        self.long_log_gap: float | None = None
        self.short_gap_count = 0
        self.long_gap_count = 0
        self.short_recurrence_log_gap: float | None = None
        self.long_recurrence_log_gap: float | None = None
        self.short_recurrence_log_m2 = 0.0
        self.long_recurrence_log_m2 = 0.0
        self.short_recurrence_weight = 0.0
        self.long_recurrence_weight = 0.0
        self.recurrence_log_mean: float | None = None
        self.recurrence_log_m2 = 0.0
        self.recurrence_weight = 0.0
        self.last_external_seed_seconds: dict[int, float] = {}
        self.current_external: dict[int, float] = {}
        self.current_event = -1
        self._transition_cache: dict[
            int,
            tuple[tuple[tuple[int, float, int], ...], float],
        ] = {}

    def grow_turn_capacity(self, turn_count: int) -> None:
        """Grow turn slots and remap engram nodes without changing relations."""

        # 1. Reject shrinkage because persisted turn identities are append-only.
        if turn_count < self.turn_count:
            raise ValueError("memory graph turn capacity cannot shrink")
        if turn_count == self.turn_count:
            return
        old_turn_count = self.turn_count
        offset = turn_count - old_turn_count
        self._transition_cache.clear()

        # 2. Shift every engram above the expanded contiguous turn range.
        def remap(node_id: int) -> int:
            return node_id + offset if node_id >= old_turn_count else node_id

        self.source = [remap(node_id) for node_id in self.source]
        self.target = [remap(node_id) for node_id in self.target]
        self.hubs = [
            HubRecord(
                node_id=remap(hub.node_id),
                created_event=hub.created_event,
                current_turn_id=hub.current_turn_id,
                threshold=hub.threshold,
                innovation_mass=hub.innovation_mass,
                member_edge_ids=hub.member_edge_ids,
            )
            for hub in self.hubs
        ]
        self.hub_members = {
            remap(node_id): edge_ids
            for node_id, edge_ids in self.hub_members.items()
        }

        # 3. Rebuild adjacency deterministically with the remapped node IDs.
        self.turn_count = turn_count
        self.max_nodes = turn_count * 2
        self.adjacency = [[] for _ in range(self.max_nodes)]
        for edge_id, (source, target, bidirectional) in enumerate(
            zip(self.source, self.target, self.bidirectional, strict=True)
        ):
            self.adjacency[source].append(edge_id)
            if bidirectional:
                self.adjacency[target].append(edge_id)

    @property
    def resource_tau_seconds(self) -> float:
        """Return the causally learned short-to-long gap boundary."""

        if self.short_log_gap is None:
            return 1.0
        if self.long_log_gap is None:
            return math.exp(self.short_log_gap)
        return math.exp((self.short_log_gap + self.long_log_gap) / 2.0)

    @property
    def threshold_tau_seconds(self) -> float:
        """Return the causally learned long activity timescale."""

        if self.long_log_gap is not None:
            return math.exp(self.long_log_gap)
        if self.short_log_gap is not None:
            return math.exp(self.short_log_gap)
        return 1.0

    @property
    def retention_tau_seconds(self) -> float:
        """Return the learned long recurrence scale for new relations."""

        return max(
            self.threshold_tau_seconds,
            self.long_recurrence_tau_seconds,
        )

    @property
    def short_recurrence_tau_seconds(self) -> float:
        """Return the learned within-burst recurrence scale."""

        if self.short_recurrence_log_gap is None:
            return self.threshold_tau_seconds
        return math.exp(self.short_recurrence_log_gap)

    @property
    def long_recurrence_tau_seconds(self) -> float:
        """Return the learned across-burst recurrence scale."""

        if self.long_recurrence_log_gap is not None:
            return math.exp(self.long_recurrence_log_gap)
        return self.short_recurrence_tau_seconds

    def prepare_retrieval(
        self,
        event: int,
        gap_seconds: float | None,
        evidence: SeedEvidence,
    ) -> None:
        """Advance causal time and prepare local plasticity eligibility."""

        # 1. Advance the real-time clock and update online gap statistics.
        if event != self.current_event + 1:
            raise ValueError("memory events must be prepared in causal order")
        if gap_seconds is not None and gap_seconds < 0.0:
            raise ValueError("memory event gap cannot be negative")
        self.current_event = event
        gap = 0.0 if gap_seconds is None else gap_seconds
        self.elapsed_seconds += gap
        self._observe_gap(gap)
        self._transition_cache.clear()

        # 2. Record only query-owned evidence as independent reactivation.
        direct_nodes = evidence.channels.get(
            "query_dense",
            frozenset(),
        ) | evidence.channels.get("query_bm25", frozenset())
        seed = dict(evidence.seed)
        self.current_external = {
            node_id: seed[node_id]
            for node_id in sorted(direct_nodes)
            if node_id in seed and seed[node_id] > 0.0
        }
        self.current_external[event] = 1.0
        for node_id, credit in self.current_external.items():
            if node_id == event:
                continue
            previous = self.last_external_seed_seconds.get(node_id)
            if previous is not None:
                self._observe_recurrence(
                    self.elapsed_seconds - previous,
                    credit,
                )
            self.last_external_seed_seconds[node_id] = self.elapsed_seconds

    def transitions(
        self,
        node_id: int,
        event: int,
    ) -> tuple[tuple[tuple[int, float, int], ...], float]:
        """Return normalized outgoing mass from the current single state."""

        if event < self.current_event:
            raise ValueError("relation state cannot move backwards in time")
        cached = self._transition_cache.get(node_id)
        if cached is not None:
            return cached
        weighted: list[tuple[int, float, int]] = []
        for edge_id in self.adjacency[node_id]:
            edge_weight = self.effective_weight(edge_id)
            if edge_weight <= 0.0:
                continue
            target = self._outgoing_target(edge_id, node_id)
            if target is not None:
                weighted.append((target, edge_weight, edge_id))
        total = math.fsum(value for _, value, _ in weighted)
        if total == 0.0:
            result = ((), 1.0)
            self._transition_cache[node_id] = result
            return result
        spread = -math.expm1(-total)
        transitions = tuple(
            (target, spread * value / total, edge_id)
            for target, value, edge_id in weighted
        )
        result = (transitions, 1.0 - spread)
        self._transition_cache[node_id] = result
        return result

    def learn(
        self,
        event: int,
        evidence: SeedEvidence,
        diffusion: DiffusionResult,
    ) -> PlasticityResult:
        """Adapt exposed relations and store the current event in one state."""

        # 1. Convert direct and completed recall into one continuous activity field.
        self._transition_cache.clear()
        observed, activity = self._event_activity(event, evidence, diffusion)
        turn_nodes = np.flatnonzero(activity[: self.turn_count] > 0.0)
        values = activity[turn_nodes]
        threshold = float(np.dot(values, values) / math.fsum(values))
        integrated = self._integrated_members(turn_nodes, activity)

        # 2. Apply Oja-like competition to every old pattern exposed by retrieval.
        hub_positive, hub_negative = self._adapt_exposed_hubs(
            event,
            observed,
            activity,
        )
        temporal_positive, temporal_negative = self._adapt_temporal_relations(
            event,
            observed,
            activity,
        )
        reactivated = hub_positive + temporal_positive

        # 3. Store novelty once and bind causal direction with the same bounded state.
        write_gain = evidence.surprise
        hub_id, hub_written = self._create_hub(
            event,
            integrated,
            observed,
            threshold,
            write_gain,
        )
        temporal_written = self._update_temporal(
            event,
            integrated,
            observed,
            evidence.continuation,
            write_gain,
        )
        observed_mass = math.fsum(
            activity[node] for node in turn_nodes if observed[node]
        )
        recurrent_mass = math.fsum(
            activity[node] for node in turn_nodes if not observed[node]
        )
        return PlasticityResult(
            hub_node_id=hub_id,
            threshold=threshold,
            integrated=integrated,
            inhibited_mass=hub_negative + temporal_negative,
            potentiated_mass=reactivated + hub_written + temporal_written,
            observed_mass=observed_mass,
            recurrent_mass=recurrent_mass,
            reactivated_mass=reactivated,
        )

    def edge_relation_name(self, edge_id: int) -> str:
        return {
            MEMBERSHIP: "episode",
            TEMPORAL_FORWARD: "temporal_forward",
            TEMPORAL_BACKWARD: "temporal_backward",
        }[self.kind[edge_id]]

    def retention_factor(self, edge_id: int) -> float:
        """Return current accessibility from the learned recurrence survival."""

        if not self.config.forgetting_enabled:
            return 1.0
        age = (
            self.elapsed_seconds
            - self.last_support_seconds[edge_id]
        )
        if age < 0.0:
            raise ValueError("retention state cannot move backwards in time")
        return self.recurrence_survival(age)

    def recurrence_survival(self, age_seconds: float) -> float:
        """Evaluate the online two-lognormal recurrence survival."""

        if age_seconds <= 0.0 or self.recurrence_log_m2 == 0.0:
            return 1.0
        total = (
            self.short_recurrence_weight
            + self.long_recurrence_weight
        )
        if total == 0.0:
            return 1.0
        log_age = math.log(age_seconds)
        survival = 0.0
        components = (
            (
                self.short_recurrence_log_gap,
                self.short_recurrence_log_m2,
                self.short_recurrence_weight,
            ),
            (
                self.long_recurrence_log_gap,
                self.long_recurrence_log_m2,
                self.long_recurrence_weight,
            ),
        )
        pooled_sigma = math.sqrt(
            self.recurrence_log_m2 / self.recurrence_weight
        )
        for mean, m2, weight in components:
            if mean is None or weight == 0.0:
                continue
            sigma = math.sqrt(m2 / weight) if m2 > 0.0 else pooled_sigma
            if sigma == 0.0:
                component = float(log_age < mean)
            else:
                z_score = (log_age - mean) / (math.sqrt(2.0) * sigma)
                component = 0.5 * math.erfc(z_score)
            survival += weight * component / total
        return survival

    def effective_weight(self, edge_id: int) -> float:
        """Return raw learned strength times current accessibility."""

        return self.weight[edge_id] * self.retention_factor(edge_id)

    def _event_activity(
        self,
        event: int,
        evidence: SeedEvidence,
        diffusion: DiffusionResult,
    ) -> tuple[np.ndarray, np.ndarray]:
        observed = np.zeros(self.max_nodes, dtype=bool)
        activity = np.zeros(self.max_nodes, dtype=np.float64)
        observed[event] = True
        activity[event] = 1.0
        for node_id, value in evidence.seed:
            observed[node_id] = True
            activity[node_id] = max(activity[node_id], value)
        active_values = diffusion.reserve[diffusion.active_nodes]
        peak = float(np.max(active_values)) if active_values.size else 1.0
        if peak > 0.0:
            activity[diffusion.active_nodes] = np.maximum(
                activity[diffusion.active_nodes],
                active_values / peak,
            )
        return observed, activity

    def _integrated_members(
        self,
        turn_nodes: np.ndarray,
        activity: np.ndarray,
    ) -> tuple[tuple[int, float], ...]:
        """Project settled activity with adaptive Tsallis sparsity."""

        powered = np.power(
            activity[turn_nodes],
            self.config.activation_power,
        )
        multiplicity_scale = math.log1p(turn_nodes.size)
        weights = _entmax15(powered * multiplicity_scale)
        return tuple(
            (int(turn_nodes[index]), value)
            for index, value in weights
        )

    def _adapt_exposed_hubs(
        self,
        event: int,
        observed: np.ndarray,
        activity: np.ndarray,
    ) -> tuple[float, float]:
        """Strengthen coactive memberships and depress unsupported members."""

        exposed = sorted(
            {
                self.target[edge_id]
                for raw_node in np.flatnonzero(activity[: self.turn_count] > 0.0)
                for edge_id in self.adjacency[int(raw_node)]
                if self.kind[edge_id] == MEMBERSHIP
            }
        )
        potentiated = 0.0
        inhibited = 0.0
        affected_members: set[int] = set()
        for hub_node in exposed:
            edges = self.hub_members[hub_node]
            hub_activity = math.fsum(
                self.effective_weight(edge_id)
                * activity[self.source[edge_id]]
                for edge_id in edges
            )
            if hub_activity == 0.0:
                continue
            external_total = math.fsum(
                self.current_external.get(self.source[edge_id], 0.0)
                for edge_id in edges
            )
            for edge_id in edges:
                old = self.weight[edge_id]
                member = self.source[edge_id]
                member_activity = float(activity[member])
                eligibility, _ = self._stimulate_edge(
                    edge_id,
                    hub_activity,
                )
                delta = (
                    self.config.learning_rate
                    * eligibility
                    * hub_activity
                    * (member_activity - hub_activity * old)
                )
                updated = max(0.0, old + delta)
                self.weight[edge_id] = updated
                self.last_updated[edge_id] = event
                potentiated += max(updated - old, 0.0)
                inhibited += max(old - updated, 0.0)
                if member_activity > 0.0:
                    if observed[member]:
                        self.observed_credit[edge_id] += member_activity
                    else:
                        self.recurrent_credit[edge_id] += member_activity
                member_external = self.current_external.get(member, 0.0)
                other_external = max(
                    0.0,
                    external_total - member_external,
                )
                independent = math.sqrt(
                    member_external * other_external
                )
                self._support_edge(
                    edge_id,
                    math.sqrt(member_activity * hub_activity),
                    independent,
                )
                affected_members.add(member)
            inhibited += self._normalize_hub(hub_node)
        for member in sorted(affected_members):
            inhibited += self._normalize_membership_source(member)
        return potentiated, inhibited

    def _adapt_temporal_relations(
        self,
        event: int,
        observed: np.ndarray,
        activity: np.ndarray,
    ) -> tuple[float, float]:
        """Apply directional Hebbian prediction and source-side LTD."""

        selected = sorted(
            {
                edge_id
                for raw_node in np.flatnonzero(activity[: self.turn_count] > 0.0)
                for edge_id in self.adjacency[int(raw_node)]
                if self.kind[edge_id] != MEMBERSHIP
            }
        )
        potentiated = 0.0
        inhibited = 0.0
        affected: set[tuple[int, int]] = set()
        for edge_id in selected:
            source = self.source[edge_id]
            target = self.target[edge_id]
            source_activity = float(activity[source])
            if source_activity == 0.0:
                continue
            old = self.weight[edge_id]
            target_activity = float(activity[target])
            eligibility, _ = self._stimulate_edge(
                edge_id,
                source_activity,
            )
            delta = (
                self.config.learning_rate
                * eligibility
                * source_activity
                * (target_activity - source_activity * old)
            )
            updated = max(0.0, old + delta)
            self.weight[edge_id] = updated
            self.last_updated[edge_id] = event
            potentiated += max(updated - old, 0.0)
            inhibited += max(old - updated, 0.0)
            credit = source_activity * target_activity
            if observed[source] and observed[target]:
                self.observed_credit[edge_id] += credit
            else:
                self.recurrent_credit[edge_id] += credit
            independent = math.sqrt(
                self.current_external.get(source, 0.0)
                * self.current_external.get(target, 0.0)
            )
            self._support_edge(
                edge_id,
                math.sqrt(source_activity * target_activity),
                independent,
            )
            affected.add((source, self.kind[edge_id]))
        for source, kind in sorted(affected):
            inhibited += self._normalize_temporal_source(source, kind)
        return potentiated, inhibited

    def _create_hub(
        self,
        event: int,
        integrated: tuple[tuple[int, float], ...],
        observed: np.ndarray,
        threshold: float,
        write_gain: float,
    ) -> tuple[int | None, float]:
        if len(integrated) < 2 or write_gain <= 0.0:
            return None, 0.0
        values = tuple(
            (node_id, weight * write_gain)
            for node_id, weight in integrated
        )
        total = math.fsum(value for _, value in values)
        scale = min(1.0, self.config.recurrent_budget / total)
        hub_node = self.turn_count + event
        member_edges: list[int] = []
        external_total = math.fsum(
            self.current_external.get(node_id, 0.0)
            for node_id, _ in values
        )
        integrated_by_node = dict(integrated)
        for node_id, value in values:
            weighted = value * scale
            member_external = self.current_external.get(node_id, 0.0)
            independent = write_gain * math.sqrt(
                member_external
                * max(0.0, external_total - member_external)
            )
            member_activity = integrated_by_node[node_id]
            support = write_gain * math.sqrt(
                member_activity * max(0.0, 1.0 - member_activity)
            )
            member_edges.append(
                self._add_edge(
                    source=node_id,
                    target=hub_node,
                    kind=MEMBERSHIP,
                    bidirectional=True,
                    event=event,
                    initial_weight=weighted,
                    observed_credit=weighted if observed[node_id] else 0.0,
                    recurrent_credit=0.0 if observed[node_id] else weighted,
                    support_credit=support,
                    independent_credit=independent,
                )
            )
        self.hub_members[hub_node] = member_edges
        for node_id, _ in values:
            self._normalize_membership_source(node_id)
        innovation = math.fsum(self.weight[edge_id] for edge_id in member_edges)
        self.hubs.append(
            HubRecord(
                node_id=hub_node,
                created_event=event,
                current_turn_id=event,
                threshold=threshold,
                innovation_mass=innovation,
                member_edge_ids=tuple(member_edges),
            )
        )
        return hub_node, innovation

    def _update_temporal(
        self,
        event: int,
        integrated: tuple[tuple[int, float], ...],
        observed: np.ndarray,
        continuation: float,
        write_gain: float,
    ) -> float:
        past = tuple((node, weight) for node, weight in integrated if node != event)
        total = math.fsum(weight for _, weight in past)
        if total == 0.0 or write_gain <= 0.0:
            return 0.0
        written = 0.0
        affected: set[tuple[int, int]] = set()
        for node_id, weight in past:
            forward = continuation * write_gain * weight / total
            reverse = self.config.reverse_temporal_ratio * forward
            written += self._upsert_temporal(
                TEMPORAL_FORWARD,
                node_id,
                event,
                event,
                forward,
                bool(observed[node_id]),
            )
            written += self._upsert_temporal(
                TEMPORAL_BACKWARD,
                event,
                node_id,
                event,
                reverse,
                bool(observed[node_id]),
            )
            affected.add((node_id, TEMPORAL_FORWARD))
            affected.add((event, TEMPORAL_BACKWARD))
        for source, kind in sorted(affected):
            self._normalize_temporal_source(source, kind)
        return written

    def _upsert_temporal(
        self,
        kind: int,
        source: int,
        target: int,
        event: int,
        signal: float,
        observed: bool,
    ) -> float:
        key = (kind, source, target)
        edge_id = self.temporal_lookup.get(key)
        if edge_id is None:
            independent = math.sqrt(
                self.current_external.get(source, 0.0)
                * self.current_external.get(target, 0.0)
            )
            edge_id = self._add_edge(
                source=source,
                target=target,
                kind=kind,
                bidirectional=False,
                event=event,
                initial_weight=signal,
                observed_credit=signal if observed else 0.0,
                recurrent_credit=0.0 if observed else signal,
                support_credit=signal,
                independent_credit=independent,
            )
            self.temporal_lookup[key] = edge_id
            return signal
        old = self.weight[edge_id]
        updated = old + self.config.learning_rate * signal * (1.0 - old)
        self.weight[edge_id] = min(1.0, updated)
        self.last_updated[edge_id] = event
        if observed:
            self.observed_credit[edge_id] += signal
        else:
            self.recurrent_credit[edge_id] += signal
        independent = math.sqrt(
            self.current_external.get(source, 0.0)
            * self.current_external.get(target, 0.0)
        )
        self._support_edge(edge_id, signal, independent)
        return self.weight[edge_id] - old

    def _normalize_hub(self, hub_node: int) -> float:
        edges = self.hub_members[hub_node]
        total = math.fsum(self.weight[edge_id] for edge_id in edges)
        budget = self.config.recurrent_budget
        if total <= budget:
            return 0.0
        scale = budget / total
        inhibited = 0.0
        for edge_id in edges:
            old = self.weight[edge_id]
            self.weight[edge_id] = old * scale
            inhibited += old - self.weight[edge_id]
        return inhibited

    def _normalize_membership_source(self, source: int) -> float:
        """Keep one turn's total episodic conductance within its local budget."""

        edges = [
            edge_id
            for edge_id in self.adjacency[source]
            if self.kind[edge_id] == MEMBERSHIP
            and self.source[edge_id] == source
        ]
        total = math.fsum(self.weight[edge_id] for edge_id in edges)
        budget = self.config.recurrent_budget
        if total <= budget:
            return 0.0
        scale = budget / total
        inhibited = 0.0
        for edge_id in edges:
            old = self.weight[edge_id]
            self.weight[edge_id] = old * scale
            inhibited += old - self.weight[edge_id]
        return inhibited

    def _normalize_temporal_source(self, source: int, kind: int) -> float:
        edges = [
            edge_id
            for edge_id in self.adjacency[source]
            if self.kind[edge_id] == kind
        ]
        total = math.fsum(self.weight[edge_id] for edge_id in edges)
        budget = (
            self.config.recurrent_budget
            if kind == TEMPORAL_FORWARD
            else self.config.reverse_temporal_ratio * self.config.recurrent_budget
        )
        if total <= budget:
            return 0.0
        scale = budget / total
        inhibited = 0.0
        for edge_id in edges:
            old = self.weight[edge_id]
            self.weight[edge_id] = old * scale
            inhibited += old - self.weight[edge_id]
        return inhibited

    def _outgoing_target(self, edge_id: int, node_id: int) -> int | None:
        if self.bidirectional[edge_id]:
            if self.source[edge_id] == node_id:
                return self.target[edge_id]
            if self.target[edge_id] == node_id:
                return self.source[edge_id]
            raise AssertionError("membership adjacency is inconsistent")
        return self.target[edge_id] if self.source[edge_id] == node_id else None

    def _add_edge(
        self,
        *,
        source: int,
        target: int,
        kind: int,
        bidirectional: bool,
        event: int,
        initial_weight: float,
        observed_credit: float,
        recurrent_credit: float,
        support_credit: float = 0.0,
        independent_credit: float = 0.0,
    ) -> int:
        edge_id = len(self.source)
        self.source.append(source)
        self.target.append(target)
        self.kind.append(kind)
        self.bidirectional.append(bidirectional)
        self.weight.append(max(0.0, initial_weight))
        self.last_updated.append(event)
        self.created_event.append(event)
        self.observed_credit.append(observed_credit)
        self.recurrent_credit.append(recurrent_credit)
        activity = max(0.0, initial_weight)
        self.resource.append(math.exp(-activity))
        self.plasticity_threshold.append(-math.expm1(-(activity * activity)))
        self.last_stimulated_seconds.append(self.elapsed_seconds)
        self.last_support_seconds.append(self.elapsed_seconds)
        self.support_credit.append(support_credit)
        self.independent_credit.append(independent_credit)
        self.adjacency[source].append(edge_id)
        if bidirectional:
            self.adjacency[target].append(edge_id)
        return edge_id

    def _observe_recurrence(self, gap_seconds: float, credit: float) -> None:
        """Update an online two-lognormal recurrence model causally."""

        if gap_seconds <= 0.0 or credit <= 0.0:
            return
        value = math.log(gap_seconds)

        # 1. Maintain pooled variance for cold-component prediction.
        if self.recurrence_log_mean is None:
            self.recurrence_log_mean = value
            self.recurrence_weight = credit
        else:
            total = self.recurrence_weight + credit
            delta = value - self.recurrence_log_mean
            self.recurrence_log_mean += credit * delta / total
            self.recurrence_log_m2 += (
                credit
                * delta
                * (value - self.recurrence_log_mean)
            )
            self.recurrence_weight = total

        # 2. Initialize the within- and across-burst components.
        if self.short_recurrence_log_gap is None:
            self.short_recurrence_log_gap = value
            self.short_recurrence_weight = credit
            return
        if self.long_recurrence_log_gap is None:
            if value < self.short_recurrence_log_gap:
                self.long_recurrence_log_gap = self.short_recurrence_log_gap
                self.long_recurrence_log_m2 = (
                    self.short_recurrence_log_m2
                )
                self.long_recurrence_weight = self.short_recurrence_weight
                self.short_recurrence_log_gap = value
                self.short_recurrence_log_m2 = 0.0
                self.short_recurrence_weight = credit
            else:
                self.long_recurrence_log_gap = value
                self.long_recurrence_weight = credit
            return

        # 3. Assign by log-distance and update weighted sufficient statistics.
        short_distance = abs(value - self.short_recurrence_log_gap)
        long_distance = abs(value - self.long_recurrence_log_gap)
        if short_distance <= long_distance:
            total = self.short_recurrence_weight + credit
            delta = value - self.short_recurrence_log_gap
            self.short_recurrence_log_gap += (
                credit * delta / total
            )
            self.short_recurrence_log_m2 += (
                credit
                * delta
                * (value - self.short_recurrence_log_gap)
            )
            self.short_recurrence_weight = total
        else:
            total = self.long_recurrence_weight + credit
            delta = value - self.long_recurrence_log_gap
            self.long_recurrence_log_gap += (
                credit * delta / total
            )
            self.long_recurrence_log_m2 += (
                credit
                * delta
                * (value - self.long_recurrence_log_gap)
            )
            self.long_recurrence_weight = total
        if self.short_recurrence_log_gap > self.long_recurrence_log_gap:
            (
                self.short_recurrence_log_gap,
                self.long_recurrence_log_gap,
            ) = (
                self.long_recurrence_log_gap,
                self.short_recurrence_log_gap,
            )
            (
                self.short_recurrence_log_m2,
                self.long_recurrence_log_m2,
            ) = (
                self.long_recurrence_log_m2,
                self.short_recurrence_log_m2,
            )
            (
                self.short_recurrence_weight,
                self.long_recurrence_weight,
            ) = (
                self.long_recurrence_weight,
                self.short_recurrence_weight,
            )

    def _support_edge(
        self,
        edge_id: int,
        credit: float,
        independent_credit: float,
    ) -> None:
        """Partially reconsolidate one edge from continuous coactivation."""

        if credit <= 0.0:
            return
        previous = self.last_support_seconds[edge_id]
        spacing = self.elapsed_seconds - previous
        if spacing < 0.0:
            raise ValueError("relation support cannot move backwards in time")
        renewal = -math.expm1(-credit)
        self.last_support_seconds[edge_id] = previous + renewal * spacing
        self.support_credit[edge_id] += credit
        self.independent_credit[edge_id] += independent_credit

    def _observe_gap(self, gap_seconds: float) -> None:
        """Update two causal geometric gap scales without future information."""

        if gap_seconds <= 0.0:
            return
        value = math.log(gap_seconds)
        if self.short_log_gap is None:
            self.short_log_gap = value
            self.short_gap_count = 1
            return
        if self.long_log_gap is None:
            if value < self.short_log_gap:
                self.long_log_gap = self.short_log_gap
                self.long_gap_count = self.short_gap_count
                self.short_log_gap = value
                self.short_gap_count = 1
            else:
                self.long_log_gap = value
                self.long_gap_count = 1
            return
        short_distance = abs(value - self.short_log_gap)
        long_distance = abs(value - self.long_log_gap)
        if short_distance <= long_distance:
            self.short_gap_count += 1
            self.short_log_gap += (
                value - self.short_log_gap
            ) / self.short_gap_count
        else:
            self.long_gap_count += 1
            self.long_log_gap += (
                value - self.long_log_gap
            ) / self.long_gap_count
        if self.short_log_gap > self.long_log_gap:
            self.short_log_gap, self.long_log_gap = (
                self.long_log_gap,
                self.short_log_gap,
            )
            self.short_gap_count, self.long_gap_count = (
                self.long_gap_count,
                self.short_gap_count,
            )

    def _stimulate_edge(
        self,
        edge_id: int,
        activity: float,
    ) -> tuple[float, float]:
        """Return resource-gated eligibility and its adaptive threshold."""

        resource, threshold = self._recover_plasticity(
            self.resource[edge_id],
            self.plasticity_threshold[edge_id],
            self.last_stimulated_seconds[edge_id],
        )
        eligibility = self._eligibility(resource, threshold, activity)
        self.resource[edge_id] = resource * math.exp(-activity)
        self.plasticity_threshold[edge_id] = self._raised_threshold(
            threshold,
            activity,
        )
        self.last_stimulated_seconds[edge_id] = self.elapsed_seconds
        return eligibility, threshold

    def _recover_plasticity(
        self,
        resource: float,
        threshold: float,
        last_stimulated: float,
    ) -> tuple[float, float]:
        if not math.isfinite(last_stimulated):
            return resource, threshold
        elapsed = self.elapsed_seconds - last_stimulated
        if elapsed < 0.0:
            raise ValueError("plasticity state cannot move backwards in time")
        recovered = 1.0 - (1.0 - resource) * math.exp(
            -elapsed / self.resource_tau_seconds
        )
        lowered = threshold * math.exp(
            -elapsed / self.threshold_tau_seconds
        )
        return recovered, lowered

    @staticmethod
    def _eligibility(resource: float, threshold: float, activity: float) -> float:
        drive = activity * max(activity - threshold, 0.0)
        return resource * -math.expm1(-drive)

    @staticmethod
    def _raised_threshold(threshold: float, activity: float) -> float:
        return threshold + (1.0 - threshold) * -math.expm1(
            -(activity * activity)
        )


def _entmax15(logits: np.ndarray) -> tuple[tuple[int, float], ...]:
    """Project logits onto a sparse simplex without a fixed support size."""

    # 1. Solve the 1.5-entmax threshold deterministically.
    alpha = 1.5
    exponent = 1.0 / (alpha - 1.0)
    lower = float(np.min(logits)) - exponent
    upper = float(np.max(logits))
    for _ in range(80):
        threshold = (lower + upper) / 2.0
        values = (
            np.maximum((alpha - 1.0) * (logits - threshold), 0.0)
            ** exponent
        )
        if float(np.sum(values)) > 1.0:
            lower = threshold
        else:
            upper = threshold

    # 2. Normalize numerical residue while preserving the learned support.
    values = (
        np.maximum((alpha - 1.0) * (logits - upper), 0.0)
        ** exponent
    )
    values /= float(np.sum(values))
    return tuple(
        (int(node), float(values[node]))
        for node in np.flatnonzero(values > 0.0)
    )
