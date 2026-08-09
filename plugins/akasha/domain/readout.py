"""Read V8 multi-basin pattern completion from a prepared memory graph."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .diffusion import residual_push
from .features import (
    FeaturePool,
    _sparsemax,
    _tail_surprisal,
    binary_entropy,
)
from .graph import (
    TEMPORAL_BACKWARD,
    TEMPORAL_FORWARD,
    DynamicMemoryGraph,
)
from .model import (
    ContextState,
    DiffusionResult,
    MemoryConfig,
    SeedEvidence,
    Turn,
)


@dataclass(frozen=True)
class Basin:
    """Hold one historical engram and its cue-conditioned evidence."""

    hub_id: int
    score: float
    members: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class _BasinStructure:
    """Cache cue-independent members and normalized weights for one read."""

    hub_id: int
    members: tuple[int, ...]
    weights: tuple[float, ...]
    member_array: np.ndarray
    normalized: np.ndarray
    conductance: float


@dataclass(frozen=True)
class RecallItem:
    """Expose one completed historical turn with auditable sources."""

    node_id: int
    score: float
    sources: tuple[str, ...]
    basin_ids: tuple[str, ...]


@dataclass(frozen=True)
class PatternCompletion:
    """Return the set-valued V8 readout and its convergence diagnostics."""

    items: tuple[RecallItem, ...]
    active_basin_count: int
    sharp_completion_count: int
    basin_direct_count: int
    basin_completion_count: int
    relative_tail_count: int
    pushes: int
    residual_l1: float


@dataclass(frozen=True)
class RecallCapture:
    """Bind one committed query turn to its explicit completion result."""

    query_node_id: int
    completion: PatternCompletion


class TemporalGraphView:
    """Expose only directed temporal relations of a prepared graph."""

    def __init__(self, graph: DynamicMemoryGraph) -> None:
        self.graph = graph
        self.max_nodes = graph.max_nodes
        self._transition_cache: dict[
            int,
            tuple[tuple[tuple[int, float, int], ...], float],
        ] = {}

    def transitions(
        self,
        node_id: int,
        event: int,
    ) -> tuple[tuple[tuple[int, float, int], ...], float]:
        cached = self._transition_cache.get(node_id)
        if cached is not None:
            return cached
        weighted: list[tuple[int, float, int]] = []
        for edge_id in self.graph.adjacency[node_id]:
            if self.graph.kind[edge_id] not in (
                TEMPORAL_FORWARD,
                TEMPORAL_BACKWARD,
            ):
                continue
            weight = self.graph.effective_weight(edge_id)
            if weight <= 0.0:
                continue
            target = self.graph._outgoing_target(  # noqa: SLF001
                edge_id,
                node_id,
            )
            if target is not None:
                weighted.append((target, weight, edge_id))
        total = math.fsum(weight for _, weight, _ in weighted)
        if total == 0.0:
            result = ((), 1.0)
            self._transition_cache[node_id] = result
            return result
        spread = -math.expm1(-total)
        result = (
            tuple(
                (target, spread * weight / total, edge_id)
                for target, weight, edge_id in weighted
            ),
            1.0 - spread,
        )
        self._transition_cache[node_id] = result
        return result


def read_pattern_completion(
    *,
    graph: DynamicMemoryGraph,
    pool: FeaturePool,
    query: Turn,
    context: ContextState,
    precomputed_fields: dict[str, np.ndarray] | None = None,
    evidence: SeedEvidence,
    diffusion: DiffusionResult,
    historical_surprise: np.ndarray,
    config: MemoryConfig,
    context_dependence: float,
    visible_nodes: tuple[int, ...] = (),
    burst_continued: bool = True,
    inhibited_nodes: frozenset[int] = frozenset(),
) -> PatternCompletion:
    """Read contextual and independent basin routes without mutating memory."""

    # 1. Snapshot cue-independent basin structure once for both routes.
    basin_structures = _active_basin_structures(graph, query.node_id)

    # 2. Preserve the contextual V8 route as the non-destructive baseline.
    contextual = _exclude_recall_items(
        _read_contextual_route(
            graph=graph,
            pool=pool,
            query=query,
            context=context,
            precomputed_fields=precomputed_fields,
            evidence=evidence,
            diffusion=diffusion,
            historical_surprise=historical_surprise,
            config=config,
            visible_nodes=visible_nodes,
            burst_continued=burst_continued,
            basin_structures=basin_structures,
        ),
        inhibited_nodes,
    )

    # 3. Inhibit query-only routing when the cue depends on its active context.
    address_mass = _independent_address_mass(context_dependence)
    if address_mass == 0.0:
        return contextual
    address = _exclude_recall_items(
        _read_independent_route(
            graph=graph,
            pool=pool,
            query=query,
            precomputed_fields=precomputed_fields,
            surprise=evidence.surprise,
            historical_surprise=historical_surprise,
            config=config,
            visible_nodes=visible_nodes,
            basin_structures=basin_structures,
        ),
        inhibited_nodes,
    )

    # 4. Let address-only completions compete without evicting baseline items.
    return _competitive_route_union(
        contextual,
        address,
        address_mass,
    )


def _exclude_recall_items(
    completion: PatternCompletion,
    excluded_nodes: frozenset[int],
) -> PatternCompletion:
    """Hide suppressed turns after they have served as graph bridges."""

    if not excluded_nodes:
        return completion
    items = tuple(
        item
        for item in completion.items
        if item.node_id not in excluded_nodes
    )
    counts = {
        source: sum(source in item.sources for item in items)
        for source in (
            "sharp_completion",
            "basin_direct",
            "basin_completion",
            "relative_tail",
        )
    }
    return PatternCompletion(
        items=items,
        active_basin_count=completion.active_basin_count,
        sharp_completion_count=counts["sharp_completion"],
        basin_direct_count=counts["basin_direct"],
        basin_completion_count=counts["basin_completion"],
        relative_tail_count=counts["relative_tail"],
        pushes=completion.pushes,
        residual_l1=completion.residual_l1,
    )


def _read_contextual_route(
    *,
    graph: DynamicMemoryGraph,
    pool: FeaturePool,
    query: Turn,
    context: ContextState,
    precomputed_fields: dict[str, np.ndarray] | None,
    evidence: SeedEvidence,
    diffusion: DiffusionResult,
    historical_surprise: np.ndarray,
    config: MemoryConfig,
    visible_nodes: tuple[int, ...],
    burst_continued: bool,
    basin_structures: tuple[_BasinStructure, ...],
) -> PatternCompletion:
    """Settle the existing current-cue plus active-context route."""

    fields = (
        _reuse_evidence_fields(
            precomputed_fields,
            include_context=burst_continued,
        )
        if precomputed_fields is not None
        else _evidence_fields(pool, query.node_id, context)
    )
    scores = fields["current"] + evidence.continuation * (
        fields["same_event"] - fields["current"]
    )
    return _read_route(
        graph=graph,
        query=query,
        fields=fields,
        basin_scores=scores,
        continuation=evidence.continuation,
        surprise=evidence.surprise,
        diffusion=diffusion,
        historical_surprise=historical_surprise,
        config=config,
        visible_nodes=visible_nodes,
        burst_continued=burst_continued,
        basin_structures=basin_structures,
    )


def _read_independent_route(
    *,
    graph: DynamicMemoryGraph,
    pool: FeaturePool,
    query: Turn,
    precomputed_fields: dict[str, np.ndarray] | None,
    surprise: float,
    historical_surprise: np.ndarray,
    config: MemoryConfig,
    visible_nodes: tuple[int, ...],
    basin_structures: tuple[_BasinStructure, ...],
) -> PatternCompletion:
    """Settle a query-only graph-address route without active context."""

    fields = (
        _reuse_evidence_fields(
            precomputed_fields,
            include_context=False,
        )
        if precomputed_fields is not None
        else _evidence_fields(
            pool,
            query.node_id,
            ContextState((), None, ()),
        )
    )
    seed = _sparsemax(fields["current"])
    diffusion = residual_push(
        graph,
        seed,
        query.node_id,
        restart=config.restart,
        tolerance=config.tolerance,
        capture_paths=False,
    )
    return _read_route(
        graph=graph,
        query=query,
        fields=fields,
        basin_scores=diffusion.reserve[: query.node_id],
        continuation=0.0,
        surprise=surprise,
        diffusion=diffusion,
        historical_surprise=historical_surprise,
        config=config,
        visible_nodes=visible_nodes,
        burst_continued=False,
        basin_structures=basin_structures,
    )


def _read_route(
    *,
    graph: DynamicMemoryGraph,
    query: Turn,
    fields: dict[str, np.ndarray],
    basin_scores: np.ndarray,
    continuation: float,
    surprise: float,
    diffusion: DiffusionResult,
    historical_surprise: np.ndarray,
    config: MemoryConfig,
    visible_nodes: tuple[int, ...],
    burst_continued: bool,
    basin_structures: tuple[_BasinStructure, ...],
) -> PatternCompletion:
    """Settle one independently normalized basin-routing hypothesis."""

    # 1. Select live engram heads from this route's evidence.
    basins = _score_active_basins(basin_structures, basin_scores)
    temperature = _surprise_temperature(
        surprise,
        historical_surprise,
    )
    if not burst_continued:
        temperature *= binary_entropy(continuation)
    heads = _accessibility_supported_heads(
        graph,
        _pooled_heads(
            basins,
            fields,
            continuation,
            temperature,
        ),
    )
    return _materialize_route_completion(
        graph=graph,
        query=query,
        fields=fields,
        diffusion=diffusion,
        heads=heads,
        continuation=continuation,
        config=config,
        visible_nodes=visible_nodes,
    )


def _materialize_route_completion(
    *,
    graph: DynamicMemoryGraph,
    query: Turn,
    fields: dict[str, np.ndarray],
    diffusion: DiffusionResult,
    heads: list[tuple[str, float, tuple[tuple[int, float], ...]]],
    continuation: float,
    config: MemoryConfig,
    visible_nodes: tuple[int, ...],
) -> PatternCompletion:
    """Diffuse selected heads and expose one route's stable readout."""

    # 1. Diffuse the sharp cue and every selected basin independently.
    components = _diffuse_heads(
        graph=graph,
        query=query,
        fields=fields,
        sharp_diffusion=diffusion,
        heads=heads,
        continuation=continuation,
        config=config,
    )

    # 2. Produce a stable set-valued readout with source provenance.
    visible = set(visible_nodes)
    items = tuple(
        item
        for item in _recall_items(
            components=components,
            sharp_reserve=diffusion.reserve,
        )
        if item.node_id not in visible
    )
    source_counts = {
        source: sum(source in item.sources for item in items)
        for source in (
            "sharp_completion",
            "basin_direct",
            "basin_completion",
            "relative_tail",
        )
    }
    return PatternCompletion(
        items=items,
        active_basin_count=len(heads),
        sharp_completion_count=source_counts["sharp_completion"],
        basin_direct_count=source_counts["basin_direct"],
        basin_completion_count=source_counts["basin_completion"],
        relative_tail_count=source_counts["relative_tail"],
        pushes=diffusion.pushes,
        residual_l1=diffusion.residual_l1,
    )


def _independent_address_mass(context_dependence: float) -> float:
    """Project contextual and independent route evidence onto a sparse simplex."""

    if context_dependence == 0.0:
        return 1.0
    if context_dependence == 1.0:
        return 0.0
    evidence = np.log(
        np.asarray(
            [1.0 - context_dependence, context_dependence],
            dtype=np.float64,
        )
    )
    return dict(_sparsemax(evidence)).get(0, 0.0)


def _competitive_route_union(
    contextual: PatternCompletion,
    address: PatternCompletion,
    address_mass: float,
) -> PatternCompletion:
    """Admit sparse address completions while preserving the baseline route."""

    # 1. Merge provenance but expose only address-side pattern completion.
    contextual_nodes = {item.node_id for item in contextual.items}
    contextual_scores = {
        item.node_id: item.score for item in contextual.items
    }
    items = _merge_address_completions(contextual, address)
    ordered = tuple(
        sorted(items.values(), key=lambda item: (-item.score, item.node_id))
    )

    # 2. Apply one shared lateral competition to address-only additions.
    supported = _competitive_address_support(
        ordered,
        contextual_scores,
        address_mass,
    )
    selected = tuple(
        item
        for item in ordered
        if item.node_id in contextual_nodes or item.node_id in supported
    )
    return _union_completion(contextual, address, selected)


def _merge_address_completions(
    contextual: PatternCompletion,
    address: PatternCompletion,
) -> dict[int, RecallItem]:
    """Merge address completion provenance into contextual recall items."""

    items = {item.node_id: item for item in contextual.items}
    for item in address.items:
        if "basin_completion" not in item.sources:
            continue
        previous = items.get(item.node_id)
        if previous is None:
            items[item.node_id] = item
            continue
        items[item.node_id] = RecallItem(
            node_id=item.node_id,
            score=max(previous.score, item.score),
            sources=tuple(sorted(set(previous.sources) | set(item.sources))),
            basin_ids=tuple(
                sorted(set(previous.basin_ids) | set(item.basin_ids))
            ),
        )
    return items


def _competitive_address_support(
    ordered: tuple[RecallItem, ...],
    contextual_scores: dict[int, float],
    address_mass: float,
) -> set[int]:
    """Return address additions surviving shared sparse competition."""

    scores = np.asarray(
        [
            max(
                contextual_scores.get(
                    item.node_id,
                    address_mass * item.score,
                ),
                0.0,
            )
            for item in ordered
        ],
        dtype=np.float64,
    )
    peak = float(np.max(scores)) if scores.size else 0.0
    supported = (
        {
            ordered[index].node_id
            for index, _ in _sparsemax(scores / peak)
        }
        if peak > 0.0
        else set()
    )
    return supported


def _union_completion(
    contextual: PatternCompletion,
    address: PatternCompletion,
    selected: tuple[RecallItem, ...],
) -> PatternCompletion:
    """Recompute observable diagnostics for one protected route union."""

    source_counts = {
        source: sum(source in item.sources for item in selected)
        for source in (
            "sharp_completion",
            "basin_direct",
            "basin_completion",
            "relative_tail",
        )
    }
    return PatternCompletion(
        items=selected,
        active_basin_count=(
            contextual.active_basin_count + address.active_basin_count
        ),
        sharp_completion_count=source_counts["sharp_completion"],
        basin_direct_count=source_counts["basin_direct"],
        basin_completion_count=source_counts["basin_completion"],
        relative_tail_count=source_counts["relative_tail"],
        pushes=contextual.pushes + address.pushes,
        residual_l1=max(contextual.residual_l1, address.residual_l1),
    )


@dataclass
class _CompletionComponents:
    sharp: set[int]
    direct: set[int]
    completion: set[int]
    tail: set[int]
    basin_mass: dict[int, float]
    tail_mass: dict[int, float]
    basin_ids: dict[int, set[str]]


def _diffuse_heads(
    *,
    graph: DynamicMemoryGraph,
    query: Turn,
    fields: dict[str, np.ndarray],
    sharp_diffusion: DiffusionResult,
    heads: list[tuple[str, float, tuple[tuple[int, float], ...]]],
    continuation: float,
    config: MemoryConfig,
) -> _CompletionComponents:
    """Diffuse selected basins and retain the V8 local entmax tail."""

    sharp = _completion_support(
        sharp_diffusion,
        (),
        query.node_id,
        config.restart,
    )
    sharp_posterior = _turn_posterior(
        sharp_diffusion,
        query.node_id,
    )
    direct = {node for _, _, seed in heads for node, _ in seed}
    completion: set[int] = set()
    basin_mass: dict[int, float] = {}
    tail_mass: dict[int, float] = {}
    basin_ids: dict[int, set[str]] = {}
    temporal = TemporalGraphView(graph)
    for head_id, mass, seed in heads:
        _diffuse_one_head(
            graph=graph,
            temporal=temporal,
            query=query,
            head_id=head_id,
            head_mass=mass,
            seed=seed,
            sharp_posterior=sharp_posterior,
            direct=direct,
            completion=completion,
            basin_mass=basin_mass,
            tail_mass=tail_mass,
            basin_ids=basin_ids,
            config=config,
        )
    shared = _shared_tail(tail_mass)
    tail = _filter_tail_by_agreement(
        shared,
        tail_mass,
        _cue_agreement(fields, continuation),
    )
    return _CompletionComponents(
        sharp,
        direct,
        completion,
        tail,
        basin_mass,
        tail_mass,
        basin_ids,
    )


def _diffuse_one_head(
    *,
    graph: DynamicMemoryGraph,
    temporal: TemporalGraphView,
    query: Turn,
    head_id: str,
    head_mass: float,
    seed: tuple[tuple[int, float], ...],
    sharp_posterior: np.ndarray,
    direct: set[int],
    completion: set[int],
    basin_mass: dict[int, float],
    tail_mass: dict[int, float],
    basin_ids: dict[int, set[str]],
    config: MemoryConfig,
) -> None:
    basin_diffusion = residual_push(
        graph,
        seed,
        query.node_id,
        restart=config.restart,
        tolerance=config.tolerance,
        capture_paths=False,
    )
    temporal_diffusion = residual_push(
        temporal,
        seed,
        query.node_id,
        restart=config.restart,
        tolerance=config.tolerance,
        capture_paths=False,
    )
    local_completion = _completion_support(
        basin_diffusion,
        seed,
        query.node_id,
        config.restart,
    )
    completion.update(local_completion)
    posterior = _turn_posterior(basin_diffusion, query.node_id)
    temporal_posterior = _turn_posterior(
        temporal_diffusion,
        query.node_id,
    )
    relative = _relative_tail_entmax15(
        posterior,
        sharp_posterior,
        temporal_posterior,
        direct,
    )
    for node, value in seed:
        basin_mass[node] = basin_mass.get(node, 0.0) + head_mass * value
        basin_ids.setdefault(node, set()).add(head_id)
    for node in local_completion:
        basin_mass[node] = (
            basin_mass.get(node, 0.0)
            + head_mass * float(posterior[node])
        )
        basin_ids.setdefault(node, set()).add(head_id)
    for node, value in relative:
        tail_mass[node] = tail_mass.get(node, 0.0) + head_mass * value
        basin_ids.setdefault(node, set()).add(head_id)


def _recall_items(
    *,
    components: _CompletionComponents,
    sharp_reserve: np.ndarray,
) -> tuple[RecallItem, ...]:
    nodes = (
        components.sharp
        | components.direct
        | components.completion
        | components.tail
    )
    items = []
    for node in nodes:
        sources = _node_sources(node, components)
        score = max(
            float(sharp_reserve[node]),
            components.basin_mass.get(node, 0.0),
            components.tail_mass.get(node, 0.0),
        )
        items.append(
            RecallItem(
                node_id=node,
                score=score,
                sources=sources,
                basin_ids=tuple(sorted(components.basin_ids.get(node, set()))),
            )
        )
    return tuple(
        sorted(items, key=lambda item: (-item.score, item.node_id))
    )


def _node_sources(
    node: int,
    components: _CompletionComponents,
) -> tuple[str, ...]:
    labels = []
    for name, members in (
        ("sharp_completion", components.sharp),
        ("basin_direct", components.direct),
        ("basin_completion", components.completion),
        ("relative_tail", components.tail),
    ):
        if node in members:
            labels.append(name)
    return tuple(labels)


def _evidence_fields(
    pool: FeaturePool,
    index: int,
    context: ContextState,
) -> dict[str, np.ndarray]:
    """Return calibrated current-cue and prior-context evidence fields."""

    query = pool.turns[index]
    term_total = math.fsum(value for _, value in query.user_terms)
    query_terms = (
        {}
        if term_total == 0.0
        else {
            term: value / term_total
            for term, value in query.user_terms
        }
    )
    current_dense = _tail_surprisal(
        pool.dense_scores(query.user_dense, index)
    )
    current_bm25 = _tail_surprisal(
        pool.bm25_scores(query_terms, index)
    )
    context_dense = _tail_surprisal(
        pool.dense_scores(context.dense, index)
    )
    context_bm25 = _tail_surprisal(
        pool.bm25_scores(dict(context.terms), index)
    )
    current = current_dense + current_bm25
    same_event = current + context_dense + context_bm25
    return {
        "current": current,
        "same_event": same_event,
        "current_dense": current_dense,
        "current_bm25": current_bm25,
        "context_dense": context_dense,
        "context_bm25": context_bm25,
    }


def _reuse_evidence_fields(
    source: dict[str, np.ndarray],
    *,
    include_context: bool,
) -> dict[str, np.ndarray]:
    """Project already calibrated seed evidence onto one readout route."""

    # 1. Reuse the exact current-cue arrays produced by burst inference.
    current_dense = source["query_dense"]
    current_bm25 = source["query_bm25"]
    current = current_dense + current_bm25

    # 2. Match the existing empty-context route with exact zero arrays.
    if include_context:
        context_dense = source["context_dense"]
        context_bm25 = source["context_bm25"]
    else:
        context_dense = np.zeros_like(current_dense)
        context_bm25 = np.zeros_like(current_bm25)
    same_event = current + context_dense + context_bm25
    return {
        "current": current,
        "same_event": same_event,
        "current_dense": current_dense,
        "current_bm25": current_bm25,
        "context_dense": context_dense,
        "context_bm25": context_bm25,
    }

def _active_basin_structures(
    graph: DynamicMemoryGraph,
    event: int,
) -> tuple[_BasinStructure, ...]:
    """Snapshot raw engram structure before route-specific scoring."""

    structures: list[_BasinStructure] = []
    for hub in graph.hubs:
        members: list[int] = []
        weights: list[float] = []
        for edge_id in hub.member_edge_ids:
            node_id = graph.source[edge_id]
            weight = graph.weight[edge_id]
            if node_id < event and weight > 0.0:
                members.append(node_id)
                weights.append(weight)
        if len(members) < 2:
            continue
        member_array = np.asarray(members, dtype=np.int32)
        total = math.fsum(weights)
        normalized = np.asarray(weights) / total
        structures.append(
            _BasinStructure(
                hub.created_event,
                tuple(members),
                tuple(weights),
                member_array,
                normalized,
                -math.expm1(-total),
            )
        )
    return tuple(structures)

def _score_active_basins(
    structures: tuple[_BasinStructure, ...],
    scores: np.ndarray,
) -> tuple[Basin, ...]:
    """Score one immutable basin snapshot with route-specific evidence."""

    basins: list[Basin] = []
    for structure in structures:
        values = scores[structure.member_array]
        peak = float(np.max(values))
        pooled = peak + math.log(
            float(np.sum(structure.normalized * np.exp(values - peak)))
        )
        pooled *= structure.conductance
        basins.append(
            Basin(
                structure.hub_id,
                pooled,
                structure.members,
                structure.weights,
            )
        )
    return tuple(basins)



def _pooled_heads(
    basins: tuple[Basin, ...],
    fields: dict[str, np.ndarray],
    continuation: float,
    temperature: float,
) -> list[tuple[str, float, tuple[tuple[int, float], ...]]]:
    """Select event basins after pooling every evidence channel."""

    positive = tuple(basin for basin in basins if basin.score > 0.0)
    if not positive:
        return []
    scores = np.asarray([basin.score for basin in positive])
    selection = _temperature_sparsemax(
        scores / float(np.max(scores)),
        temperature,
    )
    selected = [
        (
            positive[index],
            mass,
            _head_seed(positive[index], fields, continuation),
        )
        for index, mass in selection
    ]
    return _merge_overlapping_heads(selected)


def _accessibility_supported_heads(
    graph: DynamicMemoryGraph,
    heads: list[tuple[str, float, tuple[tuple[int, float], ...]]],
) -> list[tuple[str, float, tuple[tuple[int, float], ...]]]:
    """Intersect selected heads with relative live conductance support."""

    if len(heads) < 2:
        return heads
    hubs_by_event = {
        hub.created_event: hub.node_id
        for hub in graph.hubs
    }
    accessibility = np.asarray(
        [
            _head_accessibility(graph, head_id, hubs_by_event)
            for head_id, _, _ in heads
        ],
        dtype=np.float64,
    )
    support = {
        index
        for index, _ in _gain_normalized_sparsemax(accessibility)
    }
    selected = [
        head for index, head in enumerate(heads) if index in support
    ]
    total = math.fsum(mass for _, mass, _ in selected)
    return [
        (head_id, mass / total, seed)
        for head_id, mass, seed in selected
    ]


def _head_accessibility(
    graph: DynamicMemoryGraph,
    head_id: str,
    hubs_by_event: dict[int, int] | None = None,
) -> float:
    resolved = (
        {
            hub.created_event: hub.node_id
            for hub in graph.hubs
        }
        if hubs_by_event is None
        else hubs_by_event
    )
    edge_ids = tuple(
        edge_id
        for event_id in (int(value) for value in head_id.split("+"))
        for edge_id in graph.hub_members[resolved[event_id]]
    )
    raw = math.fsum(graph.weight[edge_id] for edge_id in edge_ids)
    effective = math.fsum(
        graph.effective_weight(edge_id) for edge_id in edge_ids
    )
    return effective / raw


def _head_seed(
    basin: Basin,
    fields: dict[str, np.ndarray],
    continuation: float,
) -> tuple[tuple[int, float], ...]:
    """Build one independently normalized seed inside an engram."""

    members = np.asarray(basin.members, dtype=np.int32)
    membership = np.asarray(basin.weights, dtype=np.float64)
    log_prior = np.log(membership / float(np.max(membership)))
    current = _gain_normalized_sparsemax(
        fields["current"][members] + log_prior
    )
    same_event = _gain_normalized_sparsemax(
        fields["same_event"][members] + log_prior
    )
    local = _mixed_seed(current, same_event, continuation)
    return tuple((basin.members[node], value) for node, value in local)


def _merge_overlapping_heads(
    heads: list[tuple[Basin, float, tuple[tuple[int, float], ...]]],
) -> list[tuple[str, float, tuple[tuple[int, float], ...]]]:
    """Merge heads connected by shared direct seed coordinates."""

    parent = list(range(len(heads)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    owners: dict[int, int] = {}
    for index, (_, _, seed) in enumerate(heads):
        for node, _ in seed:
            previous = owners.setdefault(node, index)
            left = find(index)
            right = find(previous)
            if left != right:
                parent[max(left, right)] = min(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(heads)):
        grouped.setdefault(find(index), []).append(index)
    return _materialize_merged_heads(heads, grouped)


def _materialize_merged_heads(
    heads: list[tuple[Basin, float, tuple[tuple[int, float], ...]]],
    grouped: dict[int, list[int]],
) -> list[tuple[str, float, tuple[tuple[int, float], ...]]]:
    merged = []
    for indices in grouped.values():
        mass = math.fsum(heads[index][1] for index in indices)
        values: dict[int, float] = {}
        hub_ids = []
        for index in indices:
            basin, head_mass, seed = heads[index]
            hub_ids.append(basin.hub_id)
            for node, value in seed:
                values[node] = values.get(node, 0.0) + head_mass * value
        total = math.fsum(values.values())
        merged.append(
            (
                "+".join(str(hub_id) for hub_id in sorted(hub_ids)),
                mass,
                tuple(
                    (node, values[node] / total)
                    for node in sorted(values)
                ),
            )
        )
    return sorted(merged, key=lambda item: (-item[1], item[0]))


def _completion_support(
    diffusion: DiffusionResult,
    seed: tuple[tuple[int, float], ...],
    end: int,
    restart: float,
) -> set[int]:
    """Select graph completion after subtracting directly clamped mass."""

    completion = diffusion.reserve[:end].copy()
    seed_nodes = {node for node, _ in seed}
    for node, value in seed:
        completion[node] = max(
            0.0,
            completion[node] - restart * value,
        )
    candidates = np.fromiter(
        (
            node
            for node in np.flatnonzero(completion > 0.0)
            if int(node) not in seed_nodes
        ),
        dtype=np.int32,
    )
    if candidates.size == 0:
        return set()
    selection = _gain_normalized_sparsemax(completion[candidates])
    return {int(candidates[index]) for index, _ in selection}


def _turn_posterior(
    diffusion: DiffusionResult,
    end: int,
) -> np.ndarray:
    turns = diffusion.reserve[:end].copy()
    total = float(np.sum(turns))
    return turns if total == 0.0 else turns / total


def _relative_tail_entmax15(
    basin: np.ndarray,
    sharp: np.ndarray,
    temporal: np.ndarray,
    direct_nodes: set[int],
) -> tuple[tuple[int, float], ...]:
    """Retain the V8 sparse local tail with temporal reachability."""

    information = np.zeros_like(basin)
    stronger = basin > sharp
    information[stronger] = basin[stronger] * np.log(
        2.0
        * basin[stronger]
        / (basin[stronger] + sharp[stronger])
    )
    if not np.any(information > 0.0):
        return ()
    information /= float(np.max(information))
    temporal_peak = float(np.max(temporal))
    if temporal_peak == 0.0:
        return ()
    temporal = temporal / temporal_peak
    score = np.sqrt(information * temporal)
    if direct_nodes:
        score[
            np.fromiter(sorted(direct_nodes), dtype=np.int32)
        ] = 0.0
    candidates = np.flatnonzero(score > 0.0)
    if candidates.size == 0:
        return ()
    selected = _entmax(
        score[candidates] / float(np.max(score[candidates])),
        alpha=1.5,
    )
    return tuple(
        (int(candidates[index]), value)
        for index, value in selected
    )


def _shared_tail(evidence: dict[int, float]) -> set[int]:
    """Select a sparse weak tail after cross-basin accumulation."""

    if not evidence:
        return set()
    nodes = np.fromiter(sorted(evidence), dtype=np.int32)
    values = np.fromiter(
        (evidence[int(node)] for node in nodes),
        dtype=np.float64,
    )
    selected = _entmax(
        values / float(np.max(values)),
        alpha=1.75,
    )
    return {int(nodes[index]) for index, _ in selected}


def _cue_agreement(
    fields: dict[str, np.ndarray],
    continuation: float,
) -> np.ndarray:
    """Return geometric agreement between dense and lexical evidence."""

    dense = (
        fields["current_dense"]
        + continuation * fields["context_dense"]
    )
    bm25 = (
        fields["current_bm25"]
        + continuation * fields["context_bm25"]
    )
    dense_peak = float(np.max(dense))
    bm25_peak = float(np.max(bm25))
    if dense_peak == 0.0 or bm25_peak == 0.0:
        return np.zeros_like(dense)
    return np.sqrt((dense / dense_peak) * (bm25 / bm25_peak))


def _filter_tail_by_agreement(
    nodes: set[int],
    evidence: dict[int, float],
    cue_agreement: np.ndarray,
) -> set[int]:
    """Remove weak-view tails without introducing a new candidate."""

    candidates = np.fromiter(
        (
            node
            for node in sorted(nodes)
            if cue_agreement[node] > 0.0
        ),
        dtype=np.int32,
    )
    if candidates.size == 0:
        return set()
    quality = np.fromiter(
        (
            evidence[int(node)] * float(cue_agreement[node])
            for node in candidates
        ),
        dtype=np.float64,
    )
    selected = _entmax(
        quality / float(np.max(quality)),
        alpha=1.90,
    )
    return {int(candidates[index]) for index, _ in selected}


def _surprise_temperature(
    surprise: float,
    historical_surprise: np.ndarray,
) -> float:
    """Map empirical novelty to the V8 competition temperature."""

    if historical_surprise.size == 0:
        return 1.0
    median = float(np.median(historical_surprise))
    robust_scale = float(
        np.median(np.abs(historical_surprise - median))
    )
    if robust_scale == 0.0:
        return 0.0 if surprise == 0.0 else 1.0
    return -math.expm1(-surprise / robust_scale)


def _temperature_sparsemax(
    logits: np.ndarray,
    temperature: float,
) -> tuple[tuple[int, float], ...]:
    if temperature == 0.0:
        winners = np.flatnonzero(logits == float(np.max(logits)))
        mass = 1.0 / winners.size
        return tuple((int(node), mass) for node in winners)
    return _sparsemax(logits / temperature)


def _gain_normalized_sparsemax(
    scores: np.ndarray,
) -> tuple[tuple[int, float], ...]:
    peak = float(np.max(scores))
    if peak <= 0.0:
        return ()
    return _sparsemax(scores / peak)


def _mixed_seed(
    current: tuple[tuple[int, float], ...],
    same_event: tuple[tuple[int, float], ...],
    continuation: float,
) -> tuple[tuple[int, float], ...]:
    mixed: dict[int, float] = {}
    for node, value in current:
        mixed[node] = mixed.get(node, 0.0) + (
            1.0 - continuation
        ) * value
    for node, value in same_event:
        mixed[node] = mixed.get(node, 0.0) + continuation * value
    total = math.fsum(mixed.values())
    if total == 0.0:
        return ()
    return tuple(
        (node, mixed[node] / total)
        for node in sorted(mixed)
        if mixed[node] > 0.0
    )


def _entmax(
    logits: np.ndarray,
    alpha: float,
) -> tuple[tuple[int, float], ...]:
    """Project logits onto an alpha-entmax sparse distribution."""

    exponent = 1.0 / (alpha - 1.0)
    lower = float(np.min(logits)) - exponent
    upper = float(np.max(logits))
    for _ in range(80):
        threshold = (lower + upper) / 2.0
        values = (
            np.maximum(
                (alpha - 1.0) * (logits - threshold),
                0.0,
            )
            ** exponent
        )
        if float(np.sum(values)) > 1.0:
            lower = threshold
        else:
            upper = threshold
    values = (
        np.maximum((alpha - 1.0) * (logits - upper), 0.0)
        ** exponent
    )
    values /= float(np.sum(values))
    return tuple(
        (int(node), float(values[node]))
        for node in np.flatnonzero(values > 0.0)
    )
