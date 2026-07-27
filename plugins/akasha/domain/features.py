"""Compute causal dense, BM25, time, and context evidence."""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .model import ContextState, SeedEvidence, Turn


class FeaturePool:
    """Serve causal evidence from contiguous dense and inverted lexical arrays."""

    def __init__(self, turns: list[Turn]) -> None:
        if not turns:
            raise ValueError("memory rebuild requires at least one turn")
        self.turns = turns
        dimension = _dense_dimension(turns)
        self.user_dense, self.user_mask = _dense_matrix(
            turns,
            "user_dense",
            dimension,
        )
        self.assistant_dense, self.assistant_mask = _dense_matrix(
            turns,
            "assistant_dense",
            dimension,
        )
        self.turn_dense, self.turn_dense_mask = _turn_dense_matrix(
            turns,
            dimension,
        )
        self.normalized_terms = tuple(_normalized_turn_terms(turn) for turn in turns)
        self.lengths = np.asarray(
            [_term_total(turn) for turn in turns],
            dtype=np.float64,
        )
        self.prefix_lengths = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(self.lengths))
        )
        self.gaps = np.asarray(
            [
                math.nan if turn.inter_gap_seconds is None else turn.inter_gap_seconds
                for turn in turns
            ],
            dtype=np.float64,
        )
        self.term_order, self.postings = _build_postings(turns)

    def infer_seed(
        self,
        index: int,
        context: ContextState,
        capture_channels: bool,
    ) -> SeedEvidence:
        """Infer a sparse seed from strictly historical content."""

        # 1. Score the current user cue and the completed previous context.
        query = self.turns[index]
        query_terms = _normalize_pairs(query.user_terms)
        query_dense = self.dense_scores(query.user_dense, index)
        query_bm25 = self.bm25_scores(query_terms, index)
        context_dense = self.dense_scores(context.dense, index)
        context_terms = dict(context.terms)
        context_bm25 = self.bm25_scores(context_terms, index)
        surprise = self.query_surprise(
            query,
            index,
            query_dense,
            query_bm25,
        )

        # 2. Infer whether the previous event continues.
        time_prior = self.time_prior(query.inter_gap_seconds, index)
        continuation = self.continuation_belief(
            query,
            context,
            index,
            time_prior,
            query_dense,
            query_bm25,
        )

        # 3. Project calibrated channel evidence onto one sparse simplex.
        evidence = {
            "query_dense": _tail_surprisal(query_dense),
            "query_bm25": _tail_surprisal(query_bm25),
            "context_dense": _tail_surprisal(context_dense),
            "context_bm25": _tail_surprisal(context_bm25),
        }
        seed = _causal_seed(evidence, continuation)
        seed_nodes = None if capture_channels else tuple(node for node, _ in seed)
        channels = _channel_support(evidence, seed_nodes)
        return SeedEvidence(seed, channels, time_prior, continuation, surprise)

    def query_surprise(
        self,
        query: Turn,
        end: int,
        dense_scores: np.ndarray,
        bm25_scores: np.ndarray,
    ) -> float:
        """Measure residual novelty after reconstructing the cue from memory."""

        # 1. Normalize dense prediction by its natural cosine upper bound.
        residuals: list[float] = []
        if query.user_dense is not None and np.any(self.user_mask[:end]):
            dense_prediction = min(
                1.0,
                max(0.0, float(np.max(dense_scores))),
            )
            residuals.append(1.0 - dense_prediction)

        # 2. Normalize lexical prediction by the query's causal self-score.
        query_terms = _normalize_pairs(query.user_terms)
        if query_terms:
            self_score = self.bm25_document_score(query_terms, query_terms, end)
            if self_score > 0.0:
                lexical_prediction = min(
                    1.0,
                    max(0.0, float(np.max(bm25_scores)) / self_score),
                )
                residuals.append(1.0 - lexical_prediction)
        if not residuals:
            return 1.0

        # 3. Root-mean-square residual preserves novelty seen by either channel.
        return math.sqrt(
            math.fsum(value * value for value in residuals) / len(residuals)
        )

    def dense_scores(self, cue: np.ndarray | None, end: int) -> np.ndarray:
        if cue is None or end == 0:
            return np.zeros(end, dtype=np.float64)
        user = self.user_dense[:end] @ cue
        assistant = self.assistant_dense[:end] @ cue
        user = np.where(self.user_mask[:end], user, -1.0)
        assistant = np.where(self.assistant_mask[:end], assistant, -1.0)
        available = self.user_mask[:end] | self.assistant_mask[:end]
        scores = np.maximum(user, assistant)
        return np.where(available, scores, 0.0).astype(np.float64, copy=False)

    def bm25_scores(
        self,
        query_terms: dict[str, float],
        end: int,
    ) -> np.ndarray:
        """Score shared postings with vectorized exact BM25."""

        scores = np.zeros(end, dtype=np.float64)
        if not query_terms or end == 0:
            return scores
        average_length = self.prefix_lengths[end] / end
        ordered_terms = sorted(
            query_terms,
            key=lambda term: self.term_order.get(term, math.inf),
        )
        for term in ordered_terms:
            posting = self.postings.get(term)
            if posting is None:
                continue
            positions, frequencies = posting
            stop = int(np.searchsorted(positions, end, side="left"))
            if stop == 0:
                continue
            selected = positions[:stop]
            tf = frequencies[:stop]
            idf = math.log1p((end - stop + 0.5) / (stop + 0.5))
            saturation = tf * 2.2 / (
                tf + 1.2 * (0.25 + 0.75 * self.lengths[selected] / average_length)
            )
            scores[selected] += query_terms[term] * idf * saturation
        return scores

    def time_prior(self, gap: float | None, end: int) -> float:
        if gap is None or end <= 1:
            return 0.5
        observed = self.gaps[1:end]
        observed = observed[np.isfinite(observed)]
        return (float(np.count_nonzero(observed >= gap)) + 0.5) / (
            observed.size + 1.0
        )

    def continuation_belief(
        self,
        query: Turn,
        context: ContextState,
        end: int,
        time_prior: float,
        query_dense: np.ndarray,
        query_bm25: np.ndarray,
    ) -> float:
        log_evidence = 0.0
        if query.user_dense is not None and context.dense is not None:
            observed = float(np.dot(query.user_dense, context.dense))
            log_evidence += _log_evidence(observed, query_dense)
        query_terms = _normalize_pairs(query.user_terms)
        context_terms = dict(context.terms)
        if query_terms and context_terms:
            observed = self.bm25_document_score(query_terms, context_terms, end)
            log_evidence += _log_evidence(observed, query_bm25)
        odds = math.log(time_prior / (1.0 - time_prior)) + log_evidence
        return _sigmoid(odds)

    def bm25_document_score(
        self,
        query_terms: dict[str, float],
        document_terms: dict[str, float],
        end: int,
    ) -> float:
        average_length = self.prefix_lengths[end] / end
        length = math.fsum(document_terms.values())
        parts: list[float] = []
        common = sorted(
            query_terms.keys() & document_terms.keys(),
            key=lambda term: self.term_order.get(term, math.inf),
        )
        for term in common:
            posting = self.postings.get(term)
            if posting is None:
                continue
            df = int(np.searchsorted(posting[0], end, side="left"))
            if df == 0:
                continue
            tf = document_terms[term]
            idf = math.log1p((end - df + 0.5) / (df + 0.5))
            saturation = tf * 2.2 / (
                tf + 1.2 * (0.25 + 0.75 * length / average_length)
            )
            parts.append(query_terms[term] * idf * saturation)
        return math.fsum(parts)

    def build_context(
        self,
        members: tuple[tuple[int, float], ...],
    ) -> ContextState:
        """Build a compact completed event context in stable node order."""

        normalized = _normalize_pairs_by_id(members)
        dense = np.zeros(self.turn_dense.shape[1], dtype=np.float64)
        has_dense = False
        lexical: dict[str, float] = defaultdict(float)
        for node_id, weight in normalized:
            if self.turn_dense_mask[node_id]:
                dense += weight * self.turn_dense[node_id]
                has_dense = True
            for term, value in self.normalized_terms[node_id]:
                lexical[term] += weight * value
        terms = tuple(sorted(lexical.items(), key=lambda item: item[0].encode("utf-8")))
        return ContextState(normalized, _unit(dense) if has_dense else None, terms)


@dataclass(frozen=True)
class BurstDecision:
    """Describe one causal boundary decision and its retrieval evidence."""

    evidence: SeedEvidence
    base_continuation: float
    context_dependence: float
    context_mass: float
    continued: bool
    visible_nodes: tuple[int, ...]
    context: ContextState


class BurstAwareFeaturePool(FeaturePool):
    """Infer query evidence against one stream-local active burst."""

    def __init__(self, turns: list[Turn]) -> None:
        super().__init__(turns)
        self.context_dependence = _causal_context_dependence(turns)

    def infer_burst_seed(
        self,
        index: int,
        candidate_context: ContextState,
        visible_nodes: tuple[int, ...],
        capture_channels: bool,
    ) -> BurstDecision:
        """Infer a causal burst boundary and mix cue with active context."""

        # 1. Score the current cue and the complete visible burst separately.
        query = self.turns[index]
        query_terms = _normalize_pairs(query.user_terms)
        query_dense = self.dense_scores(query.user_dense, index)
        query_bm25 = self.bm25_scores(query_terms, index)
        context_dense = self.dense_scores(candidate_context.dense, index)
        context_bm25 = self.bm25_scores(
            dict(candidate_context.terms),
            index,
        )
        fields = {
            "query_dense": _tail_surprisal(query_dense),
            "query_bm25": _tail_surprisal(query_bm25),
            "context_dense": _tail_surprisal(context_dense),
            "context_bm25": _tail_surprisal(context_bm25),
        }

        # 2. Rescue underspecified cues only when temporal evidence agrees.
        time_prior = self.time_prior(query.inter_gap_seconds, index)
        base = self.continuation_belief(
            query,
            candidate_context,
            index,
            time_prior,
            query_dense,
            query_bm25,
        )
        dependence = float(self.context_dependence[index])
        continuation = _burst_continuation(
            base,
            time_prior,
            dependence,
        )
        continued = bool(visible_nodes) and continuation >= 0.5
        active_context = (
            candidate_context
            if continued
            else ContextState((), None, ())
        )

        # 3. Fuse normalized cue and context sources without fixed weights.
        context_mass = (
            _combine_odds(continuation, dependence)
            if continued
            else 0.0
        )
        seed = _mix_sources(fields, context_mass)
        seed_nodes = (
            None
            if capture_channels
            else tuple(node for node, _ in seed)
        )
        return BurstDecision(
            evidence=SeedEvidence(
                seed=seed,
                channels=_channel_support(fields, seed_nodes),
                time_prior=time_prior,
                continuation=continuation,
                surprise=self.query_surprise(
                    query,
                    index,
                    query_dense,
                    query_bm25,
                ),
            ),
            base_continuation=base,
            context_dependence=dependence,
            context_mass=context_mass,
            continued=continued,
            visible_nodes=visible_nodes if continued else (),
            context=active_context,
        )


def effective_support(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    if positive.size == 0:
        return 0.0
    normalized = positive / math.fsum(float(value) for value in positive)
    entropy = -math.fsum(
        float(value) * math.log(float(value)) for value in normalized
    )
    return math.exp(entropy)


def binary_entropy(probability: float) -> float:
    """Return normalized uncertainty over continue versus new-burst."""

    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    ) / math.log(2.0)


def _burst_continuation(
    base: float,
    time_prior: float,
    context_dependence: float,
) -> float:
    rescue = time_prior * context_dependence
    return base + (1.0 - base) * rescue


def _combine_odds(first: float, second: float) -> float:
    epsilon = np.finfo(np.float64).eps
    first = min(1.0 - epsilon, max(epsilon, first))
    second = min(1.0 - epsilon, max(epsilon, second))
    log_odds = (
        math.log(first / (1.0 - first))
        + math.log(second / (1.0 - second))
    )
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exponent = math.exp(log_odds)
    return exponent / (1.0 + exponent)


def _mix_sources(
    evidence: dict[str, np.ndarray],
    context_mass: float,
) -> tuple[tuple[int, float], ...]:
    direct = _sparsemax(
        evidence["query_dense"] + evidence["query_bm25"]
    )
    context = _sparsemax(
        evidence["context_dense"] + evidence["context_bm25"]
    )
    mixed: dict[int, float] = defaultdict(float)
    for node, value in direct:
        mixed[node] += (1.0 - context_mass) * value
    for node, value in context:
        mixed[node] += context_mass * value
    return (
        _normalize_pairs_by_id(tuple(mixed.items()))
        if mixed
        else ()
    )


def _causal_context_dependence(turns: list[Turn]) -> np.ndarray:
    supports = np.asarray(
        [_term_effective_support(turn.user_terms) for turn in turns],
        dtype=np.float64,
    )
    result = np.full(len(turns), 0.5, dtype=np.float64)
    ordered: list[float] = []
    for index, value in enumerate(supports):
        if ordered and value > 0.0:
            position = bisect.bisect_left(ordered, value)
            result[index] = (len(ordered) - position + 0.5) / (
                len(ordered) + 1.0
            )
        bisect.insort_right(ordered, value)
    return result


def _term_effective_support(
    terms: tuple[tuple[str, int], ...],
) -> float:
    normalized = _normalize_pairs(terms)
    if not normalized:
        return 0.0
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in normalized.values()
    )
    return math.exp(entropy)


def _build_postings(
    turns: list[Turn],
) -> tuple[dict[str, int], dict[str, tuple[np.ndarray, np.ndarray]]]:
    positions: dict[str, list[int]] = defaultdict(list)
    frequencies: dict[str, list[float]] = defaultdict(list)
    term_order: dict[str, int] = {}
    for node_id, turn in enumerate(turns):
        combined = _combined_terms(turn)
        for term, tf in combined:
            if term not in term_order:
                term_order[term] = len(term_order)
            positions[term].append(node_id)
            frequencies[term].append(float(tf))
    postings = {
        term: (
            np.asarray(positions[term], dtype=np.int32),
            np.asarray(frequencies[term], dtype=np.float64),
        )
        for term in sorted(positions, key=term_order.__getitem__)
    }
    return term_order, postings


def _dense_matrix(
    turns: list[Turn],
    attribute: str,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(turns), dimension), dtype=np.float32)
    mask = np.zeros(len(turns), dtype=bool)
    for node_id, turn in enumerate(turns):
        vector = getattr(turn, attribute)
        if vector is not None:
            matrix[node_id] = vector
            mask[node_id] = True
    return matrix, mask


def _turn_dense_matrix(
    turns: list[Turn],
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(turns), dimension), dtype=np.float32)
    mask = np.zeros(len(turns), dtype=bool)
    for node_id, turn in enumerate(turns):
        vectors = [
            vector
            for vector in (turn.user_dense, turn.assistant_dense)
            if vector is not None
        ]
        if vectors:
            matrix[node_id] = _unit(
                sum(vectors, start=np.zeros(dimension, dtype=np.float32))
            )
            mask[node_id] = True
    return matrix, mask


def _dense_dimension(turns: list[Turn]) -> int:
    dimension = next(
        (
            vector.shape[0]
            for turn in turns
            for vector in (turn.user_dense, turn.assistant_dense)
            if vector is not None
        ),
        0,
    )
    for turn in turns:
        for vector in (turn.user_dense, turn.assistant_dense):
            if vector is not None and vector.shape != (dimension,):
                raise ValueError("dense vectors must share one dimension")
    return dimension


def _combined_terms(turn: Turn) -> tuple[tuple[str, int], ...]:
    combined: dict[str, int] = defaultdict(int)
    for term, tf in (*turn.user_terms, *turn.assistant_terms):
        combined[term] += tf
    return tuple(sorted(combined.items(), key=lambda item: item[0].encode("utf-8")))


def _normalized_turn_terms(turn: Turn) -> tuple[tuple[str, float], ...]:
    return tuple(_normalize_pairs(dict(_combined_terms(turn))).items())


def _term_total(turn: Turn) -> int:
    return sum(tf for _, tf in turn.user_terms) + sum(
        tf for _, tf in turn.assistant_terms
    )


def _normalize_pairs(
    values: tuple[tuple[str, int], ...] | dict[str, int],
) -> dict[str, float]:
    items = values.items() if isinstance(values, dict) else values
    materialized = tuple(items)
    total = math.fsum(float(value) for _, value in materialized)
    if total == 0.0:
        return {}
    return {key: float(value) / total for key, value in materialized}


def _normalize_pairs_by_id(
    values: tuple[tuple[int, float], ...],
) -> tuple[tuple[int, float], ...]:
    ordered = tuple(sorted(values))
    total = math.fsum(value for _, value in ordered if value > 0.0)
    if total == 0.0:
        raise ValueError("context members must contain positive mass")
    return tuple((node, value / total) for node, value in ordered if value > 0.0)


def _tail_surprisal(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0 or not np.any(scores != 0.0):
        return np.zeros_like(scores)
    ordered = np.sort(scores, kind="stable")
    counts = scores.size - np.searchsorted(ordered, scores, side="left")
    return -np.log(counts / scores.size)


def _log_evidence(observed: float, background: np.ndarray) -> float:
    probability = (float(np.count_nonzero(background >= observed)) + 1.0) / (
        background.size + 1.0
    )
    return math.log((1.0 - math.log(probability)) / 2.0)


def _causal_seed(
    evidence: dict[str, np.ndarray],
    continuation: float,
) -> tuple[tuple[int, float], ...]:
    if not any(np.any(values > 0.0) for values in evidence.values()):
        return ()
    current = evidence["query_dense"] + evidence["query_bm25"]
    same_event = current + evidence["context_dense"] + evidence["context_bm25"]
    current_seed = _sparsemax(current)
    event_seed = _sparsemax(same_event)
    mixed: dict[int, float] = defaultdict(float)
    for node_id, value in current_seed:
        mixed[node_id] += (1.0 - continuation) * value
    for node_id, value in event_seed:
        mixed[node_id] += continuation * value
    return _normalize_pairs_by_id(tuple(mixed.items()))


def _sparsemax(logits: np.ndarray) -> tuple[tuple[int, float], ...]:
    if logits.size == 0 or not np.any(logits != 0.0):
        return ()
    order = np.argsort(-logits, kind="stable")
    ordered = logits[order]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    condition = 1.0 + np.arange(1, logits.size + 1) * ordered > cumulative
    support_size = int(np.flatnonzero(condition)[-1]) + 1
    threshold = (cumulative[support_size - 1] - 1.0) / support_size
    support = order[:support_size]
    return tuple(
        (int(node_id), float(logits[node_id] - threshold))
        for node_id in np.sort(support)
        if logits[node_id] > threshold
    )


def _channel_support(
    evidence: dict[str, np.ndarray],
    nodes: tuple[int, ...] | None,
) -> dict[str, frozenset[int]]:
    if nodes is not None:
        return {
            name: frozenset(node for node in nodes if values[node] > 0.0)
            for name, values in evidence.items()
        }
    return {
        name: frozenset(int(node) for node in np.flatnonzero(values > 0.0))
        for name, values in evidence.items()
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("dense vector must be finite and non-zero")
    return vector / norm
