"""Deterministic indexed-heap Residual Push."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .model import DiffusionResult


class TransitionGraph(Protocol):
    """Expose the finite transition interface required by Residual Push."""

    max_nodes: int

    def transitions(
        self,
        node_id: int,
        event: int,
    ) -> tuple[tuple[tuple[int, float, int], ...], float]: ...


class IndexedMaxHeap:
    """Keep at most one residual entry per node with stable tie-breaking."""

    def __init__(self, capacity: int, residual: np.ndarray) -> None:
        self.nodes: list[int] = []
        self.position = np.full(capacity, -1, dtype=np.int32)
        self.residual = residual

    def update(self, node_id: int) -> None:
        position = int(self.position[node_id])
        if position < 0:
            self.position[node_id] = len(self.nodes)
            self.nodes.append(node_id)
            self._sift_up(len(self.nodes) - 1)
        else:
            self._sift_up(position)

    def pop(self) -> int:
        if not self.nodes:
            raise IndexError("pop from empty residual heap")
        root = self.nodes[0]
        last = self.nodes.pop()
        self.position[root] = -1
        if self.nodes:
            self.nodes[0] = last
            self.position[last] = 0
            self._sift_down(0)
        return root

    def stable_nodes(self) -> list[int]:
        return sorted(self.nodes)

    def _sift_up(self, position: int) -> None:
        node = self.nodes[position]
        while position:
            parent = (position - 1) // 2
            parent_node = self.nodes[parent]
            if not self._higher(node, parent_node):
                break
            self.nodes[position] = parent_node
            self.position[parent_node] = position
            position = parent
        self.nodes[position] = node
        self.position[node] = position

    def _sift_down(self, position: int) -> None:
        size = len(self.nodes)
        node = self.nodes[position]
        while True:
            left = position * 2 + 1
            if left >= size:
                break
            right = left + 1
            child = left
            if right < size and self._higher(self.nodes[right], self.nodes[left]):
                child = right
            child_node = self.nodes[child]
            if not self._higher(child_node, node):
                break
            self.nodes[position] = child_node
            self.position[child_node] = position
            position = child
        self.nodes[position] = node
        self.position[node] = position

    def _higher(self, left: int, right: int) -> bool:
        left_value = self.residual[left]
        right_value = self.residual[right]
        return left_value > right_value or (
            left_value == right_value and left < right
        )


def residual_push(
    graph: TransitionGraph,
    seed: tuple[tuple[int, float], ...],
    event: int,
    *,
    restart: float,
    tolerance: float,
    capture_paths: bool,
) -> DiffusionResult:
    """Settle one sparse seed while preserving an explicit L1 error."""

    if not seed:
        empty = np.zeros(graph.max_nodes, dtype=np.float64)
        return DiffusionResult(empty, np.empty(0, dtype=np.int32), 0, 0.0, None, None)

    # 1. Initialize one indexed residual entry per seed coordinate.
    residual = np.zeros(graph.max_nodes, dtype=np.float64)
    reserve = np.zeros(graph.max_nodes, dtype=np.float64)
    heap = IndexedMaxHeap(graph.max_nodes, residual)
    for node_id, value in seed:
        residual[node_id] = value
        heap.update(node_id)
    residual_total = math.fsum(value for _, value in seed)
    parent_node = (
        np.full(graph.max_nodes, -1, dtype=np.int32) if capture_paths else None
    )
    parent_edge = (
        np.full(graph.max_nodes, -1, dtype=np.int32) if capture_paths else None
    )
    parent_mass = (
        np.zeros(graph.max_nodes, dtype=np.float64) if capture_paths else None
    )

    # 2. Repeatedly settle the largest residual coordinate.
    pushes = 0
    while True:
        if residual_total <= tolerance:
            residual_total = math.fsum(
                float(residual[node]) for node in heap.stable_nodes()
            )
            if residual_total <= tolerance:
                break
        if not heap.nodes:
            raise ArithmeticError("positive residual mass has no heap entry")
        node = heap.pop()
        value = float(residual[node])
        residual[node] = 0.0
        residual_total -= value
        reserve[node] += restart * value
        propagated = (1.0 - restart) * value
        transitions, unspread = graph.transitions(node, event)
        for target, probability, edge_id in transitions:
            addition = propagated * probability
            residual_total += _add_residual(
                residual,
                heap,
                target,
                addition,
            )
            _record_parent(
                parent_node,
                parent_edge,
                parent_mass,
                target,
                node,
                edge_id,
                addition,
            )
        for target, probability in seed:
            addition = propagated * unspread * probability
            residual_total += _add_residual(
                residual,
                heap,
                target,
                addition,
            )
        pushes += 1
        if pushes % 4096 == 0:
            residual_total = math.fsum(
                float(residual[item]) for item in heap.stable_nodes()
            )

    active = np.flatnonzero(reserve > 0.0).astype(np.int32, copy=False)
    return DiffusionResult(
        reserve=reserve,
        active_nodes=active,
        pushes=pushes,
        residual_l1=residual_total,
        parent_node=parent_node,
        parent_edge=parent_edge,
    )


def _add_residual(
    residual: np.ndarray,
    heap: IndexedMaxHeap,
    node_id: int,
    addition: float,
) -> float:
    if addition == 0.0:
        return 0.0
    residual[node_id] += addition
    heap.update(node_id)
    return addition


def _record_parent(
    parent_node: np.ndarray | None,
    parent_edge: np.ndarray | None,
    parent_mass: np.ndarray | None,
    target: int,
    source: int,
    edge_id: int,
    addition: float,
) -> None:
    if parent_node is None or parent_edge is None or parent_mass is None:
        return
    if addition > parent_mass[target]:
        parent_mass[target] = addition
        parent_node[target] = source
        parent_edge[target] = edge_id
