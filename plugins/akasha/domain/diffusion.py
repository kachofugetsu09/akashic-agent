"""Deterministic indexed-heap Residual Push."""

from __future__ import annotations

import math
from heapq import heappop, heappush
from typing import Protocol

import numpy as np

from .model import DiffusionResult

try:
    from . import _push_kernel as kernel
except ImportError:  # numba 不可用时退回解释器循环，结果相同。
    kernel = None


Transitions = tuple[tuple[tuple[int, float, int], ...], float]


class PushTable:
    """Flat copy of already-computed transitions so the compiled loop can read them without Python."""

    def __init__(self, max_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.start = np.zeros(max_nodes, dtype=np.int64)
        self.count = np.full(max_nodes, -1, dtype=np.int64)
        self.unspread = np.zeros(max_nodes, dtype=np.float64)
        self.target = np.empty(4096, dtype=np.int64)
        self.probability = np.empty(4096, dtype=np.float64)
        self.size = 0

    def load(self, node_id: int, transitions: Transitions) -> None:
        items, unspread = transitions
        end = self.size + len(items)
        if end > self.target.size:
            capacity = max(end, self.target.size * 2)
            self.target = np.resize(self.target, capacity)
            self.probability = np.resize(self.probability, capacity)
        for offset, (target, probability, _) in enumerate(items, start=self.size):
            self.target[offset] = target
            self.probability[offset] = probability
        self.start[node_id] = self.size
        self.count[node_id] = len(items)
        self.unspread[node_id] = unspread
        self.size = end


class TransitionCache(dict[int, Transitions]):
    """Per-state transition cache; clearing it also drops the flat copy used by the compiled loop."""

    table: PushTable | None = None

    def clear(self) -> None:
        super().clear()
        self.table = None

    def push_table(self, max_nodes: int) -> PushTable:
        if self.table is None or self.table.max_nodes != max_nodes:
            self.table = PushTable(max_nodes)
        return self.table


class TransitionGraph(Protocol):
    """Expose the finite transition interface required by Residual Push."""

    max_nodes: int
    transition_cache: TransitionCache

    def transitions(
        self,
        node_id: int,
        event: int,
    ) -> tuple[tuple[tuple[int, float, int], ...], float]: ...


class IndexedMaxHeap:
    """Keep at most one residual entry per node with stable tie-breaking."""

    def __init__(self, capacity: int, residual: np.ndarray) -> None:
        self.nodes: list[int] = []
        self.position = [-1] * capacity
        self.residual = residual

    def update(self, node_id: int) -> None:
        position = self.position[node_id]
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
        nodes = self.nodes
        positions = self.position
        residual = self.residual
        node = nodes[position]
        node_value = residual[node]
        while position:
            parent = (position - 1) // 2
            parent_node = nodes[parent]
            parent_value = residual[parent_node]
            if not (
                node_value > parent_value
                or (node_value == parent_value and node < parent_node)
            ):
                break
            nodes[position] = parent_node
            positions[parent_node] = position
            position = parent
        nodes[position] = node
        positions[node] = position

    def _sift_down(self, position: int) -> None:
        nodes = self.nodes
        positions = self.position
        residual = self.residual
        size = len(nodes)
        node = nodes[position]
        node_value = residual[node]
        while True:
            left = position * 2 + 1
            if left >= size:
                break
            right = left + 1
            child = left
            left_node = nodes[left]
            child_node = left_node
            child_value = residual[left_node]
            if right < size:
                right_node = nodes[right]
                right_value = residual[right_node]
                if right_value > child_value or (
                    right_value == child_value and right_node < child_node
                ):
                    child = right
                    child_node = right_node
                    child_value = right_value
            if not (
                child_value > node_value
                or (child_value == node_value and child_node < node)
            ):
                break
            nodes[position] = child_node
            positions[child_node] = position
            position = child
        nodes[position] = node
        positions[node] = position


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

    if kernel is not None and not capture_paths:
        return _compiled_push(graph, seed, event, restart=restart, tolerance=tolerance)

    # 1. 残差只存正值节点；出堆次序由 (残差降序, 节点升序) 唯一确定，惰性堆与原索引堆逐次相同。
    residual: dict[int, float] = {}
    reserve: dict[int, float] = {}
    queue: list[tuple[float, int]] = []
    for node_id, value in seed:
        residual[node_id] = value
        heappush(queue, (-value, node_id))
    residual_total = math.fsum(value for _, value in seed)
    parent_node: dict[int, int] = {}
    parent_edge: dict[int, int] = {}
    parent_mass: dict[int, float] = {}

    # 2. 反复结算当前最大残差；过期堆项在出堆时丢弃。
    # 局部绑定与转移缓存只省解释器开销；每个浮点运算的操作数与结合次序保持原样。
    pushes = 0
    keep = 1.0 - restart
    lookup = residual.get
    local: dict[int, tuple[tuple[tuple[int, float, int], ...], float]] = {}
    while True:
        if residual_total <= tolerance:
            residual_total = math.fsum(residual[node] for node in sorted(residual))
            if residual_total <= tolerance:
                break
        if not residual:
            raise ArithmeticError("positive residual mass has no heap entry")
        while True:
            negative, node = heappop(queue)
            if lookup(node) == -negative:
                break
        value = residual.pop(node)
        residual_total -= value
        reserve[node] = reserve.get(node, 0.0) + restart * value
        propagated = keep * value
        cached = local.get(node)
        if cached is None:
            cached = graph.transitions(node, event)
            local[node] = cached
        transitions, unspread = cached
        for target, probability, edge_id in transitions:
            addition = propagated * probability
            if addition != 0.0:
                updated = lookup(target, 0.0) + addition
                residual[target] = updated
                heappush(queue, (-updated, target))
                residual_total += addition
                if capture_paths and addition > parent_mass.get(target, 0.0):
                    parent_mass[target] = addition
                    parent_node[target] = node
                    parent_edge[target] = edge_id
        leak = propagated * unspread
        for target, probability in seed:
            addition = leak * probability
            if addition != 0.0:
                updated = lookup(target, 0.0) + addition
                residual[target] = updated
                heappush(queue, (-updated, target))
                residual_total += addition
        pushes += 1
        if pushes % 4096 == 0:
            residual_total = math.fsum(residual[item] for item in sorted(residual))

    # 3. 稀疏结果散回定长数组，读者看到的 reserve 与路径数组不变。
    dense_reserve = np.zeros(graph.max_nodes, dtype=np.float64)
    if reserve:
        nodes = np.fromiter(reserve, dtype=np.int64, count=len(reserve))
        dense_reserve[nodes] = np.fromiter(reserve.values(), dtype=np.float64, count=len(reserve))
    active = np.flatnonzero(dense_reserve > 0.0).astype(np.int32, copy=False)
    paths_node = paths_edge = None
    if capture_paths:
        paths_node = np.full(graph.max_nodes, -1, dtype=np.int32)
        paths_edge = np.full(graph.max_nodes, -1, dtype=np.int32)
        for target, source in parent_node.items():
            paths_node[target] = source
            paths_edge[target] = parent_edge[target]
    return DiffusionResult(
        reserve=dense_reserve,
        active_nodes=active,
        pushes=pushes,
        residual_l1=residual_total,
        parent_node=paths_node,
        parent_edge=paths_edge,
    )


# 编译循环：与上面的解释器循环同一出堆全序、同一浮点运算；转移与精确求和仍由 Python 提供。
def _compiled_push(
    graph: TransitionGraph,
    seed: tuple[tuple[int, float], ...],
    event: int,
    *,
    restart: float,
    tolerance: float,
) -> DiffusionResult:
    assert kernel is not None
    # 1. 初始残差与堆；seed 内节点唯一，按 (残差降序, 节点升序) 入堆。
    table = graph.transition_cache.push_table(graph.max_nodes)
    residual = np.zeros(graph.max_nodes, dtype=np.float64)
    reserve = np.zeros(graph.max_nodes, dtype=np.float64)
    heap_value = np.empty(4096, dtype=np.float64)
    heap_node = np.empty(4096, dtype=np.int64)
    seed_nodes = np.fromiter((node for node, _ in seed), dtype=np.int64, count=len(seed))
    seed_values = np.fromiter((value for _, value in seed), dtype=np.float64, count=len(seed))
    residual[seed_nodes] = seed_values
    scalars = np.array([math.fsum(value for _, value in seed), 0.0])
    counters = np.zeros(3, dtype=np.int64)
    for node_id, value in sorted(seed, key=lambda item: (-item[1], item[0])):
        heap_value[counters[0]] = value
        heap_node[counters[0]] = node_id
        counters[0] += 1

    # 2. 内核跑到需要协助时返回：补转移、精确求和或扩堆，然后接着跑。
    while True:
        status = kernel.run(
            residual, reserve, heap_value, heap_node, table.start, table.count, table.unspread,
            table.target, table.probability, seed_nodes, seed_values, scalars, counters,
            restart, 1.0 - restart, tolerance,
        )
        if status == kernel.DONE:
            break
        if status == kernel.NEED_TRANSITIONS:
            node_id = int(counters[2])
            table.load(node_id, graph.transitions(node_id, event))
        elif status == kernel.NEED_SUM:
            scalars[0] = math.fsum(residual[np.flatnonzero(residual)].tolist())
            scalars[1] = 1.0
        else:
            heap_value = np.resize(heap_value, heap_value.size * 2)
            heap_node = np.resize(heap_node, heap_node.size * 2)
    active = np.flatnonzero(reserve > 0.0).astype(np.int32, copy=False)
    return DiffusionResult(
        reserve=reserve,
        active_nodes=active,
        pushes=int(counters[1]),
        residual_l1=float(scalars[0]),
        parent_node=None,
        parent_edge=None,
    )
