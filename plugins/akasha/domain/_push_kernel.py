"""Optional compiled Residual Push loop; same pop order and float operations as the Python loop.

Not cached to disk: the installed source tree is read-only, so each process compiles once on first use.
"""

from __future__ import annotations

import numpy as np
from numba import njit  # pyright: ignore[reportMissingTypeStubs]

DONE = 0
NEED_TRANSITIONS = 1
NEED_SUM = 2
NEED_HEAP = 3


# 堆键 (残差降序, 节点升序)；与 Python 惰性 heapq 的出堆全序相同。
@njit(inline="always")
def _before(value_a: float, node_a: int, value_b: float, node_b: int) -> bool:
    return value_a > value_b or (value_a == value_b and node_a < node_b)


@njit
def _push(heap_value: np.ndarray, heap_node: np.ndarray, size: int, value: float, node: int) -> int:
    position = size
    while position > 0:
        parent = (position - 1) // 2
        if not _before(value, node, heap_value[parent], heap_node[parent]):
            break
        heap_value[position] = heap_value[parent]
        heap_node[position] = heap_node[parent]
        position = parent
    heap_value[position] = value
    heap_node[position] = node
    return size + 1


@njit
def _pop(heap_value: np.ndarray, heap_node: np.ndarray, size: int) -> int:
    size -= 1
    value = heap_value[size]
    node = heap_node[size]
    position = 0
    while True:
        child = 2 * position + 1
        if child >= size:
            break
        right = child + 1
        if right < size and _before(heap_value[right], heap_node[right], heap_value[child], heap_node[child]):
            child = right
        if not _before(heap_value[child], heap_node[child], value, node):
            break
        heap_value[position] = heap_value[child]
        heap_node[position] = heap_node[child]
        position = child
    if size > 0:
        heap_value[position] = value
        heap_node[position] = node
    return size


# 运行到需要 Python 协助为止：缺某节点的转移、需要精确求和（math.fsum）或堆满。
# scalars: [residual_total, summed_flag]；counters: [heap_size, pushes, pending_node]
@njit
def run(
    residual: np.ndarray,
    reserve: np.ndarray,
    heap_value: np.ndarray,
    heap_node: np.ndarray,
    trans_start: np.ndarray,
    trans_count: np.ndarray,
    unspread: np.ndarray,
    trans_target: np.ndarray,
    trans_probability: np.ndarray,
    seed_nodes: np.ndarray,
    seed_values: np.ndarray,
    scalars: np.ndarray,
    counters: np.ndarray,
    restart: float,
    keep: float,
    tolerance: float,
) -> int:
    residual_total = scalars[0]
    size = counters[0]
    pushes = counters[1]
    while True:
        # 1. 增量总量跌破容差时交回 Python 精确求和；求和后仍不超过容差即结束。
        if residual_total <= tolerance:
            if scalars[1] == 0.0:
                scalars[0] = residual_total
                counters[0] = size
                counters[1] = pushes
                return NEED_SUM
            scalars[1] = 0.0
            if residual_total <= tolerance:
                scalars[0] = residual_total
                counters[0] = size
                counters[1] = pushes
                return DONE
        scalars[1] = 0.0

        # 2. 丢弃过期堆项，取当前最大残差；缺转移时放回原堆项并请求加载。
        while True:
            if size == 0:
                raise ArithmeticError("positive residual mass has no heap entry")
            value = heap_value[0]
            node = heap_node[0]
            size = _pop(heap_value, heap_node, size)
            if residual[node] == value:
                break
        if trans_count[node] < 0:
            size = _push(heap_value, heap_node, size, value, node)
            scalars[0] = residual_total
            counters[0] = size
            counters[1] = pushes
            counters[2] = node
            return NEED_TRANSITIONS
        if size + trans_count[node] + seed_nodes.size > heap_value.size:
            size = _push(heap_value, heap_node, size, value, node)
            scalars[0] = residual_total
            counters[0] = size
            counters[1] = pushes
            return NEED_HEAP

        # 3. 结算该节点：与 Python 循环相同的乘加与次序。
        residual[node] = 0.0
        residual_total -= value
        reserve[node] = reserve[node] + restart * value
        propagated = keep * value
        start = trans_start[node]
        for offset in range(trans_count[node]):
            target = int(trans_target[start + offset])
            addition = propagated * trans_probability[start + offset]
            if addition != 0.0:
                updated = residual[target] + addition
                residual[target] = updated
                size = _push(heap_value, heap_node, size, updated, target)
                residual_total += addition
        leak = propagated * unspread[node]
        for index in range(seed_nodes.size):
            target = int(seed_nodes[index])
            addition = leak * seed_values[index]
            if addition != 0.0:
                updated = residual[target] + addition
                residual[target] = updated
                size = _push(heap_value, heap_node, size, updated, target)
                residual_total += addition
        pushes += 1
        if pushes % 4096 == 0:
            scalars[0] = residual_total
            counters[0] = size
            counters[1] = pushes
            return NEED_SUM
