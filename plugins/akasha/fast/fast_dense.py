"""
fast_dense —— 把 dense_message_candidates 的 message_index=None 慢路径（对全部消息做
Python 点积循环）换成"构建一次归一矩阵 + 缓存复用 + 向量化 matmul"。

数字与 core 原函数完全一致（就是 core 自己 message_index 快路径的逻辑），只是：
  * 不再每轮重建/重算（缓存键 = (id(message_embeddings), len)，replay 全程同一 dict 对象、长度不变 → 只建一次）
  * 选择逻辑（normalize·query → 降序 → 跳过未提交节点/已见 → 取 limit）逐行照搬，保证 top-K 与排名一致。

通过 install() 同时替换 core 和 replay 两个命名空间里的引用（replay 用 from-import 持有自己的绑定）。
不修改任何源文件。
"""
from __future__ import annotations

import numpy as np

import plugins.akasha.core as _core
import plugins.akasha.replay as _replay
from plugins.akasha.core import (
    AkashaCandidate, normalize, build_dense_message_index, dense_candidates,
)

_CACHE: dict[tuple[int, int], object] = {}


def fast_dense_message_candidates(query_vec, nodes, message_embeddings, message_turn_keys,
                                  *, limit, message_index=None):
    if not message_embeddings:
        return dense_candidates(query_vec, nodes, limit=limit)
    query_norm = normalize(query_vec)
    idx = message_index
    if idx is None:
        ck = (id(message_embeddings), len(message_embeddings))
        idx = _CACHE.get(ck)
        if idx is None:
            idx = build_dense_message_index(message_embeddings)
            _CACHE[ck] = idx
    indexed = idx.by_dim.get(int(query_norm.size))
    if indexed is None:
        return []
    message_ids, matrix = indexed
    scores = np.dot(matrix, query_norm)
    order = np.argsort(-scores, kind="stable")  # 降序，平局保持原插入序（对齐 sorted reverse=True）
    candidates: list = []
    seen: set = set()
    for j in order:
        key = message_turn_keys.get(message_ids[j])
        if key is None or key not in nodes or key in seen:
            continue
        seen.add(key)
        s = float(scores[j])
        candidates.append(AkashaCandidate(
            key=key, source="Dense", ripple=0.0, direct=s, state=0.0, edge=0.0,
            long=0.0, resource=1.0, fan=0, score=s))
        if len(candidates) >= limit:
            break
    return candidates


_ORIG: dict = {}  # uninstall 还原用


def install():
    _CACHE.clear()
    if not _ORIG:
        _ORIG.update(
            core=_core.dense_message_candidates,
            replay=_replay.dense_message_candidates,
        )
    _core.dense_message_candidates = fast_dense_message_candidates
    _replay.dense_message_candidates = fast_dense_message_candidates
    print("[fast_dense] installed (向量化 dense + 缓存归一矩阵)", flush=True)


def uninstall():
    """还原 dense_message_candidates 并清缓存（同进程测试/复用时调用）。"""
    if _ORIG:
        _core.dense_message_candidates = _ORIG["core"]
        _replay.dense_message_candidates = _ORIG["replay"]
        _ORIG.clear()
    _CACHE.clear()


def clear_cache():
    _CACHE.clear()
