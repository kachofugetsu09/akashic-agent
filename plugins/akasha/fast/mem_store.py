"""
MemoryStore —— 纯内存版 AkashaStore，duck-type 出 AkashaReplayRuntime 用到的方法。

替代 replay 每轮从 sqlite 重读全库 + 反序列化（O(N²)）。线上 engine.py 本来就是内存增量。

性能增量维护（消灭每轮 O(E)）：
  * edges_by_src / fan：增量更新（engine.py 已有先例）
  * in_strength：用 exp 衰减因式分解 —— in_strength[d](t)=e^{-(t-t0)/τ}·A_dec[d]+A_const[d]，
    A 与 t 无关、只在边变化时增量；t0 在数学上完全约掉，仅用于防 exp 溢出。
    → graph_expand 的 O(E) 全边遍历降为 O(候选)。
mutating 公式逐行照搬 store.py，保证与落库版等价（差分 parity 验证）。
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from plugins.akasha.core import (
    AkashaNode, EDGE_DECAY_TAU, advance_salience_state, causal_salience,
    effective_edge_weight, bounded_add, heterosynaptic_depression, initial_strength,
    normalize, parse_ts_unix, turn_key,
)


class MemoryStore:
    def __init__(self):
        self._nodes: dict[str, AkashaNode] = {}
        self._edges: dict[tuple[str, str], float] = {}
        self._meta: dict[tuple[str, str], float] = {}
        self._cocount: dict[tuple[str, str], int] = {}
        self._csum: np.ndarray | None = None
        self._ccount: int = 0
        self._frozen = False
        # 增量结构
        self._ebs: dict[str, dict[str, float]] = {}      # edges_by_src
        self._fan: dict[str, int] = {}                   # fan 计数
        self._A_dec: dict[str, float] = {}               # Σ w·e^{(lu-t0)/τ}  (lu>0)
        self._A_const: dict[str, float] = {}             # Σ w               (lu<=0)
        self._t0: float | None = None                    # 参考纪元（数学上约掉，仅防溢出）

    # ── 读路径 ───────────────────────────────────────────────────────
    def list_nodes(self):
        return list(self._nodes.values())

    def load_edges_with_meta(self):
        return self._edges, self._meta  # 不复制：replay 只读不改

    def get_node(self, key):
        return self._nodes.get(key)

    def edges_by_src_view(self):
        return self._ebs

    def fan_view(self):
        return self._fan

    def in_strength(self, dst: str, now_ts: float):
        """= Σ_src effective_edge_weight(w,lu,now)。无入边返回 None（对齐原 dict.get 缺省语义）。"""
        in_dec = dst in self._A_dec
        if not in_dec and dst not in self._A_const:
            return None
        val = self._A_const.get(dst, 0.0)
        if in_dec and self._t0 is not None:
            val += math.exp(-(now_ts - self._t0) / EDGE_DECAY_TAU) * self._A_dec[dst]
        return val

    # ── A 的增量加减 ─────────────────────────────────────────────────
    def _contrib_add(self, dst, w, lu, sign):
        if lu > 0:
            if self._t0 is None:
                self._t0 = lu
            term = w * math.exp((lu - self._t0) / EDGE_DECAY_TAU)
            self._A_dec[dst] = self._A_dec.get(dst, 0.0) + sign * term
        else:
            self._A_const[dst] = self._A_const.get(dst, 0.0) + sign * w

    # ── 写路径：逐行照搬 store.py ────────────────────────────────────
    def upsert_message_node(self, message, embedding) -> str:
        session_key, turn_seq, key = turn_key(message.session_key, message.seq, message.role)
        vector = normalize(np.array(embedding, dtype=np.float32))
        ts_unix = parse_ts_unix(message.ts)
        prior_sum, prior_count = self._csum, self._ccount
        salience = (
            causal_salience(vector, prior_sum, prior_count)
            if getattr(message, "salience", None) is None
            else min(1.0, max(0.0, float(message.salience)))
        )
        self._csum, self._ccount = advance_salience_state(prior_sum, prior_count, vector)
        node = self._nodes.get(key)
        if node is None:
            self._nodes[key] = AkashaNode(
                key=key, anchor_id=message.id, session_key=session_key, turn_seq=turn_seq,
                first_ts_unix=ts_unix, salience=salience, strength=initial_strength(salience),
                resource=1.0, recall_count=0, last_activated_ts=ts_unix,
                last_strength_ts=ts_unix, last_resource_ts=ts_unix, embedding=vector, emb_count=1)
        else:
            old_count = max(1, node.emb_count)
            merged = normalize(node.embedding * old_count + vector)
            anchor = message.id if message.role == "user" else node.anchor_id
            self._nodes[key] = replace(node, anchor_id=anchor,
                                       salience=max(node.salience, salience),
                                       embedding=merged, emb_count=old_count + 1)
        return key

    def update_activation_batch(self, updates) -> None:
        if self._frozen:
            return
        for u in updates:
            n = self._nodes.get(u.key)
            if n is None:
                continue
            self._nodes[u.key] = replace(
                n, strength=u.strength, resource=u.resource, recall_count=u.recall_count,
                last_activated_ts=u.ts, last_strength_ts=u.ts, last_resource_ts=u.ts)

    def upsert_edges(self, updates) -> None:
        if self._frozen:
            return
        for u in updates:
            if u.src_key == u.dst_key:
                continue
            ek = (u.src_key, u.dst_key)
            if ek not in self._edges:
                neww = 0.12 * u.strength
                self._edges[ek] = neww
                self._meta[ek] = u.ts
                self._cocount[ek] = 1
                self._ebs.setdefault(u.src_key, {})[u.dst_key] = neww
                self._fan[u.src_key] = self._fan.get(u.src_key, 0) + 1
                self._fan[u.dst_key] = self._fan.get(u.dst_key, 0) + 1
                self._contrib_add(u.dst_key, neww, u.ts, +1)
            else:
                oldw, oldlu = self._edges[ek], self._meta[ek]
                neww = bounded_add(effective_edge_weight(oldw, oldlu, u.ts), 0.12 * u.strength, 2.0)
                self._contrib_add(u.dst_key, oldw, oldlu, -1)   # 撤旧
                self._contrib_add(u.dst_key, neww, u.ts, +1)    # 加新
                self._edges[ek] = neww
                self._meta[ek] = u.ts
                self._cocount[ek] = self._cocount.get(ek, 0) + 1
                self._ebs[u.src_key][u.dst_key] = neww
        # heterosynaptic：被强化节点的非活动出边按权重压抑（动 _ebs/_edges/in_strength 因式分解）
        for src_key, dst_key, new_w in heterosynaptic_depression(
            updates, lambda s: self._ebs.get(s, {})
        ):
            ek = (src_key, dst_key)
            oldw, oldlu = self._edges[ek], self._meta[ek]
            self._contrib_add(dst_key, oldw, oldlu, -1)
            self._contrib_add(dst_key, new_w, oldlu, +1)        # last_used_ts 不变：非"使用"
            self._edges[ek] = new_w
            self._ebs[src_key][dst_key] = new_w

    def insert_activation_events(self, rows) -> None:
        return None

    def insert_query_log(self, **kwargs) -> None:
        return None

    def fan(self) -> dict[str, int]:
        return dict(self._fan)


class CapturingMemoryStore(MemoryStore):
    """重放时把 query_log / activation_events 攒在内存，末尾由 dump.dump_to_db 一起落库。

    与 MemoryStore 唯一区别：override 两个诊断写入(原本是 no-op)为内存累积。
    """

    def __init__(self):
        super().__init__()
        self.query_logs: list[dict] = []
        self.activation_events: list = []

    def insert_query_log(self, **kwargs) -> None:
        self.query_logs.append(kwargs)

    def insert_activation_events(self, rows) -> None:
        self.activation_events.extend(rows)
