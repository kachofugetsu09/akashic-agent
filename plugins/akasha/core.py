"""
akasha/core.py — 纯算法核心

Akasha RAR（Ripple Activation & Recall）引擎的纯算法层。
只依赖 numpy + jieba + stdlib，不依赖任何框架代码。
"""

from __future__ import annotations

import math
import struct
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

# ── 常量 ──────────────────────────────────────────────────────────────

LONG_DECAY_TAU = 2200.0
RESOURCE_RECOVER_TAU = 14.0
RESOURCE_USE_RATE = 0.35
STRENGTH_LR = 0.18
STRENGTH_CAP = 3.0
ASSISTANT_ONLY_PENALTY = 0.12
FAN_PENALTY_POWER = 0.10
EXPANDED_DIRECT_FLOOR = 0.62
ACTIVATION_THRESHOLD = 0.22
GRAPH_EXPAND_LIMIT = 8
GRAPH_DIRECT_BIAS = 0.25
GRAPH_FAN_PENALTY_POWER = 0.15

# ── 数据类型 ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CoreConfig:
    """算法配置。字段与 AkashaConfig 保持一致的命名和默认值。"""
    dense_top_k: int = 10
    dense_seed_threshold: float = 0.675
    activation_threshold: float = 0.22
    cross_boost: float = 36.0
    nearby_time_seconds: int = 1800
    nearby_dense_threshold: float = 0.28
    soft_recall_threshold: float = 0.165
    soft_recall_direct_floor: float = 0.45
    activate_limit: int = 8


@dataclass(frozen=True)
class AkashaNode:
    key: str
    anchor_id: str
    session_key: str
    turn_seq: int
    first_ts_unix: float
    salience: float
    strength: float
    resource: float
    recall_count: int
    last_activated_seq: int
    last_strength_seq: int
    last_resource_seq: int
    embedding: np.ndarray
    emb_count: int


@dataclass(frozen=True)
class ActivationUpdate:
    key: str
    strength: float
    resource: float
    recall_count: int
    seq: int


@dataclass(frozen=True)
class SourceMessage:
    id: str
    session_key: str
    seq: int
    role: str
    content: str
    ts: str
    salience: float | None = None


@dataclass(frozen=True)
class AkashaCandidate:
    key: str
    source: str
    ripple: float
    direct: float
    state: float
    edge: float
    long: float
    resource: float
    fan: int
    score: float
    suppressed: str = ""
    path_type: str = "direct"
    seed_key: str = ""
    bridge_key: str = ""
    path_value: float = 0.0


@dataclass(frozen=True)
class ActivationTrace:
    seed_count: int
    pool_count: int


@dataclass(frozen=True)
class EdgeUpdate:
    src_key: str
    dst_key: str
    strength: float
    seq: int


@dataclass(frozen=True)
class ActivationEventRow:
    seq: int
    query_id: str
    activated_key: str
    source: str
    score: float
    direct_score: float
    state_score: float
    edge_score: float
    long_score: float
    resource: float
    fan: int


# ── 工具函数 ──────────────────────────────────────────────────────────


def normalize(vector: np.ndarray) -> np.ndarray:
    """归一化向量到单位长度。"""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _best_device() -> str:
    """选择最佳推理设备。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_turn_key(key: str) -> tuple[str, int] | None:
    """从 turn key 解析 session_key 和 seq。"""
    left, sep, right = key.rpartition(":")
    if not sep:
        return None
    try:
        return left, int(right)
    except ValueError:
        return None


def turn_key(session_key: str, seq: int, role: str) -> tuple[str, int, str]:
    """把 message 映射成 turn key。assistant 归到前一个 user turn。"""
    turn_seq = seq if role == "user" else max(0, seq - 1)
    return session_key, turn_seq, f"{session_key}:{turn_seq}"


def serialize_f32(vector: np.ndarray) -> bytes:
    """把 float32 向量打包成 BLOB。"""
    return struct.pack(f"{len(vector)}f", *vector.astype(np.float32).tolist())


def deserialize_f32(blob: bytes) -> np.ndarray:
    """从 BLOB 还原 float32 向量。"""
    if not blob:
        return np.array([], dtype=np.float32)
    return np.array(struct.unpack(f"{len(blob) // 4}f", blob), dtype=np.float32)


def parse_ts_unix(value: str) -> float:
    """把时间字符串转换成 Unix 秒。"""
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return time.time()


def message_id_to_key_from_db(cursor: sqlite3.Cursor, message_id: str) -> str:
    """从 messages 表反查 message id 对应的 turn key。"""
    cursor.execute(
        "SELECT session_key, seq, role FROM messages WHERE id = ?",
        (message_id,),
    )
    row = cursor.fetchone()
    if row:
        _, _, key = turn_key(str(row[0]), int(row[1]), str(row[2] or ""))
        return key
    return message_id  # fallback


# ── DB 工具函数 ───────────────────────────────────────────────────────


def open_source_db(path: str) -> sqlite3.Connection:
    """打开带 sqlite-vec 的源数据库。"""
    import sqlite_vec
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def has_user_turn(cursor: sqlite3.Cursor | None, key: str) -> bool:
    """判断 turn key 对应的轮次是否有 user 消息。"""
    if cursor is None:
        return True
    parsed = parse_turn_key(key)
    if parsed is None:
        return False
    session_key, seq = parsed
    cursor.execute(
        "SELECT 1 FROM messages WHERE session_key = ? AND seq = ? AND role = 'user' LIMIT 1",
        (session_key, seq),
    )
    return cursor.fetchone() is not None


def get_turn_context(cursor: sqlite3.Cursor, key: str) -> tuple[str, str]:
    """从 messages 表读取 user/assistant 消息内容（用于展示）。"""
    parsed = parse_turn_key(key)
    if parsed is None:
        return "", ""
    session_key, seq = parsed
    cursor.execute(
        "SELECT content FROM messages WHERE session_key = ? AND seq = ? AND role = 'user'",
        (session_key, seq),
    )
    user_row = cursor.fetchone()
    cursor.execute(
        "SELECT content FROM messages WHERE session_key = ? AND seq = ? AND role = 'assistant'",
        (session_key, seq + 1),
    )
    assistant_row = cursor.fetchone()
    user_text = (user_row[0] if user_row else "") or ""
    assistant_text = (assistant_row[0] if assistant_row else "") or ""
    user_text = user_text.replace("\n", " ").strip()
    assistant_text = assistant_text.replace("\n", " ").strip()
    if len(user_text) > 58:
        user_text = user_text[:55] + "..."
    if len(assistant_text) > 58:
        assistant_text = assistant_text[:55] + "..."
    return user_text, assistant_text


def load_state(path: str) -> tuple[dict[str, AkashaNode], dict[tuple[str, str], float], dict[str, tuple]]:
    """从 sidecar DB 加载全部节点、边和激活统计。"""
    db = sqlite3.connect(path)
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT key, anchor_id, session_key, turn_seq, first_ts_unix, salience,
               strength, resource, recall_count, last_activated_seq,
               last_strength_seq, last_resource_seq, embedding, emb_count
        FROM akasha_nodes
        """
    )
    nodes: dict[str, AkashaNode] = {}
    for row in cursor.fetchall():
        (
            key, anchor_id, session_key, turn_seq, first_ts_unix,
            salience, strength, resource, recall_count,
            last_activated_seq, last_strength_seq, last_resource_seq,
            embedding_blob, emb_count,
        ) = row
        embedding = deserialize_f32(embedding_blob)
        if embedding.size == 0:
            continue
        nodes[key] = AkashaNode(
            key=key,
            anchor_id=anchor_id,
            session_key=session_key,
            turn_seq=turn_seq,
            first_ts_unix=first_ts_unix,
            salience=salience,
            strength=strength,
            resource=resource,
            recall_count=recall_count,
            last_activated_seq=last_activated_seq,
            last_strength_seq=last_strength_seq,
            last_resource_seq=last_resource_seq,
            embedding=embedding,
            emb_count=emb_count,
        )

    cursor.execute("SELECT src_key, dst_key, weight FROM akasha_edges")
    edges = {(str(src_key), str(dst_key)): float(weight) for src_key, dst_key, weight in cursor.fetchall()}

    cursor.execute(
        """
        SELECT activated_key, COUNT(*) AS c, MAX(seq) AS last_seq
        FROM akasha_activation_events
        GROUP BY activated_key
        """
    )
    activation_stats = {str(key): (int(count), int(last_seq)) for key, count, last_seq in cursor.fetchall()}
    db.close()
    return nodes, edges, activation_stats


# ── 状态计算辅助函数 ──────────────────────────────────────────────────


def recover_resource(node: AkashaNode, seq: int) -> float:
    """计算短期资源恢复后的值。"""
    gap = max(0, seq - node.last_resource_seq)
    return 1.0 - (1.0 - node.resource) * math.exp(-gap / RESOURCE_RECOVER_TAU)


def decayed_strength(node: AkashaNode, seq: int) -> float:
    """计算长期强度衰减后的值。"""
    gap = max(0, seq - node.last_strength_seq)
    return node.strength * math.exp(-gap / LONG_DECAY_TAU)


def bounded_add(value: float, delta: float, cap: float) -> float:
    """有界增加：越接近 cap 增速越慢。"""
    return value + delta * max(0.0, 1.0 - value / cap)


def fan_counts(edges: dict[tuple[str, str], float]) -> dict[str, int]:
    """统计每个节点的扇入/扇出总数。"""
    fan: dict[str, int] = {}
    for src, dst in edges:
        fan[src] = fan.get(src, 0) + 1
        fan[dst] = fan.get(dst, 0) + 1
    return fan


def edges_by_src(edges: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    """把边表按源节点分组索引。"""
    grouped: dict[str, dict[str, float]] = {}
    for (src, dst), weight in edges.items():
        grouped.setdefault(src, {})[dst] = weight
    return grouped


# ── Dense 计算 ────────────────────────────────────────────────────────


def dense_scores(query_vec: np.ndarray, nodes: dict[str, AkashaNode]) -> dict[str, float]:
    """计算 query 对所有节点的余弦相似度。"""
    if not nodes:
        return {}
    keys = list(nodes.keys())
    matrix = np.vstack([nodes[key].embedding for key in keys])
    scores = np.dot(matrix, normalize(query_vec))
    return {key: float(score) for key, score in zip(keys, scores)}


def dense_candidates(
    query_vec: np.ndarray,
    nodes: dict[str, AkashaNode],
    *,
    limit: int,
) -> list[AkashaCandidate]:
    """纯 Dense top-K 候选。"""
    scores = dense_scores(query_vec, nodes)
    return [
        AkashaCandidate(key=key, source="Dense", ripple=0.0, direct=score,
                        state=0.0, edge=0.0, long=0.0, resource=1.0, fan=0, score=score)
        for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def dense_message_candidates(
    query_vec: np.ndarray,
    nodes: dict[str, AkashaNode],
    message_embeddings: dict[str, np.ndarray],
    message_turn_keys: dict[str, str],
    *,
    limit: int,
) -> list[AkashaCandidate]:
    """从 message-level embedding 命中映射回 turn 的 Dense 候选。"""
    if not message_embeddings:
        return dense_candidates(query_vec, nodes, limit=limit)

    query_norm = normalize(query_vec)
    scored: list[tuple[str, float]] = []
    for message_id, embedding in message_embeddings.items():
        if embedding.size != query_norm.size:
            continue
        score = float(np.dot(normalize(embedding), query_norm))
        scored.append((message_id, score))

    candidates: list[AkashaCandidate] = []
    seen: set[str] = set()
    for message_id, score in sorted(scored, key=lambda item: item[1], reverse=True):
        key = message_turn_keys.get(message_id)
        if key is None or key not in nodes or key in seen:
            continue
        seen.add(key)
        candidates.append(
            AkashaCandidate(key=key, source="Dense", ripple=0.0, direct=score,
                            state=0.0, edge=0.0, long=0.0, resource=1.0, fan=0, score=score)
        )
        if len(candidates) >= limit:
            break
    return candidates


# ── Seed 选择 ─────────────────────────────────────────────────────────


def get_jieba_keywords(text: str) -> str:
    """把文本切成 SQLite FTS 可用的 OR 查询。"""
    import jieba
    words: list[str] = []
    for word in jieba.cut_for_search(text):
        cleaned = "".join(
            char for char in word.strip()
            if char.isalnum() or "\u4e00" <= char <= "\u9fff"
        )
        if len(cleaned) > 1:
            words.append(f'"{cleaned}"')
    return " OR ".join(words[:20])


def seed_pool(
    query: str,
    direct_scores: dict[str, float],
    nodes: dict[str, AkashaNode],
    config: CoreConfig,
    source_cursor: sqlite3.Cursor | None,
) -> tuple[dict[str, str], dict[str, float]]:
    """Dense / FTS / BlackHole 三路种子选择。"""
    ranked = sorted(direct_scores.items(), key=lambda item: item[1], reverse=True)
    seed_sources: dict[str, str] = {}
    seed_energy: dict[str, float] = {}
    for key, score in ranked[:min(100, len(ranked))]:
        if score > config.dense_seed_threshold:
            seed_sources[key] = "Dense"
            seed_energy[key] = 1.0
    if not seed_sources:
        for key, _ in ranked[:config.dense_top_k]:
            seed_sources[key] = "Dense(FB)"
            seed_energy[key] = 1.0

    if source_cursor is not None:
        fts_query = get_jieba_keywords(query)
        if fts_query:
            source_cursor.execute(
                "SELECT rowid FROM messages_fts WHERE content MATCH ? LIMIT 10",
                (fts_query,),
            )
            rowids = [int(row[0]) for row in source_cursor.fetchall()]
            if rowids:
                placeholders = ",".join("?" for _ in rowids)
                source_cursor.execute(
                    f"SELECT session_key, seq, role FROM messages WHERE rowid IN ({placeholders})",
                    rowids,
                )
                for session_key, seq, role in source_cursor.fetchall():
                    _, _, key = turn_key(str(session_key), int(seq), str(role or ""))
                    if key not in nodes:
                        continue
                    if key not in seed_sources:
                        seed_sources[key] = "FTS"
                    elif "FTS" not in seed_sources[key].split("+"):
                        seed_sources[key] += "+FTS"
                    seed_energy[key] = 1.0

    blackhole_hits: list[tuple[str, float]] = []
    for key, node in nodes.items():
        if node.salience <= 0.8 or key in seed_sources:
            continue
        score = direct_scores.get(key, 0.0)
        if score > 0.60:
            blackhole_hits.append((key, score))
    for key, _ in sorted(blackhole_hits, key=lambda item: item[1], reverse=True)[:5]:
        seed_sources[key] = "BlackHole"
        seed_energy[key] = 1.0

    return seed_sources, seed_energy


# ── 扩散矩阵 ──────────────────────────────────────────────────────────


def state_array(
    keys: list[str],
    nodes: dict[str, AkashaNode],
    fan: dict[str, int],
    seq: int,
) -> np.ndarray:
    """计算节点状态权重（salience + 长期强度 + 短期资源 + fan 惩罚）。"""
    values = np.zeros(len(keys), dtype=np.float32)
    for index, key in enumerate(keys):
        node = nodes[key]
        long_score = min(1.0, decayed_strength(node, seq) / STRENGTH_CAP)
        resource = recover_resource(node, seq)
        values[index] = (
            math.exp(1.4 * node.salience + 1.0 * long_score)
            * resource
            / math.sqrt(1.0 + fan.get(key, 0))
        )
    return values


def cross_matrix(
    keys: list[str],
    edges: dict[tuple[str, str], float],
    index_by_key: dict[str, int],
    edges_by_src: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    """构建微型图内部的共激活边矩阵。"""
    matrix = np.zeros((len(keys), len(keys)), dtype=np.float32)
    if edges_by_src is not None:
        for src_key in keys:
            src_index = index_by_key[src_key]
            for dst_key, weight in edges_by_src.get(src_key, {}).items():
                dst_index = index_by_key.get(dst_key)
                if dst_index is not None:
                    matrix[dst_index, src_index] = max(matrix[dst_index, src_index], weight)
        return matrix
    for (src_key, dst_key), weight in edges.items():
        src_index = index_by_key.get(src_key)
        dst_index = index_by_key.get(dst_key)
        if src_index is not None and dst_index is not None:
            matrix[dst_index, src_index] = max(matrix[dst_index, src_index], weight)
    return matrix


def keep_top_edges_per_column(matrix: np.ndarray, *, top_k: int) -> np.ndarray:
    """每列只保留最强的 top_k 条边。"""
    if len(matrix) <= top_k:
        return matrix
    kth = np.partition(matrix, -top_k, axis=0)[-top_k]
    return np.where(matrix >= kth[np.newaxis, :], matrix, 0.0)


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """对转移矩阵按列归一化。"""
    sums = matrix.sum(axis=0)
    sums[sums == 0] = 1e-10
    return matrix / sums


def initial_energy(
    keys: list[str],
    seed_energy: dict[str, float],
    fan: dict[str, int],
    index_by_key: dict[str, int],
) -> np.ndarray:
    """构造 RWR 初始能量向量。"""
    energy = np.zeros(len(keys), dtype=np.float32)
    for key, value in seed_energy.items():
        index = index_by_key.get(key)
        if index is not None:
            energy[index] = value / math.sqrt(1.0 + fan.get(key, 0))
    total = float(energy.sum())
    return energy / total if total > 0 else energy


# ── 路径回溯 ──────────────────────────────────────────────────────────


def path_info(
    keys: list[str],
    transition: np.ndarray,
    e0: np.ndarray,
    te0: np.ndarray,
) -> dict[str, tuple[str, str, str, float]]:
    """回溯每个候选的能量路径（direct / 1hop / 2hop）。"""
    result: dict[str, tuple[str, str, str, float]] = {}
    seed_indices = np.where(e0 > 0)[0]
    for index, key in enumerate(keys):
        c0 = float(0.2 * e0[index])
        c1_vec = 0.16 * transition[index, :] * e0
        s1 = int(np.argmax(c1_vec))
        c1 = float(c1_vec[s1])
        c2_vec = 0.64 * transition[index, :] * te0
        c2_vec[index] = 0.0
        c2_vec[seed_indices] = 0.0
        bridge = int(np.argmax(c2_vec))
        c2 = float(c2_vec[bridge])
        s2 = int(np.argmax(transition[bridge, :] * e0))
        if c0 >= c1 and c0 >= c2:
            result[key] = ("direct", "", "", c0)
        elif c1 >= c2:
            result[key] = ("1hop", keys[s1], "", c1)
        else:
            result[key] = ("2hop", keys[s2], keys[bridge], c2)
    return result


# ── 候选评分 ──────────────────────────────────────────────────────────


def score_candidates(
    keys: list[str],
    nodes: dict[str, AkashaNode],
    direct_scores: dict[str, float],
    seed_sources: dict[str, str],
    current: np.ndarray,
    state_arr: np.ndarray,
    cross_mat: np.ndarray,
    fan: dict[str, int],
    seq: int,
    path_info_dict: dict[str, tuple[str, str, str, float]],
    config: CoreConfig,
    source_cursor: sqlite3.Cursor | None,
    *,
    soft_recall: bool,
    return_limit: int | None,
) -> tuple[list[AkashaCandidate], list[AkashaCandidate]]:
    """计算最终 Ripple 分数，返回 (candidates, suppressed)。"""
    all_candidates: dict[str, AkashaCandidate] = {}
    max_state = max(float(np.max(state_arr)), 1e-10)
    for index, key in enumerate(keys):
        node = nodes[key]
        long_score = min(1.0, decayed_strength(node, seq) / STRENGTH_CAP)
        resource = recover_resource(node, seq)
        fan_value = fan.get(key, 0)
        direct_value = max(0.0, direct_scores.get(key, 0.0))
        state_value = min(1.0, float(state_arr[index]) / max_state)
        edge_value = float(np.max(cross_mat[index])) if len(keys) else 0.0
        ptype, seed_key, bridge_key, path_value = path_info_dict.get(key, ("direct", "", "", 0.0))
        hop_penalty = {"direct": 1.0, "1hop": 0.86, "2hop": 0.62}.get(ptype, 0.62)
        source = seed_sources.get(key, "Expanded")
        direct_weight = 0.50 if source != "Expanded" else 0.18
        fan_penalty = math.pow(1.0 + fan_value, FAN_PENALTY_POWER)
        user_penalty = 1.0 if has_user_turn(source_cursor, key) else ASSISTANT_ONLY_PENALTY
        score = (
            float(current[index]) * 3.0 * state_value
            + direct_weight * direct_value
            + 0.18 * long_score
            + 0.12 * node.salience
            + 0.20 * min(1.0, edge_value)
        ) * resource * hop_penalty * user_penalty / fan_penalty
        if source == "Expanded" and ptype == "1hop" and direct_value >= EXPANDED_DIRECT_FLOOR:
            score = max(score, config.activation_threshold + 0.01)
        all_candidates[key] = AkashaCandidate(
            key=key, source=source, ripple=float(current[index]),
            direct=direct_value, state=state_value, edge=edge_value,
            long=long_score, resource=resource, fan=fan_value,
            score=score, path_type=ptype, seed_key=seed_key,
            bridge_key=bridge_key, path_value=path_value,
        )

    # Bridge 提升
    for child in list(all_candidates.values()):
        if child.path_type != "2hop" or not child.bridge_key:
            continue
        bridge = all_candidates.get(child.bridge_key)
        if bridge is None or not has_user_turn(source_cursor, bridge.key):
            continue
        bridge_score = max(
            bridge.score,
            child.score * 0.62,
            bridge.direct * 0.24 + bridge.state * 0.08,
        )
        all_candidates[bridge.key] = AkashaCandidate(
            key=bridge.key, source="Bridge",
            ripple=bridge.ripple, direct=bridge.direct,
            state=bridge.state, edge=bridge.edge,
            long=bridge.long, resource=bridge.resource,
            fan=bridge.fan, score=bridge_score, path_type="bridge",
            seed_key=child.seed_key, bridge_key="",
            path_value=max(bridge.path_value, child.path_value),
        )

    candidates: list[AkashaCandidate] = []
    suppressed: list[AkashaCandidate] = []
    for candidate in all_candidates.values():
        soft_hit = (
            soft_recall
            and candidate.score >= config.soft_recall_threshold
            and candidate.direct >= config.soft_recall_direct_floor
            and candidate.source in {"Bridge", "Expanded"}
            and candidate.path_type in {"bridge", "1hop", "2hop"}
        )
        if candidate.score >= config.activation_threshold or soft_hit:
            if soft_hit and candidate.score < config.activation_threshold:
                candidate = AkashaCandidate(
                    key=candidate.key, source=candidate.source,
                    ripple=candidate.ripple, direct=candidate.direct,
                    state=candidate.state, edge=candidate.edge,
                    long=candidate.long, resource=candidate.resource,
                    fan=candidate.fan, score=candidate.score,
                    suppressed="soft-recall", path_type=candidate.path_type,
                    seed_key=candidate.seed_key, bridge_key=candidate.bridge_key,
                    path_value=candidate.path_value,
                )
            candidates.append(candidate)
        else:
            suppressed.append(
                AkashaCandidate(
                    key=candidate.key, source=candidate.source,
                    ripple=candidate.ripple, direct=candidate.direct,
                    state=candidate.state, edge=candidate.edge,
                    long=candidate.long, resource=candidate.resource,
                    fan=candidate.fan, score=candidate.score,
                    suppressed="below-threshold", path_type=candidate.path_type,
                    seed_key=candidate.seed_key, bridge_key=candidate.bridge_key,
                    path_value=candidate.path_value,
                )
            )
    candidates.sort(key=lambda item: item.score, reverse=True)
    suppressed.sort(key=lambda item: item.score, reverse=True)
    limit = return_limit or config.activate_limit
    return candidates[:limit], suppressed[:limit]


def graph_expand_candidates(
    query_vec: np.ndarray,
    nodes: dict[str, AkashaNode],
    direct_scores: dict[str, float],
    fan: dict[str, int],
    seq: int,
    source_cursor: sqlite3.Cursor | None,
    edges_by_src: dict[str, dict[str, float]] | None,
    graph_seed_keys: list[str],
) -> list[AkashaCandidate]:
    """沿 Dense 种子的强共激活边补一跳候选。"""
    if edges_by_src is None or not graph_seed_keys:
        return []

    seed_set = {key for key in graph_seed_keys if key in nodes}
    in_strength: dict[str, float] = {}
    for src_neighbors in edges_by_src.values():
        for dst_key, edge_weight in src_neighbors.items():
            in_strength[dst_key] = in_strength.get(dst_key, 0.0) + edge_weight

    candidates: list[AkashaCandidate] = []
    for seed_key in graph_seed_keys:
        if seed_key not in nodes:
            continue
        raw_neighbors = edges_by_src.get(seed_key, {})
        out_strength = sum(raw_neighbors.values())
        if out_strength <= 0:
            continue

        scored_neighbors = []
        for key, edge_weight in raw_neighbors.items():
            if key not in nodes or key in seed_set or not has_user_turn(source_cursor, key):
                continue
            dst_strength = in_strength.get(key, edge_weight)
            edge_signal = edge_weight / math.sqrt(max(out_strength * dst_strength, 1e-9))
            direct = max(0.0, direct_scores.get(key, 0.0))
            degree_penalty = math.pow(1.0 + max(0, fan.get(key, 0)), GRAPH_FAN_PENALTY_POWER)
            candidate_signal = edge_signal * (GRAPH_DIRECT_BIAS + direct) / degree_penalty
            scored_neighbors.append((candidate_signal, edge_signal, direct, key, edge_weight))
        scored_neighbors.sort(reverse=True, key=lambda item: item[0])
        max_edge_signal = max((item[1] for item in scored_neighbors), default=0.0)

        for _, edge_signal, direct, key, edge_weight in scored_neighbors[:GRAPH_EXPAND_LIMIT]:
            if max_edge_signal <= 0:
                continue
            node = nodes[key]
            degree = max(0, fan.get(key, 0))
            resource = recover_resource(node, seq)
            long_score = min(1.0, node.strength / STRENGTH_CAP)
            normalized_edge = edge_signal / max_edge_signal
            score = (
                3.0
                * normalized_edge
                * (GRAPH_DIRECT_BIAS + direct)
                * (1.0 + 0.15 * long_score)
                / math.pow(1.0 + degree, GRAPH_FAN_PENALTY_POWER)
            )
            candidates.append(AkashaCandidate(
                key=key, source="Graph", ripple=float(edge_weight),
                direct=direct, state=0.0, edge=float(edge_signal),
                long=long_score, resource=resource, fan=degree,
                score=float(score), path_type="1hop",
                seed_key=seed_key, path_value=float(edge_signal),
            ))
    return candidates


def merge_active_candidates(
    candidates: list[AkashaCandidate],
    graph_candidates: list[AkashaCandidate],
    limit: int,
) -> list[AkashaCandidate]:
    best_by_key: dict[str, AkashaCandidate] = {}
    for item in candidates + graph_candidates:
        current = best_by_key.get(item.key)
        if current is None or item.score > current.score:
            best_by_key[item.key] = item
    merged = sorted(best_by_key.values(), key=lambda item: item.score, reverse=True)
    return merged[:limit]


# ── 主入口 ────────────────────────────────────────────────────────────


def compute_candidates(
    query: str,
    query_vec: np.ndarray,
    nodes: dict[str, AkashaNode],
    edges: dict[tuple[str, str], float],
    seq: int,
    *,
    config: CoreConfig,
    fan: dict[str, int],
    source_cursor: sqlite3.Cursor | None = None,
    edges_by_src: dict[str, dict[str, float]] | None = None,
    soft_recall: bool = False,
    return_limit: int | None = None,
    graph_seed_keys: list[str] | None = None,
) -> tuple[list[AkashaCandidate], list[AkashaCandidate], ActivationTrace]:
    """
    状态化 RAR 扩散主入口。

    参数：
        query: 查询文本
        query_vec: 查询向量（已归一化）
        nodes: 所有状态节点 {key: AkashaNode}
        edges: 共激活边 {(src, dst): weight}
        seq: 当前消息序号
        config: 算法参数
        fan: 节点扇出统计
        source_cursor: 源数据库 cursor（FTS 和 user_turn 查询）
        edges_by_src: 边按源节点索引（可选加速）
        soft_recall: 是否开启软召回（展示用）
        return_limit: 返回数量上限

    返回：
        (candidates, suppressed, trace)
    """
    if not nodes:
        return [], [], ActivationTrace(seed_count=0, pool_count=0)

    direct_scores_map = dense_scores(query_vec, nodes)
    seed_sources, seed_energy = seed_pool(
        query, direct_scores_map, nodes, config, source_cursor,
    )
    if not seed_sources:
        return [], [], ActivationTrace(seed_count=0, pool_count=0)

    micro_keys = set(seed_sources)
    for seed_key in seed_sources:
        seed_ts = nodes[seed_key].first_ts_unix
        for key, node in nodes.items():
            if key in micro_keys:
                continue
            is_near = abs(node.first_ts_unix - seed_ts) <= config.nearby_time_seconds
            if is_near and direct_scores_map.get(key, 0.0) > config.nearby_dense_threshold:
                micro_keys.add(key)
    valid_keys = list(micro_keys)
    if not valid_keys:
        return [], [], ActivationTrace(seed_count=0, pool_count=0)

    index_by_key = {key: idx for idx, key in enumerate(valid_keys)}
    embeddings = np.vstack([nodes[key].embedding for key in valid_keys])
    sim_matrix = np.maximum(np.dot(embeddings, embeddings.T), 0.0)
    np.fill_diagonal(sim_matrix, 0.0)

    state_arr = state_array(valid_keys, nodes, fan, seq)
    cross_mat = cross_matrix(valid_keys, edges, index_by_key, edges_by_src)

    transition = sim_matrix * state_arr[:, np.newaxis]
    transition *= 1.0 + config.cross_boost * cross_mat
    transition = keep_top_edges_per_column(transition, top_k=12)
    transition = normalize_columns(transition)

    e0 = initial_energy(valid_keys, seed_energy, fan, index_by_key)
    te0 = np.dot(transition, e0)
    current = e0.copy()
    for _ in range(2):
        current = 0.8 * np.dot(transition, current) + 0.2 * e0

    path_info_dict = path_info(valid_keys, transition, e0, te0)
    candidates, suppressed = score_candidates(
        valid_keys, nodes, direct_scores_map, seed_sources,
        current, state_arr, cross_mat, fan, seq,
        path_info_dict, config, source_cursor,
        soft_recall=soft_recall, return_limit=return_limit,
    )
    if graph_seed_keys:
        graph_candidates = graph_expand_candidates(
            query_vec, nodes, direct_scores_map, fan, seq,
            source_cursor, edges_by_src, graph_seed_keys,
        )
        limit = return_limit or config.activate_limit
        candidates = merge_active_candidates(candidates, graph_candidates, limit)
        active_keys = {item.key for item in candidates}
        suppressed = [item for item in suppressed if item.key not in active_keys]
    return candidates, suppressed, ActivationTrace(
        seed_count=len(seed_sources), pool_count=len(valid_keys),
    )


# ── 状态更新 ──────────────────────────────────────────────────────────


def activation_updates(
    items: list[AkashaCandidate],
    nodes: dict[str, AkashaNode],
    seq: int,
) -> list[ActivationUpdate]:
    """生成被激活节点的状态更新。"""
    updates: list[ActivationUpdate] = []
    for item in items:
        node = nodes.get(item.key)
        if node is None:
            continue
        strength = decayed_strength(node, seq)
        strength = bounded_add(strength, STRENGTH_LR * item.score, STRENGTH_CAP)
        resource = recover_resource(node, seq)
        resource *= max(0.05, 1.0 - RESOURCE_USE_RATE * min(1.0, item.score))
        updates.append(ActivationUpdate(
            key=item.key, strength=strength, resource=resource,
            recall_count=node.recall_count + 1, seq=seq,
        ))
    return updates
