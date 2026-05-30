"""
akasha/core.py — 纯算法核心

Akasha RAR（Ripple Activation & Recall）引擎的纯算法层。
只依赖 numpy + jieba + stdlib，不依赖任何框架代码。
"""

from __future__ import annotations

import math
import struct
import sqlite3
from dataclasses import dataclass
from datetime import datetime

import numpy as np

# ── 常量 ──────────────────────────────────────────────────────────────

# 时间衰减常数（秒）—— 基于 sessions.db 真实对话间隔统计
LONG_DECAY_TAU = 604800.0       # 7 天：strength 衰减到 1/e
RESOURCE_RECOVER_TAU = 1800.0   # 30 分钟：短期抑制恢复
EDGE_DECAY_TAU = 1209600.0      # 14 天：Hebbian 边衰减
RESOURCE_USE_RATE = 0.35
STRENGTH_LR = 0.18
STRENGTH_CAP = 3.0
STDP_CAUSAL_EDGE_GAIN = 1.0
STDP_ACAUSAL_EDGE_GAIN = 0.35
STDP_COACTIVE_EDGE_GAIN = 1.0
# 新事件初始 strength：编码即峰值（Ebbinghaus / ACT-R / early-LTP）
# initial_strength = STRENGTH_CAP × (BASE + SALIENCE_BONUS · σ)
INITIAL_STRENGTH_BASE = 0.70
INITIAL_STRENGTH_SALIENCE_BONUS = 0.30
SALIENCE_CENTROID_SCALE = 2.0


def initial_strength(salience: float) -> float:
    """新节点 encoding 时的 strength。高显著度事件起步更接近 cap。"""
    s = max(0.0, min(1.0, salience))
    return STRENGTH_CAP * (INITIAL_STRENGTH_BASE + INITIAL_STRENGTH_SALIENCE_BONUS * s)
ASSISTANT_ONLY_PENALTY = 0.12
FAN_PENALTY_POWER = 0.10
ACTIVATION_THRESHOLD = 0.22
GRAPH_EXPAND_LIMIT = 8
GRAPH_DIRECT_BIAS = 0.25
GRAPH_FAN_PENALTY_POWER = 0.15

# RWR 重启概率 α：扩散迭代和 path_info 各 hop 权重都由它推出，
# 不再是 0.2 / 0.16 / 0.64 三个独立字面量。
#   iteration : r = (1−α)·P·r + α·e0
#   path_info : direct = α, 1hop = α(1−α), 2hop = (1−α)²
RWR_RESTART_ALPHA = 0.2

# FTS 改进参数
FTS_MIN_IDF = 3.5          # 过滤低 IDF 常见 token
FTS_MIN_TOKEN_LEN = 3      # trigram 分词器要求至少 3 字符
FTS_MAX_TOKENS = 10        # FTS query 最多 OR 多少 token
FTS_TOP_K = 10             # BM25 取 top K
FTS_ONLY_MAX_HITS = 5      # 单跑 FTS 没 Dense 配对时最多保留几个
FTS_OVERLAP_BOOST = 1.3    # Dense ∩ FTS 时 seed_energy 倍数

# 模块级 IDF 表（由 engine 调用 set_idf_table 注入）
_IDF_TABLE: dict[str, float] = {}


def set_idf_table(table: dict[str, float] | None) -> None:
    """注入 IDF 表。空表 / None 时退化到无过滤行为。"""
    global _IDF_TABLE
    _IDF_TABLE = table or {}


def load_idf_from_db(conn: sqlite3.Connection) -> dict[str, float]:
    """从 akasha.db 的 fts_token_idf 表加载 IDF 字典。"""
    try:
        rows = conn.execute("SELECT token, idf FROM fts_token_idf").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {t: float(v) for t, v in rows}


def build_idf_table(
    sessions_db_path: str,
    target_conn: sqlite3.Connection,
) -> dict[str, float]:
    """扫描 sessions.db 所有 message 算 IDF，写入 target_conn 的 fts_token_idf 表。

    用法：当 fts_token_idf 不存在或为空时调用，一次性建表。
    增量更新场景下也可重新调用——会全表重写。
    """
    import jieba
    from collections import defaultdict

    sconn = sqlite3.connect(sessions_db_path)
    df: dict[str, int] = defaultdict(int)
    n_docs = 0
    for (content,) in sconn.execute("SELECT content FROM messages"):
        n_docs += 1
        seen: set[str] = set()
        for w in jieba.cut_for_search(content or ""):
            cleaned = "".join(
                ch for ch in w.strip()
                if ch.isalnum() or "一" <= ch <= "鿿"
            ).lower()
            if len(cleaned) > 1 and cleaned not in seen:
                seen.add(cleaned)
                df[cleaned] += 1
    sconn.close()

    idf: dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log((n_docs + 1) / (freq + 1)) + 1

    target_conn.execute("""
        CREATE TABLE IF NOT EXISTS fts_token_idf (
            token TEXT PRIMARY KEY,
            df INTEGER NOT NULL,
            idf REAL NOT NULL
        )
    """)
    target_conn.execute("""
        CREATE TABLE IF NOT EXISTS fts_token_idf_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    target_conn.execute("DELETE FROM fts_token_idf")
    target_conn.executemany(
        "INSERT INTO fts_token_idf VALUES (?, ?, ?)",
        [(t, df[t], idf[t]) for t in df],
    )
    target_conn.execute(
        "INSERT OR REPLACE INTO fts_token_idf_meta VALUES ('n_docs', ?)",
        (str(n_docs),),
    )
    target_conn.commit()
    return idf


def idf_table_is_stale(
    sessions_db_path: str,
    target_conn: sqlite3.Connection,
    drift_ratio: float = 0.20,
) -> bool:
    """判断 IDF 表是否需要重建。"""
    try:
        cnt = target_conn.execute("SELECT COUNT(*) FROM fts_token_idf").fetchone()[0]
    except sqlite3.OperationalError:
        return True
    if cnt == 0:
        return True
    try:
        row = target_conn.execute(
            "SELECT value FROM fts_token_idf_meta WHERE key='n_docs'"
        ).fetchone()
        last_n = int(row[0]) if row else None
    except sqlite3.OperationalError:
        last_n = None
    if last_n is None or last_n == 0:
        return False  # 已有数据但无 meta，先不强制重建
    sconn = sqlite3.connect(sessions_db_path)
    cur_n = sconn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    sconn.close()
    return abs(cur_n - last_n) / last_n > drift_ratio


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
    last_activated_ts: float
    last_strength_ts: float
    last_resource_ts: float
    embedding: np.ndarray
    emb_count: int


@dataclass(frozen=True)
class ActivationUpdate:
    key: str
    strength: float
    resource: float
    recall_count: int
    ts: float


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
class DenseMessageIndex:
    by_dim: dict[int, tuple[tuple[str, ...], np.ndarray]]


@dataclass
class _GraphPathAggregate:
    signal: float = 0.0
    best_signal: float = 0.0
    best_edge: float = 0.0
    best_weight: float = 0.0
    direct: float = 0.0
    paths: float = 0.0
    seed_key: str = ""


@dataclass(frozen=True)
class ActivationTrace:
    seed_count: int
    pool_count: int


@dataclass(frozen=True)
class AkashaActivationSnapshot:
    nodes: dict[str, AkashaNode]
    edges: dict[tuple[str, str], float]
    edges_meta: dict[tuple[str, str], float]
    fan: dict[str, int]
    edges_by_src: dict[str, dict[str, float]]
    message_embeddings: dict[str, np.ndarray]
    message_turn_keys: dict[str, str]
    message_index: DenseMessageIndex | None = None


@dataclass(frozen=True)
class EdgeUpdate:
    src_key: str
    dst_key: str
    strength: float
    ts: float


def activation_edge_updates(
    current_key: str,
    candidates: list[AkashaCandidate],
    ts: float,
) -> list[EdgeUpdate]:
    updates: list[EdgeUpdate] = []
    key_to_score = {item.key: item.score for item in candidates}
    for item in candidates:
        edge_strength = key_to_score.get(item.key, 1.0)
        updates.append(
            EdgeUpdate(item.key, current_key, edge_strength * STDP_CAUSAL_EDGE_GAIN, ts)
        )
        updates.append(
            EdgeUpdate(current_key, item.key, edge_strength * STDP_ACAUSAL_EDGE_GAIN, ts)
        )
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            edge_strength = math.sqrt(key_to_score[left.key] * key_to_score[right.key])
            edge_strength *= STDP_COACTIVE_EDGE_GAIN
            updates.append(EdgeUpdate(left.key, right.key, edge_strength, ts))
            updates.append(EdgeUpdate(right.key, left.key, edge_strength, ts))
    return updates


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


def causal_salience(
    embedding: list[float] | np.ndarray,
    prior_sum: np.ndarray | None,
    prior_count: int,
) -> float:
    """只用当前消息之前的全局重心计算 salience。"""
    if prior_sum is None or prior_count <= 0:
        return 0.0
    vector = normalize(np.array(embedding, dtype=np.float32))
    centroid = normalize(prior_sum / float(prior_count))
    value = (1.0 - float(np.dot(vector, centroid))) * SALIENCE_CENTROID_SCALE
    return min(1.0, max(0.0, value))


def advance_salience_state(
    prior_sum: np.ndarray | None,
    prior_count: int,
    embedding: list[float] | np.ndarray,
) -> tuple[np.ndarray, int]:
    vector = normalize(np.array(embedding, dtype=np.float32))
    total = vector.copy() if prior_sum is None else prior_sum + vector
    return total, prior_count + 1


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
    except ValueError as exc:
        raise ValueError(f"无效时间戳: {value}") from exc


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
               strength, resource, recall_count, last_activated_ts,
               last_strength_ts, last_resource_ts, embedding, emb_count
        FROM akasha_nodes
        """
    )
    nodes: dict[str, AkashaNode] = {}
    for row in cursor.fetchall():
        (
            key, anchor_id, session_key, turn_seq, first_ts_unix,
            salience, strength, resource, recall_count,
            last_activated_ts, last_strength_ts, last_resource_ts,
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
            last_activated_ts=last_activated_ts,
            last_strength_ts=last_strength_ts,
            last_resource_ts=last_resource_ts,
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


def recover_resource(node: AkashaNode, now_ts: float) -> float:
    """计算短期资源恢复后的值（按真实时间）。"""
    gap = max(0.0, now_ts - node.last_resource_ts)
    return 1.0 - (1.0 - node.resource) * math.exp(-gap / RESOURCE_RECOVER_TAU)


def decayed_strength(node: AkashaNode, now_ts: float) -> float:
    """计算长期强度衰减后的值（按真实时间）。"""
    gap = max(0.0, now_ts - node.last_strength_ts)
    return node.strength * math.exp(-gap / LONG_DECAY_TAU)


def effective_edge_weight(weight: float, last_used_ts: float, now_ts: float) -> float:
    """边的 lazy time-decay。"""
    if last_used_ts <= 0:
        return weight
    gap = max(0.0, now_ts - last_used_ts)
    return weight * math.exp(-gap / EDGE_DECAY_TAU)


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


def build_dense_message_index(
    message_embeddings: dict[str, np.ndarray],
) -> DenseMessageIndex:
    grouped: dict[int, list[tuple[str, np.ndarray]]] = {}
    for message_id, embedding in message_embeddings.items():
        grouped.setdefault(int(embedding.size), []).append((message_id, embedding))

    by_dim: dict[int, tuple[tuple[str, ...], np.ndarray]] = {}
    for dim, items in grouped.items():
        message_ids = tuple(message_id for message_id, _ in items)
        matrix = np.vstack([embedding for _, embedding in items]).astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > 0,
        )
        by_dim[dim] = (message_ids, normalized)
    return DenseMessageIndex(by_dim=by_dim)


def dense_message_candidates(
    query_vec: np.ndarray,
    nodes: dict[str, AkashaNode],
    message_embeddings: dict[str, np.ndarray],
    message_turn_keys: dict[str, str],
    *,
    limit: int,
    message_index: DenseMessageIndex | None = None,
) -> list[AkashaCandidate]:
    """从 message-level embedding 命中映射回 turn 的 Dense 候选。"""
    if not message_embeddings:
        return dense_candidates(query_vec, nodes, limit=limit)

    query_norm = normalize(query_vec)
    if message_index is not None:
        indexed = message_index.by_dim.get(int(query_norm.size))
        if indexed is None:
            return []
        message_ids, matrix = indexed
        scored = [
            (message_id, float(score))
            for message_id, score in zip(message_ids, np.dot(matrix, query_norm))
        ]
    else:
        scored = []
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


def graph_seed_keys_from_snapshot(
    query_vec: np.ndarray,
    snapshot: AkashaActivationSnapshot,
    *,
    limit: int,
) -> list[str]:
    if not snapshot.message_embeddings:
        return []
    return [
        item.key
        for item in dense_message_candidates(
            query_vec,
            snapshot.nodes,
            snapshot.message_embeddings,
            snapshot.message_turn_keys,
            limit=limit,
            message_index=snapshot.message_index,
        )
    ]


# ── Seed 选择 ─────────────────────────────────────────────────────────


def get_jieba_keywords(text: str) -> str:
    """把文本切成 SQLite FTS 可用的 OR 查询。"""
    import jieba
    import re

    pairs: list[tuple[str, float]] = []
    seen: set[str] = set()

    def _consider(token: str, default_idf: float) -> None:
        if len(token) < FTS_MIN_TOKEN_LEN or token in seen:
            return
        seen.add(token)
        if _IDF_TABLE:
            idf = _IDF_TABLE.get(token, 6.0)  # \u672a\u89c1\u8fc7\u89c6\u4e3a\u7a00\u6709
            if idf < FTS_MIN_IDF:
                return
            pairs.append((token, idf))
        else:
            pairs.append((token, default_idf))

    # 1. Latin / \u6570\u5b57 \u8bcd\u7ec4\uff08trigram \u76f4\u63a5\u80fd\u5904\u7406\uff09
    for m in re.finditer(r"[a-z0-9_]{3,}", text.lower()):
        _consider(m.group(), 5.0)

    # 2. jieba \u5207\u8bcd
    tokens = list(jieba.lcut(text))
    cleaned_tokens: list[str] = []
    for w in tokens:
        c = "".join(ch for ch in w if "\u4e00" <= ch <= "\u9fff")
        cleaned_tokens.append(c.lower())

    # 3a. 3+ \u5b57\u4e2d\u6587 token \u76f4\u63a5\u52a0\uff08IDF \u8fc7\u6ee4\uff09
    for c in cleaned_tokens:
        if len(c) >= 3:
            _consider(c, 5.0)

    # 3b. 2 \u5b57\u4e2d\u6587 token \u4e0e\u76f8\u90bb\u5b57\u7b26\u62fc\u6210 3 \u5b57 phrase\uff08trigram \u624d\u80fd\u547d\u4e2d\uff09
    for i, c in enumerate(cleaned_tokens):
        if len(c) != 2:
            continue
        # \u62fc\u63a5\u4e0b\u4e00\u4e2a token \u7684\u9996\u5b57
        if i + 1 < len(cleaned_tokens) and cleaned_tokens[i + 1]:
            combined = c + cleaned_tokens[i + 1][0]
            _consider(combined, 5.0)
        # \u4e5f\u62fc\u63a5\u4e0a\u4e00\u4e2a token \u7684\u672b\u5b57
        if i - 1 >= 0 and cleaned_tokens[i - 1]:
            combined = cleaned_tokens[i - 1][-1] + c
            _consider(combined, 5.0)

    pairs.sort(key=lambda x: -x[1])
    pairs = pairs[:FTS_MAX_TOKENS]
    return " OR ".join(f'"{w}"' for w, _ in pairs)


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
            # 用 BM25 排序拿 top K（bm25() 返回负值，越小越匹配）
            try:
                source_cursor.execute(
                    """
                    SELECT rowid, bm25(messages_fts) AS rank
                    FROM messages_fts
                    WHERE content MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, FTS_TOP_K),
                )
                rows = source_cursor.fetchall()
            except sqlite3.OperationalError:
                # FTS5 不支持 bm25 时退回旧行为
                source_cursor.execute(
                    "SELECT rowid, 0 FROM messages_fts WHERE content MATCH ? LIMIT ?",
                    (fts_query, FTS_TOP_K),
                )
                rows = source_cursor.fetchall()
            if rows:
                rowid_to_rank = {int(r[0]): float(r[1] or 0.0) for r in rows}
                placeholders = ",".join("?" for _ in rowid_to_rank)
                source_cursor.execute(
                    f"SELECT session_key, seq, role, rowid FROM messages WHERE rowid IN ({placeholders})",
                    list(rowid_to_rank.keys()),
                )
                fts_candidates: list[tuple[str, float]] = []
                for session_key, seq, role, rowid in source_cursor.fetchall():
                    _, _, key = turn_key(str(session_key), int(seq), str(role or ""))
                    if key not in nodes:
                        continue
                    fts_candidates.append((key, rowid_to_rank.get(int(rowid), 0.0)))
                # 按 BM25 |rank| 降序（更匹配的排前）
                fts_candidates.sort(key=lambda x: x[1])  # rank 是负值，小的更匹配

                fts_only_count = 0
                for key, _ in fts_candidates:
                    if key in seed_sources:
                        # Dense ∩ FTS: 加 boost（multiplicative fusion）
                        if "FTS" not in seed_sources[key].split("+"):
                            seed_sources[key] += "+FTS"
                        seed_energy[key] = min(1.5, seed_energy[key] * FTS_OVERLAP_BOOST)
                    else:
                        # FTS-only: 限制数量，只让最匹配的几个进
                        if fts_only_count >= FTS_ONLY_MAX_HITS:
                            continue
                        seed_sources[key] = "FTS"
                        seed_energy[key] = 1.0
                        fts_only_count += 1

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
    now_ts: float,
) -> np.ndarray:
    """计算节点状态权重（salience + 长期强度 + 短期资源 + fan 惩罚）。"""
    values = np.zeros(len(keys), dtype=np.float32)
    for index, key in enumerate(keys):
        node = nodes[key]
        long_score = min(1.0, decayed_strength(node, now_ts) / STRENGTH_CAP)
        resource = recover_resource(node, now_ts)
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
    edges_meta: dict[tuple[str, str], float] | None = None,
    now_ts: float = 0.0,
) -> np.ndarray:
    """构建微型图内部的共激活边矩阵。

    若提供 edges_meta + now_ts，对每条边按 last_used_ts 做 lazy 时间衰减。
    """
    matrix = np.zeros((len(keys), len(keys)), dtype=np.float32)
    apply_decay = edges_meta is not None and now_ts > 0

    def _eff(src_key: str, dst_key: str, weight: float) -> float:
        if not apply_decay:
            return weight
        last_used_ts = edges_meta.get((src_key, dst_key), 0.0)  # type: ignore[union-attr]
        return effective_edge_weight(weight, last_used_ts, now_ts)

    if edges_by_src is not None:
        for src_key in keys:
            src_index = index_by_key[src_key]
            for dst_key, weight in edges_by_src.get(src_key, {}).items():
                dst_index = index_by_key.get(dst_key)
                if dst_index is None:
                    continue
                eff_w = _eff(src_key, dst_key, weight)
                matrix[dst_index, src_index] = max(matrix[dst_index, src_index], eff_w)
        return matrix
    for (src_key, dst_key), weight in edges.items():
        src_index = index_by_key.get(src_key)
        dst_index = index_by_key.get(dst_key)
        if src_index is None or dst_index is None:
            continue
        eff_w = _eff(src_key, dst_key, weight)
        matrix[dst_index, src_index] = max(matrix[dst_index, src_index], eff_w)
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
    alpha = RWR_RESTART_ALPHA
    seed_indices = np.where(e0 > 0)[0]
    for index, key in enumerate(keys):
        c0 = float(alpha * e0[index])
        c1_vec = alpha * (1.0 - alpha) * transition[index, :] * e0
        s1 = int(np.argmax(c1_vec))
        c1 = float(c1_vec[s1])
        c2_vec = (1.0 - alpha) ** 2 * transition[index, :] * te0
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
    now_ts: float,
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
        long_score = min(1.0, decayed_strength(node, now_ts) / STRENGTH_CAP)
        resource = recover_resource(node, now_ts)
        fan_value = fan.get(key, 0)
        direct_value = max(0.0, direct_scores.get(key, 0.0))
        state_value = min(1.0, float(state_arr[index]) / max_state)
        edge_value = float(np.max(cross_mat[index])) if len(keys) else 0.0
        ptype, seed_key, bridge_key, path_value = path_info_dict.get(key, ("direct", "", "", 0.0))
        hop_penalty = {"direct": 1.0, "1hop": 0.86, "2hop": 0.62}.get(ptype, 0.62)
        source = seed_sources.get(key, "")
        direct_weight = 0.50 if source else 0.18
        fan_penalty = math.pow(1.0 + fan_value, FAN_PENALTY_POWER)
        user_penalty = 1.0 if has_user_turn(source_cursor, key) else ASSISTANT_ONLY_PENALTY
        # ── 乘算 gain modulation (Salinas & Sejnowski 2001) ───────
        # 内容基底：spreading 能量 + 直接相似度两路证据按 noisy-OR 融合。
        # P(relevant) = a + b − a·b = strong + weak·(1 − strong)，
        # 取代原 max + 0.3·min：0.3 本就是 (1 − strong) 的硬编码近似。
        # strong ≥ 1（证据已饱和）时 (1−strong) 截到 0，次要证据不再加成。
        ripple_term = float(current[index]) * 3.0 * state_value
        direct_term = direct_weight * direct_value
        strong = max(ripple_term, direct_term)
        weak = min(ripple_term, direct_term)
        content_base = strong + weak * max(0.0, 1.0 - strong)

        # 每个状态信号都是乘法 gain factor，signal=0 时 gain=1（不影响）
        # signal 高时 gain > 1（multiplicative 放大）
        gain_salience = 1.0 + 0.8 * node.salience           # σ ∈ [0,1] → gain ∈ [1, 1.8]
        gain_long     = 1.0 + 0.6 * long_score              # long ∈ [0,1] → gain ∈ [1, 1.6]
        gain_edge     = 1.0 + 0.5 * min(1.0, edge_value)    # edge → gain ∈ [1, 1.5]

        score = (
            content_base
            * gain_salience
            * gain_long
            * gain_edge
        ) * resource * hop_penalty * user_penalty / fan_penalty
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
        if not candidate.source:
            continue
        soft_hit = (
            soft_recall
            and candidate.score >= config.soft_recall_threshold
            and candidate.direct >= config.soft_recall_direct_floor
            and candidate.source == "Bridge"
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
    now_ts: float,
    source_cursor: sqlite3.Cursor | None,
    edges_by_src: dict[str, dict[str, float]] | None,
    edges_meta: dict[tuple[str, str], float] | None,
    graph_seed_keys: list[str],
) -> list[AkashaCandidate]:
    """沿 Dense 种子的强共激活边补一跳候选。"""
    if edges_by_src is None or not graph_seed_keys:
        return []

    def _eff(src_key: str, dst_key: str, weight: float) -> float:
        if edges_meta is None or now_ts <= 0:
            return weight
        return effective_edge_weight(
            weight,
            edges_meta.get((src_key, dst_key), 0.0),
            now_ts,
        )

    seed_set = {key for key in graph_seed_keys if key in nodes}
    in_strength: dict[str, float] = {}
    for src_key, src_neighbors in edges_by_src.items():
        for dst_key, edge_weight in src_neighbors.items():
            in_strength[dst_key] = in_strength.get(dst_key, 0.0) + _eff(src_key, dst_key, edge_weight)

    aggregate: dict[str, _GraphPathAggregate] = {}
    for seed_key in graph_seed_keys:
        if seed_key not in nodes:
            continue
        raw_neighbors = edges_by_src.get(seed_key, {})
        out_strength = sum(
            _eff(seed_key, dst_key, edge_weight)
            for dst_key, edge_weight in raw_neighbors.items()
        )
        if out_strength <= 0:
            continue

        scored_neighbors: list[tuple[float, float, float, str, float]] = []
        for key, edge_weight in raw_neighbors.items():
            if key not in nodes or key in seed_set or not has_user_turn(source_cursor, key):
                continue
            effective_weight = _eff(seed_key, key, edge_weight)
            dst_strength = in_strength.get(key, effective_weight)
            edge_signal = effective_weight / math.sqrt(max(out_strength * dst_strength, 1e-9))
            direct = max(0.0, direct_scores.get(key, 0.0))
            seed_direct = max(GRAPH_DIRECT_BIAS, max(0.0, direct_scores.get(seed_key, 0.0)))
            candidate_signal = edge_signal * seed_direct
            scored_neighbors.append((candidate_signal, edge_signal, direct, key, effective_weight))
        scored_neighbors.sort(reverse=True, key=lambda item: item[0])
        for candidate_signal, edge_signal, direct, key, edge_weight in scored_neighbors[:GRAPH_EXPAND_LIMIT]:
            item = aggregate.setdefault(key, _GraphPathAggregate(direct=direct, seed_key=seed_key))
            item.signal += candidate_signal
            item.paths += 1.0
            item.direct = max(item.direct, direct)
            if candidate_signal > item.best_signal:
                item.best_signal = candidate_signal
                item.best_edge = edge_signal
                item.best_weight = edge_weight
                item.seed_key = seed_key

    candidates: list[AkashaCandidate] = []
    for key, item in aggregate.items():
        node = nodes[key]
        resource = recover_resource(node, now_ts)
        long_score = min(1.0, node.strength / STRENGTH_CAP)
        direct = item.direct
        paths = max(1.0, item.paths)
        signal = item.signal * (1.0 + math.log(paths))
        score = 6.0 * signal * (GRAPH_DIRECT_BIAS + direct) * (1.0 + 0.15 * long_score)
        candidates.append(AkashaCandidate(
            key=key, source="Graph", ripple=item.best_weight,
            direct=direct, state=0.0, edge=signal,
            long=long_score, resource=resource, fan=max(0, fan.get(key, 0)),
            score=float(score * resource), path_type="1hop",
            seed_key=item.seed_key, path_value=item.best_edge,
        ))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:GRAPH_EXPAND_LIMIT]


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
    now_ts: float,
    *,
    config: CoreConfig,
    fan: dict[str, int],
    source_cursor: sqlite3.Cursor | None = None,
    edges_by_src: dict[str, dict[str, float]] | None = None,
    edges_meta: dict[tuple[str, str], float] | None = None,
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

    state_arr = state_array(valid_keys, nodes, fan, now_ts)
    cross_mat = cross_matrix(
        valid_keys, edges, index_by_key, edges_by_src,
        edges_meta=edges_meta, now_ts=now_ts,
    )

    transition = sim_matrix * state_arr[:, np.newaxis]
    transition *= 1.0 + config.cross_boost * cross_mat
    transition = keep_top_edges_per_column(transition, top_k=12)
    transition = normalize_columns(transition)

    e0 = initial_energy(valid_keys, seed_energy, fan, index_by_key)
    te0 = np.dot(transition, e0)
    current = e0.copy()
    for _ in range(2):
        current = (1.0 - RWR_RESTART_ALPHA) * np.dot(transition, current) + RWR_RESTART_ALPHA * e0

    path_info_dict = path_info(valid_keys, transition, e0, te0)
    candidates, suppressed = score_candidates(
        valid_keys, nodes, direct_scores_map, seed_sources,
        current, state_arr, cross_mat, fan, now_ts,
        path_info_dict, config, source_cursor,
        soft_recall=soft_recall, return_limit=return_limit,
    )
    if graph_seed_keys:
        graph_candidates = graph_expand_candidates(
            query_vec, nodes, direct_scores_map, fan, now_ts,
            source_cursor, edges_by_src, edges_meta, graph_seed_keys,
        )
        limit = return_limit or config.activate_limit
        candidates = merge_active_candidates(candidates, graph_candidates, limit)
        active_keys = {item.key for item in candidates}
        suppressed = [item for item in suppressed if item.key not in active_keys]
    return candidates, suppressed, ActivationTrace(
        seed_count=len(seed_sources), pool_count=len(valid_keys),
    )


def compute_candidates_from_snapshot(
    query: str,
    query_vec: np.ndarray,
    snapshot: AkashaActivationSnapshot,
    now_ts: float,
    *,
    config: CoreConfig,
    source_cursor: sqlite3.Cursor | None = None,
    soft_recall: bool = False,
    return_limit: int | None = None,
    graph_seed_keys: list[str] | None = None,
) -> tuple[list[AkashaCandidate], list[AkashaCandidate], ActivationTrace]:
    return compute_candidates(
        query,
        query_vec,
        snapshot.nodes,
        snapshot.edges,
        now_ts,
        config=config,
        fan=snapshot.fan,
        source_cursor=source_cursor,
        edges_by_src=snapshot.edges_by_src,
        edges_meta=snapshot.edges_meta,
        soft_recall=soft_recall,
        return_limit=return_limit,
        graph_seed_keys=graph_seed_keys,
    )


# ── 状态更新 ──────────────────────────────────────────────────────────


def activation_updates(
    items: list[AkashaCandidate],
    nodes: dict[str, AkashaNode],
    now_ts: float,
) -> list[ActivationUpdate]:
    """生成被激活节点的状态更新。"""
    updates: list[ActivationUpdate] = []
    for item in items:
        node = nodes.get(item.key)
        if node is None:
            continue
        strength = decayed_strength(node, now_ts)
        strength = bounded_add(strength, STRENGTH_LR * item.score, STRENGTH_CAP)
        resource = recover_resource(node, now_ts)
        resource *= max(0.05, 1.0 - RESOURCE_USE_RATE * min(1.0, item.score))
        updates.append(ActivationUpdate(
            key=item.key, strength=strength, resource=resource,
            recall_count=node.recall_count + 1, ts=now_ts,
        ))
    return updates
