# pyright: reportPrivateUsage=false

from __future__ import annotations

import argparse
import asyncio
import shutil
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.config_models import Config
from core.net.http import SharedHttpResources
from memory2.embedder import Embedder
from plugins.akasha.config import (
    AkashaConfig,
    load_akasha_config,
    resolve_akasha_db_path,
)
from plugins.akasha.engine import (
    AkashaCandidate,
    _activation_updates,
    _compute_candidates,
    _fan_counts,
)
from plugins.akasha.store import (
    ActivationEventRow,
    AkashaStore,
    EdgeUpdate,
    SourceMessage,
    SourceSessionSnapshot,
    turn_key,
)


@dataclass(frozen=True)
class MigrationStats:
    messages: int = 0
    activations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    snapshots: int = 0
    run_id: str = ""
    backup_path: Path | None = None


# 解析迁移脚本参数。
def _parse_args() -> argparse.Namespace:
    # 1. 只保留迁移必须参数，避免脚本变成另一套配置系统。
    parser = argparse.ArgumentParser(
        description="从 workspace/sessions.db 重建 Akasha sidecar 数据库。"
    )
    _ = parser.add_argument("--config", default="config.toml", help="主配置文件路径")
    _ = parser.add_argument(
        "--workspace",
        default=str(Path.home() / ".akashic" / "workspace"),
        help="Akashic workspace 路径",
    )
    _ = parser.add_argument("--sessions-db", default="", help="原始 sessions.db 路径")
    _ = parser.add_argument("--db-path", default="", help="输出 akasha.db 路径")
    _ = parser.add_argument("--batch-size", type=int, default=10, help="embedding 批大小")
    _ = parser.add_argument("--progress-every", type=int, default=500, help="进度打印间隔")
    return parser.parse_args()


# 构造 Akasha 配置，并允许命令行覆盖 db_path。
def _load_script_config(
    *,
    db_path: str,
) -> AkashaConfig:
    # 1. 插件配置仍从 plugins/akasha/config.local.toml 读取。
    config = load_akasha_config()
    if db_path.strip():
        return replace(config, db_path=db_path)
    return config


# 构造 embedding 客户端。
def _build_embedder(
    *,
    config: Config,
    http_resources: SharedHttpResources,
) -> Embedder:
    # 1. 复用运行时的 embedding 配置和 HTTP requester。
    embedding = config.memory.embedding
    return Embedder(
        base_url=embedding.base_url or config.light_base_url or config.base_url or "",
        api_key=embedding.api_key or config.light_api_key or config.api_key,
        model=embedding.model,
        requester=http_resources.external_default,
    )


# 读取 sessions.db 中的原始消息。
def _iter_source_batches(
    *,
    sessions_db: Path,
    batch_size: int,
) -> Iterator[list[SourceMessage]]:
    # 1. 按 cross CLI 的 session_key, seq 顺序 replay。
    with closing(sqlite3.connect(str(sessions_db))) as db:
        cursor = db.execute(
            """
            SELECT id, session_key, seq, role, content, ts
            FROM messages
            WHERE role IN ('user', 'assistant')
            ORDER BY session_key, seq
            """
        )
        while rows := cursor.fetchmany(max(1, batch_size)):
            yield [
                SourceMessage(
                    id=str(row[0]),
                    session_key=str(row[1]),
                    seq=int(row[2]),
                    role=str(row[3] or ""),
                    content=str(row[4] or ""),
                    ts=str(row[5] or ""),
                )
                for row in rows
            ]


# 读取 sessions.db 中全部原始消息。
def _load_source_messages(sessions_db: Path) -> list[SourceMessage]:
    # 1. salience 需要全量 embedding 分布，不能边读边 replay。
    messages: list[SourceMessage] = []
    for batch in _iter_source_batches(sessions_db=sessions_db, batch_size=1000):
        messages.extend(batch)
    return messages


# 读取迁移开始时的 session 游标快照。
def _load_session_snapshots(sessions_db: Path) -> list[SourceSessionSnapshot]:
    # 1. 只读取旧系统游标，用于回滚和迁移诊断。
    with closing(sqlite3.connect(str(sessions_db))) as db:
        rows = db.execute(
            """
            SELECT
                s.key,
                COALESCE(s.last_consolidated, 0),
                COALESCE(s.next_seq, 0),
                COALESCE(MAX(m.seq), -1)
            FROM sessions s
            LEFT JOIN messages m ON m.session_key = s.key
            GROUP BY s.key
            ORDER BY s.key
            """
        ).fetchall()
    return [
        SourceSessionSnapshot(
            session_key=str(row[0]),
            last_consolidated=int(row[1] or 0),
            next_seq=int(row[2] or 0),
            max_seq=int(row[3] or -1),
        )
        for row in rows
    ]


# 备份已有 Akasha sidecar。
def _backup_existing_db(db_path: Path) -> Path | None:
    # 1. 重建前保留旧库，避免迁移脚本误覆盖唯一状态。
    if not db_path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.bak-{stamp}")
    _ = shutil.copy2(db_path, backup_path)
    return backup_path


# 从 cache 读取 embedding，缺失时批量调用 embedding API。
async def _embed_batch_with_cache(
    *,
    store: AkashaStore,
    embedder: Embedder,
    model: str,
    batch: list[SourceMessage],
) -> tuple[list[list[float]], int, int]:
    # 1. 先按 message_id + content_hash + model 查 cache。
    embeddings: list[list[float] | None] = []
    missing: list[tuple[int, SourceMessage]] = []
    cache_hits = 0
    for index, message in enumerate(batch):
        cached = store.get_cached_embedding(message=message, model=model)
        embeddings.append(cached)
        if cached is None:
            missing.append((index, message))
        else:
            cache_hits += 1

    # 2. 只对缺口调用远端 embedding，并立刻写回 cache。
    cache_misses = len(missing)
    if missing:
        fresh = await embedder.embed_batch([
            message.content if message.content.strip() else " "
            for _, message in missing
        ])
        for (index, message), embedding in zip(missing, fresh, strict=False):
            store.upsert_cached_embedding(
                message=message,
                model=model,
                embedding=embedding,
            )
            embeddings[index] = embedding

    # 3. 返回顺序必须和 batch 一致，后面的 replay 依赖消息顺序。
    ordered: list[list[float]] = []
    for embedding in embeddings:
        if embedding is None:
            raise RuntimeError("embedding cache 写入后仍有缺口")
        ordered.append(embedding)
    return ordered, cache_hits, cache_misses


# 确保全量 message embedding 都在 cache 里。
async def _ensure_embeddings_with_cache(
    *,
    store: AkashaStore,
    embedder: Embedder,
    model: str,
    messages: list[SourceMessage],
    batch_size: int,
) -> tuple[dict[str, list[float]], int, int]:
    # 1. 按批次补齐缺失 embedding，但 replay 之前先拿到完整向量表。
    embedding_map: dict[str, list[float]] = {}
    cache_hits = 0
    cache_misses = 0
    for index in range(0, len(messages), max(1, batch_size)):
        batch = messages[index : index + max(1, batch_size)]
        embeddings, batch_hits, batch_misses = await _embed_batch_with_cache(
            store=store,
            embedder=embedder,
            model=model,
            batch=batch,
        )
        cache_hits += batch_hits
        cache_misses += batch_misses
        for message, embedding in zip(batch, embeddings, strict=False):
            embedding_map[message.id] = embedding
    return embedding_map, cache_hits, cache_misses


# 计算原始 cross activation 使用的 message salience。
def _compute_salience_map(
    messages: list[SourceMessage],
    embedding_map: dict[str, list[float]],
) -> dict[str, float]:
    # 1. 用 temporal isolation、assistant arousal、session outlier 三个分量复刻原始建库逻辑。
    available = [
        (message, embedding_map[message.id])
        for message in messages
        if message.id in embedding_map
    ]
    if not available:
        return {}
    matrix = np.vstack([
        _normalize_vector(np.array(embedding, dtype=np.float32))
        for _, embedding in available
    ])
    index_by_id = {message.id: index for index, (message, _) in enumerate(available)}
    session_sorted: dict[str, list[int]] = {}
    seq_index: dict[tuple[str, int], int] = {}
    for index, (message, _) in enumerate(available):
        session_sorted.setdefault(message.session_key, []).append(index)
        seq_index[(message.session_key, message.seq)] = index

    # 2. 先计算每个 session 的质心。
    centroids: dict[str, np.ndarray] = {}
    for session_key, indices in session_sorted.items():
        group = matrix[indices]
        centroid = group.mean(axis=0)
        centroids[session_key] = _normalize_vector(centroid)

    # 3. 三个原始分量分别做 p5-p95 归一化后加权。
    temporal_raw = np.zeros(len(available), dtype=np.float32)
    arousal_raw = np.zeros(len(available), dtype=np.float32)
    session_out_raw = np.zeros(len(available), dtype=np.float32)
    position_by_index = {
        source_index: position
        for indices in session_sorted.values()
        for position, source_index in enumerate(indices)
    }
    for index, (message, _) in enumerate(available):
        session_indices = session_sorted[message.session_key]
        position = position_by_index[index]
        neighbors = [
            float(np.dot(matrix[index], matrix[neighbor_index]))
            for neighbor_index in session_indices[max(0, position - 5) : position]
        ]
        temporal_raw[index] = 1.0 - max(neighbors) if neighbors else 1.0
        if message.role == "user":
            assistant_index = seq_index.get((message.session_key, message.seq + 1))
            if (
                assistant_index is None
                or available[assistant_index][0].role != "assistant"
            ):
                assistant_len = 0
            else:
                assistant_len = len(available[assistant_index][0].content)
            arousal_raw[index] = min(1.0, assistant_len / 300.0)
        else:
            arousal_raw[index] = min(1.0, len(message.content) / 300.0)
        centroid = centroids[message.session_key]
        session_out_raw[index] = max(0.0, (1.0 - float(np.dot(matrix[index], centroid))) / 2.0)

    salience = (
        0.4 * _percentile_normalize(temporal_raw)
        + 0.3 * _percentile_normalize(arousal_raw)
        + 0.3 * _percentile_normalize(session_out_raw)
    )
    return {
        message_id: float(value)
        for message_id, value in zip(index_by_id, np.clip(salience, 0.0, 1.0), strict=False)
    }


# p5-p95 拉伸到 0..1。
def _percentile_normalize(values: np.ndarray) -> np.ndarray:
    # 1. 常量分布保留中性值，避免除零。
    p5 = float(np.percentile(values, 5))
    p95 = float(np.percentile(values, 95))
    if p95 - p5 < 1e-8:
        return np.full_like(values, 0.5)
    return np.clip((values - p5) / (p95 - p5), 0.0, 1.0)


# 归一化向量。
def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    # 1. 原始 salience 和 dense 分数都基于单位向量。
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


# 在写入当前 user turn 前，按 cross CLI 规则激活历史节点。
def _activate_before_upsert(
    *,
    store: AkashaStore,
    message: SourceMessage,
    embedding: list[float],
    config: AkashaConfig,
    source_cursor: sqlite3.Cursor,
) -> int:
    # 1. 只有 user 输入会触发状态激活。
    if message.role != "user":
        return 0
    nodes = {node.key: node for node in store.list_nodes()}
    if not nodes:
        return 0

    # 2. 使用当前 user embedding 检索历史节点，并更新旧节点状态。
    edges = store.load_edges()
    candidates, _, _ = _compute_candidates(
        message.content,
        np.array(embedding, dtype=np.float32),
        nodes,
        edges,
        message.seq,
        config=config,
        fan=_fan_counts(edges),
        source_cursor=source_cursor,
        soft_recall=False,
        return_limit=config.activate_limit,
    )
    store.update_activation_batch(_activation_updates(candidates, nodes, message.seq))

    # 3. 当前 turn key 先参与建边，随后再被 upsert_message_node 写成节点。
    current_key = turn_key(message.session_key, message.seq, message.role)[2]
    store.upsert_edges(_edge_updates(current_key, candidates, message.seq))
    store.insert_activation_events(_activation_events(message, candidates))
    return len(candidates)


# 把一轮激活候选转成共激活边。
def _edge_updates(
    current_key: str,
    candidates: list[AkashaCandidate],
    seq: int,
) -> list[EdgeUpdate]:
    # 1. 当前输入和被激活旧节点互连。
    updates: list[EdgeUpdate] = []
    key_to_score = {item.key: item.score for item in candidates}
    for item in candidates:
        edge_strength = key_to_score.get(item.key, 1.0)
        updates.append(EdgeUpdate(current_key, item.key, edge_strength, seq))
        updates.append(EdgeUpdate(item.key, current_key, edge_strength, seq))

    # 2. 同轮共同激活的旧节点互连。
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            edge_strength = float(
                np.sqrt(
                    key_to_score.get(left.key, 1.0)
                    * key_to_score.get(right.key, 1.0)
                )
            )
            updates.append(EdgeUpdate(left.key, right.key, edge_strength, seq))
            updates.append(EdgeUpdate(right.key, left.key, edge_strength, seq))
    return updates


# 把一轮激活候选转成诊断事件。
def _activation_events(
    message: SourceMessage,
    candidates: list[AkashaCandidate],
) -> list[ActivationEventRow]:
    # 1. query_id 使用当前 user message id，便于回源诊断。
    return [
        ActivationEventRow(
            seq=message.seq,
            query_id=message.id,
            activated_key=item.key,
            source=item.source,
            score=item.score,
            direct_score=item.direct,
            state_score=item.state,
            edge_score=item.edge,
            long_score=item.long,
            resource=item.resource,
            fan=item.fan,
        )
        for item in candidates
    ]


# 执行 Akasha sidecar 重建。
async def _run() -> MigrationStats:
    # 1. 解析路径、配置和目标 sidecar。
    args = _parse_args()
    workspace = Path(str(args.workspace)).expanduser()
    sessions_db = Path(str(args.sessions_db)).expanduser() if args.sessions_db else workspace / "sessions.db"
    akasha_config = _load_script_config(db_path=str(args.db_path or ""))
    db_path = resolve_akasha_db_path(workspace=workspace, akasha_config=akasha_config)
    if not sessions_db.exists():
        raise FileNotFoundError(f"sessions.db 不存在: {sessions_db}")

    # 2. 备份旧 sidecar，并初始化本次迁移记录。
    backup_path = _backup_existing_db(db_path)
    store = AkashaStore(db_path)
    config = Config.load(str(args.config))
    embedding_model = config.memory.embedding.model
    run_id = store.start_migration_run(
        source_db_path=sessions_db,
        embedding_model=embedding_model,
    )
    snapshots = _load_session_snapshots(sessions_db)
    store.insert_session_snapshots(run_id=run_id, snapshots=snapshots)
    store.reset_schema()

    # 3. 先复用 embedding cache 和 salience，再按消息顺序 replay 激活状态。
    http_resources = SharedHttpResources()
    embedder = _build_embedder(config=config, http_resources=http_resources)
    messages = 0
    activations = 0
    cache_hits = 0
    cache_misses = 0
    status = "failed"
    try:
        source_messages = _load_source_messages(sessions_db)
        embedding_map, cache_hits, cache_misses = await _ensure_embeddings_with_cache(
            store=store,
            embedder=embedder,
            model=embedding_model,
            messages=source_messages,
            batch_size=int(args.batch_size),
        )
        salience_map = _compute_salience_map(source_messages, embedding_map)
        with closing(sqlite3.connect(str(sessions_db))) as source_db:
            source_cursor = source_db.cursor()
            for raw_message in source_messages:
                embedding = embedding_map.get(raw_message.id)
                if embedding is None:
                    continue
                message = replace(
                    raw_message,
                    salience=salience_map.get(raw_message.id, 0.0),
                )
                activations += _activate_before_upsert(
                    store=store,
                    message=message,
                    embedding=embedding,
                    config=akasha_config,
                    source_cursor=source_cursor,
                )
                _ = store.upsert_message_node(message, embedding)
                messages += 1
                if args.progress_every > 0 and messages % int(args.progress_every) == 0:
                    print(f"已处理 messages={messages} activations={activations}")
        status = "completed"
    finally:
        store.finish_migration_run(
            run_id=run_id,
            status=status,
            message_count=messages,
            activation_count=activations,
            cache_hit_count=cache_hits,
            cache_miss_count=cache_misses,
        )
        store.close()
        await embedder.aclose()
        await http_resources.aclose()

    return MigrationStats(
        messages=messages,
        activations=activations,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        snapshots=len(snapshots),
        run_id=run_id,
        backup_path=backup_path,
    )


# 脚本入口。
def main() -> None:
    # 1. asyncio 只包住 embedding HTTP 调用。
    stats = asyncio.run(_run())
    print(
        "Akasha 迁移完成: "
        f"run_id={stats.run_id} "
        f"messages={stats.messages} "
        f"activations={stats.activations} "
        f"cache_hits={stats.cache_hits} "
        f"cache_misses={stats.cache_misses} "
        f"snapshots={stats.snapshots}"
    )
    if stats.backup_path is not None:
        print(f"旧库备份: {stats.backup_path}")


if __name__ == "__main__":
    main()
