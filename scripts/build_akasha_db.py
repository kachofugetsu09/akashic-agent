# pyright: reportPrivateUsage=false

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator, cast

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.config_models import Config
from plugins.akasha.config import (
    AkashaConfig,
    load_akasha_config,
    resolve_akasha_db_path,
)
from plugins.akasha.core import SourceMessage
from plugins.akasha.replay import AkashaReplayRuntime, ReplayMessage
from plugins.akasha.store import (
    AkashaStore,
    SourceSessionSnapshot,
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


# 读取 sessions.db 中的原始消息。
def _iter_source_batches(
    *,
    sessions_db: Path,
    batch_size: int,
) -> Iterator[list[SourceMessage]]:
    # 1. 按真实时间顺序 replay，避免后来的 turn 提前进入历史图。
    with closing(sqlite3.connect(str(sessions_db))) as db:
        cursor = db.execute(
            """
            SELECT id, session_key, seq, role, content, ts
            FROM messages
            WHERE role IN ('user', 'assistant')
            ORDER BY COALESCE(datetime(ts), ts), session_key, seq
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


# 读取不应进入 Akasha 的消息。
def _load_skip_message_ids(sessions_db: Path) -> set[str]:
    result: set[str] = set()
    with closing(sqlite3.connect(str(sessions_db))) as db:
        rows = db.execute("SELECT id, extra FROM messages").fetchall()
    for message_id, raw_extra in rows:
        try:
            parsed: object = json.loads(str(raw_extra or "{}"))
        except json.JSONDecodeError:
            parsed = {}
        extra = cast(dict[str, object], parsed) if isinstance(parsed, dict) else {}
        if bool(extra.get("proactive")) or bool(extra.get("skip_post_memory")):
            result.add(str(message_id))
    return result


def _skip_message(message: SourceMessage, skip_message_ids: set[str]) -> bool:
    return message.id in skip_message_ids or message.content.startswith("[后台任务完成]")


# 备份已有 Akasha sidecar。
def _backup_existing_db(db_path: Path) -> Path | None:
    # 1. 重建前保留旧库，避免迁移脚本误覆盖唯一状态。
    if not db_path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.bak-{stamp}")
    _ = shutil.copy2(db_path, backup_path)
    return backup_path


# 从 cache 读取回放需要的 embedding，缺失时跳过对应消息。
def _load_embeddings_from_cache(
    *,
    store: AkashaStore,
    model: str,
    messages: list[SourceMessage],
) -> tuple[dict[str, list[float]], int, int]:
    embedding_map: dict[str, list[float]] = {}
    cache_hits = 0
    cache_misses = 0
    for message in messages:
        embedding = store.get_cached_embedding(message=message, model=model)
        if embedding is None:
            cache_misses += 1
        else:
            cache_hits += 1
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
    for indices in session_sorted.values():
        indices.sort(key=lambda item: available[item][0].seq)

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


# 按 user turn 聚合回放输入，assistant 归入前一个 user turn。
def _iter_replay_turns(
    messages: list[SourceMessage],
    skip_message_ids: set[str],
) -> Iterator[list[SourceMessage]]:
    by_turn = {
        (message.session_key, message.seq, message.role): message
        for message in messages
        if not _skip_message(message, skip_message_ids)
    }
    used: set[str] = set()
    for message in messages:
        if message.id in used or _skip_message(message, skip_message_ids):
            continue
        if message.role != "user":
            continue
        turn = [message]
        used.add(message.id)
        assistant = by_turn.get((message.session_key, message.seq + 1, "assistant"))
        if assistant is not None and assistant.id not in used:
            turn.append(assistant)
            used.add(assistant.id)
        yield turn


# 执行 Akasha sidecar 重建。
def _run() -> MigrationStats:
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

    # 3. 只复用 embedding cache，再按消息顺序 replay 激活状态。
    messages = 0
    activations = 0
    cache_hits = 0
    cache_misses = 0
    status = "failed"
    try:
        source_messages = _load_source_messages(sessions_db)
        skip_message_ids = _load_skip_message_ids(sessions_db)
        replay_turns = list(_iter_replay_turns(source_messages, skip_message_ids))
        replay_messages = [message for turn in replay_turns for message in turn]
        embedding_map, cache_hits, cache_misses = _load_embeddings_from_cache(
            store=store,
            model=embedding_model,
            messages=replay_messages,
        )
        salience_map = _compute_salience_map(replay_messages, embedding_map)
        with closing(sqlite3.connect(str(sessions_db))) as source_db:
            runtime = AkashaReplayRuntime(
                store=store,
                config=akasha_config,
                source_cursor=source_db.cursor(),
            )
            for raw_turn in replay_turns:
                replay_items: list[ReplayMessage] = []
                for raw_message in raw_turn:
                    embedding = embedding_map.get(raw_message.id)
                    if embedding is None:
                        continue
                    replay_items.append(ReplayMessage(
                        message=replace(
                            raw_message,
                            salience=salience_map.get(raw_message.id, 0.0),
                        ),
                        embedding=embedding,
                    ))
                if not any(item.message.role == "user" for item in replay_items):
                    continue
                result = runtime.replay_turn(replay_items)
                activations += len(result.activation_items)
                messages += len(replay_items)
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
    stats = _run()
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
