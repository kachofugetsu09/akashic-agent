# pyright: reportPrivateUsage=false

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sqlite3
import sys
from collections.abc import Callable
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
from plugins.akasha.core import (
    SourceMessage,
    parse_ts_unix,
    reinforce_boost_from_payload,
    turn_key,
)
from plugins.akasha.fast import fast_dense, graph_fast
from plugins.akasha.fast.mem_store import CapturingMemoryStore
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
    _ = parser.add_argument(
        "--embedding-model",
        default="",
        help="直接指定 embedding 模型名（缓存命中键用）；给定后不读 --config，免依赖 agent 配置",
    )
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
    # 1. 先批量读取，最终统一用核心时间解析器排序。
    with closing(sqlite3.connect(str(sessions_db))) as db:
        cursor = db.execute(
            """
            SELECT id, session_key, seq, role, content, ts
            FROM messages
            WHERE role IN ('user', 'assistant')
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
    messages: list[SourceMessage] = []
    for batch in _iter_source_batches(sessions_db=sessions_db, batch_size=1000):
        messages.extend(batch)
    messages.sort(key=lambda item: (parse_ts_unix(item.ts), item.session_key, item.seq))
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
        rows = db.execute("SELECT id, session_key, extra FROM messages").fetchall()
    for message_id, session_key, raw_extra in rows:
        try:
            parsed: object = json.loads(str(raw_extra or "{}"))
        except json.JSONDecodeError:
            parsed = {}
        extra = cast(dict[str, object], parsed) if isinstance(parsed, dict) else {}
        if _skip_session_key(str(session_key)) or bool(extra.get("proactive")) or bool(extra.get("skip_post_memory")):
            result.add(str(message_id))
    return result


def _load_reinforce_boosts(sessions_db: Path) -> dict[str, float]:
    """读取 reinforce 标记，映射成 {turn_key: boost}。reinforce 是"输入信号"，
    留痕在 sessions.db(源头)→ 重建时确定性复现，图不丢。两个来源(等价):

      1. tool_chain 里有 reinforce_memory 工具调用(线上真实调用,自动记录)；
      2. extra["akasha_reinforce"](历史纠错回填迁移用,可带 {"boost": N})。
    """
    boosts: dict[str, float] = {}
    with closing(sqlite3.connect(str(sessions_db))) as db:
        rows = db.execute("SELECT session_key, seq, role, extra, tool_chain FROM messages").fetchall()
    for session_key, seq, role, raw_extra, raw_chain in rows:
        boost = reinforce_boost_from_payload(raw_extra, raw_chain)
        if boost <= 1.0:
            continue
        key = turn_key(str(session_key), int(seq), str(role or ""))[2]
        boosts[key] = max(boosts.get(key, 1.0), boost)  # 同轮多标记取最大
    return boosts


def _skip_session_key(session_key: str) -> bool:
    return session_key.startswith("scheduler:")


def _skip_message(message: SourceMessage, skip_message_ids: set[str]) -> bool:
    return (
        message.id in skip_message_ids
        or _skip_session_key(message.session_key)
        or message.content.startswith("[后台任务完成]")
    )


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
    if str(args.embedding_model).strip():
        embedding_model = str(args.embedding_model).strip()  # 离线重建：免读 config
    else:
        config = Config.load(str(args.config))
        embedding_model = config.memory.embedding.model
    run_id = store.start_migration_run(
        source_db_path=sessions_db,
        embedding_model=embedding_model,
    )
    snapshots = _load_session_snapshots(sessions_db)
    store.insert_session_snapshots(run_id=run_id, snapshots=snapshots)
    store.reset_schema()

    # 2b. 用内存图重放，末尾一次性落库；AkashaStore 只负责 cache、迁移记录和 dump 连接。
    mem = CapturingMemoryStore()
    graph_install = cast("Callable[[CapturingMemoryStore], None]", getattr(graph_fast, "install"))
    dense_install = cast("Callable[[], None]", getattr(fast_dense, "install"))
    dump_to_db = cast(
        "Callable[[AkashaStore, CapturingMemoryStore], dict[str, int]]",
        getattr(importlib.import_module("plugins.akasha.fast.dump"), "dump_to_db"),
    )
    graph_install(mem)
    dense_install()

    # 3. 只复用 embedding cache，再按消息顺序 replay 激活状态。
    messages = 0
    activations = 0
    cache_hits = 0
    cache_misses = 0
    next_progress = int(args.progress_every)
    status = "failed"
    try:
        source_messages = _load_source_messages(sessions_db)
        skip_message_ids = _load_skip_message_ids(sessions_db)
        reinforce_boosts = _load_reinforce_boosts(sessions_db)
        skipped_source_messages = [
            message for message in source_messages if _skip_message(message, skip_message_ids)
        ]
        if skipped_source_messages:
            _ = store.delete_cached_embeddings([message.id for message in skipped_source_messages])
        replay_turns = list(_iter_replay_turns(source_messages, skip_message_ids))
        replay_messages = [message for turn in replay_turns for message in turn]
        embedding_map, cache_hits, cache_misses = _load_embeddings_from_cache(
            store=store,
            model=embedding_model,
            messages=replay_messages,
        )
        message_embeddings = {
            message_id: np.array(embedding, dtype=np.float32)
            for message_id, embedding in embedding_map.items()
        }
        message_turn_keys = {
            message.id: turn_key(message.session_key, message.seq, message.role)[2]
            for message in replay_messages
            if message.id in embedding_map
        }
        with closing(sqlite3.connect(str(sessions_db))) as source_db:
            runtime = AkashaReplayRuntime(
                store=mem,
                config=akasha_config,
                source_db_path=sessions_db,
                source_cursor=source_db.cursor(),
                message_embeddings=message_embeddings,
                message_turn_keys=message_turn_keys,
                reinforce_boosts=reinforce_boosts,
            )
            for raw_turn in replay_turns:
                replay_items: list[ReplayMessage] = []
                for raw_message in raw_turn:
                    embedding = embedding_map.get(raw_message.id)
                    if embedding is None:
                        continue
                    replay_items.append(ReplayMessage(
                        message=raw_message,
                        embedding=embedding,
                    ))
                if not any(item.message.role == "user" for item in replay_items):
                    continue
                result = runtime.replay_turn(replay_items)
                activations += len(result.activation_items)
                messages += len(replay_items)
                if next_progress > 0 and messages >= next_progress:
                    print(f"已处理 messages={messages} activations={activations}", flush=True)
                    while messages >= next_progress:
                        next_progress += int(args.progress_every)
        # 3b. 内存重放完成后一次性批量落库。
        dumped = dump_to_db(store, mem)
        print(f"dump 完成 {dumped}", flush=True)
        status = "completed"
    finally:
        graph_fast.uninstall()
        fast_dense.uninstall()
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
