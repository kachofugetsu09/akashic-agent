"""一次性登记 Akasha 学习图全量重放：dense 打分改为固定单线程 BLAS（ADR 0138）。

已有的图是在多线程 BLAS 下逐步学出来的，2026-09-29 之前与之后的 dense 规约方式还不同，
任何一份代码都无法从日志重建出它。改为单线程后，必须从日志重放一次，在线轨迹才与重建一致。
判定只看图自身：schema v3 且元数据记录了 `dense_threads = 1` 的图已经重放过。
实际重放仍由 Akasha 插件用自己的唯一实现完成，迁移只取恢复点并写下凭据。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from yoyo import step

from agent.migrations.context import current_migration_context

__depends__ = {"20260918_04_retire_removed_config_keys"}
__transactional__ = False

_BUNDLE_ID = "akasha"
_REQUEST_NAME = ".akasha-replay-request.json"
_SCHEMA_VERSION = 1
_GRAPH_SCHEMA_VERSION = 3


# 缺图表示没有可替换内容；记录过单线程 dense 的 v3 图已经重放过。
def _needs_replay(memory_path: Path) -> bool:
    if not memory_path.is_file():
        return False
    with closing(sqlite3.connect(f"file:{memory_path}?mode=ro", uri=True)) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        row = connection.execute("SELECT value FROM metadata WHERE key='dense_threads'").fetchone()
    return version != _GRAPH_SCHEMA_VERSION or row is None or row[0] != "1"


# 默认图与按 scope 物化的独立图一起判定；插件的重放会逐图重建。
def _graphs(memory_root: Path) -> list[Path]:
    return [memory_root / "akasha.db", *sorted((memory_root / "akasha-graphs").glob("*/akasha.db"))]


# 重放前建立并校验恢复点；失败不能继续。
def _backup_graph(memory_path: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(f"file:{memory_path}?mode=ro", uri=True)) as incoming:
        with closing(sqlite3.connect(target)) as outgoing:
            incoming.backup(outgoing)
            if outgoing.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError(f"Akasha 重放前的学习图备份完整性检查失败: {memory_path}")
    return str(target)


def request_akasha_replay(_connection: object) -> None:
    """登记一次性重放请求；插件成功重放后自行移除该凭据。"""

    # 1. 凭据放在 memory root：插件在 runtime 内只能稳定寻址它声明的 workspace root。
    context = current_migration_context()
    data_root = context.bundle_data_roots[_BUNDLE_ID]
    memory_root = context.workspace / "memory"
    request_path = memory_root / _REQUEST_NAME

    # 2. 没有图需要重放时不登记。
    pending = [path for path in _graphs(memory_root) if _needs_replay(path)]
    if not pending:
        return

    # 3. 先为每张待重放的图固定恢复点，再写入凭据。
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    backup_root = data_root / "backups" / "replay" / stamp
    backups = [
        _backup_graph(path, backup_root / path.relative_to(memory_root).parent / "memory-before.db")
        for path in pending
    ]
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "requested_at": datetime.now(UTC).isoformat(),
        "workspace": str(context.workspace),
        "memory_path": str(memory_root / "akasha.db"),
        "backup_path": backups[0],
        "backup_paths": backups,
        "reason": "dense_single_thread",
    }
    request_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = request_path.with_name(request_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(request_path)


steps = [step(request_akasha_replay)]
