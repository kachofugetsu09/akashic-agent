"""纠正 20260918_02：Akasha 是否启用由插件清单决定，不再看已退役的 [memory] 配置段。

02 仍然要求配置里有 `[memory] engine`，但该开关早已并进插件启用状态与
plugin-data 配置输入，workspace 里也不再存在 config.toml，于是它同样什么都不做。
本迁移只保留真正可判定的条件：学习图存在且不是当前消费版本。

原始说明：纠正 20260918_01：Akasha 选择属于 workspace 配置，凭据必须放在 memory root。

首个迁移读的是宿主配置（没有 `[memory]` 段）并把凭据写进 plugin-data，因此
在正式安装里既没有触发重放，也无法被 runtime 内的插件寻址。本纠正迁移不改写
已合并的脚本，只按同一合同重新登记一次性重放。

原始说明：一次性登记 Akasha 学习图重放：迁移只取恢复点并写下凭据。

实际重放由 Akasha 插件用自己的唯一实现完成，所以迁移不复制学习算法，
也不会与在线路径产生第二份实现。
"""
from __future__ import annotations

import json
import sqlite3
import tomllib
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from yoyo import step

from agent.migrations.context import current_migration_context

__depends__ = {"20260918_02_correct_graph_replay_request"}
__transactional__ = False

_BUNDLE_ID = "akasha"
_REQUEST_NAME = ".akasha-replay-request.json"
_SCHEMA_VERSION = 1
_CURRENT_CONSUMPTION_VERSION = 2


def _memory_engine(workspace: Path, config_path: Path) -> str:
    """读取 memory 配置；Akasha 选择属于 workspace 配置，宿主配置只作回退。"""

    for candidate in (workspace / "config.toml", config_path):
        engine = _memory_engine_from(candidate)
        if engine:
            return engine
    return ""


def _memory_engine_from(config_path: Path) -> str:
    """从一个配置文件读出 memory engine；缺失或未启用返回空。"""

    if not config_path.is_file():
        return ""
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    raw_memory = payload.get("memory", {})
    if not isinstance(raw_memory, dict):
        raise ValueError("memory 配置必须是 table")
    memory = cast(dict[str, object], raw_memory)
    if memory.get("enabled") is not True:
        return ""
    engine = memory.get("engine")
    return engine if isinstance(engine, str) else ""


def _needs_replay(memory_path: Path) -> bool:
    """缺图表示没有可替换内容；当前消费版本表示已经重放过。"""

    if not memory_path.is_file():
        return False
    with closing(sqlite3.connect(f"file:{memory_path}?mode=ro", uri=True)) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='consumer_state_json'"
        ).fetchone()
    if row is None:
        return True
    try:
        state = json.loads(str(row[0]))
    except json.JSONDecodeError:
        return True
    if not isinstance(state, dict):
        return True
    return state.get("version") != _CURRENT_CONSUMPTION_VERSION


def _backup_memory(memory_path: Path, backup_dir: Path) -> str:
    """重放前建立并校验恢复点；失败不能继续。"""

    if not memory_path.is_file():
        return ""
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / "memory-before.db"
    with closing(sqlite3.connect(f"file:{memory_path}?mode=ro", uri=True)) as incoming:
        with closing(sqlite3.connect(target)) as outgoing:
            incoming.backup(outgoing)
            if outgoing.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Akasha 重放前的学习图备份完整性检查失败")
            turns = outgoing.execute("SELECT COUNT(*) FROM turn_nodes").fetchone()
    if turns is None:
        raise ValueError("Akasha 重放前的学习图备份缺少节点表")
    return str(target)


def request_akasha_replay(_connection: object) -> None:
    """登记一次性重放请求；插件成功重放后自行移除该凭据。"""

    # 1. 本 bundle 只随 akasha 插件发布；是否需要重放由学习图自身判定。
    context = current_migration_context()
    data_root = context.bundle_data_roots[_BUNDLE_ID]
    memory_path = context.workspace / "memory" / "akasha.db"
    # 凭据放在 memory root：插件在 runtime 内只能稳定寻址它声明的 workspace root。
    request_path = memory_path.parent / _REQUEST_NAME

    # 2. 已经处于当前消费版本的图不需要再次重放。
    if not _needs_replay(memory_path):
        request_path.unlink(missing_ok=True)
        return

    # 3. 先固定恢复点，再写入凭据；凭据内容是重放的唯一事实。
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    backup = _backup_memory(memory_path, data_root / "backups" / "replay" / stamp)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "requested_at": datetime.now(UTC).isoformat(),
        "workspace": str(context.workspace),
        "memory_path": str(memory_path),
        "backup_path": backup,
    }
    request_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = request_path.with_name(request_path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(request_path)


steps = [step(request_akasha_replay)]
