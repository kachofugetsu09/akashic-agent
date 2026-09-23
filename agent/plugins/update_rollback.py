"""更新只保存旧指针与启用状态；进程死亡后回退，不续跑候选。"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from agent.plugin_composition.archive import sync_directory
from agent.plugins.artifacts import ArtifactPointer, ArtifactPointers, pointer_state_path, write_pointers
from agent.plugins.manifest import load_plugin_manifest, write_plugin_manifest
from infra.persistence.json_store import load_json

_LEGACY_PLUGIN_UPDATES = """CREATE TABLE plugin_updates (
        update_id TEXT PRIMARY KEY,
        plugin_id TEXT NOT NULL,
        plugin_base TEXT NOT NULL,
        previous_pointers_json TEXT,
        candidate_pointer TEXT NOT NULL,
        previous_enabled INTEGER CHECK (previous_enabled IN (0, 1)),
        phase TEXT NOT NULL CHECK (phase IN ('armed', 'committed', 'rolled_back')),
        reload_tx_id TEXT UNIQUE REFERENCES reload_transactions(tx_id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error TEXT NOT NULL
    )"""

_ALTERED_PLUGIN_UPDATES = """CREATE TABLE plugin_updates (
        update_id TEXT PRIMARY KEY,
        plugin_id TEXT NOT NULL,
        plugin_base TEXT NOT NULL,
        previous_pointers_json TEXT,
        candidate_pointer TEXT NOT NULL,
        previous_enabled INTEGER CHECK (previous_enabled IN (0, 1)),
        phase TEXT NOT NULL CHECK (phase IN ('armed', 'committed', 'rolled_back')),
        reload_tx_id TEXT UNIQUE REFERENCES reload_transactions(tx_id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error TEXT NOT NULL ,
        input_ref TEXT)"""

SCHEMA = {
    "plugin_updates": """CREATE TABLE plugin_updates (
        update_id TEXT PRIMARY KEY,
        plugin_id TEXT NOT NULL,
        plugin_base TEXT NOT NULL,
        previous_pointers_json TEXT,
        candidate_pointer TEXT NOT NULL,
        previous_enabled INTEGER CHECK (previous_enabled IN (0, 1)),
        phase TEXT NOT NULL CHECK (phase IN ('armed', 'committed', 'rolled_back')),
        reload_tx_id TEXT UNIQUE REFERENCES reload_transactions(tx_id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error TEXT NOT NULL,
        input_ref TEXT
    )""",
    "plugin_update_active": """CREATE UNIQUE INDEX plugin_update_active
        ON plugin_updates(plugin_id) WHERE phase='armed'""",
}


@dataclass(frozen=True)
class UpdateRollback:
    update_id: str
    input_ref: str | None
    plugin_id: str
    plugin_base: Path
    previous: ArtifactPointers | None
    candidate: ArtifactPointer
    previous_enabled: bool | None
    phase: Literal['armed', 'committed', 'rolled_back']
    reload_tx_id: str | None
    error: str


def plugin_update_schema_state(conn: sqlite3.Connection) -> Literal["missing", "old", "new"]:
    """Classify only the known plugin update table and index shapes."""
    table = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='plugin_updates'"
    ).fetchone()
    index = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='plugin_update_active'"
    ).fetchone()
    if table is None and index is None:
        return "missing"
    if table is None or index is None:
        raise ValueError("plugin update schema 不完整")
    table_sql = " ".join(str(table[0]).split())
    index_sql = " ".join(str(index[0]).split())
    expected_index = " ".join(SCHEMA["plugin_update_active"].split())
    if index_sql != expected_index:
        raise ValueError("未知 plugin update schema: plugin_update_active")
    if table_sql in {
        " ".join(SCHEMA["plugin_updates"].split()),
        " ".join(_ALTERED_PLUGIN_UPDATES.split()),
    }:
        return "new"
    if table_sql == " ".join(_LEGACY_PLUGIN_UPDATES.split()):
        return "old"
    raise ValueError("未知 plugin update schema: plugin_updates")


def check_schema(conn: sqlite3.Connection) -> bool:
    """Return true only for the current shape; old shape needs Core migration."""
    return plugin_update_schema_state(conn) == "new"


def pointer_value(pointers: ArtifactPointers | None) -> dict[str, str | None] | None:
    return None if pointers is None else {"stable": pointers.stable.path, "latest": pointers.latest.path}


def read(conn: sqlite3.Connection, update_id: str) -> UpdateRollback:
    row = conn.execute(
        "SELECT input_ref,plugin_id,plugin_base,previous_pointers_json,candidate_pointer,previous_enabled,"
        "phase,reload_tx_id,error FROM plugin_updates WHERE update_id=?", (update_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"插件更新不存在: {update_id}")
    raw = None if row[3] is None else json.loads(row[3])
    previous = None
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError("旧插件指针记录损坏")
        raw = cast(dict[str, object], raw)
        if set(raw) != {'stable', 'latest'} or any(
            value is not None and not isinstance(value, str) for value in raw.values()
        ):
            raise ValueError("旧插件指针记录损坏")
        previous = ArtifactPointers(ArtifactPointer(cast(str | None, raw['stable'])), ArtifactPointer(cast(str | None, raw['latest'])))
    return UpdateRollback(
        update_id, row[0], row[1], Path(row[2]), previous, ArtifactPointer(row[4]),
        None if row[5] is None else bool(row[5]), row[6], row[7], row[8],
    )


def arm(
    conn: sqlite3.Connection, *, update_id: str, plugin_id: str, plugin_base: Path,
    previous: ArtifactPointers | None, candidate: ArtifactPointer,
    previous_enabled: bool | None, now: str,
) -> None:
    """可见指针变化前只保存一次恢复点，禁止复用旧请求继续安装。"""
    if not update_id or candidate.path is None:
        raise ValueError("插件更新需要 ID 与实际 candidate pointer")
    if not check_schema(conn):
        raise RuntimeError("插件更新需要先执行 update rollback 迁移")
    _ = conn.execute(
        "INSERT INTO plugin_updates (update_id,input_ref,plugin_id,plugin_base,"
        "previous_pointers_json,candidate_pointer,previous_enabled,phase,reload_tx_id,"
        "created_at,updated_at,error) VALUES(?,NULL,?,?,?,?,?,'armed',NULL,?,?, '')",
        (update_id, plugin_id, str(plugin_base.resolve()),
         None if previous is None else json.dumps(pointer_value(previous), sort_keys=True),
         candidate.path, previous_enabled, now, now),
    )


def set_input_ref(conn: sqlite3.Connection, *, update_id: str, input_ref: str) -> None:
    """Fill the immutable runtime archive reference once, or reject drift."""
    if not input_ref:
        raise ValueError("插件更新 input_ref 不能为空")
    if not check_schema(conn):
        raise RuntimeError("插件更新需要先执行 update rollback 迁移")
    row = conn.execute("SELECT input_ref FROM plugin_updates WHERE update_id=?", (update_id,)).fetchone()
    if row is None:
        raise KeyError(f"插件更新不存在: {update_id}")
    if row[0] not in (None, input_ref):
        raise RuntimeError("插件更新 input_ref 不能改变")
    _ = conn.execute(
        "UPDATE plugin_updates SET input_ref=? WHERE update_id=? AND input_ref IS NULL",
        (input_ref, update_id),
    )


def link(conn: sqlite3.Connection, *, tx_id: str, plugin_id: str, candidate_pointer: str | None) -> None:
    """同事务把实际 candidate 的资源记录接到唯一未完成更新。"""
    if not check_schema(conn):
        return
    row = conn.execute(
        "SELECT update_id,candidate_pointer,reload_tx_id FROM plugin_updates WHERE plugin_id=? AND phase='armed'",
        (plugin_id,),
    ).fetchone()
    if row is None:
        return
    if candidate_pointer != row[1] or row[2] is not None:
        raise RuntimeError("未完成插件更新不能换候选或重建 reload")
    _ = conn.execute("UPDATE plugin_updates SET reload_tx_id=? WHERE update_id=?", (tx_id, row[0]))


def commit(conn: sqlite3.Connection, tx_id: str, now: str) -> None:
    if check_schema(conn):
        _ = conn.execute(
            "UPDATE plugin_updates SET phase='committed',updated_at=?,error='' WHERE reload_tx_id=? AND phase='armed'",
            (now, tx_id),
        )


def rollback(conn: sqlite3.Connection, update: UpdateRollback, plugins_home: Path, *, now: str, error: str) -> None:
    """核对旧/新状态后先恢复文件，再记录回退完成；中途死亡可再次执行。"""
    if update.phase != 'armed':
        raise RuntimeError("只有尚未提交的插件更新可以回退")
    name, separator, marketplace = update.plugin_id.rpartition('@')
    if not separator or any(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', item) is None for item in (name, marketplace)):
        raise ValueError("插件更新身份无效")
    expected_base = plugins_home.resolve() / 'cache' / marketplace / name
    if update.plugin_base != expected_base or expected_base.resolve() != expected_base:
        raise RuntimeError("插件更新恢复点不属于当前插件目录")
    path = pointer_state_path(expected_base)
    if path.is_symlink():
        raise ValueError("插件指针恢复目标不能是符号链接")
    if path.exists() and not path.is_file():
        raise ValueError("插件指针恢复目标必须是普通文件")
    missing = object()
    current = load_json(path, default=missing, domain='plugin_update_rollback')
    if current is missing:
        current = None
    elif not isinstance(current, dict):
        raise ValueError("插件指针恢复目标必须是对象")
    previous = pointer_value(update.previous)
    base = None if update.previous is None else update.previous.stable.path
    accepted = (previous, {'stable': base, 'latest': update.candidate.path},
                {'stable': update.candidate.path, 'latest': update.candidate.path})
    if current not in accepted:
        raise RuntimeError("插件指针已被其他操作改变，不能覆盖")
    entries = load_plugin_manifest(plugins_home)
    if entries.get(update.plugin_id) not in (update.previous_enabled, True):
        raise RuntimeError("插件启用状态已被其他操作改变，不能覆盖")
    # 1. 写回旧目标时由原 pointer owner 核验旧 artifact 仍然可用。
    if update.previous is None:
        path.unlink(missing_ok=True)
        sync_directory(expected_base)
    else:
        _ = write_pointers(expected_base, stable=update.previous.stable, latest=update.previous.latest)
    if update.previous_enabled is None:
        _ = entries.pop(update.plugin_id, None)
    else:
        entries[update.plugin_id] = update.previous_enabled
    _ = write_plugin_manifest(entries, plugins_home=plugins_home)
    # 2. 保留新 artifact 与全部 plugin-data；只更新本恢复点的完成事实。
    _ = conn.execute(
        "UPDATE plugin_updates SET phase='rolled_back',updated_at=?,error=? WHERE update_id=? AND phase='armed'",
        (now, error, update.update_id),
    )
