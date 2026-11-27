"""换来源：把已装的 name@旧来源 换成另一个来源里的同名包，插件 ID 随来源改成 name@新来源。

插件 ID 落在三处持久状态里：选择记录、数据目录 plugin-data/<name>-<来源>、ledger 的 owner_records（plugin:<ID>[:子空间]）。
选择记录由卸载和安装各自维护；这里补另外两处，让新来源的插件接着用原来的数据。
"""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from agent.plugins.manifest import workspace_plugin_data_dir
from agent.plugins.package_ops import Context, Outcome, remove_one, install_one
from agent.plugins.packages import list_release_heads, read_state


# 逐个换来源；ctx 是目标来源。names 写完整的旧 ID（phone-use@local）。
def move_many(ctx: Context, plugin_ids: list[str]) -> list[Outcome]:
    heads = list_release_heads(ctx.repo)
    return [_move_one(ctx, plugin_id, heads) for plugin_id in plugin_ids]


# 1. 校验；2. 卸旧（停下来，数据保留）；3. 复制数据目录、owner 记录改名；4. 装新，装不上就把 owner 记录改回去。
def _move_one(ctx: Context, old_id: str, heads: dict[str, str]) -> Outcome:
    name, _, old_marketplace = old_id.partition("@")
    new_id = ctx.identity(name)
    # 1. 旧包要装着、目标来源要发布了同名包、新 ID 不能已被占用。
    state = read_state(ctx.workspace, ctx.repo)
    installed = {item.plugin_id: item for item in state.packages}
    problem = _precheck(ctx, old_id, new_id, set(installed), heads)
    if problem:
        return Outcome(name, "failed", problem)
    # 2. 同名插件不能两个来源同时在选择里，所以先卸旧的；在线时会等它停下来。
    removed = remove_one(ctx, installed[old_id])
    if removed.result != "removed":
        return Outcome(name, "failed", f"卸载 {old_id} 未完成：{removed.detail}")
    # 3. 旧包已停，数据目录和 owner 记录都没有写入者了。
    old_dir = workspace_plugin_data_dir(ctx.workspace, name, old_marketplace)
    if old_dir.is_dir():
        _ = shutil.copytree(old_dir, workspace_plugin_data_dir(ctx.workspace, name, ctx.marketplace),
                            ignore=_special_files)
    moved = _rename_owner(ctx.workspace, old_id, new_id)
    # 4. 新包装不上时 owner 记录改回旧 ID，重新装回旧来源就能接着用；复制出的数据目录留着无害。
    outcome = install_one(ctx, read_state(ctx.workspace, ctx.repo), new_id, heads[name])
    if outcome.result not in {"installed", "restart-required"}:
        _ = _rename_owner(ctx.workspace, new_id, old_id)
        return Outcome(name, "failed", f"{new_id} 未装上（{outcome.detail}）；{old_id} 已卸载，数据和 owner 记录都在，"
                       f"可从原来源重新安装")
    return Outcome(name, outcome.result, f"{old_id} → {new_id}，迁移 {moved} 条 owner 记录")


def _precheck(ctx: Context, old_id: str, new_id: str, ids: set[str], heads: dict[str, str]) -> str:
    name, _, old_marketplace = old_id.partition("@")
    if not old_marketplace:
        return f"{old_id} 要写完整 ID（名称@来源）"
    if old_id not in ids:
        return f"{old_id} 未安装"
    if new_id == old_id:
        return f"{old_id} 已经来自这个来源"
    if new_id in ids:
        return f"{new_id} 已安装，不能再换过去"
    if name not in heads:
        return f"来源 {ctx.marketplace} 没有发布 {name}"
    target = workspace_plugin_data_dir(ctx.workspace, name, ctx.marketplace)
    if target.exists() and any(target.iterdir()):
        return f"目标数据目录 {target.name} 已有内容，先确认后再移走"
    return ""


# 数据目录里的 socket 等特殊文件是运行时产物，不复制。
def _special_files(directory: str, names: list[str]) -> list[str]:
    base = Path(directory)
    return [item for item in names if not ((base / item).is_dir() or (base / item).is_file()) or (base / item).is_symlink()]


# owner 记录改名：plugin:<旧ID> 与 plugin:<旧ID>:<子空间>；新 owner 已有同键不同内容时拒绝，整体回滚。
def _rename_owner(workspace: Path, old_id: str, new_id: str) -> int:
    path = workspace / "sessions.db"
    if not path.exists():
        return 0
    old, new = f"plugin:{old_id}", f"plugin:{new_id}"
    match = "(owner = ? OR substr(owner, 1, ?) = ?)"
    arguments = (old, len(old) + 1, old + ":")
    renamed = f"? || substr(owner, {len(old) + 1})"
    with closing(sqlite3.connect(path, timeout=60)) as db, db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='owner_records'").fetchone() is None:
            return 0
        conflicts = db.execute(
            f"""SELECT o.owner, o.key FROM owner_records AS o
                JOIN owner_records AS n ON n.owner = ? || substr(o.owner, {len(old) + 1}) AND n.key = o.key
                WHERE (o.owner = ? OR substr(o.owner, 1, ?) = ?)
                  AND (n.version IS NOT o.version OR n.value IS NOT o.value)""",
            (new, *arguments)).fetchall()
        if conflicts:
            raise RuntimeError(f"{new} 已有不同内容的同键记录，拒绝覆盖: {conflicts[:3]}")
        moved = db.execute(f"UPDATE OR IGNORE owner_records SET owner = {renamed} WHERE {match}",
                           (new, *arguments)).rowcount
        _ = db.execute(f"DELETE FROM owner_records WHERE {match}", arguments)
    return moved
