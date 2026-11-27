"""换来源：把已装的 name@旧来源 换成另一个来源里的同名包，插件 ID 随来源改成 name@新来源。

插件 ID 落在三处持久状态里：选择记录（卸载、安装各自维护）、数据目录 plugin-data/<name>-<来源>（这里复制），
以及插件经 ledger 存的状态（ledger 按新数据目录里的 .moved-from 标记自己改名，Core 不碰它的库）。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agent.plugins.manifest import workspace_plugin_data_dir
from agent.plugins.package_ops import Context, Outcome, remove_one, install_one
from agent.plugins.packages import InstalledPackage, list_release_heads, read_state


# 换来源标记：写在新数据目录里，内容是旧插件 ID；ledger 读到后把旧 ID 的 owner 记录改归新 ID，再删掉标记。
MOVED_FROM = ".moved-from"


# 逐个换来源；ctx 是目标来源。names 写完整的旧 ID（phone-use@local）。
def move_many(ctx: Context, plugin_ids: list[str]) -> list[Outcome]:
    heads = list_release_heads(ctx.repo)
    return [_move_one(ctx, plugin_id, heads) for plugin_id in plugin_ids]


# 1. 校验；2. 卸旧（停下来，数据保留）；3. 复制数据目录并留下迁移标记；4. 装新，装不上就撤掉复制出的目录。
def _move_one(ctx: Context, old_id: str, heads: dict[str, str]) -> Outcome:
    name, _, old_marketplace = old_id.partition("@")
    new_id = ctx.identity(name)
    # 1. 旧包要装着、目标来源要发布了同名包、新 ID 不能已被占用。
    state = read_state(ctx.workspace, ctx.repo)
    installed = {item.plugin_id: item for item in state.packages}
    problem = _precheck(ctx, old_id, new_id, installed, heads)
    if problem:
        return Outcome(name, "failed", problem)
    # 2. 同名插件不能两个来源同时在选择里，所以先卸旧的；在线时会等它停下来。
    removed = remove_one(ctx, installed[old_id])
    if removed.result != "removed":
        return Outcome(name, "failed", f"卸载 {old_id} 未完成：{removed.detail}")
    if new_id == old_id:
        return _reinstall(ctx, old_id, heads[name])
    # 3. 旧包已停，数据目录没有写入者了。插件存在 ledger 里的状态由 ledger 在新插件第一次打开时按标记改归新 ID。
    old_dir = workspace_plugin_data_dir(ctx.workspace, name, old_marketplace)
    new_dir = workspace_plugin_data_dir(ctx.workspace, name, ctx.marketplace)
    if old_dir.is_dir():
        _ = shutil.copytree(old_dir, new_dir, ignore=_special_files)
    new_dir.mkdir(parents=True, exist_ok=True)
    _ = (new_dir / MOVED_FROM).write_text(old_id, encoding="utf-8")
    # 4. 新包装不上：撤掉复制出的目录（含标记），旧数据和状态都没动过，可从原来源重新安装。
    outcome = install_one(ctx, read_state(ctx.workspace, ctx.repo), new_id, heads[name])
    if outcome.result not in {"installed", "restart-required"}:
        shutil.rmtree(new_dir, ignore_errors=True)
        return Outcome(name, "failed", f"{new_id} 未装上（{outcome.detail}）；{old_id} 已卸载，数据都在，可从原来源重新安装")
    return Outcome(name, outcome.result, f"{old_id} → {new_id}")


# ID 不变（只是安装来源记错了，或换到同名来源）：数据目录与 ledger 状态都按 ID 原样有效，卸了重装即可。
def _reinstall(ctx: Context, plugin_id: str, sha: str) -> Outcome:
    name = plugin_id.partition("@")[0]
    outcome = install_one(ctx, read_state(ctx.workspace, ctx.repo), plugin_id, sha)
    if outcome.result not in {"installed", "restart-required"}:
        return Outcome(name, "failed", f"{plugin_id} 未从 {ctx.marketplace} 装上（{outcome.detail}）；数据都在，可从原来源重新安装")
    return Outcome(name, outcome.result, f"{plugin_id} 改为从 {ctx.marketplace} 安装")


def _precheck(ctx: Context, old_id: str, new_id: str, installed: dict[str, InstalledPackage], heads: dict[str, str]) -> str:
    name, _, old_marketplace = old_id.partition("@")
    if not old_marketplace:
        return f"{old_id} 要写完整 ID（名称@来源）"
    if old_id not in installed:
        return f"{old_id} 未安装"
    if name not in heads:
        return f"来源 {ctx.marketplace} 没有发布 {name}"
    if new_id == old_id:
        # 已经来自这个来源（官方源记作 extra，私有源记作来源名）就不用动。
        return f"{old_id} 已经来自这个来源" if installed[old_id].source == ("extra" if ctx.official else ctx.marketplace) else ""
    if new_id in installed:
        return f"{new_id} 已安装，不能再换过去"
    target = workspace_plugin_data_dir(ctx.workspace, name, ctx.marketplace)
    if target.exists() and any(target.iterdir()):
        return f"目标数据目录 {target.name} 已有内容，先确认后再移走"
    return ""


# 数据目录里的 socket 等特殊文件是运行时产物，不复制。
def _special_files(directory: str, names: list[str]) -> list[str]:
    base = Path(directory)
    return [item for item in names if not ((base / item).is_dir() or (base / item).is_file()) or (base / item).is_symlink()]
