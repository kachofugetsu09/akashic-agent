"""包管理器的写命令：install、upgrade、remove、sync；在线走 Gateway，离线持锁直装。"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from agent.plugins.bundles import composition_rows, set_plugin_choice
from agent.plugins.distribution_sources import distribution_sources
from agent.plugins.install import PluginInstallResult, finalize_uninstall_plugin, fetch_sha_shallow, install_git_plugin
from agent.plugins.manifest import plugins_root
from agent.plugins.packages import InstalledPackage, PackageState, list_release_heads, read_state, sync_rows
from agent.plugins.selection import PluginSelection
from agent.plugins.static_manifest import load_static_plugin_manifest
from bootstrap.workspace_lock import PluginPublicationLock, WorkspaceMaintenanceLock

_MAIN = Path(__file__).resolve().parents[2] / "main.py"
_DEFAULT_MARKETPLACE = "extra"

Result = Literal["installed", "upgraded", "current", "restart-required", "removed", "failed",
                 "in-progress", "needs-switch", "inactive", "retired", "optional"]
BAD_RESULTS = frozenset({"failed", "in-progress", "needs-switch", "inactive"})


@dataclass(frozen=True)
class Outcome:
    name: str
    result: Result
    detail: str = ""


@dataclass(frozen=True)
class Context:
    workspace: Path
    config_path: Path
    repo: str
    online: bool
    timeout: float = 120.0


# 错误信息取最后一行：子进程 traceback 的结论在末尾。
def _last_line(error: BaseException) -> str:
    lines = str(error).strip().splitlines()
    return lines[-1] if lines else type(error).__name__


# 身份取组合行里该包的完整身份；不在任何行里的新包用 extra。
def identity_of(workspace: Path, name: str) -> str:
    for row in composition_rows(workspace):
        if row.plugin.partition("@")[0] == name:
            return row.plugin if "@" in row.plugin else f"{name}@{_DEFAULT_MARKETPLACE}"
    return f"{name}@{_DEFAULT_MARKETPLACE}"


# 取远端提交并读静态 manifest，校验 name 与 api_version；不写 cache。
def _check_remote_source(repo: str, sha: str, name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="pkg-check-") as directory:
        target = Path(directory)
        if not fetch_sha_shallow(repo, target, sha):
            _ = subprocess.run(["git", "clone", "--quiet", "--", repo, str(target)], check=True, capture_output=True)
            _ = subprocess.run(["git", "-C", str(target), "checkout", "--quiet", "--detach", sha],
                               check=True, capture_output=True)
        manifest = load_static_plugin_manifest(target)
        if manifest.name != name:
            raise ValueError(f"release/{name} 里的插件名是 {manifest.name}，与分支名不一致")


# 调用既有管理命令（plugin-install、plugin-status、plugin-uninstall），经 Gateway 工作。
def _main(ctx: Context, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(_MAIN), *args, "--workspace", str(ctx.workspace),
                           "--config", str(ctx.config_path)], capture_output=True, text=True, check=False)


def _json_of(done: subprocess.CompletedProcess[str]) -> dict[str, object]:
    return cast(dict[str, object], json.loads(done.stdout.strip().splitlines()[-1]))


# 在线：plugin-install 只回执"已受理"，用 update_id 轮询 plugin-status 到终态或超时。
def _install_online(ctx: Context, plugin_id: str, sha: str) -> tuple[Result, str]:
    update_id = uuid4().hex
    done = _main(ctx, "plugin-install", "--source", ctx.repo, "--marketplace", plugin_id.partition("@")[2],
                 "--ref", sha, "--update-id", update_id)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "plugin-install 失败")
    deadline = time.monotonic() + ctx.timeout
    while True:
        status = _main(ctx, "plugin-status", update_id)
        state = str(_json_of(status).get("state", "")) if status.returncode == 0 else ""
        if state == "active":
            return "installed", ""
        if state == "restart_required":
            return "restart-required", "公共合同变化，需要重启实例后生效"
        if state == "failed":
            return "failed", str(_json_of(status).get("error", "")) or "激活失败"
        if time.monotonic() >= deadline:
            return "in-progress", f"{ctx.timeout:.0f} 秒内未到终态（state={state or 'unknown'}）；plugin-status {update_id} 查询"
        time.sleep(1.0)


# 离线：直接调用安装器，下次启动生效；身份必须是普通外置身份，不传 takeover。
def _install_offline(ctx: Context, plugin_id: str, sha: str) -> PluginInstallResult:
    return install_git_plugin(workspace=ctx.workspace, source=ctx.repo, marketplace=plugin_id.partition("@")[2],
                              ref_name=sha, plugins_home=plugins_root().resolve(strict=False))


# 实例在线时取 plugin-status 里该身份的投影；离线或取不到返回 None。
def _live_view(ctx: Context, plugin_id: str) -> dict[str, object] | None:
    if not ctx.online:
        return None
    done = _main(ctx, "plugin-status")
    if done.returncode != 0:
        return None
    for item in cast(list[dict[str, object]], _json_of(done).get("plugins", [])):
        if item.get("plugin_id") == plugin_id:
            return item
    return None


# 包是否已被选择、启用并处于正常状态；在线时还核对 generation 无加载错误。
def _healthy(ctx: Context, package: InstalledPackage) -> str:
    if not package.selected:
        return "已安装但未进选择"
    enabled = {row.plugin: not row.disabled for row in composition_rows(ctx.workspace)}
    if not enabled.get(package.plugin_id, False):
        return "已安装但未启用"
    view = _live_view(ctx, package.plugin_id)
    if ctx.online and (view is None or view.get("state") is None or view.get("load_error")):
        return f"加载异常: {None if view is None else view.get('load_error')}"
    return ""


# 安装一个包到指定提交；需要换来源的身份不在这里处理。
def install_one(ctx: Context, state: PackageState, plugin_id: str, sha: str, upgrade: bool = False) -> Outcome:
    name = plugin_id.partition("@")[0]
    # 1. 发行版命名空间或历史归属的身份属于换来源，不传 takeover。
    if state.requires_switch(plugin_id):
        return Outcome(name, "needs-switch", f"{plugin_id} 属于发行版身份，需要换来源操作")
    try:
        # 2. 取代码前先校验 name 和 api_version，失败时什么都没写。
        _check_remote_source(ctx.repo, sha, name)
        # 3. 在线等到终态，离线只写 cache 与选择 patch。
        if ctx.online:
            result, detail = _install_online(ctx, plugin_id, sha)
            return Outcome(name, "upgraded" if upgrade and result == "installed" else result,
                           detail or sha[:12])
        _ = _install_offline(ctx, plugin_id, sha)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        return Outcome(name, "failed", _last_line(error))
    return Outcome(name, "upgraded" if upgrade else "installed", sha[:12])


# 逐个安装；只有已被选择、启用且正常的同版本才跳过，其余走安装路径（可复用已有 artifact）。
def install_many(ctx: Context, plugin_ids: list[str]) -> list[Outcome]:
    heads = list_release_heads(ctx.repo)
    state = read_state(ctx.workspace, ctx.repo)
    here = {item.plugin_id: item for item in state.packages if item.source == "extra"}
    outcomes: list[Outcome] = []
    for plugin_id in plugin_ids:
        name = plugin_id.partition("@")[0]
        sha = heads.get(name)
        package = here.get(plugin_id)
        if sha is None:
            outcomes.append(Outcome(name, "failed", f"远端没有 release/{name}"))
        elif package is not None and package.head == sha and not _healthy(ctx, package):
            outcomes.append(Outcome(name, "current", "已是该版本，已启用"))
        else:
            outcomes.append(install_one(ctx, state, plugin_id, sha))
    return outcomes


# 来源是配置的 extra 仓库的已装包才参与升级；HEAD 相同且正常才报已是最新。
def upgrade_many(ctx: Context, names: list[str]) -> list[Outcome]:
    state = read_state(ctx.workspace, ctx.repo)
    installed = {item.name: item for item in state.packages if item.source == "extra"}
    heads = list_release_heads(ctx.repo)
    outcomes: list[Outcome] = []
    for name in names or sorted(installed):
        package = installed.get(name)
        if package is None:
            outcomes.append(Outcome(name, "failed", "未安装，或来源不是 extra"))
        elif name not in heads:
            outcomes.append(Outcome(name, "failed", "上游已移除 release 分支"))
        elif heads[name] != package.head:
            outcomes.append(install_one(ctx, state, package.plugin_id, heads[name], upgrade=True))
        elif problem := _healthy(ctx, package):
            outcomes.append(Outcome(name, "inactive", f"版本已是最新，但{problem}；用 pkg install {name} 重试"))
        else:
            outcomes.append(Outcome(name, "current", package.head[:12]))
    return outcomes


# 在线卸载只回执"已受理"：等到选择里不再有它且 cache 已清理，超时如实说仍在进行。
def _await_removed(ctx: Context, plugin_id: str) -> Outcome:
    name, _, marketplace = plugin_id.partition("@")
    cache = plugins_root() / "cache" / marketplace / name
    deadline = time.monotonic() + ctx.timeout
    while True:
        selection = PluginSelection(ctx.workspace)
        reference = selection.read()
        selected = reference is not None and any(
            selection.read_input(ref)["plugin_id"] == plugin_id for ref in selection.components(reference))
        if not selected and not cache.exists():
            return Outcome(name, "removed", "数据目录已保留")
        if time.monotonic() >= deadline:
            return Outcome(name, "in-progress", f"{ctx.timeout:.0f} 秒内未完成清理（selected={selected}, "
                           f"cache={cache.exists()}）；用 plugin-status 查询")
        time.sleep(0.5)


# 卸载：停用选择、从选择移除、删 cache，数据目录保留；在线走 plugin-uninstall 并等待结算。
def _remove_one(ctx: Context, package: InstalledPackage) -> Outcome:
    if ctx.online:
        done = _main(ctx, "plugin-uninstall", package.plugin_id)
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip() or "plugin-uninstall 失败")
        return _await_removed(ctx, package.plugin_id)
    root = distribution_sources(ctx.workspace, plugins_root(), replacing_distribution=True).root
    _ = set_plugin_choice(ctx.workspace, package.plugin_id, enabled=False, distribution=root)
    selection = PluginSelection(ctx.workspace)
    reference = selection.read()
    if reference is not None:
        kept = tuple(ref for ref in selection.components(reference)
                     if selection.read_input(ref)["plugin_id"] != package.plugin_id)
        if len(kept) != len(selection.components(reference)):
            _ = selection.commit(kept, expected_ref=reference)
    _ = finalize_uninstall_plugin(package.plugin_id, workspace=ctx.workspace, plugins_home=plugins_root())
    return Outcome(package.name, "removed", "数据目录已保留")


def remove_many(ctx: Context, names: list[str]) -> list[Outcome]:
    installed = {item.name: item for item in read_state(ctx.workspace, ctx.repo).packages}
    outcomes: list[Outcome] = []
    for name in names:
        package = installed.get(name)
        if package is None or package.source == "core":
            outcomes.append(Outcome(name, "failed", "未安装，或是发行版内置插件"))
            continue
        try:
            outcomes.append(_remove_one(ctx, package))
        except (RuntimeError, ValueError, OSError) as error:
            outcomes.append(Outcome(name, "failed", _last_line(error)))
    return outcomes


# 同步：只装 preset 要求的缺口；换来源、可选、已退役的单独汇总，不安装也不计入失败（换来源除外）。
def sync_install(ctx: Context) -> list[Outcome]:
    rows = sync_rows(ctx.workspace, ctx.repo)
    outcomes = install_many(ctx, [row.plugin_id for row in rows if row.action == "install"])
    notes: dict[str, tuple[Result, str]] = {
        "switch": ("needs-switch", "需要换来源，已跳过"),
        "optional": ("optional", "只在用户 patch 里启用，不自动安装；需要请 pkg install"),
        "retired": ("retired", "已退役：上游已不存在，忽略"),
    }
    outcomes.extend(Outcome(row.plugin_id.partition("@")[0], notes[row.action][0],
                            f"{row.plugin_id} {notes[row.action][1]}") for row in rows if row.action in notes)
    return outcomes


# 取得执行方式并运行一条写命令：能取得 workspace 锁说明没有运行实例，持锁离线直装；
# 否则视为实例在线，经 Gateway 局部换代。
def run_write(command: str, names: list[str], *, workspace: Path, config_path: Path, repo: str,
              timeout: float = 120.0) -> list[Outcome]:
    home = plugins_root().resolve(strict=False)
    maintenance, publication = WorkspaceMaintenanceLock(workspace), PluginPublicationLock(home)
    online = False
    try:
        maintenance.acquire()
    except RuntimeError:
        online = True
    try:
        if not online:
            publication.acquire()
        ctx = Context(workspace, config_path, repo, online, timeout)
        if command == "install":
            return install_many(ctx, [identity_of(workspace, name) for name in names])
        if command == "upgrade":
            return upgrade_many(ctx, names)
        if command == "remove":
            return remove_many(ctx, names)
        return sync_install(ctx)
    finally:
        if not online:
            publication.release()
            maintenance.release()
