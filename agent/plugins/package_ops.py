"""包管理器的写命令：install、upgrade、remove、sync；在线走 Gateway，离线持锁直装。"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from agent.plugins.bundles import composition_rows, set_plugin_choice
from agent.plugins.distribution_sources import distribution_sources
from agent.plugins.install import PluginInstallResult, finalize_uninstall_plugin, fetch_sha_shallow, install_git_plugin
from agent.plugins.manifest import plugins_root
from agent.plugins.packages import InstalledPackage, list_release_heads, read_state, sync_rows
from agent.plugins.selection import PluginSelection
from agent.plugins.static_manifest import load_static_plugin_manifest
from bootstrap.workspace_lock import PluginPublicationLock, WorkspaceMaintenanceLock

_MAIN = Path(__file__).resolve().parents[2] / "main.py"
_DEFAULT_MARKETPLACE = "extra"

Result = Literal["installed", "upgraded", "current", "restart-required", "removed", "failed"]


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


# 在线：复用 plugin-install 命令经 Gateway RPC 局部换代；返回更新状态。
def _install_online(ctx: Context, plugin_id: str, sha: str) -> str:
    marketplace = plugin_id.partition("@")[2]
    command = [sys.executable, str(_MAIN), "plugin-install", "--source", ctx.repo, "--marketplace", marketplace,
               "--ref", sha, "--workspace", str(ctx.workspace), "--config", str(ctx.config_path)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "plugin-install 失败")
    return str(cast(dict[str, object], json.loads(done.stdout.strip().splitlines()[-1])).get("state", ""))


# 离线：直接调用安装器，下次启动生效。调用前已确认目标身份不由当前发行版提供，
# 所以迁出插件保留 release 身份时只对这一个身份放行防接管检查。
def _install_offline(ctx: Context, plugin_id: str, sha: str) -> PluginInstallResult:
    marketplace = plugin_id.partition("@")[2]
    return install_git_plugin(workspace=ctx.workspace, source=ctx.repo, marketplace=marketplace,
                              ref_name=sha, plugins_home=plugins_root().resolve(strict=False),
                              takeover=frozenset({plugin_id}))


# 安装一个包到指定提交；发行版内置提供的身份一律拒绝。
def install_one(ctx: Context, plugin_id: str, sha: str, upgrade: bool = False) -> Outcome:
    name = plugin_id.partition("@")[0]
    # 1. 内置身份属于换来源，不在这里处理。
    if plugin_id in {f"{item.plugin_name}@{item.marketplace}"
                     for item in distribution_sources(ctx.workspace, plugins_root()).sources}:
        return Outcome(name, "failed", f"{plugin_id} 由发行版内置提供；请使用换来源命令 pkg switch")
    try:
        # 2. 取代码前先校验 name 和 api_version，失败时什么都没写。
        _check_remote_source(ctx.repo, sha, name)
        # 3. 在线局部换代，离线只写 cache 与选择 patch。
        if ctx.online:
            state = _install_online(ctx, plugin_id, sha)
            if state == "restart_required":
                return Outcome(name, "restart-required", "公共合同变化，需要重启实例后生效")
        else:
            _ = _install_offline(ctx, plugin_id, sha)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        return Outcome(name, "failed", _last_line(error))
    return Outcome(name, "upgraded" if upgrade else "installed", sha[:12])


# 逐个安装；每个包独立成败，不互相影响。
def install_many(ctx: Context, plugin_ids: list[str]) -> list[Outcome]:
    heads = list_release_heads(ctx.repo)
    heads_here = {item.plugin_id: item.head for item in read_state(ctx.workspace, ctx.repo).packages
                  if item.source == "extra"}
    outcomes: list[Outcome] = []
    for plugin_id in plugin_ids:
        name = plugin_id.partition("@")[0]
        sha = heads.get(name)
        if sha is None:
            outcomes.append(Outcome(name, "failed", f"远端没有 release/{name}"))
        elif heads_here.get(plugin_id) == sha:
            outcomes.append(Outcome(name, "current", "已是该版本"))
        else:
            outcomes.append(install_one(ctx, plugin_id, sha))
    return outcomes


# 来源是配置的 extra 仓库的已装包才参与升级；HEAD 与远端相同的不动。
def upgrade_many(ctx: Context, names: list[str]) -> list[Outcome]:
    installed = {item.name: item for item in read_state(ctx.workspace, ctx.repo).packages if item.source == "extra"}
    heads = list_release_heads(ctx.repo)
    outcomes: list[Outcome] = []
    for name in names or sorted(installed):
        package = installed.get(name)
        if package is None:
            outcomes.append(Outcome(name, "failed", "未安装，或来源不是 extra"))
        elif name not in heads:
            outcomes.append(Outcome(name, "failed", "上游已移除 release 分支"))
        elif heads[name] == package.head:
            outcomes.append(Outcome(name, "current", package.head[:12]))
        else:
            outcomes.append(install_one(ctx, package.plugin_id, heads[name], upgrade=True))
    return outcomes


# 卸载：停用选择、从选择移除、删 cache，数据目录保留；在线走 plugin-uninstall。
def _remove_one(ctx: Context, package: InstalledPackage) -> None:
    if ctx.online:
        command = [sys.executable, str(_MAIN), "plugin-uninstall", package.plugin_id,
                   "--workspace", str(ctx.workspace), "--config", str(ctx.config_path)]
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip() or "plugin-uninstall 失败")
        return
    root = distribution_sources(ctx.workspace, plugins_root()).root
    _ = set_plugin_choice(ctx.workspace, package.plugin_id, enabled=False, distribution=root)
    selection = PluginSelection(ctx.workspace)
    reference = selection.read()
    if reference is not None:
        kept = tuple(ref for ref in selection.components(reference)
                     if selection.read_input(ref)["plugin_id"] != package.plugin_id)
        if len(kept) != len(selection.components(reference)):
            _ = selection.commit(kept, expected_ref=reference)
    _ = finalize_uninstall_plugin(package.plugin_id, workspace=ctx.workspace, plugins_home=plugins_root())


def remove_many(ctx: Context, names: list[str]) -> list[Outcome]:
    installed = {item.name: item for item in read_state(ctx.workspace, ctx.repo).packages}
    outcomes: list[Outcome] = []
    for name in names:
        package = installed.get(name)
        if package is None or package.source == "core":
            outcomes.append(Outcome(name, "failed", "未安装，或是发行版内置插件"))
            continue
        try:
            _remove_one(ctx, package)
            outcomes.append(Outcome(name, "removed", "数据目录已保留"))
        except (RuntimeError, ValueError, OSError) as error:
            outcomes.append(Outcome(name, "failed", _last_line(error)))
    return outcomes


# 同步：对应装集合逐个安装；远端没有分支的单独报失败。
def sync_install(ctx: Context) -> list[Outcome]:
    rows = sync_rows(ctx.workspace, ctx.repo)
    return install_many(ctx, [row.plugin_id for row in rows])


# 取得执行方式并运行一条写命令：能取得 workspace 锁说明没有运行实例，持锁离线直装；
# 否则视为实例在线，经 Gateway 局部换代。
def run_write(command: str, names: list[str], *, workspace: Path, config_path: Path, repo: str) -> list[Outcome]:
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
        ctx = Context(workspace, config_path, repo, online)
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
