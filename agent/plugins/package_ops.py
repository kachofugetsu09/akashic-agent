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

from agent.plugins.artifacts import read_pointers, resolve_pointer
from agent.plugins.bundles import absorbed_plugin_ids, composition_rows, set_plugin_choice
from agent.plugins.extra_repo import is_official_repo, official_marketplace
from agent.plugins.distribution_sources import distribution_sources
from agent.plugins.install import PluginInstallResult, finalize_uninstall_plugin, fetch_sha_shallow, install_git_plugin
from agent.plugins.manifest import plugins_root
from agent.plugins.package_index import closure, expand, load_index
from agent.plugins.prerequisites import clear_disabled, describe, read_disabled, record_declared, record_disabled, unmet
from agent.plugins.packages import InstalledPackage, PackageState, list_release_heads, read_state, sync_rows
from agent.plugins.resources import ResourceError, repair_resources
from agent.plugins.selection import PluginSelection
from agent.plugins.static_manifest import load_static_plugin_manifest
from bootstrap.workspace_lock import PluginPublicationLock, WorkspaceMaintenanceLock

_MAIN = Path(__file__).resolve().parents[2] / "main.py"
_DEFAULT_MARKETPLACE = "extra"

Result = Literal["installed", "upgraded", "current", "restart-required", "removed", "failed",
                 "in-progress", "needs-switch", "inactive", "retired", "optional", "retire"]
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


# 官方 extra 仓库的包，身份一律是发行版的官方 marketplace（取自 distribution.json）。
def identity_of(name: str) -> str:
    marketplace = official_marketplace()
    if marketplace is None:
        raise ValueError("无法确定官方 marketplace：未配置发行版（AKASHIC_PLUGIN_DISTRIBUTION）")
    return f"{name}@{marketplace}"


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
    before = _live_view(ctx, plugin_id)
    prior = None if before is None else before.get("generation_id")
    done = _main(ctx, "plugin-install", "--source", ctx.repo, "--marketplace", plugin_id.partition("@")[2],
                 "--ref", sha, "--update-id", update_id)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "plugin-install 失败")
    deadline = time.monotonic() + ctx.timeout
    while True:
        status = _main(ctx, "plugin-status", update_id)
        view = _json_of(status) if status.returncode == 0 else {}
        state = str(view.get("state", ""))
        if state == "active" or _settled_elsewhere(ctx, view, prior, plugin_id, sha):
            return "installed", ""
        if state == "restart_required":
            return "restart-required", "公共合同变化，需要重启实例后生效"
        if state == "failed":
            return "failed", str(_json_of(status).get("error", "")) or "激活失败"
        if time.monotonic() >= deadline:
            return "in-progress", f"{ctx.timeout:.0f} 秒内未到终态（state={state or 'unknown'}）；plugin-status {update_id} 查询"
        time.sleep(1.0)


# 更新已到终态却报 unknown：新建了 Python 环境的包，选择里落的输入与更新记下的不是同一个，plugin-status 对不上号。
# 这时以事实为准：换出了新一代且在运行、stable 指针指向目标提交，就是装好了；否则照常等到期限。
def _settled_elsewhere(ctx: Context, view: dict[str, object], prior: object, plugin_id: str, sha: str) -> bool:
    if view.get("state") != "unknown" or view.get("terminal") is not True or view.get("fiber_state") != "active":
        return False
    generation = view.get("generation_id")
    return generation is not None and generation != prior and _landed(ctx, plugin_id.partition("@")[0], plugin_id, sha)


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
    # 前置条件不满足的原因在停用之前就记下了：仍开着却有当前已装提交的记录，说明上次停用没做完，要重走收尾。
    # 只认 head 一致的记录：旧版本留下的记录在启用新版本时会被忽略但不一定清除。
    if read_disabled(ctx.workspace).get(package.plugin_id, {}).get("head") == package.head:
        return "前置条件收尾未完成"
    view = _live_view(ctx, package.plugin_id)
    if ctx.online and (view is None or view.get("state") is None or view.get("load_error")):
        return f"加载异常: {None if view is None else view.get('load_error')}"
    return ""


# 安装一个包到指定提交；需要换来源的身份不在这里处理。
def install_one(ctx: Context, state: PackageState, plugin_id: str, sha: str, upgrade: bool = False) -> Outcome:
    name = plugin_id.partition("@")[0]
    # 0. 已并入 Core 的身份不再作为扩展包存在：明确拒绝，不安装也不升级。
    if (provider := absorbed_plugin_ids().get(plugin_id)) is not None:
        return Outcome(name, "failed", f"{plugin_id} 已并入 Core 的 {provider} 插件，不再作为扩展包安装")
    # 1. 发行版命名空间或历史归属的身份属于换来源，不传 takeover。
    if state.requires_switch(plugin_id):
        return Outcome(name, "needs-switch", f"{plugin_id} 属于发行版身份，需要换来源操作")
    try:
        # 2. 取代码前先校验 name 和 api_version，失败时什么都没写。
        _check_remote_source(ctx.repo, sha, name)
        # 3. 在线等到终态，离线只写 cache 与选择 patch。
        if ctx.online:
            result, detail = _install_online(ctx, plugin_id, sha)
            # 只转换一种失败：目标已确认落地（新 artifact 已发布、指针指向它），且本机前置条件确实不满足导致激活失败
            # （如缺 Workload Controller）。其余失败（下载、校验、依赖、旧 artifact 未换成目标）保持真实失败，不动正在用的旧包。
            if result == "installed" or (result == "failed" and _landed(ctx, name, plugin_id, sha)):
                held = _hold_closed(ctx, name, plugin_id, sha)
                if held:
                    result, detail = "installed", held
            elif result == "restart-required":
                _ = _declare(ctx, name, plugin_id, sha)
            return Outcome(name, "upgraded" if upgrade and result == "installed" else result,
                           detail or sha[:12])
        _ = _install_offline(ctx, plugin_id, sha)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        return Outcome(name, "failed", _last_line(error))
    return Outcome(name, "upgraded" if upgrade else "installed", _hold_closed(ctx, name, plugin_id, sha) or sha[:12])


# 目标提交已落地：stable 指针指向的 artifact 就是 sha（失败安装会回滚指针，回滚后不算落地）。
def _landed(ctx: Context, name: str, plugin_id: str, sha: str) -> bool:
    base = plugins_root() / "cache" / plugin_id.partition("@")[2] / name
    try:
        pointers = read_pointers(base, validate_content=False)
        artifact = None if pointers is None else resolve_pointer(base, pointers.stable, validate_content=False)
    except (OSError, ValueError):
        return False
    if artifact is None:
        return False
    head = subprocess.run(["git", "-C", str(artifact), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    return head == sha


# 所有安装与升级的成功路径（installed、restart-required、失败但目标已落地）都经这一个入口写"包声明的 needs"，
# 绑定目标提交；index 取不到时不写（旧声明因提交不一致自然失效）。返回声明的 needs，没有 index 返回 None。
def _declare(ctx: Context, name: str, plugin_id: str, sha: str) -> list[str] | None:
    try:
        needs = list(load_index(ctx.repo, plugins_root()).packages[name].needs)
    except (RuntimeError, KeyError):
        return None
    record_declared(ctx.workspace, plugin_id, needs, sha)
    return needs


# 前置条件以将来运行插件的 Core 为准：在线时问 Core（pkg 可能跑在 PATH 不同的宿主 Shell 里），
# 离线时本进程就是将要启动的运行环境。实例已不在线（退出码 3）或运行中的 Core 不认识这个方法（4）
# 时退回本进程检测。其余问不到的情况（Core 报错、选中的旧 Gateway 没有这个命令等）不拿调用者的
# 环境顶替：全部当作未确认，包保持关闭，以后启用时由 Core 再检测（plugin-set-enabled 在 Core 里查）。
# 返回 (不满足或未确认的项, 未确认的原因)。RPC 每次最多 32 项，按批询问同一个 Core。
def _unmet_in_core(ctx: Context, needs: list[str]) -> tuple[list[str], str]:
    if not needs or not ctx.online:
        return unmet(needs), ""
    missing: list[str] = []
    for start in range(0, len(needs), 32):
        done = _main(ctx, "plugin-prerequisites", *needs[start:start + 32])
        if done.returncode in (3, 4):
            print(f"运行中的实例没法检测前置条件，改为在当前环境检测: {done.stderr.strip() or done.returncode}", file=sys.stderr)
            return unmet(needs), ""
        if done.returncode != 0:
            return list(needs), _last_line(RuntimeError(done.stderr.strip() or f"退出码 {done.returncode}"))
        missing.extend(str(item) for item in cast(list[object], _json_of(done).get("unmet", [])))
    return missing, ""


# 装完后：包声明了外部前置条件而本机不满足，包照装但保持关闭并记录原因（返回说明）。
# 在线时经 Gateway 热停用（plugin-set-enabled off），离线直接写 patch。
def _hold_closed(ctx: Context, name: str, plugin_id: str, sha: str) -> str:
    needs = _declare(ctx, name, plugin_id, sha)
    if needs is None:
        return ""
    missing, unconfirmed = _unmet_in_core(ctx, needs)
    if not missing:
        clear_disabled(ctx.workspace, plugin_id)
        return ""
    note = f"没能在运行中的实例里确认前置条件（{unconfirmed}），启用时会再检测" if unconfirmed else ""
    record_disabled(ctx.workspace, plugin_id, missing, sha, note)
    if ctx.online:
        # 用 plugin-set-enabled 而不是 plugin-disable：后者在 Agent 回合里被当作内部动作拒绝，
        # Agent 自己跑 pkg install 时包会因此装失败；两者在线时走同一条 Gateway 启停路径。
        done = _main(ctx, "plugin-set-enabled", plugin_id, "off")
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip() or done.stdout.strip() or "plugin-set-enabled off 失败")
    else:
        root = distribution_sources(ctx.workspace, plugins_root(), replacing_distribution=True).root
        _ = set_plugin_choice(ctx.workspace, plugin_id, enabled=False, distribution=root)
    return f"已安装但未启用：{note or describe(missing)}"


# 按名字安装：pkg 只管官方仓库；其他仓库用 plugin-install --marketplace，这里明确拒绝。
def install_named(ctx: Context, names: list[str]) -> list[Outcome]:
    if not is_official_repo(ctx.repo):
        return [Outcome(name, "failed", f"{ctx.repo} 不是官方 extra 仓库；pkg 只管官方仓库，"
                        "其他仓库请用 plugin-install --marketplace") for name in names]
    try:
        if not any(name.startswith("@") for name in names):
            return _install_planned(ctx, *_dependency_plan(ctx, names))
        names, heads = _expand_groups(ctx, names)
        return install_many(ctx, [identity_of(name) for name in names], heads)
    except (ValueError, RuntimeError) as error:
        return [Outcome(name, "failed", str(error)) for name in names]


# 参数里出现 @组 时读 index，展开成成员并补上依赖闭包（依赖在前）；没有 @ 参数不读 index，行为与此前相同。
def _expand_groups(ctx: Context, names: list[str]) -> tuple[list[str], dict[str, str] | None]:
    if not any(name.startswith("@") for name in names):
        return names, None
    index = load_index(ctx.repo, plugins_root())
    if index.note:
        print(index.note, file=sys.stderr)
    full, added = expand(index, names)
    if added:
        print(f"自动带上依赖: {', '.join(added)}", file=sys.stderr)
    # index 可能过期（缓存回落，或 extra 发布时 release 先于 index 前进）：展开的每个包都要在远端真实存在，
    # 否则整体失败、不安装任何包。不带 @ 的多包安装不走这里，保持逐个处理的既有语义。
    heads = list_release_heads(ctx.repo)
    missing = [name for name in full if name not in heads]
    if missing:
        raise RuntimeError("这些包在远端已不存在（index 可能已过期），未安装任何包: " + ", ".join(missing))
    return full, heads


# 不带 @ 的安装也按 index 补上还没装的依赖，像 pacman 一样：返回（依赖在前的安装顺序, 每个包在本次计划里的传递依赖）。
# index 只用来找依赖：取不到、或名字不在 index 里时，那些名字照旧原样安装（逐个处理的既有语义）。
def _dependency_plan(ctx: Context, names: list[str]) -> tuple[list[str], dict[str, set[str]]]:
    try:
        index = load_index(ctx.repo, plugins_root())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"读不到包组信息，不检查依赖: {error}", file=sys.stderr)
        return names, {}
    if index.note:
        print(index.note, file=sys.stderr)
    known = [name.partition("@")[0] for name in names if name.partition("@")[0] in index.packages]
    if not known:
        return names, {}
    selected = {item.name for item in read_state(ctx.workspace, ctx.repo).packages if item.selected}
    ordered = [name for name in closure(index, known) if name in known or name not in selected]
    added = [name for name in ordered if name not in known]
    if added:
        print(f"自动带上依赖: {', '.join(added)}", file=sys.stderr)
    plan = set(ordered)
    needs = {name: (set(closure(index, [name])) - {name}) & plan for name in ordered}
    return ordered + [name for name in names if name.partition("@")[0] not in index.packages], needs


# 按计划逐个安装：依赖没装好（失败、装上但因前置条件保持关闭，或要重启才生效）时，依赖它的包不装，
# 免得装上却加载失败。要重启的依赖：当前进程仍是旧合同，提示重启后再装。
def _install_planned(ctx: Context, names: list[str], needs: dict[str, set[str]]) -> list[Outcome]:
    heads = list_release_heads(ctx.repo)
    outcomes: list[Outcome] = []
    not_ready: set[str] = set()
    seen: list[tuple[str, str]] = []
    for number, name in enumerate(names, 1):
        bare = name.partition("@")[0]
        # 逐包进度写 stderr（结果表仍在 stdout 末尾）：设置页按这一行显示"第几个、正在装谁"。
        print(f"({number}/{len(names)}) 正在安装 {bare}", file=sys.stderr, flush=True)
        if blocked := sorted(needs.get(bare, set()) & not_ready):
            restart = any(dict(seen).get(dep) == "restart-required" for dep in blocked)
            hint = "（需要先重启实例，再重新安装）" if restart else ""
            outcomes.append(Outcome(bare, "failed", f"依赖 {', '.join(blocked)} 没有就绪，未安装{hint}"))
            not_ready.add(bare)
            continue
        outcome = install_many(ctx, [identity_of(name)], heads)[0]
        outcomes.append(outcome)
        seen.append((bare, outcome.result))
        if outcome.result in BAD_RESULTS or outcome.result == "restart-required" or outcome.detail.startswith("已安装但未启用"):
            not_ready.add(bare)
    return outcomes


# 逐个安装；只有已被选择、启用且正常的同版本才跳过，其余走安装路径（可复用已有 artifact）。
def install_many(ctx: Context, plugin_ids: list[str], heads: dict[str, str] | None = None) -> list[Outcome]:
    # 组安装把预检用的那份 heads 快照传进来：核对的全集与安装用的是同一份，两次读取之间分支消失不会部分安装。
    heads = list_release_heads(ctx.repo) if heads is None else heads
    state = read_state(ctx.workspace, ctx.repo)
    here = {item.plugin_id: item for item in state.packages if item.source == "extra"}
    outcomes: list[Outcome] = []
    for plugin_id in plugin_ids:
        name = plugin_id.partition("@")[0]
        sha = heads.get(name)
        package = here.get(plugin_id)
        if (provider := absorbed_plugin_ids().get(plugin_id)) is not None:
            outcomes.append(Outcome(name, "failed", f"{plugin_id} 已并入 Core 的 {provider} 插件，不再作为扩展包安装"))
        elif sha is None:
            outcomes.append(Outcome(name, "failed", f"远端没有 release/{name}"))
        elif package is not None and package.head == sha and not package.repair and not _healthy(ctx, package):
            outcomes.append(Outcome(name, "current", "已是该版本，已启用"))
        else:
            outcomes.append(install_one(ctx, state, plugin_id, sha))
    return outcomes


# 来源是配置的 extra 仓库的已装包才参与升级；HEAD 相同、环境完好且正常才报已是最新。
def upgrade_many(ctx: Context, names: list[str]) -> list[Outcome]:
    state = read_state(ctx.workspace, ctx.repo)
    installed = {item.name: item for item in state.packages if item.source == "extra"}
    heads = list_release_heads(ctx.repo)
    outcomes: list[Outcome] = []
    absorbed = absorbed_plugin_ids()
    for name in names or sorted(installed):
        package = installed.get(name)
        # 已并入 Core 的身份不升级：不带名字的 pkg upgrade 直接略过遗留包，点名时明确说明。
        if package is not None and package.plugin_id in absorbed:
            if names:
                outcomes.append(Outcome(name, "failed", f"{package.plugin_id} 已并入 Core 的 {absorbed[package.plugin_id]} 插件，不再升级"))
            continue
        if package is None:
            other = next((item for item in state.packages if item.name == name), None)
            detail = "未安装" if other is None else f"来源不同（{other.source}，目标 {ctx.repo}），不能升级"
            outcomes.append(Outcome(name, "failed", detail))
        elif name not in heads:
            outcomes.append(Outcome(name, "failed", "上游已移除 release 分支"))
        elif heads[name] != package.head or package.repair:
            # 提交没变但依赖环境需要重建（旧 Core 装的、Python 小版本变了等）：同一提交重装，建出环境。
            outcomes.append(install_one(ctx, state, package.plugin_id, heads[name], upgrade=True))
        elif package.resource_repair and package.artifact is not None:
            # 代码与依赖环境都好，只缺资源：直接对已装 artifact 的 resources.json 取回，失败如实报错。
            outcomes.append(_repair_resources(ctx, package))
        elif problem := _healthy(ctx, package):
            outcomes.append(Outcome(name, "inactive", f"版本已是最新，但{problem}；用 pkg install {name} 重试"))
        else:
            outcomes.append(Outcome(name, "current", package.head[:12]))
    return outcomes


# 只修资源：取回已装 artifact 声明的资源（不重装、不碰选择与记录）。
def _repair_resources(ctx: Context, package: InstalledPackage) -> Outcome:
    try:
        repair_resources(cast(Path, package.artifact), ctx.workspace)
    except (ResourceError, OSError) as error:
        return Outcome(package.name, "failed", _last_line(error))
    return Outcome(package.name, "upgraded", "资源已修复")


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
        "retire": ("retire", "在用的内置插件，发行版已不提供且上游不存在，部署后会消失"),
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
            return install_named(ctx, names)
        if command == "upgrade":
            return upgrade_many(ctx, names)
        if command == "remove":
            return remove_many(ctx, names)
        return sync_install(ctx)
    finally:
        if not online:
            publication.release()
            maintenance.release()
