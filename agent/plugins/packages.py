"""包管理器只读命令：列出已装包、对比远端 release 分支、预演同步；不修改选择或 cache。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from agent.plugins.bundles import composition_rows, preset_plugin_ids
from agent.plugins.distribution_sources import distribution_sources
from agent.plugins.extra_repo import DEFAULT_EXTRA_REPO as DEFAULT_EXTRA_REPO
from agent.plugins.extra_repo import classify_gaps, list_release_heads as list_release_heads
from agent.plugins.extra_repo import resolve_repo as resolve_repo
from agent.plugins.install import is_official_fresh_install, is_same_source_upgrade, same_origin
from agent.plugins.manifest import plugins_root
from agent.plugins.selection import PluginSelection, SelectionFormatError
from agent.plugins.source_resolver import ResolvedPluginSource, resolve_plugin_sources
from agent.plugins.static_manifest import load_static_plugin_manifest, read_only_manifest_facts


@dataclass(frozen=True)
class InstalledPackage:
    plugin_id: str
    name: str
    version: str
    source: str
    head: str
    selected: bool
    retired: bool = False


@dataclass(frozen=True)
class ListRow:
    plugin_id: str
    version: str
    source: str
    head: str
    selected: bool
    retired: bool
    status: str
    remote_head: str


@dataclass(frozen=True)
class SyncRow:
    plugin_id: str
    upstream: str
    action: str = "install"


# 在 artifact 目录里问 git；目录自己不是仓库根时不向上查找，避免读到宿主源码仓库。
def _git(artifact: Path, *args: str) -> str:
    if not (artifact / ".git").exists():
        return ""
    result = subprocess.run(["git", "--no-optional-locks", "-C", str(artifact), *args],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _version_of(artifact: Path) -> str:
    try:
        return load_static_plugin_manifest(artifact).version
    except (OSError, ValueError):
        return "?"


# 来源由 artifact 的 git origin 推导：core、extra、git:<url>。
def _source_of(artifact: Path, builtin: bool, repo: str) -> str:
    if builtin:
        return "core"
    origin = _git(artifact, "config", "--get", "remote.origin.url")
    if not origin:
        return "git:unknown"
    return "extra" if same_origin(origin, repo) else f"git:{origin}"


def _package(plugin_id: str, artifact: Path, builtin: bool, repo: str, selected: bool,
             retired: bool = False) -> InstalledPackage:
    return InstalledPackage(plugin_id, plugin_id.partition("@")[0], _version_of(artifact),
                            _source_of(artifact, builtin, repo), _git(artifact, "rev-parse", "HEAD") or "-",
                            selected, retired)


# 选择文件缺失（从未启动过的 workspace）等同空选择。
def _selected_inputs(workspace: Path) -> list[dict[str, object]]:
    selection = PluginSelection(workspace)
    try:
        reference = selection.read()
    except SelectionFormatError:
        if selection.path.exists():
            raise
        return []
    if reference is None:
        return []
    return [dict(selection.read_input(identity)) for identity in selection.components(reference)]


# 与 Core 一致：内置来源有 marketplace 时身份是 name@marketplace，否则就是裸 name。
def _source_id(source: ResolvedPluginSource) -> str:
    return f"{source.plugin_name}@{source.marketplace}" if source.marketplace else source.plugin_name


@dataclass(frozen=True)
class PackageState:
    packages: list[InstalledPackage]
    provided: frozenset[str]
    installed: frozenset[str]
    adopted: frozenset[str]
    marketplaces: frozenset[str]
    root: Path | None = None
    workspace: Path = Path()
    repo: str = ""

    # 发行版自带、历史归属、旧选择里的内置输入，以及发行版命名空间里既不是同源升级、
    # 也不是官方仓库全新安装的身份，只能经换来源操作处理；与安装器守卫共用同两个判定函数。
    def requires_switch(self, plugin_id: str) -> bool:
        builtin = {item.plugin_id for item in self.packages if item.source == "core"}
        if plugin_id in self.adopted or plugin_id in builtin or plugin_id in self.provided:
            return True
        if plugin_id.partition("@")[2] not in self.marketplaces:
            return False
        return not (is_same_source_upgrade(self.workspace, plugin_id, self.repo)
                    or is_official_fresh_install(self.workspace, plugin_id, self.repo))


# 已装包 = 当前选择里的输入，加上 cache 里尚未被选择引用的 artifact；
# provided 是目标发行版当前自带的身份，installed 是 cache 里可用的外置 artifact 身份。
def read_state(workspace: Path, repo: str) -> PackageState:
    home = plugins_root()
    # 与目标发行版比较：旧选择的来源路径可能已换成新源码或已不存在，只核对归属声明，
    # 不要求旧路径仍由旧发行版占据；cache、数据身份和内容摘要的核对不变。
    distribution = distribution_sources(workspace, home, replacing_distribution=True)
    provided = frozenset(_source_id(source) for source in distribution.sources)
    ignored = {root.resolve() for root in distribution.ignored_installed_roots}
    packages: dict[str, InstalledPackage] = {}
    for record in _selected_inputs(workspace):
        plugin_id = cast(str, record["plugin_id"])
        builtin = record["source_type"] == "builtin"
        retired = builtin and bool(distribution.sources) and plugin_id not in provided
        packages[plugin_id] = _package(plugin_id, Path(cast(str, record["code"])), builtin, repo, True, retired)
    marketplaces = frozenset(source.marketplace for source in distribution.sources if source.marketplace)
    installed: set[str] = set()
    for source in resolve_plugin_sources(installed_cache_root=home / "cache"):
        if source.plugin_root.resolve() in ignored:
            continue
        plugin_id = _source_id(source)
        # 发行版已不提供的旧内置输入让位给 cache 里的新安装。
        if plugin_id not in packages or packages[plugin_id].retired:
            packages[plugin_id] = _package(plugin_id, source.plugin_root, False, repo, False)
        # 官方命名空间里只有来源是 extra 仓库的 artifact 才算"已装"，旧内置遗留制品不能掩盖缺口；
        # 其他 marketplace（github、local 等）的 artifact 照常算已装。
        official = plugin_id.partition("@")[2] in marketplaces
        if not official or _source_of(source.plugin_root, False, repo) == "extra":
            installed.add(plugin_id)
    ordered = sorted(packages.values(), key=lambda item: item.plugin_id)
    return PackageState(ordered, provided, frozenset(installed), distribution.legacy_ids, marketplaces, distribution.root,
                        workspace, repo)


# 应装集合 = 启用的组合行 - 发行版当前自带的 - cache 里已装的；旧选择里的内置输入不算数。
# 没有发行版时 Core 只有 patch 行（composition_rows 的既有语义）。分类与部署清单共用 classify_gaps。
def sync_rows(workspace: Path, repo: str) -> list[SyncRow]:
    state = read_state(workspace, repo)
    preset: frozenset[str] = preset_plugin_ids(state.root) if state.root is not None else frozenset[str]()
    gaps = classify_gaps(
        enabled=[row.plugin for row in composition_rows(workspace) if not row.disabled],
        present=state.provided | state.installed, preset=preset,
        heads=lambda: list_release_heads(repo), needs_switch=state.requires_switch,
        in_use=frozenset(item.plugin_id for item in state.packages if item.source == "core" and item.selected))
    return [SyncRow(gap.plugin_id, gap.upstream, gap.action) for gap in gaps]


# 本地列表；带 --remote 时只给 extra 来源的包标状态，其余标 "-"。
def list_rows(packages: list[InstalledPackage], repo: str, remote: bool) -> list[ListRow]:
    heads = list_release_heads(repo) if remote else {}
    rows: list[ListRow] = []
    for item in packages:
        status, remote_head = "-", ""
        if remote and item.source == "extra":
            remote_head = heads.get(item.name, "")
            status = ("removed-upstream" if not remote_head
                      else "current" if remote_head == item.head else "upgradable")
        rows.append(ListRow(item.plugin_id, item.version, item.source, item.head, item.selected,
                            item.retired, status, remote_head))
    return rows


def _short(sha: str) -> str:
    return sha[:12] if sha else "-"


def _print_list(rows: list[ListRow], remote: bool, as_json: bool) -> None:
    if as_json:
        print(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2))
        return
    for row in rows:
        line = f"{row.plugin_id:<32} {row.version:<10} {row.source:<10} {_short(row.head)}"
        if remote:
            line += f"  {row.status} {_short(row.remote_head)}"
        marks = ("" if row.selected else "  [已装，未进选择]") + ("  [发行版已不提供]" if row.retired else "")
        print(line + marks)
    if not rows:
        print("没有已装的包")


def _print_sync(rows: list[SyncRow], repo: str, as_json: bool) -> None:
    groups = {action: [row for row in rows if row.action == action]
              for action in ("install", "switch", "retire", "optional", "retired")}
    if as_json:
        keys = {"install": "to_install", "switch": "to_switch", "retire": "to_retire", "optional": "optional",
                "retired": "retired"}
        # to_retire 只在有在用的内置身份将因上游不存在而消失时才出现，其余输出与此前一致。
        print(json.dumps({"repo": repo, **{keys[a]: [asdict(row) for row in items] for a, items in groups.items()
                                           if items or a != "retire"}},
                         ensure_ascii=False, indent=2))
        return
    titles = {"install": "", "switch": "需要换来源（pkg sync 不处理，需专门的换来源操作）：",
              "retire": "在用的内置插件发行版已不提供、上游也不存在（部署后会消失）：",
              "optional": "只在用户 patch 里启用、上游存在（不自动安装，需要请 pkg install）：",
              "retired": "已退役（只在用户 patch 里启用，上游已不存在，忽略）："}
    for action, items in groups.items():
        if items and titles[action]:
            print(titles[action])
        for row in items:
            print(f"{row.plugin_id:<32} {row.upstream}")
    if not any(groups["install"] or groups["switch"]):
        print("无需安装")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pkg", description="包管理器")
    common = argparse.ArgumentParser(add_help=False)
    _ = common.add_argument("--repo", help="extra 仓库地址，默认取 AKASHIC_EXTRA_REPO 或内置常量")
    _ = common.add_argument("--json", action="store_true", help="输出 JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="列出已装包", parents=[common])
    _ = listing.add_argument("--remote", action="store_true", help="对比远端 release/* 分支")
    sync = commands.add_parser("sync", help="安装应装集合", parents=[common])
    _ = sync.add_argument("--dry-run", action="store_true", help="只打印，不做修改")
    _ = sync.add_argument("--timeout", type=float, default=120.0, help="在线等待每个包到终态的秒数")
    for name, help_text in (("install", "安装包"), ("upgrade", "升级 extra 来源的包"), ("remove", "卸载包，保留数据")):
        command = commands.add_parser(name, help=help_text, parents=[common])
        _ = command.add_argument("names", nargs="*" if name == "upgrade" else "+")
        _ = command.add_argument("--timeout", type=float, default=120.0, help="在线等待每个包到终态的秒数")
    return parser


# 命令入口：参数错误由 argparse 以退出码 2 结束；远端不可达或有包失败以退出码 1 结束。
def pkg_main(arguments: list[str], *, workspace: Path, config_path: Path) -> int:
    options = _parser().parse_args(arguments)
    repo = resolve_repo(cast(str | None, options.repo))
    try:
        if options.command in {"install", "upgrade", "remove"} or (options.command == "sync" and not options.dry_run):
            return _run_write(options, workspace, config_path, repo)
        with read_only_manifest_facts():
            return _dispatch(options, workspace, repo)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 1


def _run_write(options: argparse.Namespace, workspace: Path, config_path: Path, repo: str) -> int:
    from agent.plugins.package_ops import BAD_RESULTS, run_write

    outcomes = run_write(cast(str, options.command), cast(list[str], getattr(options, "names", [])),
                         workspace=workspace, config_path=config_path, repo=repo,
                         timeout=cast(float, options.timeout))
    if options.json:
        print(json.dumps([asdict(item) for item in outcomes], ensure_ascii=False, indent=2))
    else:
        for item in outcomes:
            print(f"{item.name:<28} {item.result:<17} {item.detail}".rstrip())
        if not outcomes:
            print("无需处理")
    return 1 if any(item.result in BAD_RESULTS for item in outcomes) else 0


def _dispatch(options: argparse.Namespace, workspace: Path, repo: str) -> int:
    if options.command == "list":
        remote = bool(options.remote)
        _print_list(list_rows(read_state(workspace, repo).packages, repo, remote), remote, bool(options.json))
        return 0
    _print_sync(sync_rows(workspace, repo), repo, bool(options.json))
    return 0
