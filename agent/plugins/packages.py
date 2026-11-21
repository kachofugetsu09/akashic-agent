"""包管理器只读命令：列出已装包、对比远端 release 分支、预演同步；不修改选择或 cache。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from agent.plugins.bundles import composition_rows
from agent.plugins.distribution_sources import distribution_sources
from agent.plugins.manifest import plugins_root
from agent.plugins.selection import PluginSelection, SelectionFormatError
from agent.plugins.source_resolver import ResolvedPluginSource, resolve_plugin_sources
from agent.plugins.static_manifest import load_static_plugin_manifest, read_only_manifest_facts

DEFAULT_EXTRA_REPO = "https://github.com/kachofugetsu09/akashic-plugins.git"
_BRANCH_PREFIX = "refs/heads/release/"
_GIT_TIMEOUT_S = 60


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


# 来源仓库：--repo 优先，其次环境变量，最后是默认常量。
def resolve_repo(option: str | None) -> str:
    return option or os.environ.get("AKASHIC_EXTRA_REPO", "").strip() or DEFAULT_EXTRA_REPO


# 比较仓库地址时忽略末尾的斜杠和 .git。
def _normalize_repo(url: str) -> str:
    return url.strip().rstrip("/").removesuffix(".git")


# 一次 ls-remote 取全部 release/<name> 分支，返回 name 到提交 SHA 的映射。
def list_release_heads(repo: str) -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "--", repo, _BRANCH_PREFIX + "*"],
        capture_output=True, text=True, check=False, timeout=_GIT_TIMEOUT_S,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-remote 失败: {repo}\n{result.stderr.strip()}")
    heads: dict[str, str] = {}
    for line in result.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.startswith(_BRANCH_PREFIX) and len(sha) == 40:
            heads[ref.removeprefix(_BRANCH_PREFIX)] = sha
    return heads


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
    return "extra" if _normalize_repo(origin) == _normalize_repo(repo) else f"git:{origin}"


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
class _State:
    packages: list[InstalledPackage]
    provided: frozenset[str]
    installed: frozenset[str]


# 已装包 = 当前选择里的输入，加上 cache 里尚未被选择引用的 artifact；
# provided 是目标发行版当前自带的身份，installed 是 cache 里可用的外置 artifact 身份。
def read_state(workspace: Path, repo: str) -> _State:
    home = plugins_root()
    distribution = distribution_sources(workspace, home)
    provided = frozenset(_source_id(source) for source in distribution.sources)
    ignored = {root.resolve() for root in distribution.ignored_installed_roots}
    packages: dict[str, InstalledPackage] = {}
    for record in _selected_inputs(workspace):
        plugin_id = cast(str, record["plugin_id"])
        builtin = record["source_type"] == "builtin"
        retired = builtin and bool(distribution.sources) and plugin_id not in provided
        packages[plugin_id] = _package(plugin_id, Path(cast(str, record["code"])), builtin, repo, True, retired)
    installed: set[str] = set()
    for source in resolve_plugin_sources(installed_cache_root=home / "cache"):
        if source.plugin_root.resolve() in ignored:
            continue
        plugin_id = _source_id(source)
        installed.add(plugin_id)
        if plugin_id not in packages:
            packages[plugin_id] = _package(plugin_id, source.plugin_root, False, repo, False)
    ordered = sorted(packages.values(), key=lambda item: item.plugin_id)
    return _State(ordered, provided, frozenset(installed))


# 应装集合 = 启用的组合行 - 发行版当前自带的 - cache 里已装的；旧选择里的内置输入不算数。
# 没有发行版时 Core 只有 patch 行（composition_rows 的既有语义）。
def sync_rows(workspace: Path, repo: str) -> list[SyncRow]:
    state = read_state(workspace, repo)
    present = state.provided | state.installed
    missing = [row.plugin for row in composition_rows(workspace)
               if not row.disabled and row.plugin not in present]
    if not missing:
        return []
    heads = list_release_heads(repo)
    return [SyncRow(plugin_id, "available" if plugin_id.partition("@")[0] in heads else "missing-upstream")
            for plugin_id in missing]


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
    if as_json:
        print(json.dumps({"repo": repo, "to_install": [asdict(row) for row in rows]}, ensure_ascii=False, indent=2))
        return
    for row in rows:
        print(f"{row.plugin_id:<32} {row.upstream}")
    if not rows:
        print("无需安装")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pkg", description="包管理器（只读命令）")
    common = argparse.ArgumentParser(add_help=False)
    _ = common.add_argument("--repo", help="extra 仓库地址，默认取 AKASHIC_EXTRA_REPO 或内置常量")
    _ = common.add_argument("--json", action="store_true", help="输出 JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="列出已装包", parents=[common])
    _ = listing.add_argument("--remote", action="store_true", help="对比远端 release/* 分支")
    sync = commands.add_parser("sync", help="求应装集合", parents=[common])
    _ = sync.add_argument("--dry-run", action="store_true", help="只打印，不做修改（目前必须带）")
    return parser


# 命令入口：参数错误由 argparse 以退出码 2 结束，远端不可达以退出码 1 结束。
def pkg_main(arguments: list[str], *, workspace: Path) -> int:
    options = _parser().parse_args(arguments)
    repo = resolve_repo(cast(str | None, options.repo))
    try:
        with read_only_manifest_facts():
            return _dispatch(options, workspace, repo)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 1


def _dispatch(options: argparse.Namespace, workspace: Path, repo: str) -> int:
    if options.command == "list":
        remote = bool(options.remote)
        _print_list(list_rows(read_state(workspace, repo).packages, repo, remote), remote, bool(options.json))
        return 0
    if not options.dry_run:
        print("pkg sync 目前只支持 --dry-run", file=sys.stderr)
        return 2
    _print_sync(sync_rows(workspace, repo), repo, bool(options.json))
    return 0
