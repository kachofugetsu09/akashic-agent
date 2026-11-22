"""extra 仓库的来源解析与缺口分类；只依赖标准库，宿主部署脚本与包管理命令共用。"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_EXTRA_REPO = "https://github.com/kachofugetsu09/akashic-plugins.git"
BRANCH_PREFIX = "refs/heads/release/"
INDEX_REF = "refs/heads/index"
_GIT_TIMEOUT_S = 60

# 容器里取不到宿主环境，求 wheel 的 docker run 按名字透传这些代理变量（证书文件路径在容器里没有意义，不传）。
PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy")
# 会改变 git 定位仓库的变量（见 git(1) Environment Variables）：宿主脚本在临时仓库里 init/fetch/bundle，
# 继承它们会把操作指到别处；其余变量（代理、证书、认证、GIT_CONFIG_*、LANG 等）原样继承，与 pkg 在线发现一致。
_GIT_REPOSITORY_LOCATION = frozenset({
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR", "GIT_NAMESPACE", "GIT_PREFIX", "GIT_IMPLICIT_WORK_TREE", "GIT_CEILING_DIRECTORIES",
    "GIT_GRAFT_FILE", "GIT_SHALLOW_FILE", "GIT_REPLACE_REF_BASE", "GIT_NO_REPLACE_OBJECTS", "GIT_QUARANTINE_PATH",
    "GIT_INTERNAL_SUPER_PREFIX", "GIT_DISCOVERY_ACROSS_FILESYSTEM",
})

GapAction = Literal["install", "switch", "retire", "optional", "retired"]


@dataclass(frozen=True)
class SelectedPackage:
    plugin_id: str
    builtin: bool
    origin: str
    head: str
    # 进程内依赖环境需要重建的原因（environment_repair_reason）；不需要为空。
    repair: str = ""


@dataclass(frozen=True)
class Gap:
    plugin_id: str
    upstream: str
    action: GapAction


@dataclass(frozen=True)
class ExtraAction:
    kind: Literal["install", "switch", "upgrade"]
    plugin_id: str
    commit: str


def git_env() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key not in _GIT_REPOSITORY_LOCATION}
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


# 来源仓库：显式参数优先，其次环境变量，最后是默认常量。
def resolve_repo(option: str | None) -> str:
    return option or os.environ.get("AKASHIC_EXTRA_REPO", "").strip() or DEFAULT_EXTRA_REPO


# 官方 marketplace：目标发行版 distribution.json 声明的那个；没有发行版时无法确定，返回 None。
def official_marketplace() -> str | None:
    configured = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION", "").strip()
    if not configured:
        return None
    report = json.loads((Path(configured) / "distribution.json").read_text(encoding="utf-8"))
    marketplace = report.get("marketplace")
    return marketplace if isinstance(marketplace, str) and marketplace else None


# 来源是否就是配置的官方 extra 仓库（环境变量或默认常量解析出的那个）。
def is_official_repo(repo: str) -> bool:
    return same_origin(repo, resolve_repo(None))


# 两个仓库地址是否同一来源：只去首尾空白与末尾 /（pkg、安装器守卫与部署清单共用这一口径）。
def same_origin(left: str, right: str) -> bool:
    return left.strip().rstrip("/") == right.strip().rstrip("/")


# 一次 ls-remote 取全部 release/<name> 分支与 index 分支：返回 (name 到提交 SHA 的映射, index 的 SHA，没有则空串)。
def list_heads(repo: str) -> tuple[dict[str, str], str]:
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "--", repo, BRANCH_PREFIX + "*", INDEX_REF],
        capture_output=True, text=True, check=False, timeout=_GIT_TIMEOUT_S,
        env=git_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-remote 失败: {repo}\n{result.stderr.strip()}")
    heads: dict[str, str] = {}
    index = ""
    for line in result.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if len(sha) != 40:
            continue
        if ref.startswith(BRANCH_PREFIX):
            heads[ref.removeprefix(BRANCH_PREFIX)] = sha
        elif ref == INDEX_REF:
            index = sha
    return heads, index


# 只要 release/<name> 分支的映射（pkg 的既有调用方）。
def list_release_heads(repo: str) -> dict[str, str]:
    return list_heads(repo)[0]


# 缺口分类，pkg sync 与部署清单共用。判据是"现在是否在用"，不只看 preset：
#   preset 启用的行            -> 换来源（旧选择里是内置）或首装；
#   当前选择里是内置输入（在用）-> 上游存在则换来源，上游不存在是真正的退役（switch / retire），退役会让在运行的能力消失；
#   其余只在用户 patch 里启用的行 -> 上游存在只提示、不自动装（optional），不存在记为已退役并跳过（retired）。
# heads 是惰性的：没有缺口时不联网。
def classify_gaps(
    *, enabled: Iterable[str], present: frozenset[str], preset: frozenset[str],
    heads: Callable[[], Mapping[str, str]], needs_switch: Callable[[str], bool],
    in_use: frozenset[str] = frozenset(),
) -> list[Gap]:
    missing = [plugin_id for plugin_id in enabled if plugin_id not in present]
    if not missing:
        return []
    remote = heads()
    gaps: list[Gap] = []
    for plugin_id in missing:
        upstream = plugin_id.partition("@")[0] in remote
        if plugin_id in preset:
            action: GapAction = "switch" if needs_switch(plugin_id) else "install"
        elif plugin_id in in_use:
            action = "switch" if upstream else "retire"
        else:
            action = "optional" if upstream else "retired"
        gaps.append(Gap(plugin_id, "available" if upstream else "missing-upstream", action))
    return gaps


# 已装且来源就是 extra 仓库、远端有新提交的包（升级）。
def upgrade_actions(
    selected: Mapping[str, SelectedPackage], heads: Callable[[], Mapping[str, str]], repo: str, *, gaps_only: bool = False,
) -> list[ExtraAction]:
    actions: list[ExtraAction] = []
    for plugin_id, package in sorted(selected.items()):
        if package.builtin or not same_origin(package.origin, repo):
            continue
        # 1. 提交落后：升级到上游新提交（新提交自带环境）；启动补装不升级。
        remote = "" if gaps_only else heads().get(plugin_id.partition("@")[0], "")
        if remote and remote != package.head:
            actions.append(ExtraAction("upgrade", plugin_id, remote))
        # 2. 提交没变但进程内环境需要重建（旧 Core 装的、Python 小版本变了等）：同一提交重装，kind 仍是 upgrade。
        elif package.repair and package.head:
            actions.append(ExtraAction("upgrade", plugin_id, package.head))
    return actions
