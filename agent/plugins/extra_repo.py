"""extra 仓库的来源解析与缺口分类；只依赖标准库，宿主部署脚本与包管理命令共用。"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

DEFAULT_EXTRA_REPO = "https://github.com/kachofugetsu09/akashic-plugins.git"
BRANCH_PREFIX = "refs/heads/release/"
_GIT_TIMEOUT_S = 60

GapAction = Literal["install", "switch", "optional", "retired"]


@dataclass(frozen=True)
class SelectedPackage:
    plugin_id: str
    builtin: bool
    origin: str
    head: str


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


# 来源仓库：显式参数优先，其次环境变量，最后是默认常量。
def resolve_repo(option: str | None) -> str:
    return option or os.environ.get("AKASHIC_EXTRA_REPO", "").strip() or DEFAULT_EXTRA_REPO


# 两个仓库地址是否同一来源：只去首尾空白与末尾 /（pkg、安装器守卫与部署清单共用这一口径）。
def same_origin(left: str, right: str) -> bool:
    return left.strip().rstrip("/") == right.strip().rstrip("/")


# 一次 ls-remote 取全部 release/<name> 分支，返回 name 到提交 SHA 的映射。
def list_release_heads(repo: str) -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "--", repo, BRANCH_PREFIX + "*"],
        capture_output=True, text=True, check=False, timeout=_GIT_TIMEOUT_S,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-remote 失败: {repo}\n{result.stderr.strip()}")
    heads: dict[str, str] = {}
    for line in result.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.startswith(BRANCH_PREFIX) and len(sha) == 40:
            heads[ref.removeprefix(BRANCH_PREFIX)] = sha
    return heads


# 缺口分类，pkg sync 与部署清单共用：preset 启用的行才是默认组合的承诺，缺了就要装（或需要换来源）；
# 只在用户 patch 里启用的行，上游不存在归为已退役，上游存在只提示、不自动装。
# heads 是惰性的：没有缺口时不联网。
def classify_gaps(
    *, enabled: Iterable[str], present: frozenset[str], preset: frozenset[str],
    heads: Callable[[], Mapping[str, str]], needs_switch: Callable[[str], bool],
) -> list[Gap]:
    missing = [plugin_id for plugin_id in enabled if plugin_id not in present]
    if not missing:
        return []
    remote = heads()
    gaps: list[Gap] = []
    for plugin_id in missing:
        upstream = plugin_id.partition("@")[0] in remote
        if plugin_id not in preset:
            action: GapAction = "optional" if upstream else "retired"
        else:
            action = "switch" if needs_switch(plugin_id) else "install"
        gaps.append(Gap(plugin_id, "available" if upstream else "missing-upstream", action))
    return gaps


# 已装且来源就是 extra 仓库、远端有新提交的包（升级）。
def upgrade_actions(
    selected: Mapping[str, SelectedPackage], heads: Mapping[str, str], repo: str,
) -> list[ExtraAction]:
    actions: list[ExtraAction] = []
    for plugin_id, package in sorted(selected.items()):
        remote = heads.get(plugin_id.partition("@")[0], "")
        if (not package.builtin and remote and remote != package.head
                and same_origin(package.origin, repo)):
            actions.append(ExtraAction("upgrade", plugin_id, remote))
    return actions
