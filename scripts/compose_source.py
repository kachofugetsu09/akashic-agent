#!/usr/bin/env python3
"""把 Core 提交与 plugins.lock 锁定的插件提交合成一个确定性的源提交。

两个仓库在文件层面互不重叠，并集等于原 monorepo 的布局。发布、构建、
runtime checkout 都使用合成提交；同样的两个输入永远得到同一个提交号。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib

LOCK_PATH = "plugins.lock"
COMPOSED_REF_PREFIX = "refs/akashic/composed/"
_COMMIT = re.compile(r"[0-9a-f]{40}")
# 插件仓自身的仓库元数据不进入合成树，Core 一侧同名文件保持权威。
PLUGINS_REPO_LOCAL = (".github/", ".gitignore", ".gitattributes", "README.md", "LICENSE")
_IDENTITY = {
    "GIT_AUTHOR_NAME": "Akashic compose",
    "GIT_AUTHOR_EMAIL": "compose@akashic.invalid",
    "GIT_COMMITTER_NAME": "Akashic compose",
    "GIT_COMMITTER_EMAIL": "compose@akashic.invalid",
}


@dataclass(frozen=True)
class PluginsLock:
    repository: str
    commit: str


@dataclass(frozen=True)
class ComposedSource:
    commit: str
    core_commit: str
    plugins_commit: str | None


def _git(repository: Path, *args: str, env: dict[str, str] | None = None, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True,
        env=env, input=stdin,
    ).stdout.strip()


def _has_commit(repository: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True, check=False,
    )
    return result.returncode == 0


# 读取 Core 提交里的插件锁；拆分前的提交没有锁，返回 None。
def read_plugins_lock(repository: Path, core_commit: str) -> PluginsLock | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{core_commit}:{LOCK_PATH}"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return None
    document = tomllib.loads(result.stdout)
    if set(document) != {"schema_version", "repository", "commit"} or document["schema_version"] != 1:
        raise ValueError(f"{LOCK_PATH} 格式无效")
    repository_url, commit = document["repository"], document["commit"]
    if not isinstance(repository_url, str) or not repository_url:
        raise ValueError(f"{LOCK_PATH} repository 无效")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError(f"{LOCK_PATH} commit 必须是完整 40 位 SHA")
    return PluginsLock(repository_url, commit)


# 把插件提交取进 Core 仓库的对象库；已存在时不访问网络。
def _fetch_plugins(repository: Path, source: str, commit: str) -> None:
    if _has_commit(repository, commit):
        return
    _ = _git(repository, "fetch", "--quiet", "--depth=1", "--no-tags", source, commit)
    if not _has_commit(repository, commit):
        raise RuntimeError(f"插件来源没有提供锁定提交: {source} {commit}")


def _tree_entries(repository: Path, commit: str) -> list[tuple[str, str]]:
    listing = _git(repository, "ls-tree", "-r", "-z", "--full-tree", commit)
    entries: list[tuple[str, str]] = []
    for item in listing.split("\0"):
        if not item:
            continue
        meta, path = item.split("\t", 1)
        entries.append((meta, path))
    return entries


# 在临时 index 里取两棵树的并集；任何同名路径都说明划分出错，直接拒绝。
def _union_tree(repository: Path, core_commit: str, plugins_commit: str) -> str:
    core_paths = {path for _, path in _tree_entries(repository, core_commit)}
    plugin_entries = [
        (meta, path) for meta, path in _tree_entries(repository, plugins_commit)
        if not path.startswith(PLUGINS_REPO_LOCAL)
    ]
    overlap = sorted(core_paths.intersection(path for _, path in plugin_entries))
    if overlap:
        raise RuntimeError("Core 与插件仓库存在同名路径，拒绝合成: " + ", ".join(overlap[:20]))
    with tempfile.TemporaryDirectory(prefix="akashic-compose-") as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}
        _ = _git(repository, "read-tree", f"{core_commit}^{{tree}}", env=env)
        index_info = "".join(f"{meta}\t{path}\0" for meta, path in plugin_entries)
        _ = _git(repository, "update-index", "-z", "--index-info", env=env, stdin=index_info)
        return _git(repository, "write-tree", env=env)


# 合成提交：固定身份、无父提交、说明里记录两个来源，便于回读。
def compose_source(
    repository: Path, core_commit: str, *,
    plugins_source: str | None = None, plugins_commit: str | None = None,
) -> ComposedSource:
    repository = repository.resolve(strict=True)
    core_commit = _git(repository, "rev-parse", "--verify", f"{core_commit}^{{commit}}")
    lock = read_plugins_lock(repository, core_commit)
    if lock is None and plugins_commit is None:
        return ComposedSource(core_commit, core_commit, None)
    target = plugins_commit or (lock.commit if lock is not None else "")
    if _COMMIT.fullmatch(target) is None:
        raise ValueError("插件提交必须是完整 40 位 SHA")
    source = plugins_source or (lock.repository if lock is not None else "")
    if not source:
        raise ValueError("缺少插件来源仓库")
    _fetch_plugins(repository, source, target)
    tree = _union_tree(repository, core_commit, target)
    message = f"Akashic source\n\ncore: {core_commit}\nplugins: {target}\n"
    # 时间取 Core 提交的提交时间：仍由输入唯一决定，同时让构建产物的 mtime 有意义。
    date = _git(repository, "show", "-s", "--format=%cI", core_commit)
    env = {**os.environ, **_IDENTITY, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    composed = _git(repository, "commit-tree", tree, env=env, stdin=message)
    # 引用让本地 file:// 取用与 gc 都能看到合成提交。
    _ = _git(repository, "update-ref", COMPOSED_REF_PREFIX + composed, composed)
    return ComposedSource(composed, core_commit, target)


# Core 只能锁定插件仓库主分支上的提交：分支上的提交被 squash 或删除后，旧版本就无法重建。
def require_on_branch(repository: Path, source: str, commit: str, branch: str) -> None:
    reference = f"refs/akashic/plugins-{branch}"
    _ = _git(repository, "fetch", "--quiet", "--no-tags", source, f"+refs/heads/{branch}:{reference}")
    result = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", commit, reference],
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{LOCK_PATH} 锁定的插件提交不在插件仓库 {branch} 分支上: {commit}")


# 从合成提交的说明回读来源；普通 Core 提交返回 None。
def composed_parts(repository: Path, commit: str) -> tuple[str, str] | None:
    message = _git(repository, "show", "-s", "--format=%B", commit)
    match = re.fullmatch(r"Akashic source\n\ncore: ([0-9a-f]{40})\nplugins: ([0-9a-f]{40})\n?", message)
    return None if match is None else (match.group(1), match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--commit", default="HEAD", help="Core 提交")
    parser.add_argument("--plugins-source", help="插件仓库 URL 或本地路径；默认取 plugins.lock")
    parser.add_argument("--plugins-commit", help="覆盖 plugins.lock 的插件提交")
    parser.add_argument("--checkout", type=Path, help="把合成树检出到这个新目录")
    parser.add_argument("--require-on-branch", help="要求 plugins.lock 的提交已进入插件仓库这个分支")
    args = parser.parse_args()
    if args.require_on_branch is not None:
        lock = read_plugins_lock(args.repository.resolve(), args.commit)
        if lock is not None:
            require_on_branch(args.repository.resolve(), args.plugins_source or lock.repository,
                              lock.commit, args.require_on_branch)
    result = compose_source(args.repository, args.commit, plugins_source=args.plugins_source,
                            plugins_commit=args.plugins_commit)
    if args.checkout is not None:
        _ = _git(args.repository.resolve(), "worktree", "add", "--detach", str(args.checkout), result.commit)
    print(result.commit)


if __name__ == "__main__":
    main()
