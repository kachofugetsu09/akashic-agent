#!/usr/bin/env python3
"""把 Core 提交与 plugins.lock 锁定的插件提交合成一个确定性的源提交。

两个仓库在文件层面互不重叠，并集等于原 monorepo 的布局。发布、构建、
runtime checkout 都使用合成提交；同样的输入永远得到同一个提交号。

实例可以把单个插件覆盖成插件仓库 main 上更新的提交：合成时只替换这个插件拥有的子树，
覆盖写进提交说明，锁定提交追上之后自动退役。
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib

LOCK_PATH = "plugins.lock"
COMPOSED_REF_PREFIX = "refs/akashic/composed/"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_PLUGIN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MESSAGE = re.compile(
    r"Akashic source\n\ncore: ([0-9a-f]{40})\nplugins: ([0-9a-f]{40})"
    r"((?:\noverride [A-Za-z0-9][A-Za-z0-9._-]*: [0-9a-f]{40})*)\n?"
)
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
    overrides: dict[str, str] = field(default_factory=dict)


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
# 一个插件在插件仓库里拥有的子树：后端目录与前端源码目录。
def _owned_prefixes(name: str) -> tuple[str, str]:
    return (f"plugins/{name}/", f"frontend/plugins/{name}/")


# 锁定提交的插件树，覆盖插件的子树换成覆盖提交里的版本；只能覆盖两边都有的插件。
def _plugin_entries(repository: Path, plugins_commit: str, overrides: Mapping[str, str]) -> list[tuple[str, str]]:
    def owns(path: str, name: str) -> bool:
        return path.startswith(_owned_prefixes(name))

    locked = [(meta, path) for meta, path in _tree_entries(repository, plugins_commit)
              if not path.startswith(PLUGINS_REPO_LOCAL)]
    entries = [(meta, path) for meta, path in locked if not any(owns(path, name) for name in overrides)]
    for name, commit in sorted(overrides.items()):
        owned = [(meta, path) for meta, path in _tree_entries(repository, commit) if owns(path, name)]
        for tree, label in ((locked, "锁定提交"), (owned, "覆盖提交")):
            if not any(path.startswith(f"plugins/{name}/") for _, path in tree):
                raise RuntimeError(f"{label}里没有这个插件: {name}")
        entries.extend(owned)
    return entries


def _union_tree(repository: Path, core_commit: str, plugins_commit: str,
                overrides: Mapping[str, str] | None = None) -> str:
    core_paths = {path for _, path in _tree_entries(repository, core_commit)}
    plugin_entries = _plugin_entries(repository, plugins_commit, overrides or {})
    overlap = sorted(core_paths.intersection(path for _, path in plugin_entries))
    if overlap:
        raise RuntimeError("Core 与插件仓库存在同名路径，拒绝合成: " + ", ".join(overlap[:20]))
    with tempfile.TemporaryDirectory(prefix="akashic-compose-") as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}
        _ = _git(repository, "read-tree", f"{core_commit}^{{tree}}", env=env)
        index_info = "".join(f"{meta}\t{path}\0" for meta, path in plugin_entries)
        _ = _git(repository, "update-index", "-z", "--index-info", env=env, stdin=index_info)
        return _git(repository, "write-tree", env=env)


# 合成提交：固定身份、无父提交、说明里记录全部来源，便于回读。
def compose_source(
    repository: Path, core_commit: str, *,
    plugins_source: str | None = None, plugins_commit: str | None = None,
    overrides: Mapping[str, str] | None = None,
) -> ComposedSource:
    repository = repository.resolve(strict=True)
    core_commit = _git(repository, "rev-parse", "--verify", f"{core_commit}^{{commit}}")
    lock = read_plugins_lock(repository, core_commit)
    overrides = dict(overrides or {})
    if lock is None and plugins_commit is None:
        if overrides:
            raise ValueError("目标 Core 没有 plugins.lock，不能覆盖单个插件")
        return ComposedSource(core_commit, core_commit, None)
    target = plugins_commit or (lock.commit if lock is not None else "")
    if _COMMIT.fullmatch(target) is None:
        raise ValueError("插件提交必须是完整 40 位 SHA")
    source = plugins_source or (lock.repository if lock is not None else "")
    if not source:
        raise ValueError("缺少插件来源仓库")
    for name, commit in overrides.items():
        if _PLUGIN_NAME.fullmatch(name) is None or _COMMIT.fullmatch(commit) is None:
            raise ValueError(f"插件覆盖必须是 名称@40位提交: {name}@{commit}")
    for commit in {target, *overrides.values()}:
        _fetch_plugins(repository, source, commit)
    tree = _union_tree(repository, core_commit, target, overrides)
    lines = "".join(f"override {name}: {commit}\n" for name, commit in sorted(overrides.items()))
    message = f"Akashic source\n\ncore: {core_commit}\nplugins: {target}\n{lines}"
    # 时间取 Core 提交的提交时间：仍由输入唯一决定，同时让构建产物的 mtime 有意义。
    date = _git(repository, "show", "-s", "--format=%cI", core_commit)
    env = {**os.environ, **_IDENTITY, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    # 固定提交编码：操作者的 i18n.commitEncoding 会写进提交对象，改变合成提交号。
    composed = _git(repository, "-c", "i18n.commitEncoding=UTF-8", "commit-tree", tree, env=env, stdin=message)
    # 引用让本地 file:// 取用与 gc 都能看到合成提交。
    _ = _git(repository, "update-ref", COMPOSED_REF_PREFIX + composed, composed)
    return ComposedSource(composed, core_commit, target, overrides)


# 插件仓库 main 的提交关系；只取提交和树，不取文件内容，也不碰调用方仓库的浅克隆边界。
@contextmanager
def _plugins_history(source: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="akashic-plugins-history-") as directory:
        repository = Path(directory)
        _ = _git(repository, "init", "--quiet", "--bare")
        _ = _git(repository, "-c", "gc.auto=0", "fetch", "--quiet", "--no-tags", "--filter=blob:none",
                 source, "+refs/heads/main:refs/heads/main")
        yield repository


def _is_ancestor(repository: Path, commit: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", commit, descendant],
        capture_output=True, check=False,
    )
    return result.returncode == 0


# 覆盖只接受插件仓库 main 上的提交，所以和锁定提交总在同一条历史上：锁定提交已包含它时退役，返回 (保留, 退役)。
def settle_overrides(repository: Path, core_commit: str,
                     overrides: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    if not overrides:
        return {}, {}
    lock = read_plugins_lock(repository, core_commit)
    if lock is None:
        raise ValueError("目标 Core 没有 plugins.lock，不能覆盖单个插件")
    kept: dict[str, str] = {}
    retired: dict[str, str] = {}
    with _plugins_history(lock.repository) as history:
        for name, commit in sorted(overrides.items()):
            if not _is_ancestor(history, commit, "refs/heads/main"):
                raise RuntimeError(f"插件覆盖提交不在插件仓库 main 上: {name}@{commit}")
            (retired if _is_ancestor(history, commit, lock.commit) else kept)[name] = commit
    return kept, retired


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


def _composed_message(repository: Path, commit: str) -> re.Match[str] | None:
    return _MESSAGE.fullmatch(_git(repository, "show", "-s", "--format=%B", commit))


# 从合成提交的说明回读来源；普通 Core 提交返回 None。
def composed_parts(repository: Path, commit: str) -> tuple[str, str] | None:
    match = _composed_message(repository, commit)
    return None if match is None else (match.group(1), match.group(2))


# 合成提交里的实例覆盖：插件名 -> 插件仓库提交。
def composed_overrides(repository: Path, commit: str) -> dict[str, str]:
    match = _composed_message(repository, commit)
    if match is None:
        return {}
    pairs = (line.removeprefix("override ").split(": ") for line in match.group(3).splitlines() if line)
    return {name: value for name, value in pairs}


# 命令行写法 name@sha。
def parse_override(value: str) -> tuple[str, str]:
    name, _, commit = value.partition("@")
    if _PLUGIN_NAME.fullmatch(name) is None or _COMMIT.fullmatch(commit) is None:
        raise ValueError(f"插件覆盖必须是 名称@40位提交: {value}")
    return name, commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--commit", default="HEAD", help="Core 提交")
    parser.add_argument("--plugins-source", help="插件仓库 URL 或本地路径；默认取 plugins.lock")
    parser.add_argument("--plugins-commit", help="覆盖 plugins.lock 的插件提交")
    parser.add_argument("--override", action="append", default=[], metavar="NAME@SHA",
                        help="把单个插件换成插件仓库这个提交里的版本；可重复")
    parser.add_argument("--checkout", type=Path, help="把合成树检出到这个新目录")
    parser.add_argument("--require-on-branch", help="要求 plugins.lock 的提交已进入插件仓库这个分支")
    args = parser.parse_args()
    if args.require_on_branch is not None:
        lock = read_plugins_lock(args.repository.resolve(), args.commit)
        if lock is not None:
            require_on_branch(args.repository.resolve(), args.plugins_source or lock.repository,
                              lock.commit, args.require_on_branch)
    overrides = dict(parse_override(value) for value in args.override)
    result = compose_source(args.repository, args.commit, plugins_source=args.plugins_source,
                            plugins_commit=args.plugins_commit, overrides=overrides)
    if args.checkout is not None:
        _ = _git(args.repository.resolve(), "worktree", "add", "--detach", str(args.checkout), result.commit)
    print(result.commit)


if __name__ == "__main__":
    main()
