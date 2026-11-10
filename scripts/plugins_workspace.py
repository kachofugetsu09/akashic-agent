#!/usr/bin/env python3
"""开发工作区：把插件仓库检出到 Core 的同一个工作目录。

两个仓库的文件互不重叠，于是共用一个工作目录后布局与拆分前的 monorepo 相同，
`python main.py`、`npm`、`pytest` 的用法不变。插件仓库的 git 目录放在 Core 的
git 目录里，用 `scripts/plugins-git` 操作；Core 的 git 通过 info/exclude 忽略插件文件，
`scripts/plugins_workspace.py status` 同时列出两侧修改与未归属的新文件。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from scripts.compose_source import LOCK_PATH, PLUGINS_REPO_LOCAL, read_plugins_lock

_BEGIN = "# >>> akashic-plugins (managed by scripts/plugins_workspace.py)"
_END = "# <<< akashic-plugins"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _root() -> Path:
    return Path(_git(_SOURCE_ROOT, "rev-parse", "--show-toplevel"))


def plugins_git_dir(root: Path) -> Path:
    return Path(_git(root, "rev-parse", "--absolute-git-dir")) / "akashic-plugins.git"


def _plugins_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", f"--git-dir={plugins_git_dir(root)}", f"--work-tree={root}", *args],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


# 把插件文件收拢成最少的忽略规则：目录下没有任何 Core 文件时整目录忽略。
def _exclude_patterns(core_files: set[str], plugin_files: list[str]) -> list[str]:
    core_dirs = {str(Path(path).parent) for path in core_files}
    core_dirs |= {str(parent) for path in core_files for parent in Path(path).parents}
    patterns: set[str] = set()
    for path in plugin_files:
        chosen = path
        for parent in reversed(Path(path).parents):
            if str(parent) != "." and str(parent) not in core_dirs:
                chosen = f"{parent}/"
                break
        patterns.add("/" + chosen)
    return sorted(patterns)


def _write_block(exclude: Path, patterns: list[str]) -> None:
    exclude.parent.mkdir(parents=True, exist_ok=True)
    lines = exclude.read_text().splitlines() if exclude.exists() else []
    if _BEGIN in lines:
        lines = lines[:lines.index(_BEGIN)] + lines[lines.index(_END) + 1:]
    exclude.write_text("\n".join([*lines, _BEGIN, *patterns, _END]) + "\n")


# Core 忽略插件文件。插件侧不显示未跟踪文件：Core 的 .gitignore 否定规则优先于
# info/exclude，没法在插件侧屏蔽；新文件改由 status 命令统一列出。
def _write_excludes(root: Path) -> None:
    core_files = _git(root, "ls-files").splitlines()
    plugin_files = _plugins_git(root, "ls-files").splitlines()
    core_git = Path(_git(root, "rev-parse", "--absolute-git-dir"))
    _write_block(core_git / "info/exclude", _exclude_patterns(set(core_files), plugin_files))


# 两侧的修改，加上两侧都没跟踪、也没被 .gitignore 忽略的新文件。
def status(root: Path) -> str:
    tracked = set(_plugins_git(root, "ls-files").splitlines())
    others = _git(root, "ls-files", "--others", "--exclude-per-directory=.gitignore").splitlines()
    unowned = [path for path in others if path not in tracked]
    return "\n".join([
        "== core", _git(root, "status", "--short", "--untracked-files=no"),
        "== plugins", _plugins_git(root, "status", "--short"),
        "== 两侧都未跟踪", *unowned,
    ])


# 1. 克隆插件仓库的 git 目录；2. 检出 plugins.lock 的提交并写两侧的忽略规则。
def attach(root: Path, source: str | None) -> str:
    lock = read_plugins_lock(root, "HEAD")
    if lock is None:
        raise RuntimeError(f"当前 Core 提交没有 {LOCK_PATH}，无需挂载插件仓库")
    git_dir = plugins_git_dir(root)
    if not git_dir.exists():
        _ = _git(root, "clone", "--quiet", "--bare", source or lock.repository, str(git_dir))
        _ = _plugins_git(root, "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*")
        _ = _plugins_git(root, "config", "status.showUntrackedFiles", "no")
        # 插件仓库自己的 README、LICENSE、workflow 等与 Core 同名，不检出到共用目录。
        _ = _plugins_git(root, "config", "core.sparseCheckout", "true")
        _ = _plugins_git(root, "config", "core.sparseCheckoutCone", "false")
        # 这些路径在共用目录里是 Core 的文件；不设置时 git 会因文件存在而取消跳过。
        _ = _plugins_git(root, "config", "sparse.expectFilesOutsideOfPatterns", "true")
        (git_dir / "info").mkdir(exist_ok=True)
        hidden = [f"!/{path}" for path in PLUGINS_REPO_LOCAL]
        (git_dir / "info/sparse-checkout").write_text("\n".join(["/*", *hidden]) + "\n")
    return sync(root)


def sync(root: Path) -> str:
    lock = read_plugins_lock(root, "HEAD")
    if lock is None:
        raise RuntimeError(f"当前 Core 提交没有 {LOCK_PATH}")
    if subprocess.run(["git", f"--git-dir={plugins_git_dir(root)}", "cat-file", "-e",
                       f"{lock.commit}^{{commit}}"], capture_output=True, check=False).returncode:
        _ = _plugins_git(root, "fetch", "--quiet", "origin")
    _ = _plugins_git(root, "checkout", "--quiet", "--detach", lock.commit)
    _write_excludes(root)
    return lock.commit


# 把插件仓库当前提交写回 plugins.lock；插件侧有未提交修改时拒绝。
def lock(root: Path) -> str:
    if _plugins_git(root, "status", "--porcelain"):
        raise RuntimeError("插件仓库有未提交修改，先在插件仓库提交")
    current = read_plugins_lock(root, "HEAD")
    if current is None:
        raise RuntimeError(f"当前 Core 提交没有 {LOCK_PATH}")
    commit = _plugins_git(root, "rev-parse", "HEAD")
    (root / LOCK_PATH).write_text(
        f'schema_version = 1\nrepository = "{current.repository}"\ncommit = "{commit}"\n'
    )
    _write_excludes(root)
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    attach_parser = commands.add_parser("attach", help="首次挂载插件仓库并检出锁定提交")
    attach_parser.add_argument("--source", help="插件仓库 URL 或本地路径；默认取 plugins.lock")
    _ = commands.add_parser("sync", help="切到 plugins.lock 锁定的插件提交")
    _ = commands.add_parser("lock", help="把插件仓库当前提交写入 plugins.lock")
    _ = commands.add_parser("status", help="两侧修改与未归属的新文件")
    args = parser.parse_args()
    root = _root()
    if args.command == "attach":
        print(attach(root, args.source))
    elif args.command == "sync":
        print(sync(root))
    elif args.command == "status":
        print(status(root))
    else:
        print(lock(root))


if __name__ == "__main__":
    main()
