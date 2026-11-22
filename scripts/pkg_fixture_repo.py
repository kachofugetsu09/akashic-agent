#!/usr/bin/env python3
"""把 Core 仓库里的插件构建成裸仓库的 release/<name> 分支（夹具与 extra CI 共用）。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from agent.plugins.static_manifest import load_static_plugin_manifest
from scripts.build_plugin_distribution import (
    build_web_assets,
    git,
    stage_plugin_package,
)

# 固定身份与时间：提交只由 tree 决定，不随构建时间或构建者变化。
_FIXED_ENV = {
    "GIT_AUTHOR_NAME": "Akashic release",
    "GIT_AUTHOR_EMAIL": "release@akashic.invalid",
    "GIT_COMMITTER_NAME": "Akashic release",
    "GIT_COMMITTER_EMAIL": "release@akashic.invalid",
    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
}


# 裸仓库不存在时创建；已存在则原样使用。
def ensure_bare_repo(output: Path) -> None:
    if not output.exists():
        output.mkdir(parents=True)
        _ = git(output, "init", "--bare", "--initial-branch=release")


# 把 package 目录写成 tree 对象并返回 tree 哈希；用临时 index，不碰任何工作区。
def write_tree(output: Path, package: Path, scratch: Path) -> str:
    env = {**os.environ, "GIT_DIR": str(output), "GIT_WORK_TREE": str(package),
           "GIT_INDEX_FILE": str(scratch / "index")}
    _ = subprocess.run(["git", "add", "--all", "--force"], cwd=package, env=env, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = subprocess.run(["git", "write-tree"], cwd=package, env=env, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


# 分支当前的 (tree, 提交 SHA)；分支不存在返回 ("", "")。
def branch_head(output: Path, branch: str) -> tuple[str, str]:
    try:
        sha = git(output, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}").decode().strip()
    except subprocess.CalledProcessError:
        return "", ""
    return git(output, "rev-parse", f"{sha}^{{tree}}").decode().strip(), sha


# 无父提交、固定作者和时间、说明只含名字和 tree：提交 SHA 完全由内容决定，
# 从零重建仓库也得到同一个版本号。来源提交不记在这里，由 CI 日志追溯。
def commit_tree(output: Path, tree: str, name: str) -> str:
    message = f"release {name} {tree}\n"
    env = {**os.environ, **_FIXED_ENV}
    result = git(output, "-c", "commit.gpgSign=false", "commit-tree", tree, "-m", message, env=env)
    return result.decode().strip()


# 构建单个插件并在 tree 变化时推进分支；返回状态行。
def release_plugin(
    repository: Path, commit: str, plugins_root: str, name: str, output: Path,
    generated_root: Path | None,
) -> dict[str, str]:
    root = f"{plugins_root}/{name}"
    with tempfile.TemporaryDirectory(prefix="akashic-release-") as directory:
        scratch = Path(directory)
        package = scratch / "package"
        package.mkdir()
        # 1. 暂存插件源码与前端产物；分支、提交说明与返回行都用 manifest 名（安装器要求分支名等于插件名，目录名可以不同，如 opencode_go 与 opencode-go）。
        stage_plugin_package(repository, commit, root, package, generated_root)
        plugin = load_static_plugin_manifest(package).name
        branch = f"release/{plugin}"
        # 2. tree 没变就不推进分支。
        tree = write_tree(output, package, scratch)
    old_tree, old_sha = branch_head(output, branch)
    if old_tree == tree:
        return {"name": plugin, "branch": branch, "tree": tree, "sha": old_sha, "status": "unchanged"}
    # 3. 新建无父提交并推进分支（旧值作为 CAS 条件，新分支用全零）。
    sha = commit_tree(output, tree, plugin)
    _ = git(output, "update-ref", f"refs/heads/{branch}", sha, old_sha or "0" * 40)
    return {"name": plugin, "branch": branch, "tree": tree, "sha": sha, "status": "updated"}


# 一次构建全部所选插件：前端只构建一次，逐个对比 tree。
def release_plugins(
    repository: Path, revision: str, plugins_root: str, names: list[str], output: Path,
) -> list[dict[str, str]]:
    commit = git(repository, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}").decode().strip()
    ensure_bare_repo(output)
    with tempfile.TemporaryDirectory(prefix="akashic-release-web-") as directory:
        _, generated_root, _ = build_web_assets(repository, commit, Path(directory))
        return [
            release_plugin(repository, commit, plugins_root, name, output, generated_root)
            for name in names
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="插件目录名（plugins/<目录名>）；分支与名字取自 manifest")
    parser.add_argument("--repository", type=Path, default=_SOURCE_ROOT, help="含插件源码与前端源码的 git 仓库")
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--plugins-root", default="plugins", help="仓库内插件目录的相对路径")
    parser.add_argument("--output", type=Path, required=True, help="输出裸仓库（不存在则创建）")
    args = parser.parse_args()
    rows = release_plugins(args.repository.resolve(), args.revision, args.plugins_root, args.names, args.output.resolve())
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
