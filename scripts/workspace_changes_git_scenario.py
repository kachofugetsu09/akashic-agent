"""用真实进程服务与 Git 仓库验证快照范围、filter 拒绝和用户 index 保留。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition import CompositionRoot, Context, PluginRuntime
from plugins.host_execution.contract import PROCESSES
from plugins.host_execution.processes import PluginProcesses
from plugins.host_execution.process_runtime import ShellProcessManager
from plugins.workspace_changes.git import Git, GitUnavailable


def git(path: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.STDOUT)


def repository(path: Path) -> Path:
    """建立一个已提交的仓库，不读取用户 HOME 的 Git 配置。"""
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "fixture@example.invalid")
    git(path, "config", "user.name", "Fixture")
    (path / "a.txt").write_text("old\n")
    git(path, "add", ".")
    git(path, "commit", "-qm", "fixture")
    return path


async def scenarios(output: Path) -> dict[str, object]:
    """经真实 Context 授权运行生产 Git 类，检查对象内容和受保护状态。"""
    root = CompositionRoot("git-snapshot-scenario")
    processes = PluginProcesses(factory=lambda: ShellProcessManager(output_dir=output / "process-output"))

    async def provide(ctx: Context) -> None:
        await ctx.effect(lambda: processes.close)
        await ctx.provide(PROCESSES, processes)

    fiber = await root.mount(provide, name="snapshot", runtime=PluginRuntime("snapshot", root.generation_id,
                            ROOT, output, output, {}))
    ctx = fiber.context
    snapshots = Git(ctx)
    report: dict[str, object] = {}
    try:
        async with ctx.runtime_scope():
            # 1. 相对属性路径仍然启用 filter；拒绝快照，外部 clean 程序不得启动。
            repo = repository(output / "filter")
            git(repo, "config", "core.attributesFile", "attrs")
            git(repo, "config", "filter.fixture.clean", "tee filter-ran")
            (repo / "attrs").write_text("*.txt filter=fixture\n")
            (repo / "a.txt").write_text("new\n")
            try:
                await snapshots.resolve(str(repo))
            except GitUnavailable as error:
                assert error.reason == "filter"
            else:
                raise AssertionError("相对属性文件中的 filter 未被拒绝")
            assert not (repo / "filter-ran").exists()
            report["relative_filter"] = "rejected_without_execution"

            # 2. 第 2001 个未跟踪文件也要检查；所有小文件仍进入树，展示名单才截断。
            repo = repository(output / "large")
            for number in range(2000):
                (repo / f"b{number:04d}").write_text("small\n")
            (repo / "z-large.db").write_bytes(b"x" * (1024 * 1024 + 1))
            index_before = (repo / ".git/index").read_bytes()
            head_before = git(repo, "rev-parse", "HEAD")
            resolved = await snapshots.resolve(str(repo))
            shot = await snapshots.snapshot(resolved)
            files = git(repo, "ls-tree", "-r", "--name-only", shot.tree).splitlines()
            assert "z-large.db" in shot.skipped and b"z-large.db" not in files
            assert b"b1999" in files and len(files) == 2001
            assert (repo / ".git/index").read_bytes() == index_before
            assert git(repo, "rev-parse", "HEAD") == head_before
            report["large_after_2000"] = {"excluded": list(shot.skipped), "tree_files": len(files), "index_unchanged": True}

            # 3. 旧私有缓存带有用户标志时也要修正；用户自己的 index 字节保持原样。
            repo = repository(output / "flags")
            (repo / "skip.txt").write_text("old\n")
            git(repo, "add", ".")
            git(repo, "commit", "-qm", "second file")
            git(repo, "update-index", "--assume-unchanged", "a.txt")
            git(repo, "update-index", "--skip-worktree", "skip.txt")
            shutil.copyfile(repo / ".git/index", repo / ".git/akashic-changes.index")
            for name in ("a.txt", "skip.txt"):
                (repo / name).write_text("new\n")
            index_before = (repo / ".git/index").read_bytes()
            resolved = await snapshots.resolve(str(repo))
            shot = await snapshots.snapshot(resolved)
            for name in ("a.txt", "skip.txt"):
                assert git(repo, "show", f"{shot.tree}:{name}") == b"new\n"
            (repo / "a.txt").write_text("newer\n")
            next_shot = await snapshots.snapshot(resolved)
            assert git(repo, "show", f"{next_shot.tree}:a.txt") == b"newer\n"
            assert (repo / ".git/index").read_bytes() == index_before
            report["index_flags"] = {"both_files_captured": True, "next_snapshot_updated": True, "user_index_unchanged": True}

            # 4. 尚无 HEAD 和用户 index 的仓库也能拍快照。
            repo = output / "unborn"
            repo.mkdir()
            git(repo, "init", "-q")
            (repo / "new.txt").write_text("hello\n")
            shot = await snapshots.snapshot(await snapshots.resolve(str(repo)))
            assert git(repo, "show", f"{shot.tree}:new.txt") == b"hello\n"
            assert not (repo / ".git/index").exists()
            report["unborn"] = "captured_without_user_index"

            # 5. 实际进程输出被截断时拒绝整次快照，不能把没检查过的尾部交给 add。
            repo = repository(output / "list-limit")
            for number in range(6000):
                (repo / (f"b{number:04d}-" + "x" * 180)).write_text("small\n")
            objects_before = git(repo, "count-objects", "-v")
            try:
                await snapshots.snapshot(await snapshots.resolve(str(repo)))
            except GitUnavailable as error:
                assert error.reason == "too_many_files", error.reason
            else:
                raise AssertionError("场景未触发列表输出上限")
            assert git(repo, "count-objects", "-v") == objects_before
            report["truncated_listing"] = "rejected_before_writing_blobs"
        return report
    finally:
        await root.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()
    (output / "home").mkdir()
    os.environ.update(HOME=str(output / "home"), XDG_CONFIG_HOME=str(output / "home/.config"))
    report = asyncio.run(scenarios(output))
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
