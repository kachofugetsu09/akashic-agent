"""Git worktree 隔离与按变更感知清理，用于 isolation='worktree' 子任务。"""
from __future__ import annotations

import subprocess
from pathlib import Path


# 统一执行 git；超时与启动失败按失败返回码处理，调用方据此保留现场而不是误删。
def _git(*args: str, timeout: float = 10) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(error.cmd, 124, "", f"git 超时（{timeout}s）")
    except OSError as error:
        return subprocess.CompletedProcess(["git", *args], 127, "", str(error))


def git_repo_info(path: Path) -> tuple[Path, Path] | None:
    """返回 (repo_root, rel_subdir)；非 Git 仓库返回 None，无有效 HEAD 提交抛出 ValueError。"""
    proc = _git("-C", str(path), "rev-parse", "--show-toplevel")
    top = proc.stdout.strip()
    if proc.returncode != 0 or not top:
        return None
    repo_root = Path(top).resolve()
    if _git("-C", str(repo_root), "rev-parse", "--verify", "HEAD").returncode != 0:
        raise ValueError("Git 仓库尚无初始提交（unborn HEAD），无法创建 worktree")
    try:
        rel_subdir = path.resolve().relative_to(repo_root)
    except ValueError:
        rel_subdir = Path(".")
    return repo_root, rel_subdir


def _common_dir(path: Path) -> Path | None:
    # --git-common-dir 可能返回相对 -C 目录的路径；不依赖 --path-format（git 2.31+）。
    proc = _git("-C", str(path), "rev-parse", "--git-common-dir")
    out = proc.stdout.strip()
    return None if proc.returncode != 0 or not out else (path / out).resolve()


def _is_worktree_of(wt_path: Path, repo_root: Path) -> bool:
    top = _git("-C", str(wt_path), "rev-parse", "--show-toplevel")
    common = _common_dir(wt_path)
    return (top.returncode == 0 and Path(top.stdout.strip()).resolve() == wt_path.resolve()
            and common is not None and common == _common_dir(repo_root))


def create_worktree(repo_root: Path, wt_path: Path, branch: str) -> None:
    # 1. 已存在的目录必须确实是本仓库登记的 worktree；半成品目录直接报错，不当作已创建复用。
    if wt_path.exists():
        if not _is_worktree_of(wt_path, repo_root):
            raise RuntimeError(f"worktree 目录已存在但不是该仓库的有效 worktree: {wt_path}")
        return
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    # 2. 残留同名分支（例如上次清理后续聊）直接复用；否则从当前 HEAD 新建。
    branch_exists = _git("-C", str(repo_root), "show-ref", "--verify", "--quiet", f"refs/heads/{branch}").returncode == 0
    cmd = (("worktree", "add", str(wt_path), branch) if branch_exists
           else ("worktree", "add", "-b", branch, str(wt_path), "HEAD"))
    proc = _git("-C", str(repo_root), *cmd, timeout=120)
    if proc.returncode == 0:
        return
    # 3. 失败时回收本次登记的 worktree 与新建分支，避免下次把半成品当成可用目录。
    _ = _git("-C", str(repo_root), "worktree", "remove", "--force", str(wt_path))
    _ = _git("-C", str(repo_root), "worktree", "prune")
    if not branch_exists:
        _ = _git("-C", str(repo_root), "branch", "-D", branch)
    detail = (proc.stderr or proc.stdout).strip()
    raise RuntimeError(f"创建 worktree 失败: {detail}")


def _commits_ahead(repo_root: Path, branch: str) -> int | None:
    proc = _git("-C", str(repo_root), "rev-list", "--count", f"HEAD..{branch}")
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip() or "0")
    except ValueError:
        return None


def cleanup_worktree(repo_root: Path, wt_path: Path, branch: str) -> tuple[bool, str | None]:
    """若 worktree 无未提交改动且无新增 commit，则自动清理；否则保留并给出分支路径提示。"""
    if not wt_path.exists():
        _ = _git("-C", str(repo_root), "worktree", "prune")
        ahead = _commits_ahead(repo_root, branch)
        if ahead == 0:
            _ = _git("-C", str(repo_root), "branch", "-D", branch)
            return True, None
        if ahead is not None:
            return False, f"[worktree 目录已缺失，分支保留] 分支: {branch}"
        return True, None

    status = _git("-C", str(wt_path), "status", "--porcelain")
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    ahead = _commits_ahead(repo_root, branch)
    if dirty or ahead != 0:
        return False, f"[worktree 保留] 分支: {branch}，路径: {wt_path}"
    if _git("-C", str(repo_root), "worktree", "remove", "--force", str(wt_path), timeout=60).returncode != 0:
        return False, f"[worktree 清理失败，保留] 分支: {branch}，路径: {wt_path}"
    if _git("-C", str(repo_root), "branch", "-D", branch).returncode != 0:
        return False, f"[worktree 分支清理失败，保留] 分支: {branch}"
    return True, None


def inspect_preserved_worktree(repo_root: Path, wt_path: Path, branch: str) -> str | None:
    """进程重启后按真实 git 状态恢复 worktree 保留说明。"""
    if wt_path.exists():
        return f"[worktree 保留] 分支: {branch}，路径: {wt_path}"
    if _git("-C", str(repo_root), "show-ref", "--verify", "--quiet", f"refs/heads/{branch}").returncode == 0:
        return f"[worktree 分支保留] 分支: {branch}"
    return None
