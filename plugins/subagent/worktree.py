"""Git worktree 隔离与按变更感知清理，用于 isolation='worktree' 子任务。"""
from __future__ import annotations

import subprocess
from pathlib import Path


def git_repo_info(path: Path) -> tuple[Path, Path] | None:
    """返回 (repo_root, rel_subdir)；非 Git 仓库返回 None，无有效 HEAD 提交抛出 ValueError。"""
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    top = proc.stdout.strip()
    if not top:
        return None
    repo_root = Path(top).resolve()
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if head.returncode != 0:
        raise ValueError("Git 仓库尚无初始提交（unborn HEAD），无法创建 worktree")
    try:
        rel_subdir = path.resolve().relative_to(repo_root)
    except ValueError:
        rel_subdir = Path(".")
    return repo_root, rel_subdir


def create_worktree(repo_root: Path, wt_path: Path, branch: str) -> None:
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    if wt_path.exists():
        return
    # 若之前残留同名孤立分支但目录不存在，先基于当前分支/或复用
    branch_exists = subprocess.run(
        ["git", "-C", str(repo_root), "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        capture_output=True,
        timeout=10,
        check=False,
    ).returncode == 0
    cmd = (
        ["git", "-C", str(repo_root), "worktree", "add", str(wt_path), branch]
        if branch_exists
        else ["git", "-C", str(repo_root), "worktree", "add", "-b", branch, str(wt_path), "HEAD"]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"创建 worktree 失败: {detail}")


def cleanup_worktree(repo_root: Path, wt_path: Path, branch: str) -> tuple[bool, str | None]:
    """若 worktree 无未提交改动且无新增 commit，则自动清理；否则保留并给出分支路径提示。"""
    if not wt_path.exists():
        _ = subprocess.run(["git", "-C", str(repo_root), "worktree", "prune"], capture_output=True, timeout=10, check=False)
        commits = subprocess.run(
            ["git", "-C", str(repo_root), "rev-list", "--count", f"HEAD..{branch}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if commits.returncode == 0 and int(commits.stdout.strip() or "0") == 0:
            _ = subprocess.run(["git", "-C", str(repo_root), "branch", "-D", branch], capture_output=True, timeout=10, check=False)
            return True, None
        if commits.returncode == 0:
            return False, f"[worktree 目录已缺失，分支保留] 分支: {branch}"
        return True, None

    status = subprocess.run(
        ["git", "-C", str(wt_path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    commits = subprocess.run(
        ["git", "-C", str(repo_root), "rev-list", "--count", f"HEAD..{branch}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    has_commits = commits.returncode != 0 or int(commits.stdout.strip() or "0") > 0
    if dirty or has_commits:
        return False, f"[worktree 保留] 分支: {branch}，路径: {wt_path}"

    rm_wt = subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(wt_path)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if rm_wt.returncode != 0:
        return False, f"[worktree 清理失败，保留] 分支: {branch}，路径: {wt_path}"
    rm_br = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "-D", branch],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if rm_br.returncode != 0:
        return False, f"[worktree 分支清理失败，保留] 分支: {branch}"
    return True, None


def inspect_preserved_worktree(repo_root: Path, wt_path: Path, branch: str) -> str | None:
    """进程重启后按真实 git 状态恢复 worktree 保留说明。"""
    if wt_path.exists():
        return f"[worktree 保留] 分支: {branch}，路径: {wt_path}"
    has_branch = subprocess.run(
        ["git", "-C", str(repo_root), "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        capture_output=True,
        timeout=10,
        check=False,
    ).returncode == 0
    if has_branch:
        return f"[worktree 分支保留] 分支: {branch}"
    return None
