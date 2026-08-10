from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _run(*arguments: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def prepare_runtime_checkout(
    source_repository: Path,
    commit: str,
    target: Path,
    canonical_origin: str,
) -> Path:
    """Create a one-commit runtime checkout without reachable repository history."""

    # 1. Refuse mutable identities and any existing destination.
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("runtime checkout commit 必须是完整 40 位 SHA")
    source_repository = source_repository.resolve(strict=True)
    if target.exists():
        raise FileExistsError(f"runtime checkout target 已存在: {target}")
    resolved = _run(
        "git", "rev-parse", "--verify", f"{commit}^{{commit}}", cwd=source_repository
    )
    if resolved != commit:
        raise RuntimeError("runtime checkout commit 解析不一致")

    # 2. Fetch depth one from the local object owner, then replace its remote URL.
    target.mkdir(parents=True)
    _run("git", "init", "-q", cwd=target)
    _run(
        "git",
        "fetch",
        "--quiet",
        "--depth=1",
        f"file://{source_repository}",
        commit,
        cwd=target,
    )
    _run("git", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=target)
    _run("git", "remote", "add", "origin", canonical_origin, cwd=target)

    # 3. Prove the checkout contains only the approved generation and is clean.
    if _run("git", "rev-list", "--all", "--count", cwd=target) != "1":
        raise RuntimeError("runtime checkout 意外包含额外历史")
    if _run("git", "status", "--porcelain", "--untracked-files=all", cwd=target):
        raise RuntimeError("runtime checkout 创建后不 clean")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an exact runtime checkout")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    args = parser.parse_args()
    print(
        prepare_runtime_checkout(args.repository, args.commit, args.target, args.origin)
    )


if __name__ == "__main__":
    main()
