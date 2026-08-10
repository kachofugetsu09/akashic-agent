from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
Run = Callable[..., subprocess.CompletedProcess[str]]


def resolve_target(origin: str, requested_commit: str | None, *, run: Run) -> str:
    """Resolve and verify one immutable commit from the canonical origin."""

    # 1. Resolve mutable main only at the remote boundary.
    if requested_commit is None:
        result = run(
            ["git", "ls-remote", "--exit-code", origin, "refs/heads/main"],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = result.stdout.strip().split()
        if len(fields) != 2 or fields[1] != "refs/heads/main":
            raise RuntimeError("origin/main 未解析为唯一远端 ref")
        commit = fields[0]
    else:
        commit = requested_commit
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("--commit 必须是 40 位小写 SHA")

    # 2. A depth-one fetch proves the object is reachable from the remote.
    with tempfile.TemporaryDirectory(prefix="akashic-release-probe-") as temporary:
        repository = Path(temporary)
        run(["git", "init", "-q"], cwd=repository, check=True)
        run(
            ["git", "fetch", "--quiet", "--depth=1", origin, commit],
            cwd=repository,
            check=True,
        )
        fetched = run(
            ["git", "rev-parse", "FETCH_HEAD^{commit}"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if fetched != commit:
            raise RuntimeError("远端 fetch identity 与目标 commit 不一致")
    return commit


def verify_bootstrap_checkout(
    checkout: Path, commit: str, origin: str, *, run: Run
) -> None:
    """Prove the bootstrap supplied a clean exact-commit checkout."""

    checkout = checkout.resolve(strict=True)
    head = run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != commit:
        raise RuntimeError(f"bootstrap checkout identity 不一致: {head}")
    dirty = run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("bootstrap checkout 必须 clean")
    remote = run(
        ["git", "remote", "get-url", "origin"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remote != origin:
        raise RuntimeError("bootstrap checkout origin 与批准 origin 不一致")


def commit_subject(checkout: Path, *, run: Run) -> str:
    return run(
        ["git", "show", "-s", "--format=%s", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
