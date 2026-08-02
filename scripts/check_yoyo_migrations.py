#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PREFIX = "migrations/yoyo/"


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Git 命令失败: git {' '.join(arguments)}: {result.stderr.strip()}"
        )
    return result.stdout


def violations(base: str) -> list[str]:
    """拒绝修改基线中已经存在的 Yoyo 迁移。"""

    # 1. 只有基线中已注册的 Python 迁移不可变
    base_paths = {
        path
        for path in _git(
            "ls-tree", "-r", "--name-only", base, "--", "migrations/yoyo"
        ).splitlines()
        if path.startswith(CATALOG_PREFIX) and path.endswith(".py")
    }

    # 2. 允许新增，拒绝修改、移动或删除既有迁移
    problems: list[str] = []
    changes = _git(
        "diff",
        "--name-status",
        "--find-renames",
        base,
        "--",
        "migrations/yoyo",
    )
    for line in changes.splitlines():
        fields = line.split("\t")
        status, paths = fields[0], fields[1:]
        if status == "A":
            continue
        immutable = sorted(base_paths.intersection(paths))
        if immutable:
            problems.append(f"registered Yoyo migration changed: {immutable[0]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    problems = violations(str(args.base))
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
