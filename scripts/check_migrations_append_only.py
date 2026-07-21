from __future__ import annotations

import argparse
import hashlib
import subprocess
import tomllib
from pathlib import Path


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Git 命令失败: git {' '.join(arguments)}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _base_bundles(base: str) -> set[str]:
    files = _git("ls-tree", "-r", "--name-only", base, "--", "migrations/")
    return {
        Path(path).parts[1]
        for path in files.splitlines()
        if path.endswith("/migration.py") and len(Path(path).parts) >= 3
    }


def _framework_exists(base: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{base}:migrations/.root"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _authorized_repairs(base: str) -> dict[str, tuple[str, str]]:
    """读取本 diff 新增的精确 hash 修复声明。"""

    output = _git(
        "diff",
        "--name-only",
        "--diff-filter=A",
        f"{base}...HEAD",
        "--",
        "migrations/repairs/*.toml",
    )
    repairs: dict[str, tuple[str, str]] = {}
    for raw_path in output.splitlines():
        document = tomllib.loads(Path(raw_path).read_text(encoding="utf-8"))
        path = str(document["path"])
        repairs[path] = (
            str(document["base_sha256"]),
            str(document["head_sha256"]),
        )
    return repairs


def _matches_repair(base: str, path: str, hashes: tuple[str, str]) -> bool:
    base_content = subprocess.run(
        ["git", "show", f"{base}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if base_content.returncode != 0:
        return False
    current = Path(path)
    return current.is_file() and hashes == (
        _sha256_bytes(base_content.stdout),
        _sha256_bytes(current.read_bytes()),
    )


def check_append_only(base: str) -> list[str]:
    """返回违反迁移历史只追加规则的 Git diff 记录。"""

    if not _framework_exists(base):
        return []
    existing_bundles = _base_bundles(base)
    repairs = _authorized_repairs(base)
    used_repairs: set[str] = set()
    output = _git(
        "diff",
        "--name-status",
        f"{base}...HEAD",
        "--",
        "migrations/",
    )
    violations: list[str] = []
    for line in output.splitlines():
        fields = line.split("\t")
        status = fields[0]
        paths = fields[1:]
        if (
            status == "M"
            and len(paths) == 1
            and paths[0] in repairs
            and _matches_repair(base, paths[0], repairs[paths[0]])
        ):
            used_repairs.add(paths[0])
            continue
        if status != "A":
            violations.append(line)
            continue
        for raw_path in paths:
            parts = Path(raw_path).parts
            if len(parts) < 3 or parts[1] in existing_bundles:
                violations.append(line)
                break
    for path in sorted(repairs.keys() - used_repairs):
        violations.append(f"unused repair declaration\t{path}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 migration bundle 只追加合同")
    parser.add_argument("--base", default="origin/main")
    args = parser.parse_args()
    violations = check_append_only(str(args.base))
    if not violations:
        print("migration append-only: passed")
        return 0
    print("既有 migration bundle 不得修改、移动、删除或追加文件:")
    for violation in violations:
        print(f"  {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
