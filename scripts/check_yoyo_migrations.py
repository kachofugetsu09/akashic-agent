#!/usr/bin/env python3
from __future__ import annotations

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PREFIX = "migrations/yoyo/"
CORE_PREFIX = "migrations/core/"
BUNDLE_PREFIX = "plugins/legacy_upgrade/legacy_upgrade_migrations/"
REGISTERED_PREFIXES = (LEGACY_PREFIX, CORE_PREFIX, BUNDLE_PREFIX)


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


def _registered_paths(base: str) -> set[str]:
    """Return direct Yoyo files that were registered by a base tree."""

    paths: set[str] = set()
    for prefix in REGISTERED_PREFIXES:
        for path in _git("ls-tree", "-r", "--name-only", base, "--", prefix).splitlines():
            if not path.startswith(prefix) or not path.endswith(".py"):
                continue
            if "/" not in path[len(prefix) :]:
                paths.add(path)
    return paths


def _relocated_path(path: str) -> str | None:
    """Map the retired checkout path to its immutable external owner."""

    if not path.startswith(LEGACY_PREFIX):
        return None
    filename = path[len(LEGACY_PREFIX) :]
    if filename == "20260802_01_yoyo_origin.py":
        return CORE_PREFIX + filename
    return BUNDLE_PREFIX + filename


def _source_at(base: str, path: str) -> str:
    return _git("show", f"{base}:{path}")


def _migration_identity(source: str, path: str) -> tuple[str, tuple[str, ...], bool]:
    """Read only ID, dependency and transaction metadata for relocation checks."""

    tree = ast.parse(source, filename=path)
    depends: tuple[str, ...] = ()
    transactional = True
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in {"__depends__", "__transactional__"}:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            raise ValueError(f"迁移元数据不是静态字面量: {path}") from None
        if name == "__depends__":
            if isinstance(value, str):
                value = (value,)
            if not isinstance(value, (tuple, list, set, frozenset)):
                raise ValueError(f"迁移依赖不是字符串集合: {path}")
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f"迁移依赖不是字符串集合: {path}")
            depends = tuple(sorted(value))
        else:
            if not isinstance(value, bool):
                raise ValueError(f"迁移 transactional 不是 bool: {path}")
            transactional = value
    return Path(path).stem, depends, transactional


def _relocation_is_unchanged(base: str, old_path: str, new_path: str) -> bool:
    target = ROOT / new_path
    if not target.is_file():
        return False
    return _migration_identity(_source_at(base, old_path), old_path) == _migration_identity(
        target.read_text(encoding="utf-8"), new_path
    )


def violations(base: str) -> list[str]:
    """拒绝修改已注册迁移；只允许一次有元数据证明的 owner relocation。"""

    base_paths = _registered_paths(str(base))
    problems: list[str] = []
    for old_path in sorted(base_paths):
        current = ROOT / old_path
        if current.is_file():
            try:
                unchanged = _source_at(str(base), old_path) == current.read_text(
                    encoding="utf-8"
                )
            except (OSError, UnicodeError, SyntaxError, ValueError):
                unchanged = False
            if not unchanged:
                problems.append(f"registered Yoyo migration changed: {old_path}")
            continue
        replacement = _relocated_path(old_path)
        if replacement is None or not _relocation_is_unchanged(
            str(base), old_path, replacement
        ):
            problems.append(f"registered Yoyo migration changed: {old_path}")

    # A retired yoyo path cannot receive new registrations after the split.
    changes = _git("diff", "--name-status", str(base), "--", LEGACY_PREFIX)
    for line in changes.splitlines():
        fields = line.split("\t")
        if len(fields) < 2 or fields[0] != "A":
            continue
        problems.append(f"new migration must use external bundle: {fields[1]}")
    return problems


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    problems = violations(str(args.base))
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
