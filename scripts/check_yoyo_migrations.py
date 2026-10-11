#!/usr/bin/env python3
"""已发布的 Yoyo 迁移只能追加：用 Git 基线逐字节比较，不在目录中保存第二份摘要。"""
from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CORE_PREFIX = "migrations/core/"
_MIGRATION_CATALOG = "migration.catalog.toml"
# ADR-0066 退役的历史命名空间；源码只保存在 Git 恢复点，不能悄悄回来。
RETIRED_PREFIXES = ("migrations/yoyo/", "plugins/legacy_upgrade/")
RETIRED_EXACT_PATHS = frozenset({"migrations/core/20260802_01_yoyo_origin.py"})
_BUNDLE_FIELDS = ("bundle_id", "migration_root", "package_name")
_ENTRY_FIELDS = ("path", "depends", "transactional")


def _git(*arguments: str) -> str:
    return _git_bytes(*arguments).decode("utf-8")


def _git_bytes(*arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git 命令失败: git {' '.join(arguments)}: {detail}")
    return result.stdout


def _tree_files(base: str, prefix: str) -> list[str]:
    return _git("ls-tree", "-r", "--name-only", base, "--", prefix).splitlines()


def _catalog(source: bytes, label: str) -> dict[str, Any]:
    try:
        value = tomllib.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"TOML 无法解析: {label}") from error
    return value


# 只取决定执行语义的字段；基线可以是任意历史格式的 catalog。
def _catalog_facts(raw: dict[str, Any]) -> tuple[tuple[object, ...], dict[str, tuple[object, ...]]]:
    bundle: tuple[object, ...] = tuple(raw.get(field) for field in _BUNDLE_FIELDS)
    entries: dict[str, tuple[object, ...]] = {}
    for item in raw.get("migrations", []):
        if isinstance(item, dict):
            values = (item.get(field) for field in _ENTRY_FIELDS)
            entries[str(item.get("id"))] = tuple(
                tuple(value) if isinstance(value, list) else value for value in values
            )
    return bundle, entries


# 基线中已发布的全部迁移源码：Core 迁移文件与每个插件 bundle 的整个 package 目录。
def _registered_paths(base: str) -> tuple[set[str], list[str]]:
    paths = {
        path for path in _tree_files(base, CORE_PREFIX)
        if path.endswith(".py") and "/" not in path[len(CORE_PREFIX):]
    }
    catalogs = [
        path for path in _tree_files(base, "plugins")
        if PurePosixPath(path).name == _MIGRATION_CATALOG
    ]
    for catalog_path in catalogs:
        raw = _catalog(_git_bytes("show", f"{base}:{catalog_path}"), catalog_path)
        package_root = PurePosixPath(catalog_path).parent / str(raw["migration_root"])
        paths.update(_tree_files(base, package_root.as_posix() + "/"))
    return paths, catalogs


def _catalog_violations(base: str, catalogs: list[str]) -> list[str]:
    problems: list[str] = []
    for catalog_path in catalogs:
        current = ROOT / catalog_path
        if current.is_symlink() or not current.is_file():
            problems.append(f"registered migration bundle removed: {catalog_path}")
            continue
        old_bundle, old_entries = _catalog_facts(
            _catalog(_git_bytes("show", f"{base}:{catalog_path}"), catalog_path)
        )
        new_bundle, new_entries = _catalog_facts(_catalog(current.read_bytes(), catalog_path))
        if old_bundle != new_bundle:
            problems.append(f"registered migration bundle changed: {catalog_path}")
        problems.extend(
            f"registered Yoyo migration changed: {catalog_path}#{migration_id}"
            for migration_id, entry in old_entries.items()
            if new_entries.get(migration_id) != entry
        )
    return problems


def _retired_path_reintroductions(base: str) -> list[str]:
    problems: list[str] = []
    for line in _git("diff", "--name-status", base, "--", ".").splitlines():
        fields = line.split("\t")
        status = fields[0][:1]
        if len(fields) < 2 or status not in {"A", "C", "R"}:
            continue
        # rename/copy 先报告旧路径；只有目标路径可能回到退役命名空间。
        for path in fields[1:] if status == "A" else fields[2:]:
            if path in RETIRED_EXACT_PATHS or path.startswith(RETIRED_PREFIXES):
                problems.append(f"retired migration path reintroduced: {path}")
    return problems


def violations(base: str) -> list[str]:
    """拒绝修改或删除基线中已发布的迁移源码与 catalog 声明。"""
    paths, catalogs = _registered_paths(base)
    problems: list[str] = []
    for path in sorted(paths):
        current = ROOT / path
        if not current.is_file() or _git_bytes("show", f"{base}:{path}") != current.read_bytes():
            problems.append(f"registered Yoyo migration changed: {path}")
    problems.extend(_catalog_violations(base, catalogs))
    problems.extend(_retired_path_reintroductions(base))
    return sorted(set(problems))


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
