#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# These are the checkout paths used by the historical Core migration chain.
# New migrations must be owned by a plugin manifest.
LEGACY_PREFIX = "migrations/yoyo/"
CORE_PREFIX = "migrations/core/"
LEGACY_PLUGIN_PREFIX = "plugins/legacy_upgrade/"
# Core remains an owner for future neutral migrations.  Only this historical
# Core origin is retired; the old checkout and legacy bundle namespaces are
# closed so the deleted implementation cannot quietly return.
RETIRED_PREFIXES = (LEGACY_PREFIX, LEGACY_PLUGIN_PREFIX)
RETIRED_EXACT_PATHS = frozenset({"migrations/core/20260802_01_yoyo_origin.py"})

RETIREMENT_METADATA = "migrations/retired.toml"
# Both commits are real migration identities in the stacked change.  The first
# is the published checkout path; the second is the intermediate plugin-owned
# path.  No descendant or date range is accepted.
RETIREMENT_RECOVERY_COMMITS = frozenset(
    {
        "816dcdf33c0c3e064c46290a3bcd22c66a812f09",
        "5cb8e8bbff5c2f8a2f37456b3433038091cada09",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SourceReader = Callable[[str], bytes]


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


def _resolve_commit(base: str) -> str:
    """Resolve a gate base to one immutable commit identity."""

    return _git("rev-parse", f"{base}^{{commit}}").strip()


def _source_bytes(base: str, path: str) -> bytes:
    return _git_bytes("show", f"{base}:{path}")


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} 必须是非空相对路径")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"{label} 必须是 artifact 内的相对路径: {value!r}")
    return path.as_posix()


def _repo_path(parent: str, relative: str) -> str:
    if parent == ".":
        return relative
    return f"{parent}/{relative}"


def _toml_bytes(source: bytes, label: str) -> dict[str, Any]:
    try:
        value = tomllib.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"TOML 无法解析: {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"TOML 顶层必须是表: {label}")
    return value


def _manifest_migration(raw: dict[str, Any], path: str) -> tuple[str, str]:
    migration = raw.get("migration")
    if not isinstance(migration, dict):
        raise ValueError(f"插件 migration 声明缺失: {path}")
    catalog = _safe_relative(migration.get("catalog"), f"{path}:migration.catalog")
    digest = migration.get("catalog_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{path}:migration.catalog_sha256 无效")
    return catalog, digest


def _catalog_spec(
    raw: dict[str, Any],
    *,
    read_source: SourceReader,
    catalog_path: str,
    plugin_path: str,
) -> dict[str, Any]:
    """Read the immutable prefix of one plugin-owned migration catalog."""

    if raw.get("schema_version") != 1:
        raise ValueError(f"migration catalog schema_version 无效: {catalog_path}")
    bundle_id = raw.get("bundle_id")
    version = raw.get("version")
    migration_root_value = raw.get("migration_root")
    package_name = raw.get("package_name")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ValueError(f"migration catalog bundle_id 无效: {catalog_path}")
    if not isinstance(version, str) or not version:
        raise ValueError(f"migration catalog version 无效: {catalog_path}")
    if not isinstance(package_name, str) or not package_name:
        raise ValueError(f"migration catalog package_name 无效: {catalog_path}")
    migration_root = _safe_relative(
        migration_root_value, f"{catalog_path}:migration_root"
    )
    plugin_root = str(Path(plugin_path).parent).replace("\\", "/")
    package_root = _repo_path(plugin_root, migration_root)

    raw_files = raw.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError(f"migration catalog.files 无效: {catalog_path}")
    package_files: dict[str, str] = {}
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict):
            raise ValueError(f"migration catalog.files[{index}] 无效: {catalog_path}")
        relative = _safe_relative(
            item.get("path"), f"{catalog_path}:files[{index}].path"
        )
        digest = item.get("sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(
                f"migration catalog.files[{index}].sha256 无效: {catalog_path}"
            )
        path = _repo_path(package_root, relative)
        if path in package_files:
            raise ValueError(f"migration catalog.files 重复: {path}")
        actual = hashlib.sha256(read_source(path)).hexdigest()
        if actual != digest:
            raise ValueError(f"migration source digest 漂移: {path}")
        package_files[path] = digest

    raw_migrations = raw.get("migrations")
    if not isinstance(raw_migrations, list) or not raw_migrations:
        raise ValueError(f"migration catalog.migrations 无效: {catalog_path}")
    migrations: dict[str, tuple[str, tuple[str, ...], bool, str]] = {}
    for index, item in enumerate(raw_migrations):
        if not isinstance(item, dict):
            raise ValueError(
                f"migration catalog.migrations[{index}] 无效: {catalog_path}"
            )
        migration_id = item.get("id")
        if not isinstance(migration_id, str) or not migration_id:
            raise ValueError(
                f"migration catalog.migrations[{index}].id 无效: {catalog_path}"
            )
        relative = _safe_relative(
            item.get("path"), f"{catalog_path}:migrations[{index}].path"
        )
        depends = item.get("depends")
        if not isinstance(depends, list) or not all(
            isinstance(value, str) for value in depends
        ):
            raise ValueError(
                f"migration catalog.migrations[{index}].depends 无效: {catalog_path}"
            )
        transactional = item.get("transactional")
        if not isinstance(transactional, bool):
            raise ValueError(
                f"migration catalog.migrations[{index}].transactional 无效: {catalog_path}"
            )
        digest = item.get("sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(
                f"migration catalog.migrations[{index}].sha256 无效: {catalog_path}"
            )
        if migration_id in migrations:
            raise ValueError(f"migration ID 重复: {migration_id}")
        path = _repo_path(package_root, relative)
        if path not in package_files:
            raise ValueError(f"migration 不在 catalog.files 中: {path}")
        if package_files[path] != digest:
            raise ValueError(f"migration 与 package file digest 不一致: {path}")
        migrations[migration_id] = (relative, tuple(depends), transactional, digest)

    return {
        "manifest_path": plugin_path,
        "catalog_path": catalog_path,
        "catalog_relative": _safe_relative(
            _manifest_migration(
                _toml_bytes(read_source(plugin_path), plugin_path), plugin_path
            )[0],
            f"{plugin_path}:migration.catalog",
        ),
        "bundle_id": bundle_id,
        "migration_root": migration_root,
        "package_name": package_name,
        "package_files": package_files,
        "migrations": migrations,
    }


def _bundle_specs(base: str) -> list[dict[str, Any]]:
    """Discover every migration catalog declared by a plugin in a tree."""

    paths = _git("ls-tree", "-r", "--name-only", base, "--", "plugins").splitlines()
    specs: list[dict[str, Any]] = []
    for manifest_path in paths:
        if not manifest_path.endswith("/akashic.plugin.toml"):
            continue
        manifest = _toml_bytes(_source_bytes(base, manifest_path), manifest_path)
        migration = manifest.get("migration")
        if migration is None:
            continue
        catalog_relative, declared_digest = _manifest_migration(manifest, manifest_path)
        plugin_root = str(Path(manifest_path).parent).replace("\\", "/")
        catalog_path = _repo_path(plugin_root, catalog_relative)
        catalog_source = _source_bytes(base, catalog_path)
        actual_digest = hashlib.sha256(catalog_source).hexdigest()
        if actual_digest != declared_digest:
            raise ValueError(f"migration catalog digest 漂移: {catalog_path}")
        catalog = _toml_bytes(catalog_source, catalog_path)
        spec = _catalog_spec(
            catalog,
            read_source=lambda path, tree=base: _source_bytes(tree, path),
            catalog_path=catalog_path,
            plugin_path=manifest_path,
        )
        specs.append(spec)
    return specs


def _registered_paths(base: str) -> set[str]:
    """Return migration sources and package files registered by a base tree."""

    paths: set[str] = set()
    for prefix in (LEGACY_PREFIX, CORE_PREFIX):
        for path in _git(
            "ls-tree", "-r", "--name-only", base, "--", prefix
        ).splitlines():
            if (
                path.startswith(prefix)
                and path.endswith(".py")
                and "/" not in path[len(prefix) :]
            ):
                paths.add(path)
    for spec in _bundle_specs(base):
        paths.update(spec["package_files"])
    return paths


def _retirement_inventory(base: str) -> tuple[tuple[str, str], ...]:
    """Build the exact path and digest inventory for one cutover base."""

    paths = _registered_paths(base)
    for spec in _bundle_specs(base):
        paths.add(spec["manifest_path"])
        paths.add(spec["catalog_path"])
    return tuple(
        (path, hashlib.sha256(_source_bytes(base, path)).hexdigest())
        for path in sorted(paths)
    )


def _read_retirement_metadata() -> dict[str, tuple[tuple[str, str], ...]]:
    path = ROOT / RETIREMENT_METADATA
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"无法读取 {RETIREMENT_METADATA}") from error
    if set(raw) != {"schema_version", "sources"}:
        raise ValueError(f"{RETIREMENT_METADATA} 顶层字段必须固定")
    if raw["schema_version"] != 1:
        raise ValueError(f"{RETIREMENT_METADATA} schema_version 必须为 1")
    sources = raw["sources"]
    if not isinstance(sources, list):
        raise ValueError(f"{RETIREMENT_METADATA} sources 必须是数组")
    result: dict[str, tuple[tuple[str, str], ...]] = {}
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != {"recovery_commit", "files"}:
            raise ValueError(f"{RETIREMENT_METADATA} sources[{index}] 字段错误")
        commit = source["recovery_commit"]
        if (
            not isinstance(commit, str)
            or commit not in RETIREMENT_RECOVERY_COMMITS
            or commit in result
        ):
            raise ValueError(f"{RETIREMENT_METADATA} recovery_commit 无效")
        entries = source["files"]
        if not isinstance(entries, list):
            raise ValueError(f"{RETIREMENT_METADATA} sources[{index}].files 必须是数组")
        parsed: list[tuple[str, str]] = []
        seen: set[str] = set()
        for file_index, item in enumerate(entries):
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise ValueError(
                    f"{RETIREMENT_METADATA} sources[{index}].files[{file_index}] 字段错误"
                )
            path = _safe_relative(
                item["path"],
                f"{RETIREMENT_METADATA}:sources[{index}].files[{file_index}].path",
            )
            digest = item["sha256"]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(
                    f"{RETIREMENT_METADATA}:sources[{index}].files[{file_index}].sha256 无效"
                )
            if path in seen:
                raise ValueError(f"{RETIREMENT_METADATA} path 重复: {path}")
            seen.add(path)
            parsed.append((path, digest))
        result[commit] = tuple(sorted(parsed))
    if set(result) != RETIREMENT_RECOVERY_COMMITS:
        raise ValueError(f"{RETIREMENT_METADATA} 必须完整覆盖固定恢复提交集合")
    return result


def _retirement_paths(base: str) -> tuple[set[str], list[str]]:
    """Enable retirement only for one of the two exact recovery commits."""

    resolved = _resolve_commit(base)
    if resolved not in RETIREMENT_RECOVERY_COMMITS:
        return set(), []
    try:
        metadata = _read_retirement_metadata()
        expected = _retirement_inventory(resolved)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        return set(), [f"invalid migration retirement metadata: {error}"]
    if metadata[resolved] != expected:
        return set(), [
            "invalid migration retirement metadata: path/digest inventory does not "
            f"match recovery commit {resolved}"
        ]
    return {path for path, _digest in expected}, []


def _current_toml(path: Path) -> dict[str, Any]:
    return _toml_bytes(path.read_bytes(), path.as_posix())


def _bundle_violations(base: str, retirement_paths: set[str]) -> list[str]:
    problems: list[str] = []
    for spec in _bundle_specs(base):
        manifest_path = spec["manifest_path"]
        if manifest_path in retirement_paths:
            continue
        manifest_file = ROOT / manifest_path
        if not manifest_file.is_file():
            problems.append(f"registered migration bundle changed: {manifest_path}")
            continue
        try:
            manifest = _current_toml(manifest_file)
            catalog_relative, declared_digest = _manifest_migration(
                manifest, manifest_path
            )
            if catalog_relative != spec["catalog_relative"]:
                problems.append(f"registered migration bundle changed: {manifest_path}")
                continue
            catalog_path = ROOT / spec["catalog_path"]
            if not catalog_path.is_file():
                problems.append(
                    f"registered migration bundle changed: {spec['catalog_path']}"
                )
                continue
            catalog_bytes = catalog_path.read_bytes()
            actual_digest = hashlib.sha256(catalog_bytes).hexdigest()
            if actual_digest != declared_digest:
                problems.append(
                    f"registered migration bundle changed: {spec['catalog_path']}"
                )
                continue
            current = _catalog_spec(
                _toml_bytes(catalog_bytes, spec["catalog_path"]),
                read_source=lambda path: (ROOT / path).read_bytes(),
                catalog_path=spec["catalog_path"],
                plugin_path=manifest_path,
            )
            for field in ("bundle_id", "migration_root", "package_name"):
                if current[field] != spec[field]:
                    problems.append(
                        f"registered migration bundle changed: {spec['catalog_path']}"
                    )
                    break
            else:
                for old_path, old_digest in spec["package_files"].items():
                    current_digest = current["package_files"].get(old_path)
                    file_path = ROOT / old_path
                    if current_digest != old_digest or not file_path.is_file():
                        problems.append(
                            f"registered Yoyo migration changed: {old_path}"
                        )
                for migration_id, old_entry in spec["migrations"].items():
                    if current["migrations"].get(migration_id) != old_entry:
                        problems.append(
                            "registered Yoyo migration changed: "
                            f"{spec['catalog_path']}#{migration_id}"
                        )
        except (OSError, UnicodeError, ValueError, RuntimeError):
            problems.append(f"registered migration bundle changed: {manifest_path}")
    return problems


def _retired_path_reintroductions(base: str) -> list[str]:
    changes = _git("diff", "--name-status", str(base), "--", ".").splitlines()
    problems: list[str] = []
    for line in changes:
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status = fields[0][:1]
        if status not in {"A", "C", "R"}:
            continue
        # A rename/copy reports the old path first; only its destination is
        # a possible reintroduction into a retired namespace.
        paths = fields[1:] if status == "A" else fields[2:]
        for path in paths:
            if path in RETIRED_EXACT_PATHS or any(
                path.startswith(prefix) for prefix in RETIRED_PREFIXES
            ):
                problems.append(f"retired migration path reintroduced: {path}")
    return problems


def violations(base: str) -> list[str]:
    """Reject historical migration edits and permit one exact retirement cutover."""

    retirement_paths, problems = _retirement_paths(str(base))
    base_paths = _registered_paths(str(base))
    for old_path in sorted(base_paths):
        current = ROOT / old_path
        if old_path in retirement_paths:
            if current.exists() or current.is_symlink():
                problems.append(f"retired migration must be deleted: {old_path}")
            continue
        try:
            unchanged = (
                current.is_file()
                and _source_bytes(str(base), old_path) == current.read_bytes()
            )
        except (OSError, RuntimeError):
            unchanged = False
        if not unchanged:
            problems.append(f"registered Yoyo migration changed: {old_path}")

    problems.extend(_bundle_violations(str(base), retirement_paths))
    problems.extend(_retired_path_reintroductions(str(base)))
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
