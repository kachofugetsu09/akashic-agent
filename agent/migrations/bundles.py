"""发现并校验外部 artifact 的离线 Yoyo migration bundle。"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, cast

from agent.plugins.source_resolver import (
    ResolvedPluginSource,
    resolve_plugin_sources,
)
from agent.plugins.static_manifest import (
    StaticMigrationDeclaration,
    StaticPluginManifest,
)


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CATALOG_KEYS = {"schema_version", "bundle_id", "version", "migration_root", "migrations"}
_MIGRATION_KEYS = {"id", "path", "depends", "transactional", "sha256"}
_REJECTED_IMPORT_PREFIXES = (
    "plugins",
    "agent.model_runtime",
    "agent.plugins",
    "infra.mobile_realtime",
    "bootstrap",
)
_ALLOWED_CORE_MIGRATION_MODULES = {"agent.migrations.context"}


class MigrationBundleError(ValueError):
    """迁移 bundle 的静态内容不满足安装合同。"""


class MigrationBundleBlocked(RuntimeError):
    """workspace 有待执行迁移，但实现 bundle 尚未安装。"""

    code = "migration_blocked"

    def __init__(
        self,
        *,
        bundle_id: str,
        migration_ids: Sequence[str],
        missing_dependencies: Sequence[str] = (),
        artifact_digest: str | None = None,
    ) -> None:
        self.bundle_id = bundle_id
        self.migration_ids = tuple(migration_ids)
        self.missing_dependencies = tuple(missing_dependencies)
        self.artifact_digest = artifact_digest
        detail = [
            self.code,
            f"bundle={bundle_id}",
            f"migrations={','.join(self.migration_ids)}",
        ]
        if self.missing_dependencies:
            detail.append(
                f"missing_dependencies={','.join(self.missing_dependencies)}"
            )
        if artifact_digest:
            detail.append(f"artifact_digest={artifact_digest}")
        super().__init__(" ".join(detail))


@dataclass(frozen=True, slots=True)
class MigrationSpec:
    """一个 bundle 中已冻结的 migration 元数据。"""

    migration_id: str
    path: str
    depends: tuple[str, ...]
    transactional: bool
    sha256: str


@dataclass(frozen=True, slots=True)
class MigrationBundle:
    """一个不可变 artifact 提供的 Yoyo source 和其完整性证明。"""

    bundle_id: str
    version: str
    artifact_root: Path
    migration_root: Path
    catalog_path: Path
    catalog_sha256: str
    bundle_sha256: str
    migrations: tuple[MigrationSpec, ...]

    @property
    def migration_ids(self) -> tuple[str, ...]:
        return tuple(item.migration_id for item in self.migrations)


@dataclass(frozen=True, slots=True)
class MigrationRequirement:
    """Core 中立索引记录的 ID、依赖和外部 owner token。"""

    migration_id: str
    bundle_id: str | None
    depends: tuple[str, ...]
    transactional: bool


def discover_migration_bundles(
    *,
    plugin_dirs: Sequence[Path] = (),
    installed_cache_root: Path | None = None,
) -> tuple[MigrationBundle, ...]:
    """只从明确的插件 source 读取 bundle，不扫描 checkout/plugins。"""

    sources = resolve_plugin_sources(
        plugin_dirs,
        installed_cache_root=installed_cache_root,
        installed_selector="stable",
    )
    bundles: list[MigrationBundle] = []
    seen_bundles: set[str] = set()
    seen_migrations: set[str] = set()
    for source in sources:
        declaration = source.static_manifest.migration if source.static_manifest else None
        if declaration is None:
            continue
        bundle = load_migration_bundle(source, declaration)
        if bundle.bundle_id in seen_bundles:
            raise MigrationBundleError(f"重复 migration bundle: {bundle.bundle_id}")
        overlap = seen_migrations.intersection(bundle.migration_ids)
        if overlap:
            raise MigrationBundleError(
                "重复 migration ID: " + ", ".join(sorted(overlap))
            )
        seen_bundles.add(bundle.bundle_id)
        seen_migrations.update(bundle.migration_ids)
        bundles.append(bundle)
    return tuple(sorted(bundles, key=lambda item: item.bundle_id))


def load_migration_bundle(
    source: ResolvedPluginSource,
    declaration: StaticMigrationDeclaration,
) -> MigrationBundle:
    """解析一个静态 bundle 并核对每个 migration 文件的 digest。"""

    root = source.plugin_root.resolve(strict=True)
    catalog_path = _inside_file(root, root / declaration.catalog, "migration catalog")
    catalog_bytes = catalog_path.read_bytes()
    actual_catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
    if actual_catalog_sha256 != declaration.catalog_sha256:
        raise MigrationBundleError(
            "migration catalog digest 漂移: "
            f"expected={declaration.catalog_sha256}, actual={actual_catalog_sha256}"
        )
    try:
        raw = tomllib.loads(catalog_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise MigrationBundleError(f"migration catalog 无法解析: {catalog_path}") from error
    if set(raw) != _CATALOG_KEYS:
        raise MigrationBundleError(
            f"migration catalog 顶层字段错误: {sorted(set(raw) - _CATALOG_KEYS)}"
        )
    if raw.get("schema_version") != 1:
        raise MigrationBundleError("migration catalog schema_version 必须为 1")
    bundle_id = _id(raw.get("bundle_id"), "bundle_id")
    version = _version(raw.get("version"), "version")
    migration_root_value = raw.get("migration_root")
    if not isinstance(migration_root_value, str):
        raise MigrationBundleError("migration_root 必须是字符串路径")
    migration_root = _inside_directory(
        root,
        root / _relative_path(migration_root_value, "migration_root"),
        "migration_root",
    )
    raw_migrations = raw.get("migrations")
    if not isinstance(raw_migrations, list) or not raw_migrations:
        raise MigrationBundleError("migration catalog.migrations 必须是非空数组")

    specs: list[MigrationSpec] = []
    ids: set[str] = set()
    for index, item in enumerate(raw_migrations):
        if not isinstance(item, dict) or set(item) != _MIGRATION_KEYS:
            raise MigrationBundleError(
                f"migrations[{index}] 字段必须是 {sorted(_MIGRATION_KEYS)}"
            )
        migration_id = _id(item.get("id"), f"migrations[{index}].id")
        if migration_id in ids:
            raise MigrationBundleError(f"migration ID 重复: {migration_id}")
        path_value = item.get("path")
        if not isinstance(path_value, str):
            raise MigrationBundleError(f"migrations[{index}].path 必须是字符串")
        relative_path = _relative_path(path_value, f"migrations[{index}].path")
        path = _inside_file(
            migration_root,
            migration_root / relative_path,
            f"migrations[{index}].path",
        )
        if path.suffix != ".py" or path.name == "__init__.py":
            raise MigrationBundleError(
                f"migrations[{index}].path 必须是普通 Python migration 文件"
            )
        depends_value = item.get("depends")
        if not isinstance(depends_value, list) or not all(
            isinstance(value, str) for value in depends_value
        ):
            raise MigrationBundleError(f"migrations[{index}].depends 无效")
        depends = tuple(depends_value)
        if len(set(depends)) != len(depends) or migration_id in depends:
            raise MigrationBundleError(f"migrations[{index}].depends 无效")
        transactional = item.get("transactional")
        if not isinstance(transactional, bool):
            raise MigrationBundleError(f"migrations[{index}].transactional 无效")
        sha256 = item.get("sha256")
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
            raise MigrationBundleError(f"migrations[{index}].sha256 无效")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != sha256:
            raise MigrationBundleError(
                f"migration source digest 漂移: id={migration_id}"
            )
        _validate_source_imports(path)
        ids.add(migration_id)
        specs.append(
            MigrationSpec(
                migration_id=migration_id,
                path=relative_path,
                depends=depends,
                transactional=transactional,
                sha256=sha256,
            )
        )

    # 1. Yoyo 会扫描 source directory，未列入 catalog 的 .py 不能藏入执行面。
    declared_paths = {item.path for item in specs}
    actual_paths = {
        path.relative_to(migration_root).as_posix()
        for path in migration_root.glob("*.py")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != declared_paths:
        raise MigrationBundleError(
            "migration_root 文件集合与 catalog 不一致: "
            f"missing={sorted(declared_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - declared_paths)}"
        )
    _validate_local_dependency_graph(specs, bundle_id)
    bundle_sha256 = _bundle_digest(catalog_bytes, specs, migration_root)
    return MigrationBundle(
        bundle_id=bundle_id,
        version=version,
        artifact_root=root,
        migration_root=migration_root,
        catalog_path=catalog_path,
        catalog_sha256=actual_catalog_sha256,
        bundle_sha256=bundle_sha256,
        migrations=tuple(specs),
    )


def validate_migration_artifact(
    plugin_root: Path,
    *,
    static_manifest: StaticPluginManifest,
) -> MigrationBundle | None:
    """在 artifact 发布前校验其声明的 migration bundle。"""

    declaration = static_manifest.migration
    if declaration is None:
        return None
    source = ResolvedPluginSource(
        plugin_root=plugin_root.resolve(strict=True),
        source_type="installed",
        plugin_name=static_manifest.name,
        entrypoint=static_manifest.entrypoint,
        static_manifest=static_manifest,
    )
    return load_migration_bundle(source, declaration)


def load_migration_requirements(path: Path) -> tuple[MigrationRequirement, ...]:
    """读取 Core 的中立 ID 索引；索引没有任何 migration 实现。"""

    if not path.exists():
        return ()
    if path.is_symlink() or not path.is_file():
        raise MigrationBundleError(f"migration requirement catalog 必须是普通文件: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise MigrationBundleError(f"migration requirement catalog 无法解析: {path}") from error
    if set(raw) != {"schema_version", "migrations"} or raw.get("schema_version") != 1:
        raise MigrationBundleError("migration requirement catalog schema 无效")
    values = raw.get("migrations")
    if not isinstance(values, list):
        raise MigrationBundleError("migration requirement catalog.migrations 必须是数组")
    result: list[MigrationRequirement] = []
    seen: set[str] = set()
    for index, item in enumerate(values):
        if not isinstance(item, dict) or set(item) != {
            "id", "bundle", "depends", "transactional"
        }:
            raise MigrationBundleError(f"requirements[{index}] 字段错误")
        migration_id = _id(item.get("id"), f"requirements[{index}].id")
        if migration_id in seen:
            raise MigrationBundleError(f"requirement ID 重复: {migration_id}")
        bundle = item.get("bundle")
        if bundle is not None:
            bundle = _id(bundle, f"requirements[{index}].bundle")
        depends = item.get("depends")
        if not isinstance(depends, list) or not all(
            isinstance(value, str) for value in depends
        ):
            raise MigrationBundleError(f"requirements[{index}].depends 无效")
        transactional = item.get("transactional")
        if not isinstance(transactional, bool):
            raise MigrationBundleError(f"requirements[{index}].transactional 无效")
        seen.add(migration_id)
        result.append(
            MigrationRequirement(
                migration_id=migration_id,
                bundle_id=cast(str | None, bundle),
                depends=tuple(depends),
                transactional=transactional,
            )
        )
    return tuple(result)


def validate_bundle_dependencies(
    bundles: Sequence[MigrationBundle],
    *,
    core_migration_ids: Sequence[str],
    requirements: Sequence[MigrationRequirement] = (),
) -> None:
    """在 Yoyo 执行前拒绝缺失 owner、重复元数据和断裂依赖。"""

    available_ids = set(core_migration_ids)
    available_ids.update(item.migration_id for bundle in bundles for item in bundle.migrations)
    bundle_by_id = {bundle.bundle_id: bundle for bundle in bundles}
    for requirement in requirements:
        missing_requirements = tuple(
            sorted(dependency for dependency in requirement.depends if dependency not in available_ids)
        )
        if missing_requirements:
            raise MigrationBundleBlocked(
                bundle_id=requirement.bundle_id or "core",
                migration_ids=(requirement.migration_id,),
                missing_dependencies=missing_requirements,
            )
        if requirement.migration_id not in available_ids:
            continue
        if requirement.bundle_id is None:
            if requirement.migration_id not in set(core_migration_ids):
                raise MigrationBundleError(
                    f"Core requirement 错误地缺少实现: {requirement.migration_id}"
                )
            continue
        bundle = bundle_by_id.get(requirement.bundle_id)
        if bundle is None:
            continue
        spec = next(
            (
                item
                for item in bundle.migrations
                if item.migration_id == requirement.migration_id
            ),
            None,
        )
        if spec is None:
            raise MigrationBundleError(
                f"bundle 未提供 requirement ID: {requirement.migration_id}"
            )
        if spec.depends != requirement.depends or spec.transactional != requirement.transactional:
            raise MigrationBundleError(
                f"bundle metadata 与 Core requirement 不一致: {requirement.migration_id}"
            )
    for bundle in bundles:
        missing: set[str] = set()
        for spec in bundle.migrations:
            missing.update(dep for dep in spec.depends if dep not in available_ids)
        if missing:
            raise MigrationBundleBlocked(
                bundle_id=bundle.bundle_id,
                migration_ids=bundle.migration_ids,
                missing_dependencies=tuple(sorted(missing)),
                artifact_digest=bundle.bundle_sha256,
            )
def validate_pending_requirements(
    requirements: Sequence[MigrationRequirement],
    *,
    loaded_ids: Sequence[str],
    applied_ids: Sequence[str],
    bundles: Sequence[MigrationBundle],
) -> None:
    """对缺少实现的未落账 ID 返回明确的 blocked，而不是伪造成功。"""

    loaded = set(loaded_ids)
    applied = set(applied_ids)
    bundles_by_id = {bundle.bundle_id: bundle for bundle in bundles}
    for requirement in requirements:
        if requirement.migration_id in loaded or requirement.migration_id in applied:
            continue
        owner = requirement.bundle_id or "core"
        bundle = bundles_by_id.get(owner)
        digest = bundle.bundle_sha256 if bundle is not None else None
        raise MigrationBundleBlocked(
            bundle_id=owner,
            migration_ids=(requirement.migration_id,),
            artifact_digest=digest,
        )


@contextmanager
def migration_import_paths(bundles: Sequence[MigrationBundle]) -> Iterator[None]:
    """只在离线 Yoyo load/apply 窗口暴露 bundle 根，不污染 runtime import path。"""

    paths = [str(bundle.artifact_root) for bundle in bundles]
    original = list(sys.path)
    try:
        for path in reversed(paths):
            if path not in sys.path:
                sys.path.insert(0, path)
        yield
    finally:
        sys.path[:] = original


def _validate_local_dependency_graph(
    specs: Sequence[MigrationSpec], bundle_id: str
) -> None:
    local_ids = {item.migration_id for item in specs}
    graph = {
        item.migration_id: {dep for dep in item.depends if dep in local_ids}
        for item in specs
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise MigrationBundleError(f"bundle 依赖环: {bundle_id}:{node}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def _validate_source_imports(path: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise MigrationBundleError(f"migration source 无法解析: {path}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            dynamic_import = (
                isinstance(function, ast.Name)
                and function.id in {"__import__", "import_module"}
            ) or (
                isinstance(function, ast.Attribute)
                and function.attr == "import_module"
            )
            if dynamic_import:
                raise MigrationBundleError(
                    f"migration 不允许动态 import: {path}"
                )
        if isinstance(node, ast.Import):
            modules = [item.name for item in node.names]
            levels = [0] * len(modules)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise MigrationBundleError(f"migration 不允许 relative import: {path}")
            modules = [node.module or ""]
            levels = [node.level]
        else:
            continue
        for module, level in zip(modules, levels):
            if level:
                continue
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in _REJECTED_IMPORT_PREFIXES
            ):
                raise MigrationBundleError(
                    f"migration 不得 import 当前 runtime/插件 namespace: {path}:{module}"
                )
            if (
                module == "agent.migrations"
                or module.startswith("agent.migrations.")
            ) and module not in _ALLOWED_CORE_MIGRATION_MODULES:
                raise MigrationBundleError(
                    f"migration 不得 import Core 业务 migration helper: {path}:{module}"
                )


def _bundle_digest(
    catalog_bytes: bytes,
    specs: Sequence[MigrationSpec],
    migration_root: Path,
) -> str:
    digest = hashlib.sha256()
    digest.update(catalog_bytes)
    for spec in sorted(specs, key=lambda item: item.migration_id):
        digest.update(spec.migration_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update((migration_root / spec.path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _id(raw: object, label: str) -> str:
    if not isinstance(raw, str) or _ID.fullmatch(raw) is None:
        raise MigrationBundleError(f"{label} 无效")
    return raw


def _version(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise MigrationBundleError(f"{label} 无效")
    return raw


def _relative_path(raw: str, label: str) -> str:
    path = Path(raw)
    if (
        not raw
        or raw != raw.strip()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in raw
    ):
        raise MigrationBundleError(f"{label} 必须是 artifact 内的相对路径")
    return "/".join(path.parts)


def _inside_directory(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_dir() or not resolved.is_relative_to(root):
        raise MigrationBundleError(f"{label} 必须是 artifact 内的普通目录: {path}")
    _reject_symlink_ancestors(root, path, label)
    return resolved


def _inside_file(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(root):
        raise MigrationBundleError(f"{label} 必须是 artifact 内的普通文件: {path}")
    _reject_symlink_ancestors(root, path, label)
    return resolved


def _reject_symlink_ancestors(root: Path, path: Path, label: str) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise MigrationBundleError(f"{label} 不能穿过符号链接: {current}")
