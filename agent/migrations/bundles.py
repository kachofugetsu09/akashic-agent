"""发现并校验外部 artifact 的离线 Yoyo migration bundle。"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, cast

from yoyo import read_migrations
from yoyo.migrations import Migration, MigrationList, StepCollector, exceptions

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
_CATALOG_KEYS = {"schema_version", "bundle_id", "version", "migration_root", "package_name", "files", "migrations"}
_FILE_KEYS = {"path", "sha256"}
_MIGRATION_KEYS = {"id", "path", "depends", "transactional", "sha256"}
_PACKAGE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")
_REJECTED_IMPORT_PREFIXES = (
    "plugins",
    "agent.model_runtime",
    "agent.plugins",
    "infra.mobile_realtime",
    "bootstrap",
)
_ALLOWED_CORE_MIGRATION_MODULES = frozenset({
    "agent.migrations.context",
    "agent.plugin_composition",
    "agent.plugin_composition.artifacts",
    "agent.plugin_composition.messages",
    "agent.plugin_contracts",
    "agent.plugin_contracts.message",
    "agent.turn_effects",
    "core.common.timekit",
    "core.net.http",
    "infra.persistence.json_store",
    "memory2.embedder",
    "session.embedding_store",
    "session.identities",
    "session.log",
    "session.message",
    "session.message_codec",
})


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
    package_name: str
    package_files: tuple[tuple[str, str], ...]

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
    seen_packages: set[str] = set()
    seen_migrations: set[str] = set()
    for source in sources:
        declaration = source.static_manifest.migration if source.static_manifest else None
        if declaration is None:
            continue
        bundle = load_migration_bundle(source, declaration)
        if bundle.bundle_id in seen_bundles:
            raise MigrationBundleError(f"重复 migration bundle: {bundle.bundle_id}")
        if bundle.package_name in seen_packages:
            raise MigrationBundleError(
                f"重复 migration package_name: {bundle.package_name}"
            )
        overlap = seen_migrations.intersection(bundle.migration_ids)
        if overlap:
            raise MigrationBundleError(
                "重复 migration ID: " + ", ".join(sorted(overlap))
            )
        seen_bundles.add(bundle.bundle_id)
        seen_packages.add(bundle.package_name)
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
    package_name = raw.get("package_name")
    if not isinstance(package_name, str) or _PACKAGE_NAME.fullmatch(package_name) is None:
        raise MigrationBundleError("package_name 必须是安全 Python 包名")
    migration_root_value = raw.get("migration_root")
    if not isinstance(migration_root_value, str):
        raise MigrationBundleError("migration_root 必须是字符串路径")
    migration_root = _inside_directory(
        root,
        root / _relative_path(migration_root_value, "migration_root"),
        "migration_root",
    )
    if package_name != migration_root.name:
        raise MigrationBundleError("package_name 必须等于 migration_root 目录名")
    package_init = migration_root / "__init__.py"
    if not package_init.is_file() or package_init.is_symlink():
        raise MigrationBundleError("migration_root 必须包含普通 __init__.py")
    raw_files = raw.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise MigrationBundleError("migration catalog.files 必须是非空数组")
    package_files: list[tuple[str, str]] = []
    seen_files: set[str] = set()
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict) or set(item) != _FILE_KEYS:
            raise MigrationBundleError(f"files[{index}] 字段必须是 {sorted(_FILE_KEYS)}")
        path_value = item.get("path")
        if not isinstance(path_value, str):
            raise MigrationBundleError(f"files[{index}].path 必须是字符串")
        relative = _relative_path(path_value, f"files[{index}].path")
        if "__pycache__" in Path(relative).parts or relative.endswith(".pyc"):
            raise MigrationBundleError(
                f"files[{index}].path 不能声明 Python 运行时缓存"
            )
        path = _inside_file(migration_root, migration_root / relative, f"files[{index}].path")
        digest = item.get("sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise MigrationBundleError(f"files[{index}].sha256 无效")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise MigrationBundleError(f"bundle helper source digest 漂移: {relative}")
        if relative in seen_files:
            raise MigrationBundleError(f"bundle 文件重复: {relative}")
        seen_files.add(relative)
        package_files.append((relative, digest))
    actual_files = {
        path.relative_to(migration_root).as_posix()
        for path in migration_root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and "__pycache__" not in path.relative_to(migration_root).parts
        and path.suffix != ".pyc"
    }
    if actual_files != seen_files:
        raise MigrationBundleError(
            "migration_root 文件集合与 catalog.files 不一致: "
            f"missing={sorted(seen_files - actual_files)}, extra={sorted(actual_files - seen_files)}"
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
        _validate_source_imports(path, migration_root=migration_root)
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
        if path.name != "__init__.py" and (path.is_file() or path.is_symlink())
    }
    if actual_paths != declared_paths:
        raise MigrationBundleError(
            "migration_root 文件集合与 catalog 不一致: "
            f"missing={sorted(declared_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - declared_paths)}"
        )
    for relative, _digest in package_files:
        _validate_source_imports(migration_root / relative, migration_root=migration_root)
    _validate_local_dependency_graph(specs, bundle_id)
    bundle_sha256 = _bundle_digest(catalog_bytes, migration_root, package_files)
    return MigrationBundle(
        bundle_id=bundle_id,
        version=version,
        artifact_root=root,
        migration_root=migration_root,
        catalog_path=catalog_path,
        catalog_sha256=actual_catalog_sha256,
        bundle_sha256=bundle_sha256,
        migrations=tuple(specs),
        package_name=package_name,
        package_files=tuple(package_files),
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
    require_missing_bundles: bool = True,
) -> None:
    """在 Yoyo 执行前拒绝缺失 owner、重复元数据和断裂依赖。"""

    available_ids = set(core_migration_ids)
    available_ids.update(item.migration_id for bundle in bundles for item in bundle.migrations)
    bundle_by_id = {bundle.bundle_id: bundle for bundle in bundles}
    for requirement in requirements:
        if (
            requirement.bundle_id is not None
            and requirement.bundle_id not in bundle_by_id
            and not require_missing_bundles
        ):
            continue
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
    require_missing_bundles: bool = True,
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
        if (
            bundle is None
            and requirement.bundle_id is not None
            and not require_missing_bundles
        ):
            continue
        digest = bundle.bundle_sha256 if bundle is not None else None
        raise MigrationBundleBlocked(
            bundle_id=owner,
            migration_ids=(requirement.migration_id,),
            artifact_digest=digest,
        )


@contextmanager
def migration_import_paths(bundles: Sequence[MigrationBundle]) -> Iterator[None]:
    """在 Yoyo load/apply 窗口安装精确的 bundle package 身份。"""

    package_names = tuple(bundle.package_name for bundle in bundles)
    if len(set(package_names)) != len(package_names):
        raise MigrationBundleError("migration package_name 必须在一次加载中唯一")
    loaded_names = set(sys.modules)
    for package_name in package_names:
        if any(
            name == package_name or name.startswith(package_name + ".")
            for name in loaded_names
        ):
            raise MigrationBundleError(
                f"migration package_name 已在当前进程加载: {package_name}"
            )
        if importlib.util.find_spec(package_name) is not None:
            raise MigrationBundleError(
                f"migration package_name 与现有模块冲突: {package_name}"
            )

    original_modules = set(sys.modules)
    try:
        for bundle in bundles:
            package_name = bundle.package_name
            init_path = bundle.migration_root / "__init__.py"
            spec = importlib.util.spec_from_file_location(
                package_name,
                init_path,
                submodule_search_locations=[str(bundle.migration_root)],
            )
            if spec is None or spec.loader is None:
                raise MigrationBundleError(
                    f"migration package 无法加载: {package_name}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)
        yield
    finally:
        for module_name in tuple(sys.modules):
            if module_name in original_modules or not any(
                module_name == package_name
                or module_name.startswith(package_name + ".")
                for package_name in package_names
            ):
                continue
            sys.modules.pop(module_name, None)


class _BundleMigration(Migration):
    """Load one Yoyo file under its validated external package identity."""

    def __init__(self, migration_id: str, path: str, source_dir: str, package_name: str):
        super().__init__(migration_id, path, source_dir)
        self._package_name = package_name

    def load(self) -> None:
        if self.loaded:
            return
        collector = StepCollector(migration=self)
        with open(self.path, "r", encoding="utf-8") as stream:
            self.source = stream.read()
        if self.is_raw_sql():
            super().load()
            return
        module_name = f"{self._package_name}.{Path(self.path).stem}"
        spec = importlib.util.spec_from_file_location(module_name, self.path)
        if spec is None or spec.loader is None:
            raise exceptions.BadMigration(self.path)
        module = importlib.util.module_from_spec(spec)
        module.step = collector.add_step  # type: ignore[attr-defined]
        module.group = collector.add_step_group  # type: ignore[attr-defined]
        module.transaction = collector.add_step_group  # type: ignore[attr-defined]
        module.__yoyo_collector__ = collector  # type: ignore[attr-defined]
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise exceptions.BadMigration(self.path, error) from error
        depends = getattr(module, "__depends__", [])
        if isinstance(depends, (str, bytes)):
            depends = [depends]
        self._depends = {
            Migration._Migration__all_migrations.get(identifier)  # type: ignore[attr-defined]
            for identifier in depends
        }
        if None in self._depends:
            raise exceptions.BadMigration(
                f"Could not resolve dependencies in {self.path}"
            )
        self.module = module
        self.use_transactions = getattr(module, "__transactional__", True)
        self.steps = collector.create_steps(self.use_transactions)


def _read_migrations(
    core_source: str,
    bundles: Sequence[MigrationBundle] = (),
) -> MigrationList:
    """Load Core and external files without changing Yoyo/importlib globals."""

    core = read_migrations(core_source)
    migrations = list(core)
    for bundle in bundles:
        source = read_migrations(str(bundle.migration_root))
        migrations.extend(
            _BundleMigration(
                migration.id,
                migration.path,
                migration.source_dir,
                bundle.package_name,
            )
            for migration in source
            if migration.id != "__init__"
        )
    return MigrationList(migrations)


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


def _validate_source_imports(path: Path, *, migration_root: Path) -> None:
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
                _validate_relative_import(path, migration_root, node)
                continue
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
            if _is_disallowed_migration_host(module):
                raise MigrationBundleError(
                    f"migration 不得 import Core 业务 migration helper: {path}:{module}"
                )


def _is_disallowed_migration_host(module: str) -> bool:
    """Reject private Core imports except the audited migration host atoms."""

    top = module.split(".", 1)[0]
    if top not in {"agent", "core", "infra", "memory2", "session"}:
        return False
    return module not in _ALLOWED_CORE_MIGRATION_MODULES


def _validate_relative_import(path: Path, migration_root: Path, node: ast.ImportFrom) -> None:
    """Allow only relative imports whose target stays inside this bundle package."""
    current_package = path.parent
    target = current_package
    for _ in range(node.level - 1):
        target = target.parent
    if node.module:
        target = target.joinpath(*node.module.split("."))
    target = target.resolve(strict=False)
    if not target.is_relative_to(migration_root):
        raise MigrationBundleError(f"migration relative import 越过 bundle: {path}")
    if target.exists() and not target.is_dir() and target.suffix != ".py":
        raise MigrationBundleError(f"migration relative import 目标无效: {path}")
    candidates = (target, target.with_suffix(".py"), target / "__init__.py")
    if not any(candidate.is_file() for candidate in candidates):
        raise MigrationBundleError(f"migration relative import 缺少 bundle helper: {path}:{node.module or ''}")


def _bundle_digest(
    catalog_bytes: bytes,
    migration_root: Path,
    package_files: Sequence[tuple[str, str]],
) -> str:
    digest = hashlib.sha256()
    digest.update(catalog_bytes)
    for relative, expected in sorted(package_files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(expected.encode("ascii"))
        digest.update(b"\0")
        digest.update((migration_root / relative).read_bytes())
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
