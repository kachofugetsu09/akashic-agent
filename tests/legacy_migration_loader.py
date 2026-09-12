"""在测试中按外部 migration bundle 的包身份加载冻结迁移。"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shutil
import sys
from types import ModuleType

import yoyo

from agent.migrations.bundles import (
    _read_migrations,
    migration_import_paths,
)


_PROJECT_ROOT = Path(__file__).parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "plugins/legacy_upgrade/legacy_upgrade_migrations"
_PACKAGE_NAME = "test_legacy_upgrade_migrations"


@dataclass(frozen=True)
class _TestBundle:
    """提供生产 loader 所需的最小只读 bundle 形状。"""

    package_name: str
    migration_root: Path


def _bundle(root: Path) -> _TestBundle:
    return _TestBundle(package_name=_PACKAGE_NAME, migration_root=root)


def _clear_package() -> None:
    """清掉上一项测试留下的 bundle 包，避免跨用例复用模块身份。"""

    for name in tuple(sys.modules):
        if name == _PACKAGE_NAME or name.startswith(f"{_PACKAGE_NAME}."):
            del sys.modules[name]


def prepare_migration_directory(directory: Path) -> Path:
    """为临时迁移目录补上冻结包的相对 helper。"""

    directory.mkdir(parents=True, exist_ok=True)
    init = directory / "__init__.py"
    if not init.exists():
        init.write_text(
            '"""Test copy of the legacy migration package."""\n',
            encoding="utf-8",
        )
    shutil.copytree(
        _SOURCE_ROOT / "support",
        directory / "support",
        dirs_exist_ok=True,
    )
    return directory


def load_migration_object(stem: str) -> ModuleType:
    """按完整包名加载单个迁移，并保留可替换的模块全局。"""

    _clear_package()
    path = _SOURCE_ROOT / f"{stem.removesuffix('.py')}.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    bundle = _bundle(_SOURCE_ROOT)
    module_name = f"{bundle.package_name}.{path.stem}"
    with migration_import_paths((bundle,)):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载迁移: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        original_step = yoyo.step
        yoyo.step = lambda *args, **kwargs: args[0] if len(args) == 1 else args
        try:
            spec.loader.exec_module(module)
        finally:
            yoyo.step = original_step
        loaded = {
            name: value
            for name, value in sys.modules.items()
            if name == _PACKAGE_NAME or name.startswith(f"{_PACKAGE_NAME}.")
        }
    # Production keeps this package alive while Yoyo invokes the callback. The
    # callback fixtures run after this function returns, so retain the exact
    # loaded package modules until the next fixture load clears them.
    sys.modules.update(loaded)
    return module


def load_migration_module(stem: str) -> dict[str, object]:
    """按完整包名加载单个迁移，返回其模块命名空间映射。"""

    return dict(vars(load_migration_object(stem)))


def load_migration_namespace(stem: str) -> ModuleType:
    """Load one migration object so monkeypatches reach its global scope."""

    return load_migration_object(stem)


def load_bundle_migrations(directory: Path):
    """加载临时 bundle 的所有 Yoyo 对象并在返回前完成相对导入。"""

    _clear_package()
    prepare_migration_directory(directory)
    bundle = _bundle(directory)
    empty_core = directory.parent / "_empty_core_migrations"
    empty_core.mkdir(exist_ok=True)
    with migration_import_paths((bundle,)):
        migrations = _read_migrations(str(empty_core), (bundle,))
        for migration in migrations:
            migration.load()
    return migrations
