from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence
from urllib.parse import quote

from yoyo import get_backend, read_migrations
from yoyo.migrations import MigrationList

from agent.migrations.context import bind_migration_context
from agent.migrations.bundles import (
    MigrationBundleBlocked,
    MigrationBundleError,
    discover_migration_bundles,
    load_migration_requirements,
    migration_import_paths,
    validate_bundle_dependencies,
    validate_pending_requirements,
)
from agent.plugins.manifest import plugins_root
from bootstrap.workspace_lock import WorkspaceInstanceLock


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_USERNAME_ENV_KEYS = ("LOGNAME", "USER", "LNAME", "USERNAME")


@dataclass(frozen=True)
class MigrationOutcome:
    state: Literal["current", "migrated"]
    migrations: tuple[str, ...] = ()


class MigrationRunner:
    """在 runtime 启动前执行缺失的 Yoyo 迁移。"""

    def __init__(
        self,
        *,
        repo_root: Path,
        config_path: Path,
        workspace: Path,
        plugin_dirs: Sequence[Path] = (),
        installed_cache_root: Path | None = None,
        migration_catalog: Path | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.config_path = config_path.expanduser().resolve()
        self.workspace = workspace.expanduser().resolve()
        # The Core source is intentionally a separate, implementation-free root;
        # silently falling back to a checkout's retired business migrations would
        # defeat the external bundle boundary.
        self.migrations_root = self.repo_root / "migrations" / "core"
        self.ledger_path = self.workspace / "migrations.sqlite3"
        # 保留路径上的 symlink 形状，让 source resolver 能够拒绝它，而不是
        # 先 resolve 后把越界路径伪装成普通 cache 根。
        self.plugin_dirs = tuple(path.expanduser() for path in plugin_dirs)
        self.installed_cache_root = (
            (installed_cache_root or (plugins_root() / "cache"))
            .expanduser()
        )
        self.migration_catalog = (
            migration_catalog
            if migration_catalog is not None
            else self.repo_root / "migrations" / "catalog.toml"
        ).expanduser()

    def run(self) -> MigrationOutcome:
        """执行当前目录并返回本次落账的迁移 ID。"""

        # 1. 复用 workspace 锁串行化迁移与 runtime 启动
        workspace_lock = WorkspaceInstanceLock(self.workspace)
        workspace_lock.acquire()
        try:
            return self._apply_pending()
        finally:
            workspace_lock.release()

    def _apply_pending(self) -> MigrationOutcome:
        """加载不可变目录并提交全部缺失迁移。"""

        # 1. 初始化由 workspace 持有的迁移账本
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            core_migrations = _read_migrations(str(self.migrations_root))
            bundles = discover_migration_bundles(
                plugin_dirs=self.plugin_dirs,
                installed_cache_root=self.installed_cache_root,
            )
            requirements = load_migration_requirements(self.migration_catalog)
            core_ids = tuple(migration.id for migration in core_migrations)
            bundle_ids = tuple(
                migration_id
                for bundle in bundles
                for migration_id in bundle.migration_ids
            )
            validate_bundle_dependencies(
                bundles,
                core_migration_ids=core_ids,
                requirements=requirements,
            )
            validate_pending_requirements(
                requirements,
                loaded_ids=core_ids + bundle_ids,
                applied_ids=_read_applied_ids(self.ledger_path),
                bundles=bundles,
            )
            backend = get_backend(self._ledger_uri())
            os.chmod(self.ledger_path, 0o600)

            # 2. 为 Yoyo Python step 绑定明确的安装路径
            with (
                _bind_yoyo_username(),
                backend,
                bind_migration_context(
                    config_path=self.config_path,
                    workspace=self.workspace,
                ),
                migration_import_paths(bundles),
            ):
                migrations = _read_migrations(
                    str(self.migrations_root),
                    *(str(bundle.migration_root) for bundle in bundles),
                )
                pending = backend.to_apply(migrations)
                migration_ids = tuple(migration.id for migration in pending)
                backend.apply_migrations(pending)
        except (MigrationBundleBlocked, MigrationBundleError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Yoyo 迁移失败: ledger={self.ledger_path} detail={exc}"
            ) from exc

        state: Literal["current", "migrated"] = (
            "migrated" if migration_ids else "current"
        )
        return MigrationOutcome(state=state, migrations=migration_ids)

    def _ledger_uri(self) -> str:
        encoded = quote(self.ledger_path.as_posix(), safe="/:")
        return f"sqlite:///{encoded}"


@contextmanager
def _bind_yoyo_username() -> Iterator[None]:
    """为没有 OS 用户记录的容器提供稳定的 Yoyo 审计身份。"""
    if any(os.environ.get(key) for key in _USERNAME_ENV_KEYS):
        yield
        return

    os.environ["USER"] = "akashic"
    try:
        yield
    finally:
        del os.environ["USER"]


def migrate_installation(config_path: Path, workspace: Path) -> MigrationOutcome:
    return MigrationRunner(
        repo_root=_PROJECT_ROOT,
        config_path=config_path,
        workspace=workspace,
    ).run()


def _read_applied_ids(path: Path) -> tuple[str, ...]:
    """只读 ledger 已成功 ID，缺表视为空账本。"""

    if not path.exists():
        return ()
    connection = sqlite3.connect(path)
    try:
        try:
            rows = connection.execute(
                "SELECT migration_id FROM _yoyo_migration"
            ).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error).lower():
                raise
            return ()
    finally:
        connection.close()
    return tuple(str(row[0]) for row in rows)


def _read_migrations(*sources: str) -> MigrationList:
    """Load Yoyo files while excluding the package marker from execution."""

    migrations = read_migrations(*sources)
    return MigrationList(
        migration for migration in migrations if migration.id != "__init__"
    )
