from __future__ import annotations

import json
import os
import tempfile
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence
from urllib.parse import quote

from yoyo import get_backend

from agent.migrations.context import bind_migration_context
from agent.migrations.bundles import (
    MigrationBundleBlocked,
    MigrationBundleError,
    _read_migrations,
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
        fresh = _workspace_is_empty(self.workspace, self.config_path)
        baseline = _read_baseline(self.ledger_path)
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
            if baseline is None and fresh:
                baseline = tuple(item.migration_id for item in requirements
                                 if item.migration_id not in core_ids + bundle_ids)
            applicable = tuple(item for item in requirements
                               if item.migration_id not in (baseline or ()))
            validate_bundle_dependencies(
                bundles,
                core_migration_ids=core_ids,
                requirements=requirements,
                require_missing_bundles=False,
            )
            validate_pending_requirements(
                applicable,
                loaded_ids=core_ids + bundle_ids,
                applied_ids=_read_applied_ids(self.ledger_path),
                bundles=bundles,
            )
            if fresh and baseline is not None:
                _save_baseline(self.ledger_path, baseline)
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
                    str(self.migrations_root), bundles
                )
                selected = type(migrations)(
                    (item for item in migrations if item.id not in (baseline or ())),
                    migrations.post_apply,
                )
                pending = backend.to_apply(selected)
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


def _workspace_is_empty(workspace: Path, config_path: Path) -> bool:
    """只有锁与调用者配置之外没有任何文件时才能建立新 workspace 起点。"""
    ignored = {WorkspaceInstanceLock(workspace).path, config_path}
    return not any(path not in ignored and (path.is_file() or path.is_symlink())
                   for path in workspace.rglob("*"))


def _read_baseline(path: Path) -> tuple[str, ...] | None:
    """读取明确的新建起点；不根据运行后出现的数据文件推测历史。"""
    if not path.exists():
        return None
    with closing(sqlite3.connect(f"file:{quote(path.as_posix(), safe='/')}?mode=ro", uri=True)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='_akashic_workspace_origin'"
        ).fetchone()
        if exists is None:
            return None
        rows = connection.execute("SELECT version, baseline_ids FROM _akashic_workspace_origin").fetchall()
    if len(rows) != 1 or rows[0][0] != 1:
        raise ValueError("workspace 迁移起点损坏")
    values = json.loads(rows[0][1])
    if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
        raise ValueError("workspace 迁移起点 ID 无效")
    if len(set(values)) != len(values):
        raise ValueError("workspace 迁移起点 ID 重复")
    return tuple(values)


def _save_baseline(path: Path, baseline: tuple[str, ...]) -> None:
    """单独记录不适用于新 workspace 的历史 ID，不伪造 Yoyo 已执行记录。"""
    # 先完成私有临时账本，再以不覆盖既有文件的方式发布。
    descriptor, temporary_name = tempfile.mkstemp(prefix=".workspace-origin-", dir=path.parent)
    temporary = Path(temporary_name)
    os.close(descriptor)
    try:
        with closing(sqlite3.connect(temporary)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE _akashic_workspace_origin (version INTEGER NOT NULL, baseline_ids TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO _akashic_workspace_origin VALUES (1, ?)", (json.dumps(baseline),))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink()


def initialize_empty_workspace(*, repo_root: Path, workspace: Path, config_path: Path) -> None:
    """显式 init 在任何插件写入前记录起点；既有 workspace 保持原样。"""
    workspace = workspace.resolve()
    config_path = config_path.resolve()
    lock = WorkspaceInstanceLock(workspace)
    lock.acquire()
    try:
        if _workspace_is_empty(workspace, config_path):
            requirements = load_migration_requirements(repo_root / "migrations/catalog.toml")
            _save_baseline(workspace / "migrations.sqlite3", tuple(
                item.migration_id for item in requirements if item.bundle_id is not None
            ))
    finally:
        lock.release()


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
