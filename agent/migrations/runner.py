from __future__ import annotations

import gc
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
    load_migration_bundle,
    load_migration_requirements,
    migration_import_paths,
    validate_bundle_dependencies,
    validate_pending_requirements,
)
from agent.plugins.source_resolver import ResolvedPluginSource
from agent.plugins.manifest import plugins_root, workspace_plugin_data_dir
from agent.plugins.distribution_sources import (
    DistributionSources,
    distribution_marketplace,
    distribution_sources,
    distribution_migration_sources,
    official_installed_sources,
)
from agent.plugins.reload_journal import check_pending_publication
from agent.plugins.selection import PluginSelection
from bootstrap.workspace_lock import WorkspaceInstanceLock, WorkspaceMaintenanceLock


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 新建 workspace 不继承旧全局启停；起点记录适用范围，不伪造 Yoyo 成功回执。
_NEW_WORKSPACE_BASELINE = ("20261011_02_bundle_choices",)
_USERNAME_ENV_KEYS = ("LOGNAME", "USER", "LNAME", "USERNAME")


@dataclass(frozen=True)
class MigrationOutcome:
    state: Literal["current", "migrated"]
    migrations: tuple[str, ...] = ()


class MigrationRunner:
    """在启动前推进 Core 和内置插件的 Yoyo 账本。"""

    def __init__(
        self,
        *,
        repo_root: Path,
        config_path: Path,
        workspace: Path,
        plugin_dirs: Sequence[Path] = (),
        plugins_home: Path | None = None,
        bundle_directory: Path | None = None,
        migration_catalog: Path | None = None,
        fixed_sources: Sequence[ResolvedPluginSource] | None = None,
        startup_selection: bool = False,
        replacement_sources: Sequence[ResolvedPluginSource] = (),
        distribution_state: DistributionSources | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.config_path = config_path.expanduser().resolve()
        self.workspace = workspace.expanduser().resolve()
        self.plugins_home = plugins_root(plugins_home).resolve()
        distribution = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION")
        self.bundle_directory = bundle_directory or (Path(distribution) / "bundles" if distribution else None)
        # The Core source is intentionally a separate, implementation-free root;
        # silently falling back to a checkout's retired business migrations would
        # defeat the external bundle boundary.
        self.migrations_root = self.repo_root / "migrations" / "core"
        self.ledger_path = self.workspace / "migrations.sqlite3"
        # 保留路径上的 symlink 形状，让 source resolver 能够拒绝它，而不是
        # 先 resolve 后把越界路径伪装成普通 cache 根。
        self.plugin_dirs = tuple(path.expanduser() for path in plugin_dirs)
        self.fixed_sources = None if fixed_sources is None else tuple(fixed_sources)
        self.startup_selection = startup_selection
        # 本次发布将换入的已装目标；它们的迁移取目标代码，而不是选择里的旧代码。
        self.replacement_sources = tuple(replacement_sources)
        # 部署发布阶段按目标发行版语义取得的来源；缺省按运行态语义读取。
        self.distribution_state = distribution_state
        self.migration_catalog = (
            migration_catalog
            if migration_catalog is not None
            else self.repo_root / "migrations" / "catalog.toml"
        ).expanduser()

    def run(self) -> MigrationOutcome:
        """先迁移 Core 和当前发行版内置数据，再允许 runtime 启动。"""

        # 1. 复用 workspace 锁串行化迁移与 runtime 启动
        workspace_lock = WorkspaceInstanceLock(self.workspace)
        workspace_lock.acquire()
        try:
            if self.startup_selection:
                self.fixed_sources = distribution_migration_sources(self.workspace, plugins_root())
                # Core 先升级自己拥有的账本结构，才能读取完整的配置恢复事实。
                core = self._apply_pending(core_only=True)
                selection = PluginSelection(self.workspace)
                if self.check() and selection.path.exists() and selection.read() is not None:
                    check_pending_publication(self.workspace)
                plugins = self._apply_pending()
                migrations = core.migrations + plugins.migrations
                return MigrationOutcome(state="migrated" if migrations else "current", migrations=migrations)
            return self._apply_pending()
        finally:
            workspace_lock.release()

    def run_under_maintenance(
        self, maintenance: WorkspaceMaintenanceLock, *, core_only: bool = False,
    ) -> MigrationOutcome:
        """Apply migrations while the release owns the workspace maintenance lock."""

        if (maintenance.paths != (
            self.workspace / ".supervisor.lock", self.workspace / ".instance.lock",
        ) or len(maintenance._streams) != 2):
            raise RuntimeError("release migration 缺少当前 workspace maintenance owner")
        return self._apply_pending(core_only=core_only)

    def check(self) -> tuple[str, ...]:
        """只读核对目标代码的待迁移集合，不创建账本或运行 step。"""
        return self._apply_pending(check_only=True).migrations

    def _apply_pending(
        self, *, core_only: bool = False,
        check_only: bool = False,
    ) -> MigrationOutcome:
        """加载不可变目录并提交全部缺失迁移。"""

        # 1. 初始化由 workspace 持有的迁移账本
        fresh = _workspace_is_empty(self.workspace, self.config_path)
        baseline = _read_baseline(self.ledger_path)
        try:
            core_migrations = _read_migrations(str(self.migrations_root))
            bundles = discover_migration_bundles(
                plugin_dirs=self.plugin_dirs,
                fixed_sources=self._migration_sources(core_only),
                trusted_marketplaces=self._official_marketplaces(),
            )
            requirements = load_migration_requirements(self.migration_catalog)
            applied_ids = _read_applied_ids(self.ledger_path)
            core_ids = tuple(migration.id for migration in core_migrations)
            bundle_ids = tuple(
                migration_id
                for bundle in bundles
                for migration_id in bundle.migration_ids
            )
            if baseline is None and fresh:
                baseline = (*_NEW_WORKSPACE_BASELINE, *(item.migration_id for item in requirements
                                 if item.migration_id not in core_ids + bundle_ids))
            applicable = tuple(item for item in requirements
                               if item.migration_id not in (baseline or ()))
            validate_bundle_dependencies(
                bundles,
                core_migration_ids=core_ids,
                requirements=requirements,
                applied_ids=applied_ids,
                require_missing_bundles=False,
            )
            validate_pending_requirements(
                applicable,
                loaded_ids=core_ids + bundle_ids,
                applied_ids=applied_ids,
                bundles=bundles,
            )
            pending_ids = set(core_ids + bundle_ids) - set(applied_ids) - set(baseline or ())
            if check_only:
                return MigrationOutcome(state="current", migrations=tuple(sorted(pending_ids)))
            if fresh and baseline is not None:
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                _save_baseline(self.ledger_path, baseline)
            if not pending_ids:
                return MigrationOutcome(state="current")
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            backend = get_backend(self._ledger_uri())
            os.chmod(self.ledger_path, 0o600)

            # 2. 为 Yoyo Python step 绑定明确的安装路径
            bundle_data_roots = {
                bundle.bundle_id: workspace_plugin_data_dir(
                    self.workspace,
                    bundle.plugin_name or bundle.bundle_id,
                    bundle.marketplace or "builtin",
                )
                for bundle in bundles
            }
            with (
                _bind_yoyo_username(),
                backend,
                bind_migration_context(
                    config_path=self.config_path,
                    workspace=self.workspace,
                    plugins_home=self.plugins_home, bundle_directory=self.bundle_directory,
                    bundle_data_roots=bundle_data_roots,
                ),
                migration_import_paths(bundles),
            ):
                migrations = _read_migrations(
                    str(self.migrations_root), bundles, (*applied_ids, *(baseline or ()))
                )
                selected_ids = set(core_ids) if core_only else {item.id for item in migrations}
                if core_only:
                    # Core 迁移的依赖可能已经移交插件；先执行真实依赖，不越过它落账。
                    available = {item.id: item for item in migrations}
                    pending = list(selected_ids)
                    while pending:
                        for dependency in available[pending.pop()].depends:
                            if dependency.id in available and dependency.id not in selected_ids:
                                selected_ids.add(dependency.id)
                                pending.append(dependency.id)
                selected = type(migrations)(
                    (item for item in migrations if item.id in selected_ids
                     and item.id not in (baseline or ())),
                    migrations.post_apply,
                )
                pending = backend.to_apply(selected)
                migration_ids = tuple(migration.id for migration in pending)
                backend.apply_migrations(pending)
                if migration_ids:
                    # 已发布的迁移用 `with sqlite3.connect(...)` 打开 reload journal：只提交、不关闭，
                    # 连接留到垃圾回收才释放 WAL 旁文件。迁移不可改，这里收一次，
                    # 之后的 journal 快照不会在读取途中看到旁文件消失。
                    _ = gc.collect()
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

    # 官方 marketplace 来自当前发行版声明；没有发行版时没有可信的已装来源。
    def _official_marketplaces(self) -> frozenset[str]:
        if self.bundle_directory is None or not (self.bundle_directory.parent / "distribution.json").is_file():
            return frozenset()
        return frozenset({distribution_marketplace(self.bundle_directory.parent)})

    # 内置来源加官方已装来源；Core 先行阶段不读选择，保持原有的失败顺序。
    def _migration_sources(self, core_only: bool) -> tuple[ResolvedPluginSource, ...] | None:
        if self.fixed_sources is None or core_only or not self._official_marketplaces():
            return self.fixed_sources
        assert self.bundle_directory is not None
        replaced = frozenset(f"{item.plugin_name}@{item.marketplace}" for item in self.replacement_sources)
        distribution = self.bundle_directory.parent
        state = self.distribution_state or distribution_sources(self.workspace, self.plugins_home, distribution)
        installed = official_installed_sources(
            self.workspace, self.plugins_home, distribution, state, exclude_ids=replaced,
        )
        return (*self.fixed_sources, *installed, *self.replacement_sources)

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
    """Run startup migrations from the same durable inputs the runtime loads."""

    return MigrationRunner(
        repo_root=_PROJECT_ROOT,
        config_path=config_path,
        workspace=workspace,
        startup_selection=True,
    ).run()


# 在线安装前核对：官方来源的新制品带有尚未执行的迁移时返回它们；在线实例不执行迁移。
def pending_artifact_migrations(workspace: Path, source: ResolvedPluginSource) -> tuple[str, ...]:
    # 1. 只有官方 marketplace 的已装来源才会被宿主执行迁移，其他来源保持现状。
    distribution = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION")
    if not distribution or source.marketplace != distribution_marketplace(Path(distribution)):
        return ()
    catalog = source.plugin_root / "migration.catalog.toml"
    if not catalog.exists() and not catalog.is_symlink():
        return ()
    # 2. 与启动时相同的口径：目录里的 ID 减去账本已成功与新建起点。
    ledger = workspace / "migrations.sqlite3"
    done = {*_read_applied_ids(ledger), *(_read_baseline(ledger) or ())}
    bundle = load_migration_bundle(source)
    return tuple(item for item in bundle.migration_ids if item not in done)


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
            _save_baseline(workspace / "migrations.sqlite3", (*_NEW_WORKSPACE_BASELINE, *(
                item.migration_id for item in requirements if item.bundle_id is not None
            )))
    finally:
        lock.release()


# 首装挑选窗口里在线安装从没装过的插件时，把它目标版本的迁移并入新建起点：它的数据还不存在，由当前代码直接建成
# 最新形态，历史迁移不适用（与新 workspace 记录起点同一口径）。返回并入前的起点，安装失败时用它还原；
# 没有起点的旧 workspace 不改动并返回 None。
def adopt_fresh_plugin_migrations(workspace: Path, ids: tuple[str, ...]) -> tuple[str, ...] | None:
    baseline = _read_baseline(workspace / "migrations.sqlite3")
    if baseline is None:
        return None
    replace_workspace_baseline(workspace, (*baseline, *(item for item in ids if item not in baseline)))
    return baseline


# 整体改写已有的新建起点：只用于上面的并入，以及安装失败后的还原。
def replace_workspace_baseline(workspace: Path, ids: tuple[str, ...]) -> None:
    with closing(sqlite3.connect(workspace / "migrations.sqlite3")) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("UPDATE _akashic_workspace_origin SET baseline_ids=?", (json.dumps(list(ids)),))


def _read_applied_ids(path: Path) -> tuple[str, ...]:
    """只读 ledger 已成功 ID，缺表视为空账本。"""

    if not path.exists():
        return ()
    connection = sqlite3.connect(f"file:{quote(path.as_posix(), safe='/')}?mode=ro", uri=True)
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
