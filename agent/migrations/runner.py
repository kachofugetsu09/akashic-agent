from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal
from urllib.parse import quote

from yoyo import get_backend, read_migrations

from agent.migrations.context import bind_migration_context
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
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.config_path = config_path.expanduser().resolve()
        self.workspace = workspace.expanduser().resolve()
        self.migrations_root = self.repo_root / "migrations" / "yoyo"
        self.ledger_path = self.workspace / "migrations.sqlite3"

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
            migrations = read_migrations(str(self.migrations_root))
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
            ):
                pending = backend.to_apply(migrations)
                migration_ids = tuple(migration.id for migration in pending)
                backend.apply_migrations(pending)
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
