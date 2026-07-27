from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, cast
from uuid import uuid4

from bootstrap.workspace_lock import WorkspaceInstanceLock


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_DURABLE_WORKSPACE_FILES = (
    "sessions.db",
    "schedules.json",
    "proactive.db",
    "wake_proactive.db",
    "memory/MEMORY.md",
    "memory/SELF.md",
    "memory/veda.md",
    "memory/memory2.db",
    "memory/akasha.db",
    "drift/drift.db",
)


@dataclass(frozen=True)
class MigrationOutcome:
    state: Literal["current", "fresh", "migrated"]
    head: str
    commits: tuple[str, ...] = ()


class _MigrationLock:
    """为同一个配置路径串行化迁移决策。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: IO[str] | None = None

    def __enter__(self) -> _MigrationLock:
        # 1. 内核锁拥有并发语义，文件正文只记录诊断 owner。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.seek(0)
            owner = stream.read().strip() or "unknown"
            stream.close()
            raise RuntimeError(
                f"配置迁移已由其他进程执行: {self.path} owner={owner}"
            ) from exc

        # 2. 持锁后再刷新诊断 owner。
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        self._stream = stream
        return self

    def __exit__(self, *_args: object) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


class MigrationRunner:
    """在 runtime 启动前发现并执行仓库迁移。"""

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
        self.migrations_root = self.repo_root / "migrations"
        self.cursor_path = self.config_path.with_name(
            f"{self.config_path.name}.migration-cursor"
        )
        self.lock_path = self.config_path.with_name(
            f"{self.config_path.name}.migration-lock"
        )
        self.backup_root = self.config_path.with_name(
            f"{self.config_path.name}.migration-backups"
        )

    def run(self) -> MigrationOutcome:
        """分类安装、执行待处理 bundle，并返回最终迁移状态。"""

        # 1. 代码与 cursor 一致时保持常数时间快速路径。
        head = self._current_head()
        cursor = self._read_cursor()
        if cursor == head:
            return MigrationOutcome(state="current", head=head)

        # 2. 串行化慢路径，并在锁内重新作出决定。
        with _MigrationLock(self.lock_path):
            head = self._current_head()
            cursor = self._read_cursor()
            if cursor == head:
                return MigrationOutcome(state="current", head=head)
            workspace_lock = WorkspaceInstanceLock(self.workspace)
            workspace_lock.acquire()
            try:
                return self._run_slow_path(head, cursor)
            finally:
                workspace_lock.release()

    def _run_slow_path(self, head: str, cursor: str | None) -> MigrationOutcome:
        """在 config 与 workspace 双重持锁后执行一次迁移慢路径。"""

        # 1. cursor 缺失时先区分全新安装与旧状态接管。
        if cursor is None:
            if not self.config_path.exists() and not self._has_durable_workspace_state():
                return MigrationOutcome(state="fresh", head=head)
            cursor = self._baseline()
            self._write_cursor(cursor)

        # 2. 逐个执行迁移提交，并发布 cursor 进度。
        self._require_ancestor(cursor, head)
        completed: list[str] = []
        for commit in self._migration_commits(cursor, head):
            for bundle in self._added_bundles(commit):
                self._run_bundle(commit, bundle)
            self._write_cursor(commit)
            completed.append(commit)
        self._write_cursor(head)
        return MigrationOutcome(
            state="migrated",
            head=head,
            commits=tuple(completed),
        )

    def mark_current(self) -> str:
        """全新初始化验证成功后，把配置标记为当前提交。"""

        head = self._current_head()
        with _MigrationLock(self.lock_path):
            workspace_lock = WorkspaceInstanceLock(self.workspace)
            workspace_lock.acquire()
            try:
                cursor = self._read_cursor()
                if cursor is not None and cursor != head:
                    raise RuntimeError(
                        "不能把已有 migration cursor 当作全新安装覆盖: "
                        f"cursor={cursor} head={head}"
                    )
                self._write_cursor(head)
            finally:
                workspace_lock.release()
        return head

    def _current_head(self) -> str:
        return self._git("rev-parse", "--verify", "HEAD")

    def _baseline(self) -> str:
        root = self.migrations_root / ".root"
        try:
            baseline = root.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(f"缺少迁移根: {root}") from exc
        self._validate_revision(baseline, source=str(root))
        _ = self._git("cat-file", "-e", f"{baseline}^{{commit}}")
        return baseline

    def _read_cursor(self) -> str | None:
        try:
            value = self.cursor_path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return None
        self._validate_revision(value, source=str(self.cursor_path))
        return value

    def _write_cursor(self, revision: str) -> None:
        self._validate_revision(revision, source="Git")
        self.cursor_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cursor_path.with_name(
            f".{self.cursor_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(f"{revision}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.cursor_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _require_ancestor(self, cursor: str, head: str) -> None:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(self.repo_root),
                "merge-base",
                "--is-ancestor",
                cursor,
                head,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return
        if result.returncode == 1:
            raise RuntimeError(
                "migration cursor 不是当前 HEAD 的祖先，禁止自动降级或跨分支迁移: "
                f"cursor={cursor} head={head}"
            )
        raise RuntimeError(
            "Git 祖先关系检查失败: "
            f"cursor={cursor} head={head} stderr={result.stderr.strip()}"
        )

    def _migration_commits(self, cursor: str, head: str) -> list[str]:
        output = self._git(
            "rev-list",
            "--reverse",
            "--first-parent",
            f"{cursor}..{head}",
            "--",
            "migrations/",
        )
        return [line for line in output.splitlines() if line]

    def _added_bundles(self, commit: str) -> list[Path]:
        output = self._git(
            "diff",
            "--name-only",
            "--diff-filter=A",
            f"{commit}^1",
            commit,
            "--",
            "migrations/",
        )
        bundles = {
            Path(path).parent
            for path in output.splitlines()
            if path.startswith("migrations/") and path.endswith("/migration.py")
        }
        return sorted(bundles, key=lambda path: path.as_posix().encode("utf-8"))

    def _run_bundle(self, commit: str, bundle: Path) -> None:
        script = self.repo_root / bundle / "migration.py"
        if not script.is_file():
            raise RuntimeError(f"迁移脚本在当前 checkout 中不存在: {script}")
        assessment = self._invoke(script, "assess", commit, bundle, backup_dir=None)
        status = assessment.get("status")
        if status == "blocked":
            reason = str(assessment.get("reason") or "未说明原因")
            raise RuntimeError(f"迁移被阻止: bundle={bundle} reason={reason}")
        if status not in {"needed", "satisfied"}:
            raise RuntimeError(f"迁移 assess 返回未知状态: bundle={bundle} status={status!r}")

        backup_dir: Path | None = None
        if status == "needed":
            self.backup_root.mkdir(parents=True, mode=0o700, exist_ok=True)
            os.chmod(self.backup_root, 0o700)
            backup_dir = self.backup_root / (
                f"{commit[:12]}-{bundle.name}-{uuid4().hex}"
            )
            _ = self._invoke(script, "apply", commit, bundle, backup_dir=backup_dir)
        _ = self._invoke(script, "verify", commit, bundle, backup_dir=backup_dir)

    def _invoke(
        self,
        script: Path,
        action: Literal["assess", "apply", "verify"],
        commit: str,
        bundle: Path,
        *,
        backup_dir: Path | None,
    ) -> dict[str, object]:
        arguments = [
            sys.executable,
            str(script),
            action,
            "--config",
            str(self.config_path),
            "--workspace",
            str(self.workspace),
            "--migration-commit",
            commit,
        ]
        if backup_dir is not None:
            arguments.extend(["--backup-dir", str(backup_dir)])
        environment = os.environ.copy()
        existing_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(self.repo_root)
            if not existing_path
            else f"{self.repo_root}{os.pathsep}{existing_path}"
        )
        result = subprocess.run(
            arguments,
            cwd=self.repo_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"迁移命令失败: bundle={bundle} action={action} "
                f"exit={result.returncode} detail={detail[-4000:]}"
            )
        if action != "assess":
            return {}
        try:
            payload: object = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"迁移 assess 输出不是 JSON: bundle={bundle}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"迁移 assess 输出必须是对象: bundle={bundle}")
        return cast(dict[str, object], payload)

    def _has_durable_workspace_state(self) -> bool:
        if any((self.workspace / relative).exists() for relative in _DURABLE_WORKSPACE_FILES):
            return True
        plugin_data = self.workspace / "plugin-data"
        return plugin_data.is_dir() and next(plugin_data.iterdir(), None) is not None

    def _git(self, *arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repo_root), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Git 命令失败: git {' '.join(arguments)}: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    @staticmethod
    def _validate_revision(value: str, *, source: str) -> None:
        if not _REVISION_RE.fullmatch(value):
            raise RuntimeError(f"迁移 Git revision 无效: source={source} value={value!r}")


def migrate_installation(config_path: Path, workspace: Path) -> MigrationOutcome:
    return MigrationRunner(
        repo_root=_PROJECT_ROOT,
        config_path=config_path,
        workspace=workspace,
    ).run()


def mark_fresh_installation_current(config_path: Path, workspace: Path) -> str:
    return MigrationRunner(
        repo_root=_PROJECT_ROOT,
        config_path=config_path,
        workspace=workspace,
    ).mark_current()
