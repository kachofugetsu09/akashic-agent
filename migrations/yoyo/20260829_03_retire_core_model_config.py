from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path
from uuid import uuid4

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context
from agent.model_runtime.store import ModelRegistryStore

__depends__ = {"20260829_02_backfill_explicit_programmatic_effects"}
__transactional__ = False

_MIGRATION = "retire-core-model-config"


class _ConfigSnapshot:
    def __init__(
        self,
        *,
        path: Path,
        target: Path,
        content: bytes,
        mode: int,
        symlink_target: str | None,
    ) -> None:
        self.path = path
        self.target = target
        self.content = content
        self.mode = mode
        self.symlink_target = symlink_target


def _snapshot(path: Path) -> _ConfigSnapshot | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        link = os.readlink(path)
        target = path.resolve(strict=True)
        if not target.is_file():
            raise RuntimeError(f"模型配置软链接目标不是普通文件: {path}")
        return _ConfigSnapshot(
            path=path,
            target=target,
            content=target.read_bytes(),
            mode=stat.S_IMODE(target.stat().st_mode),
            symlink_target=link,
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"模型配置必须是普通文件或软链接: {path}")
    return _ConfigSnapshot(
        path=path,
        target=path,
        content=path.read_bytes(),
        mode=stat.S_IMODE(metadata.st_mode),
        symlink_target=None,
    )


def _render(content: bytes, workspace: Path) -> bytes | None:
    """Remove only config facts already handed to ordinary plugins."""

    parsed = tomllib.loads(content.decode("utf-8"))
    retired = [name for name in ("llm", "memory") if name in parsed]
    if not retired:
        return None
    llm = parsed.get("llm")
    if llm is not None:
        if llm != {"registry": "workspace"}:
            raise RuntimeError("[llm] 尚未完成模型注册库 handoff，拒绝删除")
        store = ModelRegistryStore.for_workspace(workspace)
        if not store.exists():
            raise RuntimeError("[llm] 已指向 workspace，但模型注册库不存在")
        store.integrity_check()
        if store.read_snapshot() is None:
            raise RuntimeError("[llm] handoff 模型注册库为空，拒绝删除")
    memory = parsed.get("memory")
    if memory is not None:
        if not isinstance(memory, dict) or set(memory) - {"enabled", "embedding"}:
            raise RuntimeError("[memory] 尚未完成 Akasha/models handoff，拒绝删除")
        if not isinstance(memory.get("enabled"), bool):
            raise RuntimeError("memory.enabled 尚未完成校验，拒绝删除")
        embedding = memory.get("embedding")
        if embedding is not None and not isinstance(embedding, dict):
            raise RuntimeError("memory.embedding 尚未完成校验，拒绝删除")
    document = tomlkit.parse(content.decode("utf-8"))
    for name in retired:
        del document[name]
    rendered = tomlkit.dumps(document).encode("utf-8")
    final = tomllib.loads(rendered.decode("utf-8"))
    if any(name in final for name in retired):
        raise RuntimeError("Core 模型配置迁移后仍含 retired table")
    return rendered


def _write_atomic(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(candidate, path)
        os.chmod(path, mode)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        candidate.unlink(missing_ok=True)


def _check_identity(snapshot: _ConfigSnapshot) -> None:
    if snapshot.symlink_target is not None and (
        not snapshot.path.is_symlink()
        or os.readlink(snapshot.path) != snapshot.symlink_target
    ):
        raise RuntimeError(f"模型配置软链接身份改变: {snapshot.path}")


def retire_core_model_config(_connection: object) -> None:
    """Retire Core model tables after their durable plugin handoffs."""

    _ = _connection
    current = current_migration_context()
    snapshot = _snapshot(current.config_path)
    if snapshot is None:
        return
    rendered = _render(snapshot.content, current.workspace)
    if rendered is None:
        return

    backup_root = current.workspace / "backups" / _MIGRATION / uuid4().hex
    backup_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(backup_root, 0o700)
    backup = backup_root / "config.toml.before"
    _write_atomic(backup, snapshot.content, 0o600)
    if backup.read_bytes() != snapshot.content:
        raise RuntimeError("Core 模型配置备份校验失败")

    try:
        _write_atomic(snapshot.target, rendered, snapshot.mode)
        _check_identity(snapshot)
        if snapshot.path.read_bytes() != rendered:
            raise RuntimeError("Core 模型配置发布校验失败")
    except BaseException as migration_error:
        try:
            _write_atomic(snapshot.target, backup.read_bytes(), snapshot.mode)
            _check_identity(snapshot)
            if snapshot.path.read_bytes() != snapshot.content:
                raise RuntimeError("Core 模型配置恢复校验失败")
        except BaseException as restore_error:
            raise RuntimeError(
                f"Core 模型配置迁移失败且恢复失败: {migration_error}"
            ) from restore_error
        raise


steps = [step(retire_core_model_config)]
