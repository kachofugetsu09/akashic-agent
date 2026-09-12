import hashlib
import json
import os
import stat
import tomllib
from pathlib import Path
from uuid import uuid4

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context

__depends__ = {"20260826_01_migrate_turn_effects"}
__transactional__ = False

_MIGRATION_NAME = "select-akasha-embedding-plugin"
_AKASHA_PLUGIN = "akasha"
_DEFAULT_MEMORY_PLUGIN = "default_memory"
_MEMORY_DEPENDENTS = ("wake",)
_LEGACY_ENGINES = {None, "", "default", "akasha"}


class _ConfigSnapshot:
    """Capture config bytes, permissions, and symbolic-link identity."""

    def __init__(
        self,
        *,
        path: Path,
        content: bytes,
        kind: str,
        mode: int,
        resolved_target: Path,
        symlink_target: str | None,
    ) -> None:
        self.path = path
        self.content = content
        self.kind = kind
        self.mode = mode
        self.resolved_target = resolved_target
        self.symlink_target = symlink_target


def _snapshot_config(path: Path) -> _ConfigSnapshot | None:
    """Read one regular config without losing a symbolic-link boundary."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"无法读取配置迁移源元数据: {path}") from exc

    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        resolved = path.resolve(strict=False)
        if not path.is_file() or not resolved.is_file():
            raise RuntimeError(f"配置迁移源软链接不是可读文件: {path}")
        return _ConfigSnapshot(
            path=path,
            content=path.read_bytes(),
            kind="symlink",
            mode=stat.S_IMODE(resolved.stat().st_mode),
            resolved_target=resolved,
            symlink_target=target,
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"配置迁移源必须是普通文件或软链接: {path}")
    return _ConfigSnapshot(
        path=path,
        content=path.read_bytes(),
        kind="file",
        mode=stat.S_IMODE(metadata.st_mode),
        resolved_target=path,
        symlink_target=None,
    )


def _disabled_builtin(document: object, *, memory_enabled: bool) -> list[str]:
    """Project the legacy memory switch onto ordinary plugin exclusions."""

    if not isinstance(document, dict):
        raise ValueError("配置根必须是 table")
    agent = document.get("agent")
    if agent is None:
        agent = tomlkit.table()
        document["agent"] = agent
    elif not isinstance(agent, dict):
        raise ValueError("agent 必须是 table")

    plugins = agent.get("plugins")
    if plugins is None:
        plugins = tomlkit.table()
        agent["plugins"] = plugins
    elif not isinstance(plugins, dict):
        raise ValueError("agent.plugins 必须是 table")

    raw = plugins.get("disabled_builtin")
    if raw is None:
        disabled: list[str] = []
    elif isinstance(raw, list) and all(
        isinstance(item, str) and item and item.strip() == item for item in raw
    ):
        disabled = list(raw)
    else:
        raise ValueError("agent.plugins.disabled_builtin 必须是合法字符串数组")
    if len(disabled) != len(set(disabled)):
        raise ValueError("agent.plugins.disabled_builtin 不允许重复插件名")
    disabled = [item for item in disabled if item != _DEFAULT_MEMORY_PLUGIN]
    if memory_enabled:
        disabled = [item for item in disabled if item != _AKASHA_PLUGIN]
    else:
        for plugin_id in (_AKASHA_PLUGIN, *_MEMORY_DEPENDENTS):
            if plugin_id not in disabled:
                disabled.append(plugin_id)
    plugins["disabled_builtin"] = disabled
    return disabled


def _render_config(raw: bytes) -> bytes | None:
    """Translate the retired memory selector into Akasha plugin activation."""

    text = raw.decode("utf-8")
    # 1. Match the exact legacy state before constructing any mutation.
    parsed = tomllib.loads(text)
    memory = parsed.get("memory")
    if not isinstance(memory, dict) or memory.get("engine") not in _LEGACY_ENGINES:
        return None
    memory_enabled = memory.get("enabled")
    if not isinstance(memory_enabled, bool):
        return None

    # 2. Select Akasha through the ordinary plugin competition primitive.
    document = tomlkit.parse(text)
    document_memory = document.get("memory")
    if not isinstance(document_memory, dict):
        raise RuntimeError("已匹配的 memory 配置无法重新解析")
    if "engine" in document_memory:
        del document_memory["engine"]
    _disabled_builtin(document, memory_enabled=memory_enabled)
    rendered = tomlkit.dumps(document).encode("utf-8")

    # 3. Reparse and prove the exact post-migration invariants.
    final = tomllib.loads(rendered.decode("utf-8"))
    final_memory = final.get("memory")
    final_agent = final.get("agent")
    final_plugins = (
        final_agent.get("plugins") if isinstance(final_agent, dict) else None
    )
    disabled = (
        final_plugins.get("disabled_builtin")
        if isinstance(final_plugins, dict)
        else None
    )
    if not isinstance(final_memory, dict) or "engine" in final_memory:
        raise RuntimeError("配置迁移后仍含 memory.engine")
    if not isinstance(disabled, list) or _DEFAULT_MEMORY_PLUGIN in disabled:
        raise RuntimeError("配置迁移后仍引用已移除的 Default Memory")
    if (_AKASHA_PLUGIN in disabled) is memory_enabled:
        raise RuntimeError("配置迁移后的 Akasha 启用状态与 memory.enabled 不一致")
    if not memory_enabled and any(item not in disabled for item in _MEMORY_DEPENDENTS):
        raise RuntimeError("配置迁移后仍启用了依赖记忆的插件")
    return None if rendered == raw else rendered


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes, mode: int) -> None:
    """Publish one fsynced file through a same-directory replacement."""

    candidate = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        candidate.write_bytes(payload)
        candidate.chmod(mode)
        with candidate.open("rb") as stream:
            os.fsync(stream.fileno())
        candidate.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise


def _write_backup(snapshot: _ConfigSnapshot, root: Path) -> Path:
    """Store a verified 0600 source copy and manifest before publication."""

    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(root, 0o700)
    backup = root / "config.toml.before"
    _write_atomic(backup, snapshot.content, 0o600)
    digest = _sha256(snapshot.content)
    if _sha256(backup.read_bytes()) != digest:
        raise RuntimeError(f"配置迁移备份校验失败: {snapshot.path}")
    manifest = {
        "schema_version": 1,
        "migration": _MIGRATION_NAME,
        "source": {
            "path": str(snapshot.path),
            "kind": snapshot.kind,
            "resolved_target": str(snapshot.resolved_target),
            "symlink_target": snapshot.symlink_target,
            "mode": snapshot.mode,
            "backup": backup.name,
            "sha256": digest,
        },
    }
    payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_atomic(root / "manifest.json", payload, 0o600)
    if json.loads((root / "manifest.json").read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"配置迁移备份 manifest 校验失败: {root}")
    return backup


def _publish_config(snapshot: _ConfigSnapshot, rendered: bytes) -> None:
    """Publish config bytes while preserving an existing symbolic link."""

    _write_atomic(snapshot.resolved_target, rendered, snapshot.mode)
    if snapshot.kind == "symlink" and (
        not snapshot.path.is_symlink()
        or os.readlink(snapshot.path) != snapshot.symlink_target
    ):
        raise RuntimeError(f"配置软链接身份发生变化: {snapshot.path}")
    if snapshot.path.read_bytes() != rendered:
        raise RuntimeError(f"配置迁移发布校验失败: {snapshot.path}")


def _restore_config(snapshot: _ConfigSnapshot, backup: Path) -> None:
    """Restore exact pre-migration bytes after a failed publication."""

    payload = backup.read_bytes()
    if _sha256(payload) != _sha256(snapshot.content):
        raise RuntimeError(f"配置迁移恢复备份摘要不匹配: {snapshot.path}")
    _write_atomic(snapshot.resolved_target, payload, snapshot.mode)
    if snapshot.kind == "symlink" and (
        not snapshot.path.is_symlink()
        or os.readlink(snapshot.path) != snapshot.symlink_target
    ):
        raise RuntimeError(f"配置软链接恢复身份不匹配: {snapshot.path}")
    if snapshot.path.read_bytes() != snapshot.content:
        raise RuntimeError(f"配置迁移恢复校验失败: {snapshot.path}")


def select_akasha_embedding_plugin(_connection: object) -> None:
    """Preserve the legacy memory switch through ordinary Akasha activation."""

    _ = _connection
    current = current_migration_context()
    snapshot = _snapshot_config(current.config_path)
    if snapshot is None:
        return
    rendered = _render_config(snapshot.content)
    if rendered is None:
        return

    # 1. Persist a recoverable source snapshot before the only external write.
    backup_root = current.workspace / "backups" / _MIGRATION_NAME / uuid4().hex
    backup = _write_backup(snapshot, backup_root)

    # 2. Publish the narrow replacement and restore it if verification fails.
    try:
        _publish_config(snapshot, rendered)
    except BaseException as migration_error:
        try:
            _restore_config(snapshot, backup)
        except BaseException as restore_error:
            raise RuntimeError(
                f"Akasha plugin selection migration failed and restore failed: {migration_error}"
            ) from restore_error
        raise


steps = [step(select_akasha_embedding_plugin)]
