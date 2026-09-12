from __future__ import annotations

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


__depends__ = {"20260808_03_remove_compaction_trigger"}
__transactional__ = False

_MIGRATION_NAME = "retire-legacy-context-state"
_KEEP_RECENT_TOKENS = 20_000


class _PathSnapshot:
    """Capture path identity and bytes without following it during restore."""

    def __init__(
        self,
        path: Path,
        existed: bool,
        kind: str,
        mode: int | None,
        symlink_target: str | None,
        resolved_target: Path | None,
        target_mode: int | None,
        content: bytes | None,
    ) -> None:
        self.path = path
        self.existed = existed
        self.kind = kind
        self.mode = mode
        self.symlink_target = symlink_target
        self.resolved_target = resolved_target
        self.target_mode = target_mode
        self.content = content


def _path_exists(path: Path) -> bool:
    """Return whether a path exists, including a dangling symbolic link."""

    return path.exists() or path.is_symlink()


def _snapshot_path(path: Path) -> _PathSnapshot:
    """Capture one regular file or symlink and its readable target bytes."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return _PathSnapshot(path, False, "missing", None, None, None, None, None)
    except OSError as exc:
        raise RuntimeError(f"无法读取迁移源元数据: {path}") from exc

    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        resolved = path.resolve(strict=False)
        if not path.is_file() or not resolved.is_file():
            raise RuntimeError(f"迁移源软链接不是可读文件: {path}")
        return _PathSnapshot(
            path,
            True,
            "symlink",
            mode,
            target,
            resolved,
            stat.S_IMODE(resolved.stat().st_mode),
            path.read_bytes(),
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"迁移源必须是普通文件或软链接: {path}")
    return _PathSnapshot(path, True, "file", mode, None, path, mode, path.read_bytes())


def _sha256(payload: bytes) -> str:
    """Return the immutable digest used by backup and restore checks."""

    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Durably publish an atomic replacement in one directory."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes, mode: int) -> None:
    """Write one fsynced regular file and atomically replace its target."""

    path.parent.mkdir(parents=True, exist_ok=True)
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


def _archive_snapshot(snapshot: _PathSnapshot, target: Path) -> str | None:
    """Archive source bytes with a strict 0600 artifact and digest check."""

    if not snapshot.existed:
        return None
    if snapshot.content is None:
        raise RuntimeError(f"迁移源内容不可读: {snapshot.path}")
    _write_atomic(target, snapshot.content, 0o600)
    actual = _sha256(target.read_bytes())
    expected = _sha256(snapshot.content)
    if actual != expected:
        raise RuntimeError(f"备份内容校验失败: {snapshot.path}")
    return actual


def _legacy_keys_present(document: dict) -> bool:
    """Return whether the rendered TOML still exposes a retired context key."""

    agent = document.get("agent")
    context = agent.get("context") if isinstance(agent, dict) else None
    compaction = context.get("compaction") if isinstance(context, dict) else None
    if "memory_window" in document:
        return True
    if isinstance(context, dict) and "memory_window" in context:
        return True
    if isinstance(compaction, dict) and (
        "memory_window" in compaction or "trigger_percent" in compaction
    ):
        return True
    llm = document.get("llm")
    if not isinstance(llm, dict):
        return False
    locations = [llm, llm.get("main")]
    runtimes = llm.get("runtimes")
    if isinstance(runtimes, dict):
        locations.extend(runtimes.values())
    return any(
        isinstance(location, dict)
        and any(
            key in location
            for key in ("effective_context_percent", "compaction_trigger_percent")
        )
        for location in locations
    )


def _render_config(raw: bytes) -> bytes:
    """Remove retired context keys while preserving an existing valid tail size."""

    text = raw.decode("utf-8")
    # 1. Parse at the trust boundary so malformed configuration fails before writes.
    tomllib.loads(text)
    document = tomlkit.parse(text)
    agent = document.setdefault("agent", tomlkit.table())
    if not isinstance(agent, dict):
        raise ValueError("agent 配置必须是 TOML table")
    context = agent.setdefault("context", tomlkit.table())
    if not isinstance(context, dict):
        raise ValueError("agent.context 配置必须是 TOML table")
    compaction = context.setdefault("compaction", tomlkit.table())
    if not isinstance(compaction, dict):
        raise ValueError("agent.context.compaction 配置必须是 TOML table")

    # 2. Keep user-selected closed-unit tail values; only supply a missing default.
    compaction.setdefault("keep_recent_tokens", _KEEP_RECENT_TOKENS)
    document.pop("memory_window", None)
    context.pop("memory_window", None)
    compaction.pop("memory_window", None)
    compaction.pop("trigger_percent", None)
    llm = document.get("llm")
    if isinstance(llm, dict):
        for key in ("effective_context_percent", "compaction_trigger_percent"):
            llm.pop(key, None)
        main = llm.get("main")
        if isinstance(main, dict):
            for key in ("effective_context_percent", "compaction_trigger_percent"):
                main.pop(key, None)
        runtimes = llm.get("runtimes")
        if isinstance(runtimes, dict):
            for runtime in runtimes.values():
                if isinstance(runtime, dict):
                    runtime.pop("effective_context_percent", None)
                    runtime.pop("compaction_trigger_percent", None)

    rendered = tomlkit.dumps(document).encode("utf-8")
    parsed = tomllib.loads(rendered.decode("utf-8"))
    final_compaction = parsed["agent"]["context"]["compaction"]
    if _legacy_keys_present(parsed):
        raise RuntimeError("配置迁移后仍存在已退役 context key")
    if final_compaction.get("keep_recent_tokens") != _KEEP_RECENT_TOKENS:
        # Existing values above the 20k watermark are valid and intentionally kept.
        value = final_compaction.get("keep_recent_tokens")
        if not isinstance(value, int) or isinstance(value, bool) or value < _KEEP_RECENT_TOKENS:
            raise RuntimeError("配置迁移后 keep_recent_tokens 不满足水位")
    return rendered


def _write_manifest(
    root: Path,
    records: dict[str, tuple[_PathSnapshot, Path | None, str | None]],
) -> None:
    """Persist path identity, archive locations, and content digests."""

    sources: dict[str, dict[str, object]] = {}
    for name, (snapshot, backup, digest) in records.items():
        source: dict[str, object] = {
            "path": str(snapshot.path),
            "existed": snapshot.existed,
            "kind": snapshot.kind,
            "mode": snapshot.mode,
            "symlink_target": snapshot.symlink_target,
            "sha256": digest,
        }
        if snapshot.resolved_target is not None:
            source["resolved_target"] = str(snapshot.resolved_target)
        if snapshot.target_mode is not None:
            source["target_mode"] = snapshot.target_mode
        if backup is not None:
            source["backup"] = str(backup.relative_to(root))
        sources[name] = source
    payload = {
        "schema_version": 1,
        "migration": _MIGRATION_NAME,
        "sources": sources,
    }
    rendered = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    _write_atomic(root / "manifest.json", rendered, 0o600)
    if json.loads((root / "manifest.json").read_text(encoding="utf-8")) != payload:
        raise RuntimeError(f"迁移备份 manifest 校验失败: {root / 'manifest.json'}")


def _remove_current_path(path: Path) -> None:
    """Remove one file or symlink without ever recursively deleting a directory."""

    if not _path_exists(path):
        return
    if path.is_dir() and not path.is_symlink():
        raise RuntimeError(f"无法删除目录路径: {path}")
    path.unlink()
    _fsync_directory(path.parent)


def _restore_snapshot(snapshot: _PathSnapshot, backup: Path | None, digest: str | None) -> None:
    """Restore original bytes and symlink identity after a failed publish."""

    if not snapshot.existed:
        _remove_current_path(snapshot.path)
        return
    if backup is None or digest is None or not backup.is_file():
        raise RuntimeError(f"恢复备份缺失: {snapshot.path}")
    content = backup.read_bytes()
    if _sha256(content) != digest:
        raise RuntimeError(f"恢复备份摘要不匹配: {snapshot.path}")
    if snapshot.kind == "symlink":
        if snapshot.resolved_target is None or snapshot.symlink_target is None:
            raise RuntimeError(f"软链接快照不完整: {snapshot.path}")
        _write_atomic(snapshot.resolved_target, content, snapshot.target_mode or 0o600)
        if not snapshot.path.is_symlink() or os.readlink(snapshot.path) != snapshot.symlink_target:
            _remove_current_path(snapshot.path)
            snapshot.path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(snapshot.symlink_target, snapshot.path)
            _fsync_directory(snapshot.path.parent)
        if snapshot.path.read_bytes() != content:
            raise RuntimeError(f"软链接恢复校验失败: {snapshot.path}")
        return
    _remove_current_path(snapshot.path)
    _write_atomic(snapshot.path, content, snapshot.mode or 0o600)
    if snapshot.path.read_bytes() != content:
        raise RuntimeError(f"文件恢复校验失败: {snapshot.path}")


def _publish_config(snapshot: _PathSnapshot, rendered: bytes) -> None:
    """Publish config bytes through the resolved target while preserving links."""

    if not snapshot.existed or snapshot.resolved_target is None:
        return
    _write_atomic(snapshot.resolved_target, rendered, snapshot.target_mode or 0o600)
    if snapshot.kind == "symlink" and (
        not snapshot.path.is_symlink() or os.readlink(snapshot.path) != snapshot.symlink_target
    ):
        raise RuntimeError(f"配置软链接身份发生变化: {snapshot.path}")
    if snapshot.resolved_target.read_bytes() != rendered:
        raise RuntimeError(f"配置迁移发布校验失败: {snapshot.path}")


def retire_legacy_context_state(_connection: object) -> None:
    """Archive config/RECENT, remove legacy keys, and retire RECENT last."""

    _ = _connection
    current = current_migration_context()
    config_snapshot = _snapshot_path(current.config_path)
    recent_path = current.workspace / "memory" / "RECENT_CONTEXT.md"
    recent_snapshot = _snapshot_path(recent_path)
    rendered_config: bytes | None = None
    if config_snapshot.existed:
        rendered_config = _render_config(config_snapshot.content or b"")
    config_needs_publish = rendered_config != config_snapshot.content
    if not config_needs_publish and not recent_snapshot.existed:
        return

    # 1. Snapshot and archive both external paths before publishing either one.
    backup_root = current.workspace / "backups" / _MIGRATION_NAME / uuid4().hex
    backup_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(backup_root, 0o700)
    config_backup = backup_root / "config" / "config.toml"
    recent_backup = backup_root / "memory" / "RECENT_CONTEXT.md"
    config_digest = _archive_snapshot(config_snapshot, config_backup)
    recent_digest = _archive_snapshot(recent_snapshot, recent_backup)
    records = {
        "config": (config_snapshot, config_backup if config_snapshot.existed else None, config_digest),
        "recent_context": (
            recent_snapshot,
            recent_backup if recent_snapshot.existed else None,
            recent_digest,
        ),
    }
    _write_manifest(backup_root, records)

    try:
        # 2. Parse and publish config first; only then remove the legacy projection.
        if config_needs_publish:
            _publish_config(config_snapshot, rendered_config or b"")
        if recent_snapshot.existed:
            _remove_current_path(recent_path)
            if _path_exists(recent_path):
                raise RuntimeError(f"RECENT_CONTEXT.md 删除失败: {recent_path}")
    except BaseException as migration_error:
        # 3. Restore both path identities so Yoyo cannot record a false success.
        try:
            _restore_snapshot(
                config_snapshot,
                config_backup if config_snapshot.existed else None,
                config_digest,
            )
            _restore_snapshot(
                recent_snapshot,
                recent_backup if recent_snapshot.existed else None,
                recent_digest,
            )
        except BaseException as restore_error:
            raise RuntimeError(
                f"legacy context retirement failed and restore failed: {migration_error}"
            ) from restore_error
        raise


steps = [step(retire_legacy_context_state)]
