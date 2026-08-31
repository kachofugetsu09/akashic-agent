import hashlib
import json
import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context
from agent.plugins.manifest import validate_workspace_plugin_data_path

__depends__ = {"20260829_03_retire_core_model_config"}
__transactional__ = False

_MIGRATION = "migrate-compaction-plugin-config"


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    target: Path
    content: bytes | None
    mode: int
    symlink_target: str | None


def _snapshot(path: Path, *, absent_mode: int = 0o600) -> _Snapshot:
    """Capture exact bytes and link identity before publication."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return _Snapshot(path, path, None, absent_mode, None)
    if stat.S_ISLNK(metadata.st_mode):
        link = os.readlink(path)
        target = path.resolve(strict=True)
        if not target.is_file():
            raise RuntimeError(f"compaction 配置软链接目标不是普通文件: {path}")
        return _Snapshot(
            path,
            target,
            target.read_bytes(),
            stat.S_IMODE(target.stat().st_mode),
            link,
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"compaction 配置必须是普通文件或软链接: {path}")
    return _Snapshot(path, path, path.read_bytes(), stat.S_IMODE(metadata.st_mode), None)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write(path: Path, payload: bytes, mode: int) -> None:
    """Publish one fsynced file with a same-directory atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        candidate.write_bytes(payload)
        candidate.chmod(mode)
        with candidate.open("rb") as stream:
            os.fsync(stream.fileno())
        candidate.replace(path)
        _fsync_dir(path.parent)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise


def _render(
    config_raw: bytes,
    plugin_raw: bytes | None,
) -> tuple[bytes, bytes] | None:
    """Move the exact retired Core policy into ordinary plugin config."""

    config_text = config_raw.decode("utf-8")
    parsed = tomllib.loads(config_text)
    agent = parsed.get("agent")
    context = agent.get("context") if isinstance(agent, dict) else None
    legacy = context.get("compaction") if isinstance(context, dict) else None
    if legacy is None:
        return None
    if not isinstance(legacy, dict) or set(legacy) != {"keep_recent_tokens"}:
        raise RuntimeError("agent.context.compaction 含未识别字段，拒绝迁移")
    value = legacy["keep_recent_tokens"]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError("agent.context.compaction.keep_recent_tokens 必须是正整数")

    document = tomlkit.parse(config_text)
    document_agent = document.get("agent")
    document_context = (
        document_agent.get("context") if isinstance(document_agent, dict) else None
    )
    if not isinstance(document_context, dict):
        raise RuntimeError("agent.context 配置结构在解析期间变化")
    del document_context["compaction"]
    if not document_context:
        del document_agent["context"]
    rendered_config = tomlkit.dumps(document).encode("utf-8")

    plugin_document = (
        tomlkit.document()
        if plugin_raw is None
        else tomlkit.parse(plugin_raw.decode("utf-8"))
    )
    existing = plugin_document.get("keep_recent_tokens")
    if existing is not None and existing != value:
        raise RuntimeError("compaction plugin keep_recent_tokens 与旧 Core 配置冲突")
    plugin_document["keep_recent_tokens"] = value
    rendered_plugin = tomlkit.dumps(plugin_document).encode("utf-8")

    final_config = tomllib.loads(rendered_config.decode("utf-8"))
    final_agent = final_config.get("agent")
    final_context = final_agent.get("context") if isinstance(final_agent, dict) else None
    if isinstance(final_context, dict) and "compaction" in final_context:
        raise RuntimeError("迁移后 Core 配置仍含 compaction")
    if tomllib.loads(rendered_plugin.decode("utf-8")).get("keep_recent_tokens") != value:
        raise RuntimeError("迁移后 compaction plugin 配置值不一致")
    return rendered_config, rendered_plugin


def _backup(
    snapshots: tuple[_Snapshot, ...],
    rendered: tuple[bytes, ...],
    root: Path,
) -> None:
    """Write verified before/after images and a durable publication intent."""

    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(root, 0o700)
    entries: list[dict[str, object]] = []
    for index, snapshot in enumerate(snapshots):
        backup_name: str | None = None
        digest: str | None = None
        if snapshot.content is not None:
            backup_name = f"source-{index}.before"
            backup = root / backup_name
            _write(backup, snapshot.content, 0o600)
            digest = hashlib.sha256(snapshot.content).hexdigest()
            if hashlib.sha256(backup.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"compaction 配置迁移备份校验失败: {snapshot.path}")
        entries.append(
            {
                "path": str(snapshot.path),
                "target": str(snapshot.target),
                "symlink_target": snapshot.symlink_target,
                "mode": snapshot.mode,
                "backup": backup_name,
                "sha256": digest,
            }
        )
        _write(root / f"source-{index}.after", rendered[index], 0o600)
    manifest = {"schema_version": 1, "migration": _MIGRATION, "sources": entries}
    _write(
        root / "manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
        0o600,
    )
    _write(
        root / "intent.json",
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "migration": _MIGRATION,
                    "paths": [str(snapshot.path) for snapshot in snapshots],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        0o600,
    )


def _complete_intent(root: Path, snapshots: tuple[_Snapshot, ...]) -> None:
    """Forward-complete one prepared two-file publication after any crash."""

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != len(snapshots):
        raise RuntimeError("compaction 配置迁移 intent sources 无效")
    after_images: list[bytes] = []
    current_snapshots: list[_Snapshot] = []
    for index, (snapshot, raw_source) in enumerate(zip(snapshots, sources, strict=True)):
        current = _snapshot(snapshot.path)
        if (
            not isinstance(raw_source, dict)
            or raw_source.get("path") != str(current.path)
            or raw_source.get("target") != str(current.target)
            or raw_source.get("symlink_target") != current.symlink_target
        ):
            raise RuntimeError("compaction 配置迁移 intent path 冲突")
        backup_name = raw_source.get("backup")
        before = None if backup_name is None else (root / str(backup_name)).read_bytes()
        after = (root / f"source-{index}.after").read_bytes()
        if current.content not in (before, after):
            raise RuntimeError(f"compaction 配置迁移恢复时目标已变化: {current.path}")
        current_snapshots.append(current)
        after_images.append(after)
    for current, raw_source, after in zip(
        current_snapshots, sources, after_images, strict=True
    ):
        assert isinstance(raw_source, dict)
        mode = raw_source.get("mode")
        if not isinstance(mode, int):
            raise RuntimeError("compaction 配置迁移 intent mode 无效")
        _write(current.target, after, mode)
    if any(
        current.path.read_bytes() != after
        for current, after in zip(current_snapshots, after_images, strict=True)
    ):
        raise RuntimeError("compaction 配置迁移发布校验失败")
    _write(root / "complete", b"complete\n", 0o600)


def _recover_intents(
    *,
    config_path: Path,
    plugin_path: Path,
    backup_root: Path,
) -> None:
    """Finish every prepared, non-terminal publication before reading legacy config."""

    if not backup_root.exists():
        return
    expected_paths = [str(config_path), str(plugin_path)]
    for intent_path in sorted(backup_root.glob("*/intent.json")):
        root = intent_path.parent
        if (root / "conflict").exists():
            raise RuntimeError(f"compaction 配置迁移存在未解决目标冲突: {root}")
        if (root / "complete").exists() or (root / "rolled_back").exists():
            continue
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        if intent.get("migration") != _MIGRATION or intent.get("paths") != expected_paths:
            raise RuntimeError("compaction 配置迁移 intent identity 冲突")
        _complete_intent(root, (_snapshot(config_path), _snapshot(plugin_path)))


def _restore(snapshot: _Snapshot) -> None:
    """Restore exact bytes or absence after a failed two-file publication."""

    if snapshot.content is None:
        snapshot.path.unlink(missing_ok=True)
        return
    _write(snapshot.target, snapshot.content, snapshot.mode)
    if snapshot.symlink_target is not None and (
        not snapshot.path.is_symlink()
        or os.readlink(snapshot.path) != snapshot.symlink_target
    ):
        raise RuntimeError(f"compaction 配置软链接恢复失败: {snapshot.path}")


def migrate_compaction_plugin_config(_connection: object) -> None:
    """Move Core compaction policy to its ordinary plugin with recovery proof."""

    _ = _connection
    current = current_migration_context()
    workspace = current.workspace.resolve(strict=False)
    plugin_path = (
        workspace
        / "plugin-data"
        / "compaction-builtin"
        / "config.local.toml"
    )
    validate_workspace_plugin_data_path(plugin_path, workspace)
    backup_root = workspace / "backups" / _MIGRATION
    _recover_intents(
        config_path=current.config_path,
        plugin_path=plugin_path,
        backup_root=backup_root,
    )
    config = _snapshot(current.config_path)
    if config.content is None:
        return
    plugin = _snapshot(plugin_path)
    rendered = _render(config.content, plugin.content)
    if rendered is None:
        return

    publication_root = backup_root / uuid4().hex
    _backup((config, plugin), rendered, publication_root)
    try:
        _complete_intent(publication_root, (config, plugin))
    except BaseException as migration_error:
        try:
            current_snapshots = (_snapshot(config.path), _snapshot(plugin.path))
            safe_to_restore = all(
                current_snapshot.path == before_snapshot.path
                and current_snapshot.target == before_snapshot.target
                and current_snapshot.symlink_target == before_snapshot.symlink_target
                and current_snapshot.content in (before_snapshot.content, after)
                for current_snapshot, before_snapshot, after in zip(
                    current_snapshots,
                    (config, plugin),
                    rendered,
                    strict=True,
                )
            )
        except BaseException:
            safe_to_restore = False
        if not safe_to_restore:
            _write(publication_root / "conflict", b"conflict\n", 0o600)
            raise
        try:
            _restore(plugin)
            _restore(config)
            _write(publication_root / "rolled_back", b"rolled_back\n", 0o600)
        except BaseException as restore_error:
            raise RuntimeError(
                f"compaction 配置迁移失败且恢复失败: {migration_error}"
            ) from restore_error
        raise


steps = [step(migrate_compaction_plugin_config)]
