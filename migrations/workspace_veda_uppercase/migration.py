from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

_LEGACY_RELATIVE_PATH = Path("memory/veda.md")
_TARGET_RELATIVE_PATH = Path("memory/VEDA.md")


@dataclass(frozen=True)
class MigrationContext:
    config_path: Path
    workspace: Path
    migration_commit: str
    backup_dir: Path | None


def _parse_args() -> tuple[str, MigrationContext]:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "action",
        choices=("assess", "apply", "verify", "revert"),
    )
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--workspace", type=Path, required=True)
    _ = parser.add_argument("--migration-commit", required=True)
    _ = parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    return str(args.action), MigrationContext(
        config_path=Path(args.config).expanduser().resolve(),
        workspace=Path(args.workspace).expanduser().resolve(),
        migration_commit=str(args.migration_commit),
        backup_dir=Path(args.backup_dir).resolve() if args.backup_dir else None,
    )


def _paths(context: MigrationContext) -> tuple[Path, Path]:
    return (
        context.workspace / _LEGACY_RELATIVE_PATH,
        context.workspace / _TARGET_RELATIVE_PATH,
    )


def _read_valid(path: Path) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"VEDA 无法读取: {path} ({type(exc).__name__})") from exc
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"VEDA 不是合法 UTF-8: {path}") from exc
    if not content.strip():
        raise RuntimeError(f"VEDA 内容为空: {path}")
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assess(context: MigrationContext) -> dict[str, str]:
    legacy, target = _paths(context)
    legacy_exists = legacy.exists()
    target_exists = target.exists()
    if not legacy_exists and not target_exists:
        return {
            "status": "blocked",
            "reason": f"已有 workspace 同时缺少 {legacy} 与 {target}",
        }
    try:
        legacy_payload = _read_valid(legacy) if legacy_exists else None
        target_payload = _read_valid(target) if target_exists else None
    except RuntimeError as exc:
        return {"status": "blocked", "reason": str(exc)}
    if target_payload is not None and legacy_payload is None:
        return {"status": "satisfied"}
    if target_payload is not None and legacy_payload != target_payload:
        return {
            "status": "blocked",
            "reason": f"大小写 VEDA 内容冲突: legacy={legacy} target={target}",
        }
    return {"status": "needed"}


def _atomic_write(path: Path, payload: bytes) -> None:
    """在同目录刷写文件并原子发布。"""

    # 1. 写入唯一临时文件并同步正文
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            _ = stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        # 2. 原子发布并同步目录项
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _apply(context: MigrationContext) -> None:
    if context.backup_dir is None:
        raise RuntimeError("apply 缺少 --backup-dir")
    legacy, target = _paths(context)
    assessment = _assess(context)
    if assessment["status"] != "needed":
        raise RuntimeError(f"VEDA 当前状态不可迁移: {assessment}")
    payload = _read_valid(legacy)
    target_preexisting = target.exists()

    # 1. 在更改 workspace 前持久化原文件与恢复 manifest
    context.backup_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(context.backup_dir, 0o700)
    _atomic_write(context.backup_dir / "veda.md", payload)
    manifest: dict[str, object] = {
        "migrationCommit": context.migration_commit,
        "legacyPath": str(legacy),
        "targetPath": str(target),
        "sha256": _sha256(payload),
        "targetPreexisting": target_preexisting,
    }
    _atomic_write(
        context.backup_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )

    # 2. 仅一份真源：重命名或删除字节完全相同的旧路径
    if target_preexisting:
        legacy.unlink()
    else:
        os.replace(legacy, target)
    _fsync_directory(target.parent)


def _verify(context: MigrationContext) -> None:
    legacy, target = _paths(context)
    if legacy.exists():
        raise RuntimeError(f"VEDA 迁移验证失败，旧路径仍存在: {legacy}")
    _ = _read_valid(target)


def _load_manifest(backup_dir: Path) -> tuple[Path, Path, str, bool]:
    manifest_path = backup_dir / "manifest.json"
    payload: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"VEDA manifest 必须是对象: {manifest_path}")
    document = cast(dict[object, object], payload)
    legacy = document.get("legacyPath")
    target = document.get("targetPath")
    digest = document.get("sha256")
    target_preexisting = document.get("targetPreexisting")
    if (
        not isinstance(legacy, str)
        or not isinstance(target, str)
        or not isinstance(digest, str)
        or not isinstance(target_preexisting, bool)
    ):
        raise RuntimeError(f"VEDA manifest 字段无效: {manifest_path}")
    return Path(legacy), Path(target), digest, target_preexisting


def _revert(context: MigrationContext) -> None:
    if context.backup_dir is None or not context.backup_dir.is_dir():
        raise RuntimeError("revert 需要有效的 --backup-dir")
    legacy, target, digest, target_preexisting = _load_manifest(context.backup_dir)
    if (legacy, target) != _paths(context):
        raise RuntimeError("VEDA manifest 路径与当前 workspace 不匹配")
    target_payload = _read_valid(target)
    if _sha256(target_payload) != digest:
        raise RuntimeError(f"VEDA 已在迁移后修改，拒绝回滚: {target}")
    if legacy.exists():
        raise RuntimeError(f"VEDA 旧路径已被重新创建，拒绝覆盖: {legacy}")

    # 1. 恢复迁移前的小写文件
    backup_payload = _read_valid(context.backup_dir / "veda.md")
    if _sha256(backup_payload) != digest:
        raise RuntimeError("VEDA 迁移备份摘要不匹配")
    _atomic_write(legacy, backup_payload)

    # 2. 仅删除本次重命名创建的目标
    if not target_preexisting:
        target.unlink()
    _fsync_directory(target.parent)


def main() -> None:
    action, context = _parse_args()
    if action == "assess":
        print(json.dumps(_assess(context), ensure_ascii=False))
    elif action == "apply":
        _apply(context)
    elif action == "verify":
        _verify(context)
    else:
        _revert(context)


if __name__ == "__main__":
    main()
