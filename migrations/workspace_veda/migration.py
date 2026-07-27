from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4


_VEDA_RELATIVE_PATH = Path("memory/veda.md")
_TEMPLATE_PATH = Path(__file__).with_name("veda.md")


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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_nonempty(payload: bytes, *, source: Path) -> str:
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Veda 不是合法 UTF-8: {source}") from exc
    if not content.strip():
        raise RuntimeError(f"Veda 内容为空: {source}")
    return content


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """在同目录完成刷写和原子发布。"""

    # 1. 创建唯一 candidate，写入完整内容。
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            _ = stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        # 2. 原子替换并同步目录项。
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _target(context: MigrationContext) -> Path:
    return context.workspace / _VEDA_RELATIVE_PATH


def _assess(context: MigrationContext) -> dict[str, str]:
    target = _target(context)
    try:
        payload = target.read_bytes()
    except FileNotFoundError:
        return {"status": "needed"}
    except OSError as exc:
        return {
            "status": "blocked",
            "reason": f"Veda 无法读取: {target} ({type(exc).__name__})",
        }
    try:
        _ = _decode_nonempty(payload, source=target)
    except RuntimeError as exc:
        return {"status": "blocked", "reason": str(exc)}
    return {"status": "satisfied"}


def _apply(context: MigrationContext) -> None:
    if context.backup_dir is None:
        raise RuntimeError("apply 缺少 --backup-dir")
    target = _target(context)
    if target.exists():
        raise RuntimeError(f"Veda 已存在，拒绝迁移覆盖: {target}")

    # 1. 验证 bundle 自带的不可变默认快照。
    template = _TEMPLATE_PATH.read_bytes()
    _ = _decode_nonempty(template, source=_TEMPLATE_PATH)

    # 2. 先发布恢复 manifest，再原子创建目标文件。
    context.backup_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(context.backup_dir, 0o700)
    manifest: dict[str, object] = {
        "migrationCommit": context.migration_commit,
        "created": [
            {
                "path": str(target),
                "sha256": _sha256(template),
            }
        ],
    }
    _atomic_write(
        context.backup_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    _atomic_write(target, template)


def _verify(context: MigrationContext) -> None:
    target = _target(context)
    try:
        payload = target.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Veda 迁移验证失败，文件不存在: {target}") from exc
    _ = _decode_nonempty(payload, source=target)


def _load_manifest(backup_dir: Path) -> tuple[Path, str]:
    manifest_path = backup_dir / "manifest.json"
    payload: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Veda migration manifest 必须是对象: {manifest_path}")
    document = cast(dict[object, object], payload)
    created = document.get("created")
    if not isinstance(created, list):
        raise RuntimeError(f"Veda migration manifest created 无效: {manifest_path}")
    created_records = cast(list[object], created)
    if len(created_records) != 1:
        raise RuntimeError(f"Veda migration manifest created 无效: {manifest_path}")
    record_value = created_records[0]
    if not isinstance(record_value, dict):
        raise RuntimeError(f"Veda migration manifest record 无效: {manifest_path}")
    record = cast(dict[object, object], record_value)
    path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise RuntimeError(f"Veda migration manifest 字段无效: {manifest_path}")
    return Path(path), digest


def _revert(context: MigrationContext) -> None:
    if context.backup_dir is None or not context.backup_dir.is_dir():
        raise RuntimeError("revert 需要有效的 --backup-dir")
    target, expected_digest = _load_manifest(context.backup_dir)
    if target != _target(context):
        raise RuntimeError(f"Veda migration manifest 目标不匹配: {target}")
    try:
        payload = target.read_bytes()
    except FileNotFoundError:
        return
    if _sha256(payload) != expected_digest:
        raise RuntimeError(f"Veda 已在迁移后修改，拒绝删除: {target}")
    target.unlink()
    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    action, context = _parse_args()
    if action == "assess":
        print(json.dumps(_assess(context), ensure_ascii=False))
        return
    if action == "apply":
        _apply(context)
        return
    if action == "verify":
        _verify(context)
        return
    _revert(context)


if __name__ == "__main__":
    main()
