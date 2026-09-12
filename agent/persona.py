from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from infra.persistence.json_store import atomic_write_text


VEDA_RELATIVE_PATH = Path("memory/VEDA.md")
DEFAULT_VEDA_PATH = Path(__file__).resolve().parents[1] / "prompts" / "VEDA.md"


class VedaLoadError(RuntimeError):
    """报告 Veda 边界损坏，并提供显式恢复入口。"""


@dataclass(frozen=True)
class VedaResetResult:
    path: Path
    backup_path: Path | None
    previous_sha256: str | None
    default_sha256: str
    changed: bool


def veda_path(workspace: Path) -> Path:
    return workspace.expanduser().resolve() / VEDA_RELATIVE_PATH


def _decode_veda(payload: bytes, *, path: Path) -> str:
    """校验并返回非空 UTF-8 Veda 正文。"""

    # 1. 在文件边界严格解码，不把损坏内容解释成默认人格。
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VedaLoadError(
            f"Veda 不是合法 UTF-8: {path}；"
            "请运行 `python main.py veda-reset` 恢复默认人格"
        ) from exc

    # 2. 空人格没有可执行语义，必须由显式命令恢复。
    content = content.strip()
    if not content:
        raise VedaLoadError(
            f"Veda 内容为空: {path}；"
            "请运行 `python main.py veda-reset` 恢复默认人格"
        )
    return content


def read_veda(workspace: Path) -> str:
    return read_veda_file(veda_path(workspace))


def read_veda_file(path: Path) -> str:
    """读取已获授的人格文件；缺失和损坏必须由显式恢复命令处理。"""
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise VedaLoadError(
            f"缺少 Veda: {path}；"
            "请运行 `python main.py veda-reset` 恢复默认人格"
        ) from exc
    return _decode_veda(payload, path=path)


def read_default_veda() -> str:
    try:
        payload = DEFAULT_VEDA_PATH.read_bytes()
    except FileNotFoundError as exc:
        raise VedaLoadError(f"缺少默认 Veda 模板: {DEFAULT_VEDA_PATH}") from exc
    return _decode_veda(payload, path=DEFAULT_VEDA_PATH)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_backup(path: Path, payload: bytes) -> None:
    """以不可覆盖文件保存 Veda 原始字节。"""

    # 1. 备份目录和文件只由本次 reset 创建。
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=False)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        # 2. 完整刷写原始字节，非法 UTF-8 也能精确恢复。
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            _ = stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor != -1:
            os.close(descriptor)


def reset_veda(workspace: Path) -> VedaResetResult:
    """备份当前 Veda，并原子恢复仓库默认人格。"""

    # 1. 先验证默认模板，模板损坏时禁止触碰 workspace。
    default_content = read_default_veda()
    default_payload = f"{default_content}\n".encode("utf-8")
    target = veda_path(workspace)
    try:
        previous_payload = target.read_bytes()
    except FileNotFoundError:
        previous_payload = None

    default_digest = _sha256(default_payload)
    if previous_payload == default_payload:
        return VedaResetResult(
            path=target,
            backup_path=None,
            previous_sha256=default_digest,
            default_sha256=default_digest,
            changed=False,
        )

    # 2. 现有内容先形成独立恢复点，备份失败时不覆盖。
    backup_path: Path | None = None
    previous_digest: str | None = None
    if previous_payload is not None:
        previous_digest = _sha256(previous_payload)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        backup_root = target.parent / "veda-backups"
        backup_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(backup_root, 0o700)
        backup_path = backup_root / timestamp / "VEDA.md"
        _write_backup(backup_path, previous_payload)

    # 3. 原子发布默认内容；正在进行的轮次仍持有此前 prompt。
    atomic_write_text(target, f"{default_content}\n", domain="veda_reset")
    return VedaResetResult(
        path=target,
        backup_path=backup_path,
        previous_sha256=previous_digest,
        default_sha256=default_digest,
        changed=True,
    )
