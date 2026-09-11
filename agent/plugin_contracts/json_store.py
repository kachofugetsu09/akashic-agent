"""
infra.persistence.json_store — 统一 JSON 文件持久化基础工具。

替代散落在各模块的 _load()/_save() 重复实现，提供：
- 原子写（tmp 文件 + rename）
- 读取边界（仅文件缺失返回 default）
- 统一日志格式
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO, cast

logger = logging.getLogger(__name__)
_ATOMIC_WRITE_TEMP_ATTEMPTS = 100

__all__ = [
    "load_json",
    "save_json",
    "atomic_save_json",
    "atomic_write_text",
]


def load_json(
    path: Path,
    default: Any = None,
    *,
    domain: str = "json_store",
) -> Any:
    """
    从文件读取 JSON，仅文件缺失返回 default，其他失败带上下文抛出异常。

    Args:
        path: JSON 文件路径。
        default: 文件不存在时的返回值（默认 None）。
        domain: 日志标识域，格式 "[domain] ..."。
    """
    # 1. 读取并解析；只有文件不存在属于可选状态
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        return default
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"[{domain}] 读取 JSON 失败: path={path} err={error}"
        ) from error


def save_json(
    path: Path,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    domain: str = "json_store",
) -> None:
    """
    将数据写入 JSON 文件（非原子写，适合对崩溃不敏感的场景）。

    Args:
        path: 目标文件路径，父目录不存在时自动创建。
        data: 可序列化对象。
        indent: JSON 缩进。
        ensure_ascii: 是否转义非 ASCII。
        domain: 日志标识域。
    """
    # 1. 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    # 2. 写入
    try:
        path.write_text(
            json.dumps(data, indent=indent, ensure_ascii=ensure_ascii),
            encoding="utf-8",
        )
        logger.debug("[%s] 已写入 path=%s", domain, path)
    except Exception as e:
        logger.warning("[%s] 写入 JSON 失败: path=%s err=%s", domain, path, e)
        raise


def atomic_save_json(
    path: Path,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    domain: str = "json_store",
) -> None:
    """
    原子写：先写到 .tmp 再 rename，避免写到一半崩溃损坏文件。

    Args:
        path: 目标文件路径，父目录不存在时自动创建。
        data: 可序列化对象。
        indent: JSON 缩进。
        ensure_ascii: 是否转义非 ASCII。
        domain: 日志标识域。
    """
    _atomic_write(
        path,
        lambda stream: json.dump(
            data,
            stream,
            indent=indent,
            ensure_ascii=ensure_ascii,
        ),
        domain=domain,
    )


def atomic_write_text(
    path: Path,
    content: str,
    *,
    domain: str = "json_store",
) -> None:
    """原子写入 UTF-8 文本，并在替换后持久化父目录。"""

    def write_content(stream: TextIO) -> None:
        stream.write(content)

    _atomic_write(path, write_content, domain=domain)


def _create_atomic_temp(path: Path) -> tuple[int, Path]:
    """使用内核 umask 创建同目录的唯一临时文件。"""

    # 1. 用高熵文件名和 O_EXCL 排除临时文件碰撞
    for _ in range(_ATOMIC_WRITE_TEMP_ATTEMPTS):
        temporary = path.parent / f"{path.name}.{secrets.token_hex(16)}.tmp"
        try:
            fd = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o666,
            )
        except FileExistsError:
            continue
        return fd, temporary

    raise FileExistsError(f"无法为 {path} 创建唯一临时文件")


def _atomic_write(
    path: Path,
    writer: Callable[[TextIO], None],
    *,
    domain: str,
) -> None:
    """在同目录临时文件中完成写入、替换和目录同步。"""

    # 1. 创建父目录并读取现有目标的权限位
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        target_mode = None

    # 2. 创建同目录唯一临时文件；新文件权限由内核直接应用 umask
    fd, temporary = _create_atomic_temp(path)
    try:
        if target_mode is not None:
            os.fchmod(fd, target_mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            fd = -1
            # 3. 序列化、写入、刷写并同步临时文件
            writer(cast(TextIO, stream))
            stream.flush()
            os.fsync(stream.fileno())

        # 4. 原子替换并同步目录项
        _ = temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            logger.warning(
                "[%s] 原子写清理临时文件失败: tmp=%s err=%s",
                domain,
                temporary,
                cleanup_error,
            )
        raise
    finally:
        if fd != -1:
            os.close(fd)

    logger.debug("[%s] 原子写完成 path=%s", domain, path)
