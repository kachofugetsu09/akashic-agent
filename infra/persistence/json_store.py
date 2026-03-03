"""
infra.persistence.json_store — 统一 JSON 文件持久化基础工具。

替代散落在各模块的 _load()/_save() 重复实现，提供：
- 原子写（tmp 文件 + rename）
- 读取容错（坏文件不崩溃，返回 default）
- 统一日志格式
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "load_json",
    "save_json",
    "atomic_save_json",
]


def load_json(
    path: Path,
    default: Any = None,
    *,
    domain: str = "json_store",
) -> Any:
    """
    从文件读取 JSON，失败时返回 default。

    Args:
        path: JSON 文件路径。
        default: 文件不存在或解析失败时的返回值（默认 None）。
        domain: 日志标识域，格式 "[domain] ..."。
    """
    # 1. 文件不存在直接返回默认值
    if not path.exists():
        return default

    # 2. 读取并解析
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)
    except Exception as e:
        logger.warning(
            "[%s] 读取 JSON 失败，返回默认值: path=%s err=%s", domain, path, e
        )
        return default


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
    # 1. 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    # 2. 写到临时文件
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=indent, ensure_ascii=ensure_ascii),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("[%s] 原子写临时文件失败: path=%s err=%s", domain, tmp, e)
        raise

    # 3. 原子替换
    try:
        tmp.replace(path)
        logger.debug("[%s] 原子写完成 path=%s", domain, path)
    except Exception as e:
        logger.warning(
            "[%s] 原子替换失败: tmp=%s target=%s err=%s", domain, tmp, path, e
        )
        # 尝试清理临时文件
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise
