from __future__ import annotations

import hashlib
from pathlib import Path


def read_config_source(path: Path) -> tuple[bytes | None, str]:
    """同一次读取产生配置正文和版本，避免先校验 A 再使用 B。"""
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        content = None
    digest = hashlib.sha256()
    digest.update(str(path.resolve(strict=False)).encode())
    digest.update(content if content is not None else b"<missing>")
    return content, digest.hexdigest()
