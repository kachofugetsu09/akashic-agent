from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class PluginConfig:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = dict(values)

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._values[key]
        except KeyError as e:
            raise AttributeError(key) from e


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
