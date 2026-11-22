"""已被 Core 吸收的插件身份清单（随发行版的 bundles/absorbed.toml）的唯一读取与校验；只依赖标准库，宿主脚本与 Core 共用。

文件格式坏了就明确失败并指出文件：静默当空清单会让旧外置包与 Core 重复注册的问题重现。文件不存在等同空清单（旧发行版）。
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

_IDENTITY = re.compile(r"[a-z][a-z0-9_-]{0,63}@[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


# 身份到提供它的 Core 插件名。
def read_absorbed(distribution: Path | None) -> dict[str, str]:
    if distribution is None:
        return {}
    path = distribution / "bundles" / "absorbed.toml"
    if not path.is_file():
        return {}
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"已吸收身份清单无法读取: {path}: {error}") from error
    if document.get("schema_version") != 1 or set(document) - {"schema_version", "absorbed"}:
        raise ValueError(f"已吸收身份清单格式无效（schema_version 必须是 1，只允许 absorbed 表）: {path}")
    table = document.get("absorbed", {})
    if not isinstance(table, dict):
        raise ValueError(f"已吸收身份清单格式无效（absorbed 必须是表）: {path}")
    found: dict[str, str] = {}
    for identity, row in table.items():
        provider = row.get("provider") if isinstance(row, dict) else None
        if _IDENTITY.fullmatch(identity) is None or not isinstance(provider, str) or not provider:
            raise ValueError(f"已吸收身份清单条目无效（{identity!r} 需要非空字符串 provider）: {path}")
        found[identity] = provider
    return found
