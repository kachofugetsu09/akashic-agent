"""Frozen path checks used by historical plugin-data migrations."""
from __future__ import annotations

import re
from pathlib import Path


def builtin_plugin_data_dir(plugin_name: str, workspace: Path) -> Path:
    """Resolve one historical builtin plugin-data root without creating it."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", plugin_name) is None:
        raise ValueError(f"插件 name 不是安全路径段: {plugin_name!r}")
    return workspace.resolve(strict=False) / "plugin-data" / f"{plugin_name}-builtin"


def validate_workspace_plugin_data_path(path: Path, workspace: Path) -> None:
    """Reject historical plugin data outside the selected workspace or symlink chain."""
    root = workspace.resolve(strict=False)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"插件数据目录越界: {path}") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"插件数据目录不能穿过符号链接: {current}")
