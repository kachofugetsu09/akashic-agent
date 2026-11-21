"""插件数据目录以选择记录为准；没有记录时按实例是否已归一化取默认目录。"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from agent.plugins.selection import PluginSelection, SelectionFormatError
from agent.plugins.manifest import default_plugin_data_label


# 读取当前选择里每个已选插件的 data_dir；选择缺失或损坏时视为没有记录。
def read_recorded_data_dirs(selection: PluginSelection) -> Mapping[str, str]:
    recorded: dict[str, str] = {}
    try:
        ref = selection.read() if selection.path.exists() else None
        components = () if ref is None else selection.components(ref)
    except SelectionFormatError:
        components = ()
    for component in components:
        record = selection.read_input(component)
        recorded[cast(str, record["plugin_id"])] = cast(str, record["data_dir"])
    return recorded


# 不探测磁盘、不会失败：有记录用记录，否则用实例的默认目录。
def recorded_data_dir(workspace: Path, plugin_id: str, recorded: Mapping[str, str]) -> Path:
    label = recorded.get(plugin_id) or default_plugin_data_label(workspace, plugin_id)
    return workspace.resolve(strict=False) / label
