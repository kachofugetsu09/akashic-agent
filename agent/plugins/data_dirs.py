"""插件数据目录以选择记录为准；换来源时继承，只有安装单个插件时才按磁盘解析。"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agent.plugins.manifest import plugins_root, workspace_plugin_data_dir
from agent.plugins.selection import PluginSelection, SelectionFormatError
from agent.plugins.static_manifest import PluginSourceContentError

# 早期内置与发行版来源在磁盘上留下的旧目录后缀。
_KNOWN_SOURCES = ("builtin", "release")


class PluginDataAmbiguous(PluginSourceContentError):
    """同一逻辑名有多个非空候选目录；只让这一次安装或准备失败，不影响启动。"""


@dataclass(frozen=True)
class RecordedDataDirs:
    by_id: Mapping[str, str]
    by_name: Mapping[str, str]

    # 自己的记录优先；换来源时同逻辑名的旧记录让位给新来源并被继承。
    def lookup(self, plugin_id: str) -> str | None:
        return self.by_id.get(plugin_id) or self.by_name.get(plugin_id.partition("@")[0])


# 读取当前选择里每个已选插件的 data_dir；选择缺失或损坏时视为没有记录。
def read_recorded_data_dirs(selection: PluginSelection) -> RecordedDataDirs:
    by_id: dict[str, str] = {}
    by_name: dict[str, str] = {}
    try:
        ref = selection.read() if selection.path.exists() else None
        components = () if ref is None else selection.components(ref)
    except SelectionFormatError:
        components = ()
    for component in components:
        record = selection.read_input(component)
        plugin_id = cast(str, record["plugin_id"])
        by_id[plugin_id] = cast(str, record["data_dir"])
        by_name[plugin_id.partition("@")[0]] = cast(str, record["data_dir"])
    return RecordedDataDirs(by_id, by_name)


def _label_path(workspace: Path, label: str) -> Path:
    return workspace.resolve(strict=False) / label


# 不探测磁盘、不会失败：有记录用记录，否则用该来源的默认目录。
def recorded_data_dir(workspace: Path, plugin_id: str, recorded: RecordedDataDirs) -> Path:
    label = recorded.lookup(plugin_id)
    if label is not None:
        return _label_path(workspace, label)
    name, _, marketplace = plugin_id.partition("@")
    return workspace_plugin_data_dir(workspace, name, marketplace or "builtin")


def _nonempty(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink() and any(path.iterdir())


# 仅在安装/准备单个尚未选择的插件时调用。
# 1. 有记录用记录；2. 该来源自己的旧目录非空就接回；3. 其余已知来源恰好一个非空就接回；
# 4. 多个非空则失败并列出候选；5. 都没有就新建默认目录 <name>-<marketplace>。
def resolve_install_data_dir(
    workspace: Path, plugin_id: str, recorded: RecordedDataDirs,
    *, marketplaces: Iterable[str] = (),
) -> Path:
    label = recorded.lookup(plugin_id)
    if label is not None:
        return _label_path(workspace, label)
    name, _, marketplace = plugin_id.partition("@")
    own = workspace_plugin_data_dir(workspace, name, marketplace or "builtin")
    if _nonempty(own):
        return own
    sources = {*_KNOWN_SOURCES, *marketplaces, *_cache_marketplaces()} - {marketplace or "builtin"}
    bare = workspace.resolve(strict=False) / "plugin-data" / name
    found = [path for path in (
        bare, *(workspace_plugin_data_dir(workspace, name, source) for source in sorted(sources)),
    ) if _nonempty(path)]
    if len(found) > 1:
        raise PluginDataAmbiguous(
            f"插件 {name} 在 plugin-data 下有多个非空数据目录，无法确定接回哪个: "
            + ", ".join(path.name for path in found) + "；请移走多余的目录后重试")
    return found[0] if found else own


def _cache_marketplaces() -> tuple[str, ...]:
    cache = plugins_root() / "cache"
    return tuple(path.name for path in cache.iterdir() if path.is_dir()) if cache.is_dir() else ()
