"""声明式组合只生成选择输入，不发现或自动补齐 provider。"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal
from pathlib import Path
import os
import re
import tomllib

import tomlkit
from typing import cast

from infra.persistence.json_store import atomic_write_text
from core.common.frozen_json import freeze_json

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PLUGIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:@[A-Za-z0-9][A-Za-z0-9._-]*)?")


@dataclass(frozen=True)
class BundleRow:
    id: str
    plugin: str
    config: Mapping[str, object]
    disabled: bool = False


# 在文件边界校验完整 row；缺席的 config 和 disabled 使用格式默认值。
def load_bundle(path: Path) -> tuple[BundleRow, ...]:
    return tuple(row for row, _ in _read_rows(path))


# 同 load_bundle，并带回 row 显式写出的字段，供用户 patch 按字段继承 preset。
def _read_rows(path: Path) -> tuple[tuple[BundleRow, frozenset[str]], ...]:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(document) != {"schema_version", "rows"} or type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError(f"bundle schema 无效: {path}")
    raw_rows = document["rows"]
    if not isinstance(raw_rows, dict):
        raise ValueError(f"bundle rows 必须是表: {path}")
    rows: list[tuple[BundleRow, frozenset[str]]] = []
    for identity, value in raw_rows.items():
        if _PLUGIN.fullmatch(identity) is None or not isinstance(value, dict):
            raise ValueError(f"bundle row 无效: {path}:{identity}")
        if set(value) - {"plugin", "config", "disabled"} or "plugin" not in value:
            raise ValueError(f"bundle row 字段无效: {path}:{identity}")
        plugin, config, disabled = value["plugin"], value.get("config", {}), value.get("disabled", False)
        if not isinstance(plugin, str) or _PLUGIN.fullmatch(plugin) is None:
            raise ValueError(f"bundle plugin 身份无效: {path}:{identity}")
        if not isinstance(config, dict) or not isinstance(disabled, bool):
            raise ValueError(f"bundle config/disabled 无效: {path}:{identity}")
        rows.append((BundleRow(identity, plugin, cast(Mapping[str, object], freeze_json(config)), disabled),
                     frozenset(cast(dict[str, object], value))))
    _check_plugins(row for row, _ in rows)
    return tuple(rows)


def _check_plugins(rows: Iterable[BundleRow]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.plugin in seen:
            raise ValueError(f"多个 bundle row 指向同一插件: {row.plugin}")
        seen.add(row.plugin)


# base、mode 由 preset 作者写全，整行替换；用户 patch 按字段覆盖，没写的 disabled/config 继续跟随 preset
# （同 DSH 的 cordis.patch.yml）。config 写了就整份替换，不深合并；换了 provider 的 row 不继承旧 config。
def load_bundles(directory: Path, *, mode: str = "base", patch: Path | None = None) -> tuple[BundleRow, ...]:
    selected = {row.id: row for row in preset_rows(directory, mode=mode)}
    if patch is not None and patch.exists():
        for row, fields in _read_rows(patch):
            base = preset_row_for(selected, row)
            if base is not None and base.plugin == row.plugin:
                row = BundleRow(base.id, row.plugin, row.config if "config" in fields else base.config,
                                row.disabled if "disabled" in fields else base.disabled)
            selected[row.id if base is None else base.id] = row
    _check_plugins(selected.values())
    return tuple(selected.values())


# patch row 覆盖哪条 preset row：先按 id；id 不同但插件相同时也算同一条（用户先启用了可选插件、
# preset 后来才收录它，两边 id 写法不同），否则会出现两条 row 指向同一插件。
def preset_row_for(preset: Mapping[str, BundleRow], row: BundleRow) -> BundleRow | None:
    if row.id in preset:
        return preset[row.id]
    return next((base for base in preset.values() if base.plugin == row.plugin), None)


# preset 自己的组合（base 加 mode），不含用户 patch。
def preset_rows(directory: Path, *, mode: str = "base") -> tuple[BundleRow, ...]:
    if _NAME.fullmatch(mode) is None:
        raise ValueError(f"bundle mode 无效: {mode!r}")
    paths = [directory / "base.toml"]
    if mode != "base":
        paths.append(directory / f"{mode}.toml")
    selected: dict[str, BundleRow] = {}
    for path in paths:
        selected.update((row.id, row) for row in load_bundle(path))
    return tuple(selected.values())


def distribution_bundle(directory: Path) -> tuple[BundleRow, ...]:
    """读取进程显式选择的 mode，不修改运行选择。"""
    return load_bundles(directory, mode=os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base"))


# 发行版 preset 自己声明的插件；用户 patch 新增的行不属于 preset。
def preset_plugin_ids(distribution: Path) -> frozenset[str]:
    return frozenset(row.plugin for row in distribution_bundle(distribution / "bundles"))


def patch_rows(workspace: Path) -> tuple[BundleRow, ...]:
    path = workspace / "bundle.patch.toml"
    return load_bundle(path) if path.exists() else ()


def composition_rows(workspace: Path, distribution: Path | None = None) -> tuple[BundleRow, ...]:
    """发行声明与用户 patch 生成输入；安装 cache 不决定启用。"""
    configured = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION")
    if distribution is None and configured:
        distribution = Path(configured)
    if distribution is None:
        return patch_rows(workspace)
    return load_bundles(distribution / "bundles", mode=os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base"),
                        patch=workspace / "bundle.patch.toml")


def plugin_choices(workspace: Path, distribution: Path | None = None) -> dict[str, bool]:
    return {row.plugin: not row.disabled for row in composition_rows(workspace, distribution)}


# 只改启停，不抄 config：没写的字段继续跟随 preset。改完和 preset 相同时删掉这一行，回到跟随 preset；
# None 撤销用户对这个插件的整行选择。
def set_plugin_choice(workspace: Path, plugin_id: str, *, enabled: bool | None,
                      distribution: Path | None = None) -> Path:
    if _PLUGIN.fullmatch(plugin_id) is None:
        raise ValueError(f"插件身份无效: {plugin_id}")
    path = workspace / "bundle.patch.toml"
    current = next((row for row in patch_rows(workspace) if row.plugin == plugin_id), None)
    document = tomlkit.parse(path.read_text()) if path.exists() else tomlkit.parse("schema_version = 1\n[rows]\n")
    table = cast(dict[str, dict[str, object]], document["rows"])
    preset = {row.id: row for row in _distribution_preset(distribution)}
    if enabled is None:
        if current is not None:
            del table[current.id]
    else:
        base = next((row for row in preset.values() if row.plugin == plugin_id), None)
        identity = current.id if current is not None else base.id if base is not None else plugin_id
        if identity not in table:
            table[identity] = {"plugin": plugin_id}
        table[identity]["disabled"] = not enabled
        if base is not None and base.plugin == plugin_id and set(table[identity]) == {"plugin", "disabled"} \
                and base.disabled == (not enabled):
            del table[identity]
    atomic_write_text(path, tomlkit.dumps(document), domain="plugin_bundle")
    return path


# 当前进程（或指定）发行版的 preset 行；没有发行版时没有 preset。
def _distribution_preset(distribution: Path | None) -> tuple[BundleRow, ...]:
    configured = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION")
    if distribution is None and configured:
        distribution = Path(configured)
    return () if distribution is None else distribution_bundle(distribution / "bundles")


@dataclass(frozen=True)
class PresetDifference:
    plugin: str
    kind: Literal["changed", "added", "same"]
    fields: tuple[str, ...]
    preset: BundleRow | None
    user: BundleRow


# 用户 patch 相对 preset 的差异（同 DSH 的 --dump-config 对照 --dump-default-config）：
# changed 是覆盖了 preset row 的字段；added 是 preset 没有的 row；same 是和 preset 相同、可以删掉的 row。
# preset_root 是含 bundles/ 的目录：镜像内发行版或源码检出。
def preset_differences(workspace: Path, preset_root: Path) -> tuple[PresetDifference, ...]:
    path = workspace / "bundle.patch.toml"
    if not path.exists():
        return ()
    preset = {row.id: row for row in distribution_bundle(preset_root / "bundles")}
    differences: list[PresetDifference] = []
    for row, fields in _read_rows(path):
        base = preset_row_for(preset, row)
        if base is None:
            differences.append(PresetDifference(row.plugin, "added", tuple(sorted(fields - {"plugin"})), None, row))
            continue
        changed = tuple(name for name, mine, theirs in (
            ("plugin", row.plugin, base.plugin),
            ("disabled", row.disabled, base.disabled),
            ("config", row.config, base.config),
        ) if (name == "plugin" or name in fields) and mine != theirs)
        differences.append(PresetDifference(row.plugin, "changed" if changed else "same", changed, base, row))
    return tuple(differences)
