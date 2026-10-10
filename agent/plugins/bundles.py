"""声明式组合只生成选择输入，不发现或自动补齐 provider。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import os
import re
import tomllib
from typing import cast

from core.common.frozen_json import freeze_json

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PLUGIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:@[A-Za-z0-9][A-Za-z0-9._-]*)?")


@dataclass(frozen=True)
class BundleRow:
    id: str
    plugin: str
    config: Mapping[str, object]
    disabled: bool = False


def load_bundle(path: Path) -> tuple[BundleRow, ...]:
    """在文件边界校验完整 row；缺席的 config 和 disabled 使用格式默认值。"""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(document) != {"schema_version", "rows"} or type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError(f"bundle schema 无效: {path}")
    raw_rows = document["rows"]
    if not isinstance(raw_rows, dict):
        raise ValueError(f"bundle rows 必须是表: {path}")
    rows = []
    for identity, value in raw_rows.items():
        if _NAME.fullmatch(identity) is None or not isinstance(value, dict):
            raise ValueError(f"bundle row 无效: {path}:{identity}")
        if set(value) - {"plugin", "config", "disabled"} or "plugin" not in value:
            raise ValueError(f"bundle row 字段无效: {path}:{identity}")
        plugin, config, disabled = value["plugin"], value.get("config", {}), value.get("disabled", False)
        if not isinstance(plugin, str) or _PLUGIN.fullmatch(plugin) is None:
            raise ValueError(f"bundle plugin 身份无效: {path}:{identity}")
        if not isinstance(config, dict) or not isinstance(disabled, bool):
            raise ValueError(f"bundle config/disabled 无效: {path}:{identity}")
        rows.append(BundleRow(identity, plugin, cast(Mapping[str, object], freeze_json(config)), disabled))
    return tuple(rows)


def load_bundles(directory: Path, *, mode: str = "base", patch: Path | None = None) -> tuple[BundleRow, ...]:
    """按 base、mode、用户 patch 顺序整行替换；不深合并 config。"""
    if _NAME.fullmatch(mode) is None:
        raise ValueError(f"bundle mode 无效: {mode!r}")
    paths = [directory / "base.toml"]
    if mode != "base":
        paths.append(directory / f"{mode}.toml")
    if patch is not None and patch.exists():
        paths.append(patch)
    selected: dict[str, BundleRow] = {}
    for path in paths:
        selected.update((row.id, row) for row in load_bundle(path))
    plugins: set[str] = set()
    for row in selected.values():
        if row.plugin in plugins:
            raise ValueError(f"多个 bundle row 指向同一插件: {row.plugin}")
        plugins.add(row.plugin)
    return tuple(selected.values())


def distribution_bundle(directory: Path) -> tuple[BundleRow, ...]:
    """读取进程显式选择的 mode，不修改运行选择。"""
    return load_bundles(directory, mode=os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base"))
