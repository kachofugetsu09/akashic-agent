from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from infra.persistence.json_store import atomic_save_json, load_json

ArtifactSelector = Literal["stable", "latest"]
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class ArtifactPointer:
    path: str | None


@dataclass(frozen=True)
class ArtifactPointers:
    stable: ArtifactPointer
    latest: ArtifactPointer


def pointer_state_path(plugin_base: Path) -> Path:
    return plugin_base / ".pointers.json"


def read_pointers(plugin_base: Path) -> ArtifactPointers | None:
    """从一个原子状态文件读取完整 stable/latest 指针对。"""

    path = pointer_state_path(plugin_base)
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"插件 artifact pointer state 必须是普通文件: {path}")
    raw = load_json(path, default=None, domain=f"plugin_artifact_pointers:{path}")
    if not isinstance(raw, dict):
        raise ValueError(f"插件 artifact pointer state 结构无效: {path}")
    values = cast(dict[str, object], raw)
    if set(values) != {"stable", "latest"}:
        raise ValueError(f"插件 artifact pointer state 结构无效: {path}")
    stable_value = values["stable"]
    latest_value = values["latest"]
    if stable_value is not None and not isinstance(stable_value, str):
        raise ValueError(f"插件 stable artifact pointer 无效: {path}")
    if latest_value is not None and not isinstance(latest_value, str):
        raise ValueError(f"插件 latest artifact pointer 无效: {path}")
    pointers = ArtifactPointers(
        stable=ArtifactPointer(stable_value),
        latest=ArtifactPointer(latest_value),
    )
    _validate_pointers(plugin_base, pointers)
    return pointers


def read_pointer(
    plugin_base: Path,
    selector: ArtifactSelector,
) -> ArtifactPointer | None:
    """从原子指针对中读取一个 selector。"""

    pointers = read_pointers(plugin_base)
    return None if pointers is None else getattr(pointers, selector)


def write_pointers(
    plugin_base: Path,
    *,
    stable: ArtifactPointer,
    latest: ArtifactPointer,
) -> Path:
    """原子持久化完整 stable/latest 指针对。"""

    pointers = ArtifactPointers(stable=stable, latest=latest)
    _validate_pointers(plugin_base, pointers)
    path = pointer_state_path(plugin_base)
    atomic_save_json(
        path,
        {"stable": stable.path, "latest": latest.path},
        ensure_ascii=False,
        domain=f"plugin_artifact_pointers:{path}",
    )
    return path


def resolve_pointer(plugin_base: Path, pointer: ArtifactPointer) -> Path | None:
    """把指针解析为 plugin_base 内的不可变插件根目录。"""

    if pointer.path is None:
        return None
    relative = PurePosixPath(pointer.path)
    parts = relative.parts
    valid_legacy = len(parts) == 1 and _safe_segment(parts[0])
    valid_artifact = (
        len(parts) == 2 and parts[0] == ".artifacts" and _safe_segment(parts[1])
    )
    if relative.is_absolute() or not (valid_legacy or valid_artifact):
        raise ValueError(f"插件 artifact pointer 越界: {pointer.path}")
    target = plugin_base.joinpath(*parts)
    current = plugin_base
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"插件 artifact pointer 不能经过符号链接: {current}")
    if not target.is_dir():
        raise FileNotFoundError(f"插件 artifact pointer 目标不存在: {target}")
    plugin_file = target / "plugin.py"
    if plugin_file.is_symlink() or not plugin_file.is_file():
        raise ValueError(f"插件 artifact 缺少普通 plugin.py: {plugin_file}")
    return target


def relative_artifact_pointer(
    plugin_base: Path, artifact_root: Path
) -> ArtifactPointer:
    """为不可变 artifact 目录的直接子目录构造安全指针。"""

    relative = artifact_root.relative_to(plugin_base).as_posix()
    pointer = ArtifactPointer(relative)
    _ = resolve_pointer(plugin_base, pointer)
    return pointer


def promote_latest_pointer(plugin_base: Path) -> ArtifactPointer:
    pointers = read_pointers(plugin_base)
    if pointers is None or pointers.latest.path is None:
        raise RuntimeError(f"插件没有可 promote 的 latest artifact: {plugin_base}")
    _ = write_pointers(
        plugin_base,
        stable=pointers.latest,
        latest=pointers.latest,
    )
    return pointers.latest


def discard_latest_pointer(plugin_base: Path) -> ArtifactPointer:
    pointers = read_pointers(plugin_base)
    if pointers is None:
        raise RuntimeError(f"插件缺少 stable artifact pointer: {plugin_base}")
    if pointers.stable.path is None:
        _ = write_pointers(
            plugin_base,
            stable=pointers.stable,
            latest=pointers.stable,
        )
        return pointers.stable
    _ = write_pointers(
        plugin_base,
        stable=pointers.stable,
        latest=pointers.stable,
    )
    return pointers.stable


def _validate_pointers(plugin_base: Path, pointers: ArtifactPointers) -> None:
    if pointers.latest.path is None and pointers.stable.path is not None:
        raise ValueError(f"插件 latest 为空时 stable 也必须为空: {plugin_base}")
    _ = resolve_pointer(plugin_base, pointers.stable)
    _ = resolve_pointer(plugin_base, pointers.latest)


def _safe_segment(value: str) -> bool:
    return _SAFE_SEGMENT.fullmatch(value) is not None
