from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import ModuleType
from typing import Any, cast

from agent.plugin_composition import Context, ServiceKey
from agent.plugins.static_manifest import StaticPluginManifest

_CORE_RESERVED_WORKSPACE_ROOTS = frozenset({"plugin-data", "runtime"})


@dataclass
class ComposablePlugin:
    """Adapt one v3 namespace module to the composition kernel's apply contract."""

    module: ModuleType
    name: str
    version: str
    desc: str
    author: str
    inject: tuple[ServiceKey[Any], ...]
    workspace_roots: tuple[str, ...]
    workspace_files: tuple[str, ...]
    _apply: Callable[[Context], object] = field(repr=False)
    api_version: int

    @classmethod
    def from_module(cls, module: ModuleType, identity: StaticPluginManifest) -> ComposablePlugin:
        """Validate and freeze the named exports of one v3 plugin module."""

        # 1. 身份由导入前的 loader 拥有，模块只提供实际能力。
        name = identity.name
        version = identity.version
        apply = getattr(module, "apply", None)
        if not callable(apply):
            raise ValueError("插件模块必须导出 apply(ctx)")

        # 2. Dependencies are typed ServiceKeys; ordering comes from providers.
        raw_inject = cast(object, getattr(module, "inject", ()))
        if not isinstance(raw_inject, (tuple, list)):
            raise ValueError("v3 插件 inject 必须是 ServiceKey 序列")
        raw_items = cast(tuple[object, ...] | list[object], raw_inject)
        if not all(isinstance(item, ServiceKey) for item in raw_items):
            raise ValueError("v3 插件 inject 必须是 ServiceKey 序列")
        inject = tuple(
            cast(ServiceKey[Any], item)
            for item in raw_items
            if isinstance(item, ServiceKey)
        )
        if len(set(inject)) != len(inject):
            raise ValueError(f"v3 插件依赖重复: {name}")
        workspace_roots = _workspace_roots_export(module)
        workspace_files = _workspace_files_export(module)
        return cls(
            module=module,
            name=name,
            version=version,
            api_version=identity.api_version,
            desc=str(getattr(module, "desc", "")),
            author=str(getattr(module, "author", "")),
            inject=inject,
            workspace_roots=workspace_roots,
            workspace_files=workspace_files,
            _apply=cast(Callable[[Context], object], apply),
        )

    async def apply(self, ctx: Context) -> None:
        result = self._apply(ctx)
        if inspect.isawaitable(result):
            await result

def _string_tuple_export(module: ModuleType, name: str) -> tuple[str, ...]:
    raw = cast(object, getattr(module, name, ()))
    if not isinstance(raw, (tuple, list)):
        raise ValueError(f"v3 插件 {name} 必须是字符串序列")
    items = cast(tuple[object, ...] | list[object], raw)
    if any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in items
    ):
        raise ValueError(f"v3 插件 {name} 必须只包含非空字符串")
    typed = tuple(cast(str, item) for item in items)
    if len(set(typed)) != len(typed):
        raise ValueError(f"v3 插件 {name} 不得重复")
    return typed


def _workspace_roots_export(module: ModuleType) -> tuple[str, ...]:
    roots = _string_tuple_export(module, "workspace_roots")
    for root in roots:
        if root in _CORE_RESERVED_WORKSPACE_ROOTS:
            raise ValueError(f"v3 插件 workspace_roots 不得声明 Core 保留目录 {root}")
        path = PurePosixPath(root)
        if (
            path.is_absolute()
            or len(path.parts) != 1
            or path.name in {".", ".."}
            or "/" in root
            or "\\" in root
        ):
            raise ValueError("v3 插件 workspace_roots 必须是顶层目录名")
    return roots


def _workspace_files_export(module: ModuleType) -> tuple[str, ...]:
    files = _string_tuple_export(module, "workspace_files")
    for name in files:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {".", ".."} for part in path.parts)
            or "\\" in name
            or path.parts[0] in _CORE_RESERVED_WORKSPACE_ROOTS
        ):
            raise ValueError("v3 插件 workspace_files 必须是 workspace 内相对文件路径")
    return files
