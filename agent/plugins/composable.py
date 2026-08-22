from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import ModuleType
from typing import TYPE_CHECKING, cast

from agent.plugin_composition import Context, ServiceKey, ServiceView

if TYPE_CHECKING:
    from agent.plugins.generation import PluginSemanticCheck

_CORE_RESERVED_WORKSPACE_ROOTS = frozenset({"plugin-data", "runtime"})


@dataclass
class ComposablePlugin:
    """Adapt one v3 namespace module to the composition kernel's apply contract."""

    module: ModuleType
    name: str
    version: str
    desc: str
    author: str
    inject: tuple[ServiceKey[object], ...]
    skill_roots: tuple[str, ...]
    drift_skill_roots: tuple[str, ...]
    workspace_roots: tuple[str, ...]
    workspace_files: tuple[str, ...]
    dashboard_module: str | None
    _apply: Callable[[Context, object], object] = field(repr=False)
    _service_view: ServiceView | None = field(default=None, init=False, repr=False)
    _static_active: bool | None = field(default=None, init=False, repr=False)
    api_version: int = field(default=3, init=False)

    @classmethod
    def from_module(cls, module: ModuleType) -> ComposablePlugin:
        """Validate and freeze the named exports of one v3 plugin module."""

        # 1. Validate the namespace shape before Manager state is created.
        if getattr(module, "api_version", None) != 3:
            raise ValueError("v3 插件模块必须声明 api_version = 3")
        name = getattr(module, "name", None)
        version = getattr(module, "version", None)
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("v3 插件 name 必须是非空且无首尾空白的字符串")
        if (
            not isinstance(version, str)
            or not version.strip()
            or version != version.strip()
        ):
            raise ValueError("v3 插件 version 必须是非空且无首尾空白的字符串")
        apply = getattr(module, "apply", None)
        if not callable(apply):
            raise ValueError("v3 插件模块必须导出 apply(ctx, config)")
        _validate_apply_signature(apply)

        # 2. Dependencies are typed ServiceKeys; ordering comes from providers.
        raw_inject = cast(object, getattr(module, "inject", ()))
        if not isinstance(raw_inject, (tuple, list)):
            raise ValueError("v3 插件 inject 必须是 ServiceKey 序列")
        raw_items = cast(tuple[object, ...] | list[object], raw_inject)
        if not all(isinstance(item, ServiceKey) for item in raw_items):
            raise ValueError("v3 插件 inject 必须是 ServiceKey 序列")
        inject = tuple(
            cast(ServiceKey[object], item)
            for item in raw_items
            if isinstance(item, ServiceKey)
        )
        if len(set(inject)) != len(inject):
            raise ValueError(f"v3 插件依赖重复: {name}")
        static_checks = getattr(module, "static_semantic_checks", None)
        if static_checks is not None and not callable(static_checks):
            raise ValueError("v3 插件 static_semantic_checks 必须可调用")
        active = getattr(module, "is_active", None)
        if active is not None and not callable(active):
            raise ValueError("v3 插件 is_active 必须是可调用对象")
        skill_roots = _string_tuple_export(module, "skill_roots")
        drift_skill_roots = _string_tuple_export(module, "drift_skill_roots")
        workspace_roots = _workspace_roots_export(module)
        workspace_files = _workspace_files_export(module)
        dashboard_module = getattr(module, "dashboard_module", None)
        if dashboard_module is not None and (
            not isinstance(dashboard_module, str)
            or not dashboard_module.strip()
            or dashboard_module != dashboard_module.strip()
        ):
            raise ValueError("v3 插件 dashboard_module 必须是非空字符串或 None")
        return cls(
            module=module,
            name=name,
            version=version,
            desc=str(getattr(module, "desc", "")),
            author=str(getattr(module, "author", "")),
            inject=inject,
            skill_roots=skill_roots,
            drift_skill_roots=drift_skill_roots,
            workspace_roots=workspace_roots,
            workspace_files=workspace_files,
            dashboard_module=cast(str | None, dashboard_module),
            _apply=cast(Callable[[Context, object], object], apply),
        )

    @property
    def ConfigModel(self) -> type[object] | None:
        return cast(type[object] | None, getattr(self.module, "Config", None))

    async def apply(self, ctx: Context) -> None:
        active = self.is_active()
        ctx._set_static_active(active)  # pyright: ignore[reportPrivateUsage]
        if not active:
            return
        result = self._apply(ctx, ctx.runtime.config)
        if inspect.isawaitable(result):
            await result

    def bind_static_services(self, services: ServiceView) -> None:
        """使用冻结的 Core services 计算静态贡献准入。"""

        if self._service_view is not None:
            raise RuntimeError("v3 插件 static services 不能重复绑定")
        self._service_view = services
        provider = getattr(self.module, "is_active", None)
        if provider is None:
            self._static_active = True
            return
        result = provider(services)
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                _ = close()
            raise RuntimeError("v3 插件 is_active 不支持 async")
        if not isinstance(result, bool):
            raise RuntimeError("v3 插件 is_active 必须返回 bool")
        self._static_active = result

    def is_active(self) -> bool:
        """返回插件自己决定的静态 contribution 发布状态。"""

        if getattr(self.module, "is_active", None) is None:
            return True
        if self._static_active is None:
            raise RuntimeError("v3 插件 is_active 尚未绑定 Core static services")
        return self._static_active

    @property
    def static_active(self) -> bool:
        return self.is_active()

    def static_semantic_checks(self) -> list[PluginSemanticCheck]:
        provider = getattr(self.module, "static_semantic_checks", None)
        if provider is None:
            return []
        return cast(list[PluginSemanticCheck], provider())


def _validate_apply_signature(apply: Callable[..., object]) -> None:
    """Reject v3 apply callables that Core cannot invoke as apply(ctx, config)."""

    try:
        signature = inspect.signature(apply)
    except (TypeError, ValueError) as error:
        raise ValueError("v3 插件 apply 必须精确声明 apply(ctx, config)") from error
    parameters = tuple(signature.parameters.values())
    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    if (
        tuple(parameter.name for parameter in parameters) != ("ctx", "config")
        or any(parameter.kind not in positional_kinds for parameter in parameters)
        or any(
            parameter.default is not inspect.Parameter.empty for parameter in parameters
        )
    ):
        raise ValueError("v3 插件 apply 必须精确声明 apply(ctx, config)")


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
            or len(path.parts) != 1
            or path.name in {".", ".."}
            or "/" in name
            or "\\" in name
            or name in _CORE_RESERVED_WORKSPACE_ROOTS
        ):
            raise ValueError("v3 插件 workspace_files 必须是顶层文件名")
    return files
