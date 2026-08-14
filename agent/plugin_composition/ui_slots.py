from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import (
    CompositionError,
    PluginRuntime,
    ServiceKey,
)
from agent.plugins.mobile_ui_assets import (
    MobileUiAsset,
    MobileUiQueryHandler,
    resolve_mobile_ui_asset,
)


MobileUiSlot = Literal[
    "turn.before_reasoning",
    "turn.before_tool",
    "turn.after_answer",
    "drawer.panel",
]


@dataclass(frozen=True, slots=True)
class MobileUiNavigation:
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class MobileUiDefinition:
    module: str
    stylesheet: str | None = None
    navigation: MobileUiNavigation | None = None
    slots: tuple[MobileUiSlot, ...] = ()


@dataclass(frozen=True, slots=True)
class PluginUiSlotContribution:
    dashboard_module: Path | None = None
    mobile_ui_asset: MobileUiAsset | None = None
    mobile_ui_query: MobileUiQueryHandler | None = None
    mobile_ui_available: Callable[[], bool] | None = None


@dataclass(frozen=True, slots=True)
class _DashboardRegistration:
    token: int
    plugin_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class _MobileUiRegistration:
    token: int
    plugin_id: str
    asset: MobileUiAsset
    query: MobileUiQueryHandler
    available: Callable[[], bool]


UI_SLOTS = ServiceKey["PluginUiSlots"]("core.ui_slots")


class PluginUiSlots:
    """Collect plugin-owned UI surface declarations for one composition Root."""

    def __init__(self) -> None:
        self._next_token = 1
        self._dashboard_registrations: dict[int, _DashboardRegistration] = {}
        self._mobile_registrations: dict[int, _MobileUiRegistration] = {}
        self._frozen: Mapping[str, PluginUiSlotContribution] | None = None

    async def register_dashboard(
        self,
        ctx: Context,
        relative_path: str,
    ) -> None:
        """Register one plugin-owned Dashboard module as a Fiber Effect."""

        runtime = ctx.runtime
        path = _resolve_dashboard_path(runtime, relative_path)
        _ = await ctx.effect(
            lambda: self._register_dashboard(runtime.plugin_id, path),
            label=f"ui-slot:dashboard:{relative_path}",
        )

    async def register_mobile(
        self,
        ctx: Context,
        definition: MobileUiDefinition,
        *,
        query: MobileUiQueryHandler,
        available: Callable[[], bool] | None = None,
    ) -> None:
        """把一组插件自有的 Mobile UI 声明登记为 Fiber Effect。"""

        if not isinstance(definition, MobileUiDefinition):
            raise TypeError("插件 Mobile UI 声明必须是 MobileUiDefinition")
        if not callable(query):
            raise TypeError("插件 Mobile UI query 必须可调用")
        if _is_async_callable(query):
            raise TypeError("插件 Mobile UI query 必须是同步函数")
        if available is not None and not callable(available):
            raise TypeError("插件 Mobile UI available 必须可调用")
        if available is not None and _is_async_callable(available):
            raise TypeError("插件 Mobile UI available 必须是同步函数")
        runtime = ctx.runtime
        navigation = definition.navigation
        if navigation is not None and not isinstance(
            navigation,
            MobileUiNavigation,
        ):
            raise TypeError("插件 Mobile UI navigation 必须是 MobileUiNavigation")
        asset = resolve_mobile_ui_asset(
            runtime.plugin_dir,
            module=definition.module,
            stylesheet=definition.stylesheet,
            navigation_label=None if navigation is None else navigation.label,
            navigation_description=(
                None if navigation is None else navigation.description
            ),
            slots=tuple(definition.slots),
        )
        _ = await ctx.effect(
            lambda: self._register_mobile(
                runtime.plugin_id,
                asset,
                query,
                available if available is not None else _always_available,
            ),
            label=f"ui-slot:mobile:{definition.module}",
        )

    def freeze(self) -> Mapping[str, PluginUiSlotContribution]:
        """Freeze the UI declarations that Core may compile into a snapshot."""

        if self._frozen is not None:
            return self._frozen

        # 1. Preserve one declaration of each UI surface per plugin.
        dashboards: dict[str, Path] = {}
        for registration in sorted(
            self._dashboard_registrations.values(),
            key=lambda item: item.token,
        ):
            dashboards[registration.plugin_id] = registration.path
        mobile = {
            registration.plugin_id: registration
            for registration in sorted(
                self._mobile_registrations.values(),
                key=lambda item: item.token,
            )
        }

        # 2. Publish an immutable value; Effect cleanup remains Root-owned.
        self._frozen = MappingProxyType(
            {
                plugin_id: PluginUiSlotContribution(
                    dashboard_module=dashboards.get(plugin_id),
                    mobile_ui_asset=(
                        None if plugin_id not in mobile else mobile[plugin_id].asset
                    ),
                    mobile_ui_query=(
                        None if plugin_id not in mobile else mobile[plugin_id].query
                    ),
                    mobile_ui_available=(
                        None
                        if plugin_id not in mobile
                        else mobile[plugin_id].available
                    ),
                )
                for plugin_id in sorted(set(dashboards) | set(mobile))
            }
        )
        return self._frozen

    def _register_dashboard(
        self,
        plugin_id: str,
        path: Path,
    ) -> Callable[[], None]:
        """Add one declaration and return its exact inverse."""

        # 1. Registration closes before snapshot compilation.
        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_UI_SLOTS_FROZEN",
                "插件 UI Slot 声明已冻结，不能在 snapshot 发布后新增",
            )
        existing = tuple(self._dashboard_registrations.values())
        if any(item.plugin_id == plugin_id and item.path == path for item in existing):
            raise CompositionError(
                "DUPLICATE_PLUGIN_UI_SLOT",
                f"插件 Dashboard slot 重复: {plugin_id} {path}",
            )
        if any(item.plugin_id == plugin_id for item in existing):
            raise CompositionError(
                "DUPLICATE_PLUGIN_DASHBOARD",
                f"插件只能声明一个 Dashboard module: {plugin_id}",
            )

        # 2. The disposer owns only this registration token.
        token = self._next_token
        self._next_token += 1
        self._dashboard_registrations[token] = _DashboardRegistration(
            token=token,
            plugin_id=plugin_id,
            path=path,
        )

        def cleanup() -> None:
            _ = self._dashboard_registrations.pop(token, None)

        return cleanup

    def _register_mobile(
        self,
        plugin_id: str,
        asset: MobileUiAsset,
        query: MobileUiQueryHandler,
        available: Callable[[], bool],
    ) -> Callable[[], None]:
        """登记一组 Mobile UI 声明并返回精确逆操作。"""

        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_UI_SLOTS_FROZEN",
                "插件 UI Slot 声明已冻结，不能在 snapshot 发布后新增",
            )
        if any(
            item.plugin_id == plugin_id
            for item in self._mobile_registrations.values()
        ):
            raise CompositionError(
                "DUPLICATE_PLUGIN_MOBILE_UI",
                f"插件只能声明一个 Mobile UI: {plugin_id}",
            )

        token = self._next_token
        self._next_token += 1
        self._mobile_registrations[token] = _MobileUiRegistration(
            token=token,
            plugin_id=plugin_id,
            asset=asset,
            query=query,
            available=available,
        )

        def cleanup() -> None:
            _ = self._mobile_registrations.pop(token, None)

        return cleanup


def _resolve_dashboard_path(runtime: PluginRuntime, relative_path: str) -> Path:
    """Resolve one Dashboard module without allowing plugin-source escape."""

    # 1. Resolve symlinks before the containment check.
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or relative_path != relative_path.strip()
        or Path(relative_path).is_absolute()
    ):
        raise ValueError("插件 Dashboard 路径必须是非空相对路径")
    plugin_root = runtime.plugin_dir.resolve(strict=True)
    try:
        path = (plugin_root / relative_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"插件 Dashboard 不存在: {relative_path}") from error
    if not path.is_relative_to(plugin_root):
        raise RuntimeError(f"插件 Dashboard 路径越界: {relative_path}")

    # 2. Validate the concrete host contract at the registration boundary.
    if path.suffix != ".py" or not path.is_file():
        raise RuntimeError(f"插件 Dashboard module 无效: {relative_path}")
    return path


def _always_available() -> bool:
    return True


def _is_async_callable(value: object) -> bool:
    return inspect.iscoroutinefunction(value) or inspect.iscoroutinefunction(
        getattr(value, "__call__", None)
    )
