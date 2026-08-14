from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from agent.plugin_composition.context import Context
from agent.plugin_composition.model import (
    CompositionError,
    PluginRuntime,
    ServiceKey,
)


@dataclass(frozen=True, slots=True)
class PluginAssetContribution:
    dashboard_module: Path | None = None


@dataclass(frozen=True, slots=True)
class _AssetRegistration:
    token: int
    plugin_id: str
    path: Path


PLUGIN_ASSETS = ServiceKey["PluginAssets"]("core.plugin_assets")


class PluginAssets:
    """Collect the transitional Dashboard declaration for one Root."""

    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _AssetRegistration] = {}
        self._frozen: Mapping[str, PluginAssetContribution] | None = None

    async def register_dashboard(
        self,
        ctx: Context,
        relative_path: str,
    ) -> None:
        """Register one plugin-owned Dashboard module as a Fiber Effect."""

        runtime = ctx.runtime
        path = _resolve_dashboard_path(runtime, relative_path)
        await ctx.effect(
            lambda: self._register(runtime.plugin_id, path),
            label=f"asset:dashboard:{relative_path}",
        )

    def freeze(self) -> Mapping[str, PluginAssetContribution]:
        """Freeze the declarations that Core may compile into a snapshot."""

        if self._frozen is not None:
            return self._frozen

        # 1. Preserve one Dashboard declaration per plugin.
        dashboards: dict[str, Path] = {}
        for registration in sorted(
            self._registrations.values(),
            key=lambda item: item.token,
        ):
            dashboards[registration.plugin_id] = registration.path

        # 2. Publish an immutable value; Effect cleanup remains Root-owned.
        self._frozen = MappingProxyType(
            {
                plugin_id: PluginAssetContribution(
                    dashboard_module=dashboard_module,
                )
                for plugin_id, dashboard_module in sorted(dashboards.items())
            }
        )
        return self._frozen

    def _register(
        self,
        plugin_id: str,
        path: Path,
    ) -> Callable[[], None]:
        """Add one declaration and return its exact inverse."""

        # 1. Declarations close before snapshot compilation.
        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_ASSETS_FROZEN",
                "插件资产声明已冻结，不能在 snapshot 发布后新增",
            )
        existing = tuple(self._registrations.values())
        if any(item.plugin_id == plugin_id and item.path == path for item in existing):
            raise CompositionError(
                "DUPLICATE_PLUGIN_ASSET",
                f"插件 Dashboard 重复: {plugin_id} {path}",
            )
        if any(item.plugin_id == plugin_id for item in existing):
            raise CompositionError(
                "DUPLICATE_PLUGIN_DASHBOARD",
                f"插件只能声明一个 Dashboard module: {plugin_id}",
            )

        # 2. The returned cleanup owns only this registration token.
        token = self._next_token
        self._next_token += 1
        self._registrations[token] = _AssetRegistration(
            token=token,
            plugin_id=plugin_id,
            path=path,
        )

        def cleanup() -> None:
            _ = self._registrations.pop(token, None)

        return cleanup


def _resolve_dashboard_path(runtime: PluginRuntime, relative_path: str) -> Path:
    """Resolve one Dashboard module without allowing plugin-root escape."""

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
