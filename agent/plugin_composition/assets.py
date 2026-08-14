from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class PluginAssetContribution:
    skill_roots: tuple[Path, ...] = ()
    drift_skill_roots: tuple[Path, ...] = ()
    dashboard_module: Path | None = None


@dataclass(frozen=True, slots=True)
class _AssetRegistration:
    token: int
    plugin_id: str
    kind: Literal["skill", "drift_skill", "dashboard"]
    path: Path


PLUGIN_ASSETS = ServiceKey["PluginAssets"]("core.plugin_assets")


class PluginAssets:
    """Collect plugin-owned Skill and Dashboard declarations for one Root."""

    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _AssetRegistration] = {}
        self._frozen: Mapping[str, PluginAssetContribution] | None = None

    async def register_skill(
        self,
        ctx: Context,
        relative_path: str,
        *,
        drift: bool = False,
    ) -> None:
        """Register one plugin-owned Skill root as a Fiber Effect."""

        runtime = ctx.runtime
        path = _resolve_asset_path(runtime, relative_path, kind="skill")
        kind: Literal["skill", "drift_skill"] = "drift_skill" if drift else "skill"
        await ctx.effect(
            lambda: self._register(runtime.plugin_id, kind, path),
            label=f"asset:{kind}:{relative_path}",
        )

    async def register_dashboard(
        self,
        ctx: Context,
        relative_path: str,
    ) -> None:
        """Register one plugin-owned Dashboard module as a Fiber Effect."""

        runtime = ctx.runtime
        path = _resolve_asset_path(runtime, relative_path, kind="dashboard")
        await ctx.effect(
            lambda: self._register(runtime.plugin_id, "dashboard", path),
            label=f"asset:dashboard:{relative_path}",
        )

    def freeze(self) -> Mapping[str, PluginAssetContribution]:
        """Freeze the declarations that Core may compile into a snapshot."""

        if self._frozen is not None:
            return self._frozen

        # 1. Preserve registration order within each plugin contribution.
        skills: dict[str, list[Path]] = {}
        drift_skills: dict[str, list[Path]] = {}
        dashboards: dict[str, Path] = {}
        for registration in sorted(
            self._registrations.values(),
            key=lambda item: item.token,
        ):
            if registration.kind == "skill":
                skills.setdefault(registration.plugin_id, []).append(registration.path)
            elif registration.kind == "drift_skill":
                drift_skills.setdefault(registration.plugin_id, []).append(
                    registration.path
                )
            else:
                dashboards[registration.plugin_id] = registration.path

        # 2. Publish an immutable value; Effect cleanup remains Root-owned.
        plugin_ids = {*skills, *drift_skills, *dashboards}
        self._frozen = MappingProxyType(
            {
                plugin_id: PluginAssetContribution(
                    skill_roots=tuple(skills.get(plugin_id, ())),
                    drift_skill_roots=tuple(drift_skills.get(plugin_id, ())),
                    dashboard_module=dashboards.get(plugin_id),
                )
                for plugin_id in sorted(plugin_ids)
            }
        )
        return self._frozen

    def _register(
        self,
        plugin_id: str,
        kind: Literal["skill", "drift_skill", "dashboard"],
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
        if any(
            item.plugin_id == plugin_id and item.kind == kind and item.path == path
            for item in existing
        ):
            raise CompositionError(
                "DUPLICATE_PLUGIN_ASSET",
                f"插件资产重复: {plugin_id} {kind} {path}",
            )
        if kind == "dashboard" and any(
            item.plugin_id == plugin_id and item.kind == "dashboard"
            for item in existing
        ):
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
            kind=kind,
            path=path,
        )

        def cleanup() -> None:
            _ = self._registrations.pop(token, None)

        return cleanup


def _resolve_asset_path(
    runtime: PluginRuntime,
    relative_path: str,
    *,
    kind: Literal["skill", "dashboard"],
) -> Path:
    """Resolve one declared asset without allowing plugin-root escape."""

    # 1. Resolve symlinks before the containment check.
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or relative_path != relative_path.strip()
        or Path(relative_path).is_absolute()
    ):
        raise ValueError("插件资产路径必须是非空相对路径")
    plugin_root = runtime.plugin_dir.resolve(strict=True)
    try:
        path = (plugin_root / relative_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"插件资产不存在: {relative_path}") from error
    if not path.is_relative_to(plugin_root):
        raise RuntimeError(f"插件资产路径越界: {relative_path}")

    # 2. Validate the concrete host contract at the registration boundary.
    if kind == "skill" and not path.is_dir():
        raise RuntimeError(f"插件 Skill root 不是目录: {relative_path}")
    if kind == "dashboard" and (path.suffix != ".py" or not path.is_file()):
        raise RuntimeError(f"插件 Dashboard module 无效: {relative_path}")
    return path
