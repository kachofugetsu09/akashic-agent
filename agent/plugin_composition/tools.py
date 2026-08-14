from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import CompositionError, ServiceKey
from agent.tools.base import Tool

ToolRisk = Literal["read-only", "write", "external-side-effect"]


@dataclass(frozen=True, slots=True)
class PluginToolContribution:
    tool: Tool
    risk: ToolRisk
    always_on: bool
    preloadable: bool
    requires_turn_search: bool
    search_hint: str | None


@dataclass(frozen=True, slots=True)
class _ToolRegistration:
    token: int
    plugin_id: str
    contribution: PluginToolContribution


PLUGIN_TOOLS = ServiceKey["PluginTools"]("core.tools")


class PluginTools:
    """Collect Fiber-owned Tool declarations for one candidate Root."""

    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _ToolRegistration] = {}
        self._frozen: Mapping[str, tuple[PluginToolContribution, ...]] | None = None

    async def register(
        self,
        ctx: Context,
        tool: Tool,
        *,
        risk: ToolRisk,
        always_on: bool = False,
        preloadable: bool = True,
        requires_turn_search: bool = False,
        search_hint: str | None = None,
    ) -> None:
        """Register one Tool as an Effect of the calling plugin Fiber."""

        contribution = _validate_contribution(
            tool,
            risk=risk,
            always_on=always_on,
            preloadable=preloadable,
            requires_turn_search=requires_turn_search,
            search_hint=search_hint,
        )
        _ = await ctx.effect(
            lambda: self._register(ctx.runtime.plugin_id, contribution),
            label=f"tool:{tool.name}",
        )

    def freeze(self) -> Mapping[str, tuple[PluginToolContribution, ...]]:
        """Freeze registration order into an immutable snapshot input."""

        if self._frozen is not None:
            return self._frozen
        grouped: dict[str, list[PluginToolContribution]] = {}
        for registration in sorted(
            self._registrations.values(),
            key=lambda item: item.token,
        ):
            grouped.setdefault(registration.plugin_id, []).append(
                registration.contribution
            )
        self._frozen = MappingProxyType(
            {
                plugin_id: tuple(contributions)
                for plugin_id, contributions in sorted(grouped.items())
            }
        )
        return self._frozen

    def _register(
        self,
        plugin_id: str,
        contribution: PluginToolContribution,
    ) -> Callable[[], None]:
        """Add one candidate Tool and return its exact inverse."""

        # 1. Candidate declarations close before snapshot compilation.
        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_TOOLS_FROZEN",
                "插件 Tool 声明已冻结，不能在 snapshot 发布后新增",
            )
        name = contribution.tool.name
        existing = tuple(self._registrations.values())
        if any(item.contribution.tool.name == name for item in existing):
            raise CompositionError(
                "DUPLICATE_PLUGIN_TOOL",
                f"插件 Tool 名称重复: {name}",
            )

        # 2. Cleanup removes only the registration owned by this Effect.
        token = self._next_token
        self._next_token += 1
        self._registrations[token] = _ToolRegistration(
            token=token,
            plugin_id=plugin_id,
            contribution=contribution,
        )

        def cleanup() -> None:
            _ = self._registrations.pop(token, None)

        return cleanup


def _validate_contribution(
    tool: Tool,
    *,
    risk: ToolRisk,
    always_on: bool,
    preloadable: bool,
    requires_turn_search: bool,
    search_hint: str | None,
) -> PluginToolContribution:
    """Validate the public plugin boundary once before an Effect is created."""

    # 1. ToolRegistry owns schema validation; this boundary owns declaration shape.
    if not isinstance(tool, Tool):
        raise TypeError("PluginTools.register 只接受 Tool 实例")
    if risk not in {"read-only", "write", "external-side-effect"}:
        raise ValueError(f"插件 Tool risk 无效: {risk}")
    for field_name, value in (
        ("always_on", always_on),
        ("preloadable", preloadable),
        ("requires_turn_search", requires_turn_search),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"插件 Tool {field_name} 必须是 bool")
    if search_hint is not None and (
        not isinstance(search_hint, str)
        or not search_hint
        or search_hint != search_hint.strip()
    ):
        raise ValueError("插件 Tool search_hint 必须是非空且无首尾空白的字符串")

    # 2. Freeze metadata while preserving the Tool instance's runtime state.
    return PluginToolContribution(
        tool=tool,
        risk=risk,
        always_on=always_on,
        preloadable=preloadable,
        requires_turn_search=requires_turn_search,
        search_hint=search_hint,
    )
