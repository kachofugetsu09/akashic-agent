from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel
    from infra.channels.contract import Channel
    from agent.plugins.context import PluginContext
    from agent.plugins.jobs import PluginJobSpec
    from agent.plugins.specs import ManagedServiceSpec, McpServerSpec, ProactiveSourceSpec
    from agent.plugins.generation import PluginSemanticCheck
    from agent.plugins.generation import PluginReadinessContext


class Plugin(ABC):
    api_version: int = 1
    name: str | None = None
    version: str | None = None
    desc: str | None = None
    author: str | None = None
    ConfigModel: "type[BaseModel] | None" = None
    context: "PluginContext"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        from agent.plugins.registry import plugin_registry
        plugin_registry.register_class(cls)

    async def initialize(self) -> None: ...
    async def terminate(self) -> None: ...

    def static_semantic_checks(self) -> list["PluginSemanticCheck"]:
        return []

    async def readiness_semantic_checks(
        self,
        context: "PluginReadinessContext",
    ) -> list["PluginSemanticCheck"]:
        return []

    @classmethod
    def skill_roots(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def drift_skill_roots(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def mcp_servers(cls) -> list["McpServerSpec"]:
        return []

    @classmethod
    def managed_services(cls) -> list["ManagedServiceSpec"]:
        return []

    def proactive_sources(self) -> list["ProactiveSourceSpec"]:
        return []

    def before_turn_modules(self) -> list[object]:
        return []

    def before_reasoning_modules(self) -> list[object]:
        return []

    def prompt_render_modules(self) -> list[object]:
        return []

    def before_step_modules(self) -> list[object]:
        return []

    def after_step_modules(self) -> list[object]:
        return []

    def after_reasoning_modules(self) -> list[object]:
        return []

    def after_turn_modules(self) -> list[object]:
        return []

    def proactive_modules(self) -> list[object]:
        return []

    def proactive_lifecycles(self) -> list[object]:
        return []

    def proactive_module_factories(self) -> list[object]:
        return []

    def proactive_runtime_factories(self) -> list[object]:
        return []

    def jobs(self) -> list["PluginJobSpec"]:
        return []

    def channels(self) -> list["Channel"]:
        return []

    @classmethod
    def dashboard_module(cls) -> str | None:
        return None
