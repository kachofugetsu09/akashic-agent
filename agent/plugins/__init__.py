# 插件兼容接口：外部插件从本模块导入下列对象。核心重构可以在内部适配，
# 但迁移现有插件及其测试前，不要仅因核心调用减少而删除或改名。
from agent.plugins.base import Plugin
from agent.plugins.config import PluginConfig
from agent.plugins.context import PluginContext, PluginKVStore
from agent.plugins.scope import CleanupFailure, PluginScope
from agent.plugins.generation import (
    GateCheckResult,
    GateResult,
    PluginGeneration,
    PluginReadinessContext,
    PluginSemanticCheck,
)
from agent.plugins.decorators import (
    on_before_turn,
    on_before_reasoning,
    on_before_step,
    on_prompt_render,
    on_after_step,
    on_after_reasoning,
    on_after_turn,
    on_tool_call,
    on_tool_pre,
    on_tool_result,
    tool,
)
from agent.plugins.jobs import (
    EventTrigger,
    IntervalTrigger,
    PluginJobContext,
    PluginJobSpec,
)
from agent.plugins.specs import (
    ManagedServiceSpec,
    McpServerSpec,
    MobileUiContribution,
    MobileUiNavigation,
    ProactiveSourceSpec,
    RegisteredProactiveSource,
)

__all__ = [
    "Plugin",
    "PluginConfig",
    "PluginContext",
    "PluginKVStore",
    "CleanupFailure",
    "PluginScope",
    "GateCheckResult",
    "GateResult",
    "PluginGeneration",
    "PluginReadinessContext",
    "PluginSemanticCheck",
    "EventTrigger",
    "IntervalTrigger",
    "PluginJobContext",
    "PluginJobSpec",
    "McpServerSpec",
    "ManagedServiceSpec",
    "MobileUiContribution",
    "MobileUiNavigation",
    "ProactiveSourceSpec",
    "RegisteredProactiveSource",
    "on_before_turn",
    "on_before_reasoning",
    "on_before_step",
    "on_prompt_render",
    "on_after_step",
    "on_after_reasoning",
    "on_after_turn",
    "on_tool_call",
    "on_tool_pre",
    "on_tool_result",
    "tool",
]
