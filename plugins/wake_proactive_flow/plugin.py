from __future__ import annotations

from agent.plugins import Plugin
from plugins.wake_proactive.modules import build_wake_content_modules
from plugins.wake_proactive.runtime import WakeRuntime


class WakeContentModuleFactory:
    lifecycle_id = "wake"

    def __call__(self, runtime: object) -> list[object]:
        if not isinstance(runtime, WakeRuntime):
            raise RuntimeError("wake proactive flow 收到未知 Runtime")
        return build_wake_content_modules(runtime)


class WakeProactiveFlowPlugin(Plugin):
    name = "wake_proactive_flow"

    def proactive_module_factories(self) -> list[object]:
        return [WakeContentModuleFactory()]
