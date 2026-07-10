from __future__ import annotations

from plugins.default_proactive.runtime import ProactiveFlowRuntime
from plugins.proactive_flow.modules import build_proactive_flow_modules
from agent.plugins import Plugin


class ProactiveModuleFactory:
    lifecycle_id = "default"

    def __call__(self, runtime: object) -> list[object]:
        if not isinstance(runtime, ProactiveFlowRuntime):
            raise RuntimeError("proactive flow 收到未知 Runtime")
        return build_proactive_flow_modules(runtime)


class ProactiveFlowPlugin(Plugin):
    name = "proactive_flow"

    def proactive_module_factories(self) -> list[object]:
        return [ProactiveModuleFactory()]
