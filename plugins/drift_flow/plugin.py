from __future__ import annotations

from plugins.default_proactive.runtime import ProactiveFlowRuntime
from plugins.drift_flow.modules import build_drift_flow_modules
from agent.plugins import Plugin


class DriftModuleFactory:
    lifecycle_id = "default"

    def __call__(self, runtime: object) -> list[object]:
        if not isinstance(runtime, ProactiveFlowRuntime):
            raise RuntimeError("drift flow 收到未知 Runtime")
        return build_drift_flow_modules(runtime)


class DriftFlowPlugin(Plugin):
    name = "drift_flow"

    def proactive_module_factories(self) -> list[object]:
        return [DriftModuleFactory()]
