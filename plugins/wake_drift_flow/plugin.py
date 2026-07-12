from __future__ import annotations

from agent.plugins import Plugin
from plugins.wake_proactive.modules import build_wake_drift_modules
from plugins.wake_proactive.runtime import WakeRuntime


class WakeDriftModuleFactory:
    lifecycle_id = "wake"

    def __call__(self, runtime: object) -> list[object]:
        if not isinstance(runtime, WakeRuntime):
            raise RuntimeError("wake drift flow 收到未知 Runtime")
        return build_wake_drift_modules(runtime)


class WakeDriftFlowPlugin(Plugin):
    name = "wake_drift_flow"

    def proactive_module_factories(self) -> list[object]:
        return [WakeDriftModuleFactory()]
