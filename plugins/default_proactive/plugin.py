from __future__ import annotations

from plugins.default_proactive.runtime import (
    ProactiveFlowRuntime,
    build_default_proactive_modules,
)
from plugins.default_proactive.factory import AgentTickFactory
from agent.plugins import Plugin
from proactive_v2.lifecycle import ProactiveLifecycleSpec
from proactive_v2.runtime_scope import ProactiveRuntimeScope


class DefaultRuntimeFactory:
    lifecycle_id = "default"

    def __call__(self, scope: ProactiveRuntimeScope) -> object:
        return AgentTickFactory(scope).build_runtime()


class DefaultModuleFactory:
    lifecycle_id = "default"

    def __call__(self, runtime: object) -> list[object]:
        if not isinstance(runtime, ProactiveFlowRuntime):
            raise RuntimeError("default proactive 收到未知 Runtime")
        return build_default_proactive_modules(runtime)


class DefaultProactivePlugin(Plugin):
    name = "default_proactive"

    def proactive_lifecycles(self) -> list[object]:
        return [
            ProactiveLifecycleSpec(
                id="default",
                initial_slots=(
                    "proactive:cfg",
                    "proactive:session_key",
                    "proactive:started_at",
                    "proactive:last_user_at",
                    "proactive:base_judge_send_threshold",
                ),
                terminal_slots=("run:next_wakeup",),
            )
        ]

    def proactive_module_factories(self) -> list[object]:
        return [DefaultModuleFactory()]

    def proactive_runtime_factories(self) -> list[object]:
        return [DefaultRuntimeFactory()]
