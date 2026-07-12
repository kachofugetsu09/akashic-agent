from __future__ import annotations

from typing import cast

from agent.plugins import Plugin
from plugins.wake_proactive.runtime import WakeRuntime
from proactive_v2.lifecycle import ProactiveLifecycleSpec
from proactive_v2.runtime_scope import ProactiveRuntimeScope


class WakeRuntimeFactory:
    lifecycle_id = "wake"

    def __call__(self, scope: ProactiveRuntimeScope) -> object:
        return WakeRuntime(scope)


class WakeProactiveModuleFactory:
    lifecycle_id = "wake"

    def __call__(self, runtime: object) -> list[object]:
        from plugins.wake_proactive.modules import build_wake_runtime_modules
        return cast(list[object], build_wake_runtime_modules(runtime))


class WakeProactivePlugin(Plugin):
    name = "wake_proactive"

    def proactive_module_factories(self) -> list[object]:
        return [WakeProactiveModuleFactory()]

    def proactive_runtime_factories(self) -> list[object]:
        return [WakeRuntimeFactory()]

    def proactive_lifecycles(self) -> list[object]:
        return [
            ProactiveLifecycleSpec(
                id="wake",
                initial_slots=(
                    "proactive:cfg",
                    "proactive:session_key",
                    "proactive:started_at",
                    "proactive:last_user_at",
                ),
                terminal_slots=("run:next_wakeup",),
            )
        ]
