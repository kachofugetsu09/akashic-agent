from __future__ import annotations

from datetime import datetime

from agent.control.timer import OneShotTimer, TimerHandle
from agent.plugin_composition.model import ServiceKey


class PluginTimers:
    """Expose source-neutral one-shot waits without recurrence or job knowledge."""

    def __init__(self, timer: OneShotTimer | None) -> None:
        self._timer = timer

    @classmethod
    def candidate_validation(cls) -> PluginTimers:
        return cls(None)

    @property
    def formal(self) -> bool:
        return self._timer is not None

    def schedule(self, deadline: datetime) -> TimerHandle:
        timer = self._timer
        if timer is None:
            raise RuntimeError("candidate 验证期禁止登记 timer")
        return timer.schedule(deadline)


TIMERS = ServiceKey[PluginTimers]("core.timers")
