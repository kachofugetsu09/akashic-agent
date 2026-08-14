from __future__ import annotations

from typing import TYPE_CHECKING

from agent.plugin_composition import SerialEventKey

if TYPE_CHECKING:
    from agent.lifecycle.types import AfterReasoningCtx


AFTER_REASONING_BEFORE_EVENT_BUS: SerialEventKey["AfterReasoningCtx", object] = (
    SerialEventKey("turn.after_reasoning.before_event_bus")
)
AFTER_REASONING_BEFORE_PERSIST: SerialEventKey["AfterReasoningCtx", object] = (
    SerialEventKey("turn.after_reasoning.before_persist")
)
