from __future__ import annotations

from typing import cast

from agent.lifecycle.types import AfterStepCtx
from agent.plugins import Plugin

_CTX_SLOT = "step:ctx"
_EARLY_STOP_REASON_SLOT = "step:early_stop_reason"
_TELEMETRY_PREFIX = "step:telemetry:"

_MODEL_CONTEXT_WINDOW_TOKENS = 1_000_000
_CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS = _MODEL_CONTEXT_WINDOW_TOKENS * 80 // 100


class ContextPressureStopModule:
    slot = "context_pressure.stop"
    requires = ("after_step.copy_input", _CTX_SLOT)
    produces = (
        _EARLY_STOP_REASON_SLOT,
        f"{_TELEMETRY_PREFIX}context_pressure_tokens",
        f"{_TELEMETRY_PREFIX}context_pressure_threshold",
    )

    async def run(self, frame: object) -> object:
        raw_slots = getattr(frame, "slots", None)
        if not isinstance(raw_slots, dict):
            return frame
        slots = cast(dict[str, object], raw_slots)
        ctx = slots.get(_CTX_SLOT)
        if not isinstance(ctx, AfterStepCtx) or not ctx.has_more:
            return frame
        tokens = ctx.context_tokens_estimate
        if tokens <= _CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS:
            return frame
        slots[_EARLY_STOP_REASON_SLOT] = "context_pressure"
        slots[f"{_TELEMETRY_PREFIX}context_pressure_tokens"] = tokens
        slots[f"{_TELEMETRY_PREFIX}context_pressure_threshold"] = (
            _CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS
        )
        return frame


class ContextPressurePlugin(Plugin):
    name = "context_pressure"
    version = "0.1.0"
    desc = "上下文压力过高时请求被动循环阶段性收尾"

    def after_step_modules(self) -> list[object]:
        return [ContextPressureStopModule()]
