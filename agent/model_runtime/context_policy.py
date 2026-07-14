from __future__ import annotations

import math
from dataclasses import dataclass


_REFERENCE_CONTEXT_WINDOW = 1_000_000
_REFERENCE_EFFECTIVE_CONTEXT_PERCENT = 0.9
_REFERENCE_MEMORY_WINDOW = 160
_REFERENCE_OUTPUT_RESERVE = 32_768
_MIN_MEMORY_WINDOW = 20
_MIN_OUTPUT_RESERVE = 4_096


@dataclass(frozen=True)
class ContextWindowSettings:
    memory_window: int
    output_reserve: int


def recommended_context_settings(
    context_window: int,
    effective_context_percent: float = _REFERENCE_EFFECTIVE_CONTEXT_PERCENT,
) -> ContextWindowSettings:
    """按 1M 基准等比例计算历史窗口与输出预留。"""

    # 1. 在配置边界拒绝无效模型容量。
    if context_window <= 0:
        raise ValueError("context_window 必须大于 0")
    if not 0 < effective_context_percent <= 1:
        raise ValueError("effective_context_percent 必须在 (0, 1] 内")

    effective_context = context_window * effective_context_percent
    reference_effective_context = (
        _REFERENCE_CONTEXT_WINDOW * _REFERENCE_EFFECTIVE_CONTEXT_PERCENT
    )

    # 2. 历史条数按四条对齐，避免不同上下文档位产生细碎配置。
    scaled_memory = round(
        effective_context * _REFERENCE_MEMORY_WINDOW / reference_effective_context
    )
    memory_window = max(_MIN_MEMORY_WINDOW, ((scaled_memory + 2) // 4) * 4)

    # 3. 输出预留按 1024 tokens 向下对齐，且不超过 1M 基准值。
    scaled_output = int(
        effective_context * _REFERENCE_OUTPUT_RESERVE / reference_effective_context
    )
    output_reserve = max(
        _MIN_OUTPUT_RESERVE,
        min(_REFERENCE_OUTPUT_RESERVE, scaled_output // 1024 * 1024),
    )
    return ContextWindowSettings(memory_window, output_reserve)


@dataclass(frozen=True)
class ContextBudget:
    effective_context: int
    input_budget: int
    reserved_output: int


def build_runtime_context_budget(
    context_window: int,
    effective_context_percent: float,
    max_output_tokens: int,
) -> ContextBudget:
    """按 runtime 实际配置计算统一输入预算。"""
    effective = math.floor(context_window * effective_context_percent)
    output = max_output_tokens
    if output >= effective:
        raise ValueError("max_output_tokens 必须小于有效上下文")
    return ContextBudget(effective, effective - output, output)
