from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .types import ModelCapabilities


def recommended_memory_window(context_window: int) -> int:
    """根据模型上下文给出有界的历史消息建议值。"""
    if context_window <= 0:
        raise ValueError("context_window 必须大于 0")
    return max(20, min(160, round(context_window / 1600)))


@dataclass(frozen=True)
class ContextBudget:
    effective_context: int
    input_budget: int
    reserved_output: int


class ApproximateTokenEstimator:
    quality = "approximate"

    def estimate_messages(self, messages: list[dict]) -> int:
        payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        return max(1, len(payload) // 3)


def build_context_budget(capabilities: ModelCapabilities, max_output_tokens: int) -> ContextBudget:
    """预留输出容量并计算本次请求的输入预算。"""
    output = min(max_output_tokens, capabilities.max_output_tokens)
    effective = math.floor(
        capabilities.context_window * capabilities.effective_context_percent
    )
    if output >= effective:
        raise ValueError("max_output_tokens 必须小于有效上下文")
    return ContextBudget(effective, effective - output, output)
