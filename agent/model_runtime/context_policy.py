from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    """Direct provider input/output boundary for one model request."""

    effective_context: int
    input_budget: int
    reserved_output: int


def build_runtime_context_budget(
    context_window: int,
    max_output_tokens: int,
) -> ContextBudget:
    """Compute the exact input edge from model capacity and requested output."""

    if context_window <= 0:
        raise ValueError("context_window 必须大于 0")
    if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool):
        raise ValueError("max_output_tokens 必须是整数")
    if max_output_tokens < 0 or max_output_tokens >= context_window:
        raise ValueError("max_output_tokens 必须在 [0, context_window) 内")
    return ContextBudget(
        effective_context=context_window,
        input_budget=context_window - max_output_tokens,
        reserved_output=max_output_tokens,
    )
