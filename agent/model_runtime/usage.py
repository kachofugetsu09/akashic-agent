from __future__ import annotations

from .types import ModelUsage, UsageCoverage


def aggregate_usage(items: list[ModelUsage]) -> ModelUsage:
    """聚合多次模型请求，并保留未知字段的未知语义。"""

    def total(field: str) -> int | None:
        values = [getattr(item, field) for item in items]
        known = [value for value in values if value is not None]
        return sum(known) if known else None

    if not items:
        return ModelUsage(request_count=0)
    request_count = sum(item.request_count for item in items)
    covered = sum(item.covered_request_count for item in items)
    coverage = (
        UsageCoverage.UNAVAILABLE
        if covered == 0
        else UsageCoverage.EXACT
        if covered == request_count and all(item.coverage is UsageCoverage.EXACT for item in items)
        else UsageCoverage.PARTIAL
    )
    return ModelUsage(
        input_tokens=total("input_tokens"),
        cached_input_tokens=total("cached_input_tokens"),
        output_tokens=total("output_tokens"),
        reasoning_output_tokens=total("reasoning_output_tokens"),
        request_count=request_count,
        covered_request_count=covered,
        coverage=coverage,
    )
