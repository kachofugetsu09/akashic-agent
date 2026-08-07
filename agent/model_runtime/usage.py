from __future__ import annotations

import logging
from typing import Any

from genai_prices import extract_usage

from .types import ModelUsage, UsageCoverage

logger = logging.getLogger(__name__)


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
        if all(item.coverage is UsageCoverage.UNAVAILABLE for item in items)
        else UsageCoverage.EXACT
        if covered == request_count and all(item.coverage is UsageCoverage.EXACT for item in items)
        else UsageCoverage.PARTIAL
    )
    return ModelUsage(
        input_tokens=total("input_tokens"),
        cache_write_input_tokens=total("cache_write_input_tokens"),
        cached_input_tokens=total("cached_input_tokens"),
        output_tokens=total("output_tokens"),
        reasoning_output_tokens=total("reasoning_output_tokens"),
        request_count=request_count,
        covered_request_count=covered,
        coverage=coverage,
    )


def normalize_provider_usage(
    response_data: Any,
    *,
    provider_id: str,
    provider_api_url: str,
    api_flavor: str,
    reasoning_output_tokens: int | None = None,
) -> ModelUsage | None:
    """用 genai-prices 归一化 provider usage，并显式保留覆盖度。"""

    # 1. 优先使用 provider 标识；兼容网关名称未知时再按 API URL 识别。
    attempts: list[dict[str, str]] = []
    if provider_id:
        attempts.append({"provider_id": provider_id})
    if provider_api_url:
        attempts.append({"provider_api_url": provider_api_url})
    for identity in attempts:
        try:
            extracted = extract_usage(
                response_data,
                api_flavor=api_flavor,
                **identity,
            )
        except (LookupError, TypeError, ValueError):
            continue
        usage = extracted.usage
        known = [usage.input_tokens, usage.output_tokens]
        covered = int(all(value is not None for value in known))
        coverage = (
            UsageCoverage.EXACT
            if covered
            else UsageCoverage.PARTIAL
            if any(value is not None for value in known)
            else UsageCoverage.UNAVAILABLE
        )
        return ModelUsage(
            input_tokens=usage.input_tokens,
            cache_write_input_tokens=usage.cache_write_tokens,
            cached_input_tokens=usage.cache_read_tokens,
            output_tokens=usage.output_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
            covered_request_count=covered,
            coverage=coverage,
        )

    # 2. 未知兼容网关交还既有 wire parser；不把未知伪装为 0。
    logger.info(
        "usage normalizer unavailable provider=%s api_flavor=%s",
        provider_id or "-",
        api_flavor,
    )
    return None
