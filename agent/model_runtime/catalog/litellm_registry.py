from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

# LiteLLM 默认允许在线刷新价格表；模型设置必须只读取随固定 wheel 发布的快照。
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from genai_prices.data_snapshot import get_snapshot


@dataclass(frozen=True)
class CatalogCapabilities:
    context_window: int
    max_output_tokens: int
    input_modalities: tuple[str, ...]
    reasoning: bool
    tool_call: bool
    input_modalities_known: bool = True
    supported_reasoning_efforts: tuple[str, ...] = ()
    supports_parallel_tool_calls: bool = True
    source: str = "litellm"


_PROVIDER_ALIASES = {
    "dashscope": "dashscope",
    "opencode_go": "opencode-go",
    "xai": "x-ai",
}

_LITELLM_PROVIDER_ALIASES = {
    "x-ai": "xai",
    "z-ai": "zai",
}

_EFFORT_FLAGS = (
    ("none", "supports_none_reasoning_effort"),
    ("minimal", "supports_minimal_reasoning_effort"),
    ("xhigh", "supports_xhigh_reasoning_effort"),
    ("max", "supports_max_reasoning_effort"),
)


def resolve_catalog_capabilities(
    provider: str,
    model: str,
    *,
    base_url: str = "",
) -> CatalogCapabilities | None:
    """从固定 LiteLLM wheel 的注册表解析模型能力。"""

    provider_id = resolve_catalog_provider_id(
        provider,
        model=model,
        base_url=base_url,
    )
    raw = _model_entry(provider_id, model)
    if raw is None:
        return None
    modalities, modalities_known = _input_modalities(raw)
    reasoning = raw.get("supports_reasoning") is True
    max_input_tokens = _positive_int(raw.get("max_input_tokens"))
    max_output_tokens = _positive_int(
        raw.get("max_output_tokens") or raw.get("max_tokens")
    )
    return CatalogCapabilities(
        context_window=(
            max_input_tokens + max_output_tokens
            if max_input_tokens and max_output_tokens
            else max_input_tokens
        ),
        max_output_tokens=max_output_tokens,
        input_modalities=modalities,
        input_modalities_known=modalities_known,
        reasoning=reasoning,
        tool_call=raw.get("supports_function_calling") is True,
        supported_reasoning_efforts=_reasoning_efforts(raw) if reasoning else (),
        supports_parallel_tool_calls=(
            raw.get("supports_parallel_function_calling") is not False
        ),
    )


def resolve_catalog_provider_id(
    provider: str,
    *,
    model: str = "",
    base_url: str = "",
) -> str:
    """用成熟注册表识别 provider 身份，不改变实际 wire transport。"""

    # 1. genai-prices 的 provider API 正则优先识别官方入口。
    if base_url:
        try:
            matched_provider = get_snapshot().find_provider(None, None, base_url)
        except LookupError:
            pass
        else:
            match = re.match(matched_provider.api_pattern, base_url)
            suffix = base_url[match.end() :] if match is not None else "invalid"
            if not suffix or suffix.startswith(("/", "?", "#")):
                detected = matched_provider.id
                if detected is not None:
                    return _PROVIDER_ALIASES.get(detected, detected)

    # 2. 明确 provider 或带 provider 前缀的模型使用 LiteLLM 注册表核对。
    raw_provider = provider.strip().lower()
    normalized = _PROVIDER_ALIASES.get(raw_provider, raw_provider)
    if normalized in _known_provider_ids():
        return normalized
    if "/" in model:
        prefix = model.split("/", 1)[0].lower()
        if prefix in _known_provider_ids():
            return prefix
    return ""


@lru_cache(maxsize=1)
def _known_provider_ids() -> frozenset[str]:
    providers = {
        str(raw.get("litellm_provider") or "").lower()
        for raw in _registry().values()
    }
    providers.discard("")
    providers.update(_PROVIDER_ALIASES.values())
    return frozenset(providers)


def _model_entry(provider: str, model: str) -> dict[str, Any] | None:
    """优先匹配 provider 专属型号，再使用 LiteLLM 的 canonical 型号。"""

    normalized_model = model.strip()
    if not normalized_model:
        return None
    litellm_provider = _LITELLM_PROVIDER_ALIASES.get(provider, provider)
    candidates: list[str] = []
    if litellm_provider and not normalized_model.startswith(f"{litellm_provider}/"):
        candidates.append(f"{litellm_provider}/{normalized_model}")
    candidates.append(normalized_model)
    for candidate in candidates:
        raw = _registry().get(candidate)
        if raw is not None:
            return raw
    return None


def _input_modalities(raw: dict[str, Any]) -> tuple[tuple[str, ...], bool]:
    declared = raw.get("supported_modalities")
    if isinstance(declared, list):
        declared_items = cast(list[object], declared)
        if not all(isinstance(item, str) for item in declared_items):
            return ("text",), False
        declared_strings = cast(list[str], declared_items)
        modalities = tuple(dict.fromkeys(item.lower() for item in declared_strings))
        return (
            ("text",) + tuple(item for item in modalities if item != "text"),
            True,
        )
    vision = raw.get("supports_vision")
    if isinstance(vision, bool):
        return (("text", "image") if vision else ("text",), True)
    return ("text",), False


def _reasoning_efforts(raw: dict[str, Any]) -> tuple[str, ...]:
    """按 LiteLLM 的 effort flags 生成 Memoh 同序的选项。"""

    efforts = [
        name
        for name, flag in _EFFORT_FLAGS[:2]
        if raw.get(flag) is True
    ]
    if raw.get("supports_low_reasoning_effort") is not False:
        efforts.append("low")
    efforts.extend(("medium", "high"))
    efforts.extend(
        name
        for name, flag in _EFFORT_FLAGS[2:]
        if raw.get(flag) is True
    )
    return tuple(efforts)


def _positive_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[str, Any]]:
    return cast(dict[str, dict[str, Any]], litellm.model_cost)
