from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, cast

# LiteLLM 默认允许在线刷新价格表。这里不再强制使用固定 wheel 快照：
# 模型能力（尤其是新模型的上下文窗口）应以在线注册表为准，本地缓存兜底。
# 如需强制离线，可设置 LITELLM_LOCAL_MODEL_COST_MAP=True。

import litellm
from genai_prices.data_snapshot import get_snapshot

# 插件缓存的 LiteLLM 能力目录（由 plugins/models/litellm_catalog.py 在线拉取维护）。
# 运行时按环境变量覆盖；默认从 models 插件的 data_root 读取。
_LITELLM_CACHE_PATH = os.environ.get(
    "AKASHIC_LITELLM_CAPABILITIES_CACHE",
    "",
)


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
    # 上下文窗口 = 模型可接受的输入长度（max_input_tokens）。
    # 多数现代模型（DeepSeek/GPT/Gemini/Claude 等）官方口径的
    # "context window" 即 input 上限；max_output_tokens 是单独的输出
    # 预算，不应计入上下文窗口（否则会把 1M 误算成 1.4M，导致
    # 上下文超限）。只有 max_input_tokens 缺失时才回退到 input+output。
    if max_input_tokens:
        context_window = max_input_tokens
    else:
        context_window = max_input_tokens + max_output_tokens
    return CatalogCapabilities(
        context_window=context_window,
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


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[str, Any]]:
    """优先使用 LiteLLM 在线注册表；不可用时回退到插件本地缓存。

    在线数据可能因网络问题（SSL 超时等）不可用，此时 litellm.model_cost
    会 fallback 到随 wheel 发布的旧快照。若 wheel 快照也缺少模型，
    继续尝试 models 插件维护的 litellm-capabilities.json 缓存。
    """
    online = cast(dict[str, dict[str, Any]], litellm.model_cost)
    if online:
        return online
    cached = _load_cached_catalog()
    if cached:
        return cached
    return online


def _load_cached_catalog(
    cache_path: str | Path | None = None,
) -> dict[str, dict[str, Any]] | None:
    """读取 models 插件在线拉取并缓存的能力目录（litellm-capabilities.json）。

    cache_path 缺省时从环境变量 AKASHIC_LITELLM_CAPABILITIES_CACHE 读取；
    显式传入则优先使用（便于测试注入）。
    """
    if cache_path is None:
        cache_path = _LITELLM_CACHE_PATH.strip()
    if not cache_path:
        return None
    try:
        raw = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        return None
    models = raw["models"]
    if not all(isinstance(k, str) and isinstance(v, dict) for k, v in models.items()):
        return None
    return cast(dict[str, dict[str, Any]], models)


def _normalize_key(value: str) -> str:
    """归一化模型/供应商名：小写 + 去所有非字母数字。"""
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def _fuzzy_entry(
    models: Mapping[str, Mapping[str, Any]],
    *,
    model: str,
    provider_id: str,
) -> Mapping[str, Any] | None:
    """在线注册表中做大小写不敏感、跨供应商的模糊匹配。

    匹配顺序：
      1. 完整模型名归一化精确匹配（去掉标点/大小写差异）
      2. 型号部分（去掉供应商前缀）归一化匹配
      3. 在同一供应商前缀下，尝试供应商+型号拼合的候选
    """
    normalized_model = _normalize_key(model)
    if not normalized_model:
        return None
    short = model.split("/", 1)[-1].strip()
    normalized_short = _normalize_key(short)

    # 建一次归一化索引（lru 友好：调用方传入的是同一份 models）
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for key, entry in models.items():
        normalized_key = _normalize_key(key)
        if normalized_key == normalized_model:
            return entry
        if (
            short
            and normalized_short
            and normalized_short in normalized_key
            and len(normalized_short) >= 3
        ):
            candidates.append((key, entry))
    # 精确命中过直接返回；否则按"型号部分归一化"的候选里，优先
    # 供应商前缀匹配的（更可能是同型号同厂商）。
    if provider_id:
        normalized_provider = _normalize_key(provider_id)
        for key, entry in candidates:
            if normalized_provider and normalized_provider in _normalize_key(key):
                return entry
    if candidates:
        return candidates[0][1]
    return None


def _model_entry(
    provider: str,
    model: str,
) -> dict[str, Any] | None:
    """优先匹配 provider 专属型号，再使用 LiteLLM 的 canonical 型号。

    数据源顺序：在线注册表精确 → 在线注册表模糊 → 本地缓存精确 →
    本地缓存模糊。这样即使在线数据缺模型（网络不可用或 litellm 未收录），
    也能从 models 插件维护的缓存目录补齐。
    """

    normalized_model = model.strip()
    if not normalized_model:
        return None
    litellm_provider = _LITELLM_PROVIDER_ALIASES.get(provider, provider)
    candidates: list[str] = []
    if litellm_provider and not normalized_model.startswith(f"{litellm_provider}/"):
        candidates.append(f"{litellm_provider}/{normalized_model}")
    candidates.append(normalized_model)

    # 数据源按优先级排列：在线注册表 → 本地缓存
    sources: list[dict[str, dict[str, Any]]] = []
    online = cast(dict[str, dict[str, Any]], litellm.model_cost)
    if online:
        sources.append(online)
    cached = _load_cached_catalog()
    if cached:
        sources.append(cached)

    for source in sources:
        for candidate in candidates:
            raw = source.get(candidate)
            if raw is not None:
                return raw
        # 精确未命中，模糊匹配（大小写不敏感 + 跨供应商）
        fuzzy = _fuzzy_entry(source, model=normalized_model, provider_id=provider)
        if fuzzy is not None:
            return dict(fuzzy)
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
