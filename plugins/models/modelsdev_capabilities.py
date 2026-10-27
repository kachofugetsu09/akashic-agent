from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast
from urllib.parse import urlsplit

# 能力事实来自 models.dev（https://models.dev/api.json），按服务商分条目。
# 在线目录由 models 插件在显式同步时刷新、裁剪、校验和缓存；随包快照只是离线兜底。

_T = TypeVar("_T", bound=Hashable)
Providers = Mapping[str, Mapping[str, Any]]


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
    source: str = "models.dev"


# akashic 的驱动或连接里写的服务商名 -> models.dev 的 provider id。
_PROVIDER_ALIASES = {
    "codex": "openai",
    "gemini": "google",
    "opencode_go": "opencode-go",
    "dashscope": "alibaba-cn",
    "x-ai": "xai",
    "z-ai": "zai",
    "moonshot": "moonshotai",
}

_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def resolve_catalog_capabilities(
    provider: str,
    model: str,
    *,
    providers: Providers,
    base_url: str = "",
) -> CatalogCapabilities | None:
    # 1. 连接地址或服务商名能对上 models.dev 的某个服务商，就用它自己的条目。
    bare = _bare_id(model)
    if not bare:
        return None
    for provider_id in (_provider_by_url(providers, base_url), _provider_by_name(providers, provider)):
        entry = _provider_model(providers.get(provider_id), model, bare) if provider_id else None
        if entry is not None:
            return _capabilities(entry)
    # 2. 否则按裸模型 ID 在所有服务商里取多数口径（中转、自建网关常见）。
    entries = [entry for item in providers.values() for key, entry in _models(item).items() if _bare_id(key) == bare]
    return _majority(entries) if entries else None


# 把 models.dev 原始目录裁成只含本插件用到字段的同构子集，在线缓存与随包快照共用。
def prune_catalog(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for provider_id, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        item = cast(Mapping[str, Any], value)
        pruned: dict[str, Any] = {"models": {key: _prune_model(entry) for key, entry in _models(item).items()}}
        if isinstance(item.get("api"), str):
            pruned["api"] = item["api"]
        result[provider_id] = pruned
    return result


def _prune_model(entry: Mapping[str, Any]) -> dict[str, Any]:
    pruned: dict[str, Any] = {}
    modalities = entry.get("modalities")
    if isinstance(modalities, Mapping) and isinstance(cast(Mapping[str, Any], modalities).get("input"), list):
        pruned["modalities"] = {"input": cast(Mapping[str, Any], modalities)["input"]}
    limit = entry.get("limit")
    if isinstance(limit, Mapping):
        kept = {key: value for key, value in cast(Mapping[str, Any], limit).items() if key in ("context", "output")}
        if kept:
            pruned["limit"] = kept
    for key in ("reasoning", "tool_call"):
        if isinstance(entry.get(key), bool):
            pruned[key] = entry[key]
    options = entry.get("reasoning_options")
    if isinstance(options, list):
        efforts: list[Mapping[str, Any]] = [
            cast(Mapping[str, Any], option) for option in cast(list[object], options) if isinstance(option, Mapping)]
        efforts = [option for option in efforts if option.get("type") == "effort"]
        if efforts:
            pruned["reasoning_options"] = efforts
    return pruned


# 连接地址落在某服务商 api 之下（按路径边界）时取最长的那个：智谱按量与 Coding Plan 靠它区分。
def _provider_by_url(providers: Providers, base_url: str) -> str:
    target = _url_key(base_url)
    if not target:
        return ""
    best, best_length = "", 0
    for provider_id, item in providers.items():
        api = _url_key(str(item.get("api") or ""))
        if api and len(api) > best_length and (target == api or target.startswith(api + "/")):
            best, best_length = provider_id, len(api)
    return best


def _provider_by_name(providers: Providers, provider: str) -> str:
    name = provider.strip().lower()
    name = _PROVIDER_ALIASES.get(name, name)
    return name if name in providers else ""


def _url_key(url: str) -> str:
    parts = urlsplit(url.strip())
    if not parts.hostname:
        return ""
    return f"{parts.hostname.lower()}{parts.path.rstrip('/').lower()}"


def _provider_model(item: Mapping[str, Any] | None, model: str, bare: str) -> Mapping[str, Any] | None:
    if item is None:
        return None
    models = _models(item)
    exact = models.get(model.strip())
    if exact is not None:
        return exact
    return next((entry for key, entry in models.items() if _bare_id(key) == bare), None)


def _models(item: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = item.get("models")
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): cast(Mapping[str, Any], value)
            for key, value in cast(Mapping[str, Any], raw).items() if isinstance(value, Mapping)}


def _bare_id(model: str) -> str:
    return model.strip().rsplit("/", 1)[-1].lower()


def _capabilities(entry: Mapping[str, Any]) -> CatalogCapabilities:
    image = _image_input(entry)
    reasoning = entry.get("reasoning") is True
    return CatalogCapabilities(
        context_window=_limit(entry, "context"),
        max_output_tokens=_limit(entry, "output"),
        input_modalities=("text", "image") if image else ("text",),
        input_modalities_known=image is not None,
        reasoning=reasoning,
        tool_call=entry.get("tool_call") is True,
        supported_reasoning_efforts=_efforts(entry) if reasoning else (),
    )


# 多家服务商都列出同一模型时，每个字段各自取多数，未声明的不参与投票。
def _majority(entries: list[Mapping[str, Any]]) -> CatalogCapabilities:
    image = _vote(entries, _image_input)
    reasoning = _vote(entries, lambda entry: entry.get("reasoning") if isinstance(entry.get("reasoning"), bool) else None)
    return CatalogCapabilities(
        context_window=_vote(entries, lambda entry: _limit(entry, "context") or None) or 0,
        max_output_tokens=_vote(entries, lambda entry: _limit(entry, "output") or None) or 0,
        input_modalities=("text", "image") if image else ("text",),
        input_modalities_known=image is not None,
        reasoning=reasoning is True,
        tool_call=_vote(entries, lambda entry: entry.get("tool_call") if isinstance(entry.get("tool_call"), bool) else None) is True,
        supported_reasoning_efforts=(_vote(entries, lambda entry: _efforts(entry) or None) or ()) if reasoning else (),
    )


def _vote(entries: list[Mapping[str, Any]], read: Callable[[Mapping[str, Any]], _T | None]) -> _T | None:
    votes = Counter(value for entry in entries if (value := read(entry)) is not None)
    return votes.most_common(1)[0][0] if votes else None


def _image_input(entry: Mapping[str, Any]) -> bool | None:
    modalities = entry.get("modalities")
    if not isinstance(modalities, Mapping):
        return None
    declared = cast(Mapping[str, Any], modalities).get("input")
    if not isinstance(declared, list):
        return None
    return "image" in {str(item).lower() for item in cast(list[object], declared)}


def _efforts(entry: Mapping[str, Any]) -> tuple[str, ...]:
    options = entry.get("reasoning_options")
    for option in cast(list[object], options) if isinstance(options, list) else []:
        if isinstance(option, Mapping) and cast(Mapping[str, Any], option).get("type") == "effort":
            values = cast(Mapping[str, Any], option).get("values")
            declared = {str(item) for item in cast(list[object], values)} if isinstance(values, list) else set[str]()
            return tuple(effort for effort in _EFFORTS if effort in declared)
    return ()


def _limit(entry: Mapping[str, Any], key: str) -> int:
    limit = entry.get("limit")
    value = cast(Mapping[str, Any], limit).get(key) if isinstance(limit, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0
