from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import httpx

from agent.plugin_composition import (
    AuthenticationError,
    CapabilitySources,
    CredentialHandle,
    DiscoveredModel,
    DriverConnectionDescriptor,
    ModelCapabilities,
    ModelKind,
    TransportError,
)

from .auth import CODEX_CLIENT_VERSION, headers


async def discover(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> tuple[DiscoveredModel, ...]:
    """Load the complete visible Codex catalog for one connection."""

    payload = await _catalog_request(descriptor, credential)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise TransportError("Codex 模型目录响应缺少 models 数组")
    result: list[DiscoveredModel] = []
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise TransportError("Codex 模型目录包含无效模型项")
        if raw.get("visibility", "list") != "list":
            continue
        if raw.get("supported_in_api", True) is False:
            continue
        result.append(_parse_model(raw))
    if not result:
        raise TransportError("Codex 模型目录没有可用模型")
    return tuple(result)


async def probe(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> None:
    """Prove the credential can read a valid Codex catalog."""

    _ = await _catalog_request(descriptor, credential)


async def _catalog_request(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> Mapping[str, Any]:
    if descriptor.driver_id != "codex":
        raise ValueError(f"unexpected driver id: {descriptor.driver_id}")
    if credential.connection_id != descriptor.connection_id:
        raise AuthenticationError("credential connection scope does not match")
    if credential.auth_identity != descriptor.auth_identity:
        raise AuthenticationError("credential auth identity does not match")
    timeout = _positive_timeout(descriptor.config.get("catalog_timeout", 30.0))
    endpoint = descriptor.endpoint.rstrip("/")
    rejected: str | None = None
    for attempt in range(2):
        token, request_headers = await headers(
            credential,
            rejected_access_token=rejected,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                response = await client.get(
                    f"{endpoint}/models",
                    params={"client_version": CODEX_CLIENT_VERSION},
                    headers=request_headers,
                )
        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException as exc:
            raise TransportError("Codex 模型目录连接超时") from exc
        except httpx.TransportError as exc:
            raise TransportError("Codex 模型目录连接失败") from exc
        if response.status_code == 401 and attempt == 0:
            rejected = token
            continue
        if response.status_code in {401, 403}:
            raise AuthenticationError("Codex 模型目录认证失败，请重新登录")
        if response.status_code >= 400:
            raise TransportError(
                f"Codex 模型目录请求失败 (HTTP {response.status_code})"
            )
        try:
            value: Any = response.json()
        except json.JSONDecodeError as exc:
            raise TransportError("Codex 模型目录返回了无效 JSON") from exc
        if not isinstance(value, Mapping):
            raise TransportError("Codex 模型目录响应必须是对象")
        return value
    raise AuthenticationError("Codex 模型目录认证失败，请重新登录")


def _parse_model(raw: Mapping[str, Any]) -> DiscoveredModel:
    slug = raw.get("slug")
    if not isinstance(slug, str) or not slug or slug.strip() != slug:
        raise TransportError("Codex 模型目录包含无效 slug")
    context_window = raw.get("context_window") or raw.get("max_context_window")
    if not _positive_int(context_window):
        raise TransportError(f"模型 {slug} 缺少有效 context_window")
    modalities = raw.get("input_modalities")
    if modalities is None:
        parsed_modalities = ("text",)
        modalities_source = "default"
    elif isinstance(modalities, list) and all(
        isinstance(item, str) and item for item in modalities
    ):
        parsed_modalities = tuple(modalities)
        modalities_source = "provider"
    else:
        raise TransportError(f"模型 {slug} 的 input_modalities 无效")
    levels = raw.get("supported_reasoning_levels") or []
    if not isinstance(levels, list):
        raise TransportError(f"模型 {slug} 的 reasoning levels 无效")
    parsed_efforts: list[str] = []
    for item in levels:
        if not isinstance(item, Mapping):
            raise TransportError(f"模型 {slug} 的 reasoning level 项无效")
        effort = item.get("effort")
        if (
            not isinstance(effort, str)
            or not effort
            or effort.strip() != effort
        ):
            raise TransportError(f"模型 {slug} 的 reasoning effort 无效")
        if effort not in parsed_efforts:
            parsed_efforts.append(effort)
    efforts = tuple(parsed_efforts)
    default_effort = raw.get("default_reasoning_level")
    if default_effort is not None:
        if (
            not isinstance(default_effort, str)
            or not default_effort
            or default_effort.strip() != default_effort
        ):
            raise TransportError(f"模型 {slug} 的 default reasoning level 无效")
        if efforts and default_effort not in efforts:
            raise TransportError(
                f"模型 {slug} 的 default reasoning level 不在支持列表"
            )
    use_lite = raw.get("use_responses_lite", False)
    if not isinstance(use_lite, bool):
        raise TransportError(f"模型 {slug} 的 use_responses_lite 无效")
    supports_summary = raw.get(
        "supports_reasoning_summary_parameter",
        raw.get("supports_reasoning_summaries", False),
    )
    if not isinstance(supports_summary, bool):
        raise TransportError(f"模型 {slug} 的 reasoning summary 标记无效")
    parallel = raw.get("supports_parallel_tool_calls", False)
    if not isinstance(parallel, bool):
        raise TransportError(f"模型 {slug} 的 parallel tools 标记无效")
    return DiscoveredModel(
        kind=ModelKind.CHAT,
        model=slug,
        capabilities=ModelCapabilities(
            context_window=context_window,
            max_output_tokens=None,
            input_modalities=parsed_modalities,
            supports_tool_calls=True,
            supports_parallel_tool_calls=parallel,
            supported_reasoning_efforts=efforts,
        ),
        capability_sources=CapabilitySources(
            context_window="provider",
            max_output_tokens="unknown",
            input_modalities=modalities_source,
            tool_calls="protocol",
            parallel_tool_calls="provider",
            reasoning_efforts="provider",
        ),
        default_reasoning_effort=default_effort,
        driver_config={
            "format_version": 1,
            "use_responses_lite": use_lite,
            "reasoning_summary": "auto" if supports_summary else "none",
        },
    )


def _positive_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("catalog_timeout must be positive")
    return float(value)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
