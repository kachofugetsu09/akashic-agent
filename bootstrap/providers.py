from __future__ import annotations

from agent.config_models import Config
from infra.providers.llm_provider import LLMProvider

_MAIN_PROVIDER_TIMEOUT_S = 300.0
_LIGHT_PROVIDER_TIMEOUT_S = 180.0


def build_providers(
    config: Config,
) -> tuple[LLMProvider, LLMProvider | None, LLMProvider | None]:
    payload_snapshot_enabled = bool(getattr(config, "dev_mode", False))
    main_extra = _sanitize_extra_body(
        base_url=config.base_url,
        extra_body=config.extra_body,
    )
    provider = LLMProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        system_prompt=config.system_prompt,
        extra_body=main_extra,
        request_timeout_s=_MAIN_PROVIDER_TIMEOUT_S,
        provider_name=config.provider,
        payload_snapshot_enabled=payload_snapshot_enabled,
    )

    light_provider: LLMProvider | None = None
    if config.light_model and (config.light_api_key or config.light_base_url):
        light_url = config.light_base_url or config.base_url or ""
        light_extra: dict = (
            {}
            if "googleapis.com" in light_url or "generativelanguage" in light_url
            else {"enable_thinking": False}
        )
        light_extra = _sanitize_extra_body(
            base_url=light_url,
            extra_body=light_extra,
        )
        light_provider = LLMProvider(
            api_key=config.light_api_key or config.api_key,
            base_url=config.light_base_url or config.base_url,
            system_prompt=config.system_prompt,
            extra_body=light_extra,
            request_timeout_s=_LIGHT_PROVIDER_TIMEOUT_S,
            force_disable_thinking=True,
            payload_snapshot_enabled=payload_snapshot_enabled,
        )

    agent_provider: LLMProvider | None = None
    if config.agent_model and (config.agent_api_key or config.agent_base_url):
        agent_url = config.agent_base_url or config.base_url or ""
        agent_extra = _sanitize_extra_body(base_url=agent_url, extra_body={})
        agent_provider = LLMProvider(
            api_key=config.agent_api_key or config.api_key,
            base_url=agent_url,
            system_prompt=config.system_prompt,
            extra_body=agent_extra,
            request_timeout_s=_MAIN_PROVIDER_TIMEOUT_S,
            payload_snapshot_enabled=payload_snapshot_enabled,
        )

    return provider, light_provider, agent_provider


def build_vl_provider(config: Config) -> LLMProvider | None:
    """构建 VL 视觉模型 provider，仅当主模型不支持多模态且配置了 vl_model 时返回。"""
    if not getattr(config, "multimodal", True) and getattr(config, "vl_model", ""):
        payload_snapshot_enabled = bool(getattr(config, "dev_mode", False))
        vl_url = getattr(config, "vl_base_url", "") or getattr(config, "base_url", "") or ""
        vl_extra = _sanitize_extra_body(base_url=vl_url, extra_body={})
        return LLMProvider(
            api_key=getattr(config, "vl_api_key", "") or config.api_key,
            base_url=getattr(config, "vl_base_url", "") or config.base_url,
            system_prompt="",
            extra_body=vl_extra,
            request_timeout_s=_MAIN_PROVIDER_TIMEOUT_S,
            payload_snapshot_enabled=payload_snapshot_enabled,
        )
    return None


def _sanitize_extra_body(base_url: str | None, extra_body: dict | None) -> dict:
    cleaned = dict(extra_body or {})
    url = (base_url or "").lower()
    if "minimaxi.com" in url:
        cleaned.pop("enable_thinking", None)
    return cleaned
