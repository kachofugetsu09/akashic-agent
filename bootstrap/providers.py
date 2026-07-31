from __future__ import annotations

from agent.config_models import Config
from agent.model_runtime.fallback import ResilientLightProvider
from infra.providers.llm_provider import LLMProvider

_MAIN_NETWORK_READ_TIMEOUT_S = 120.0
_LIGHT_NETWORK_READ_TIMEOUT_S = 60.0


def build_providers(
    config: Config,
) -> tuple[LLMProvider, LLMProvider | None, LLMProvider | None]:
    payload_snapshot_enabled = config.dev_mode
    main_extra = _sanitize_extra_body(
        base_url=config.base_url,
        extra_body=config.extra_body,
    )
    main_runtime = config.model_runtimes.get(config.runtime_id)
    provider = (
        LLMProvider.from_runtime(
            main_runtime,
            system_prompt=config.system_prompt,
            extra_body=main_extra,
            read_timeout_s=_MAIN_NETWORK_READ_TIMEOUT_S,
            payload_snapshot_enabled=payload_snapshot_enabled,
        )
        if main_runtime is not None
        else LLMProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            system_prompt=config.system_prompt,
            extra_body=main_extra,
            read_timeout_s=_MAIN_NETWORK_READ_TIMEOUT_S,
            provider_name=config.provider,
            auth_id=config.auth_id,
            runtime_id=config.runtime_id,
            context_window=config.context_window,
            effective_context_percent=config.effective_context_percent,
            compaction_trigger_percent=config.compaction_trigger_percent,
            use_responses_lite=config.use_responses_lite,
            supports_parallel_tool_calls=config.supports_parallel_tool_calls,
            reasoning_summary=config.reasoning_summary,
            payload_snapshot_enabled=payload_snapshot_enabled,
        )
    )

    light_provider = _build_named_role_provider(
        config,
        config.fast_runtime_id,
        system_prompt=config.system_prompt,
        read_timeout_s=_LIGHT_NETWORK_READ_TIMEOUT_S,
        force_disable_thinking=True,
    )
    if light_provider is None and config.light_model and (config.light_api_key or config.light_base_url):
        light_url = config.light_base_url or config.base_url or ""
        light_extra: dict[str, object] = (
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
            read_timeout_s=_LIGHT_NETWORK_READ_TIMEOUT_S,
            force_disable_thinking=True,
            payload_snapshot_enabled=payload_snapshot_enabled,
        )
    if light_provider is not None:
        primary_runtime_id = config.fast_runtime_id or "legacy-fast"
        primary_model = config.light_model
        light_provider = ResilientLightProvider(
            primary=light_provider,
            primary_runtime_id=primary_runtime_id,
            primary_model=primary_model,
            fallback=provider,
            fallback_model=config.model,
        )

    agent_provider = _build_named_role_provider(
        config,
        config.agent_runtime_id,
        system_prompt=config.system_prompt,
        read_timeout_s=_MAIN_NETWORK_READ_TIMEOUT_S,
    )
    if agent_provider is None and config.agent_model and (config.agent_api_key or config.agent_base_url):
        agent_url = config.agent_base_url or config.base_url or ""
        agent_extra = _sanitize_extra_body(base_url=agent_url, extra_body={})
        agent_provider = LLMProvider(
            api_key=config.agent_api_key or config.api_key,
            base_url=agent_url,
            system_prompt=config.system_prompt,
            extra_body=agent_extra,
            read_timeout_s=_MAIN_NETWORK_READ_TIMEOUT_S,
            payload_snapshot_enabled=payload_snapshot_enabled,
        )

    return provider, light_provider, agent_provider


def build_vl_provider(config: Config) -> LLMProvider | None:
    """构建 VL 视觉模型 provider，仅当主模型不支持多模态且配置了 vl_model 时返回。"""
    if not config.multimodal and config.vl_model:
        named = _build_named_role_provider(
            config,
            config.vl_runtime_id,
            system_prompt="",
            read_timeout_s=_MAIN_NETWORK_READ_TIMEOUT_S,
        )
        if named is not None:
            return named
        payload_snapshot_enabled = config.dev_mode
        vl_url = config.vl_base_url or config.base_url or ""
        vl_extra = _sanitize_extra_body(base_url=vl_url, extra_body={})
        return LLMProvider(
            api_key=config.vl_api_key or config.api_key,
            base_url=config.vl_base_url or config.base_url,
            system_prompt="",
            extra_body=vl_extra,
            read_timeout_s=_MAIN_NETWORK_READ_TIMEOUT_S,
            payload_snapshot_enabled=payload_snapshot_enabled,
        )
    return None


def _build_named_role_provider(
    config: Config,
    runtime_id: str,
    *,
    system_prompt: str,
    read_timeout_s: float,
    force_disable_thinking: bool = False,
) -> LLMProvider | None:
    """为独立 named runtime 组装完整 provider；复用 main 时返回空。"""
    if not runtime_id or runtime_id == config.runtime_id:
        return None
    runtime = config.model_runtimes[runtime_id]
    return LLMProvider.from_runtime(
        runtime,
        system_prompt=system_prompt,
        read_timeout_s=read_timeout_s,
        force_disable_thinking=force_disable_thinking,
        payload_snapshot_enabled=config.dev_mode,
    )


def _sanitize_extra_body(
    base_url: str | None,
    extra_body: dict[str, object] | None,
) -> dict[str, object]:
    cleaned = dict(extra_body or {})
    url = (base_url or "").lower()
    if "minimaxi.com" in url:
        _ = cleaned.pop("enable_thinking", None)
    return cleaned
