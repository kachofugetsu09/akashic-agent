from __future__ import annotations

from agent.config_models import Config
from agent.model_runtime.auth.store import CredentialStore
from agent.model_runtime.fallback import ResilientLightProvider
from agent.model_runtime.registry import (
    ModelGeneration,
    ModelRegistry,
    model_config_digest,
)
from infra.providers.llm_provider import LLMProvider

_MAIN_NETWORK_READ_TIMEOUT_S = 120.0
_LIGHT_NETWORK_READ_TIMEOUT_S = 60.0


def build_model_registry(config: Config) -> ModelRegistry:
    """Build the runtime model registry from the validated configuration."""

    return ModelRegistry(config, _build_model_generation)


def _build_model_generation(config: Config, generation_id: int) -> ModelGeneration:
    """Construct every configured runtime before publishing one generation."""

    # 1. Reuse the established role builders and their fallback semantics.
    default_provider, fast_provider, agent_provider = build_providers(config)
    credential_store = CredentialStore.for_workspace(config.workspace_path)
    runtime_providers: dict[str, LLMProvider] = {config.runtime_id: default_provider}
    for runtime_id, runtime in config.model_runtimes.items():
        if runtime_id in runtime_providers:
            continue
        runtime_providers[runtime_id] = LLMProvider.from_runtime(
            runtime,
            system_prompt=config.system_prompt,
            credential_store=credential_store,
            read_timeout_s=_MAIN_NETWORK_READ_TIMEOUT_S,
            payload_snapshot_enabled=config.dev_mode,
        )

    # 2. Role-specific wrappers are part of the immutable generation.
    role_runtime_ids = {
        "default": config.runtime_id,
        "fast": config.fast_runtime_id or config.runtime_id,
        "agent": config.agent_runtime_id or config.runtime_id,
        "vision": config.vl_runtime_id or config.runtime_id,
    }
    role_providers: dict[str, object] = {"default": default_provider}
    role_providers["fast"] = fast_provider or default_provider
    role_providers["agent"] = agent_provider or default_provider
    vision_provider = build_vl_provider(config)
    role_providers["vision"] = vision_provider or default_provider
    return ModelGeneration(
        generation_id=generation_id,
        config_digest=model_config_digest(config),
        runtimes=dict(config.model_runtimes),
        providers=runtime_providers,
        role_runtime_ids=role_runtime_ids,
        role_providers=role_providers,
        registry_revision=config.model_registry_revision,
    )


def build_providers(
    config: Config,
) -> tuple[LLMProvider, LLMProvider | None, LLMProvider | None]:
    payload_snapshot_enabled = config.dev_mode
    credential_store = CredentialStore.for_workspace(config.workspace_path)
    main_extra = _sanitize_extra_body(
        base_url=config.base_url,
        extra_body=config.extra_body,
    )
    main_runtime = config.model_runtimes.get(config.runtime_id)
    provider = (
        LLMProvider.from_runtime(
            main_runtime,
            system_prompt=config.system_prompt,
            credential_store=credential_store,
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
        credential_store=CredentialStore.for_workspace(config.workspace_path),
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
