from __future__ import annotations

from agent.config_models import Config
from agent.provider import LLMProvider


def build_providers(config: Config) -> tuple[LLMProvider, LLMProvider | None]:
    provider = LLMProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        system_prompt=config.system_prompt,
        extra_body=config.extra_body,
        request_timeout_s=180.0,
    )

    light_provider: LLMProvider | None = None
    if config.light_model and (config.light_api_key or config.light_base_url):
        light_url = config.light_base_url or config.base_url or ""
        light_extra: dict = (
            {}
            if "googleapis.com" in light_url or "generativelanguage" in light_url
            else {"enable_thinking": False}
        )
        light_provider = LLMProvider(
            api_key=config.light_api_key or config.api_key,
            base_url=config.light_base_url or config.base_url,
            system_prompt=config.system_prompt,
            extra_body=light_extra,
        )

    return provider, light_provider
