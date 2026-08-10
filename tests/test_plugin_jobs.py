from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.config_models import Config, ModelRuntimeConfig
from agent.model_runtime.registry import ModelGeneration, ModelRegistry, model_config_digest
from agent.plugins.jobs import ProviderPluginLlmService


@pytest.mark.asyncio
async def test_plugin_job_distinguishes_default_and_explicit_zero_limit() -> None:
    provider = AsyncMock()
    provider.chat.return_value = SimpleNamespace(content="done", usage=None)
    service = ProviderPluginLlmService(
        provider,
        model="main",
        max_tokens=256,
    )

    assert await service.generate_text(prompt="bounded") == "done"
    assert provider.chat.await_args.kwargs["max_tokens"] == 256

    assert await service.generate_text(prompt="provider", max_tokens=0) == "done"
    assert provider.chat.await_args.kwargs["max_tokens"] == 0


@pytest.mark.asyncio
async def test_plugin_llm_reports_binding_and_preserves_explicit_model() -> None:
    runtime = ModelRuntimeConfig(
        runtime_id="a",
        provider="openai",
        model="model-a",
        max_output_tokens=777,
    )
    config = Config(
        provider="openai",
        model=runtime.model,
        api_key="test",
        system_prompt="test",
        runtime_id="a",
        model_runtimes={"a": runtime},
    )
    concrete = AsyncMock()
    concrete.chat.return_value = SimpleNamespace(content="done", usage=None)

    def build(candidate: Config, generation_id: int) -> ModelGeneration:
        return ModelGeneration(
            generation_id=generation_id,
            config_digest=model_config_digest(candidate),
            runtimes=candidate.model_runtimes,
            providers={"a": concrete},
            role_runtime_ids={
                "default": "a",
                "fast": "a",
                "agent": "a",
                "vision": "a",
            },
        )

    service = ProviderPluginLlmService(
        ModelRegistry(config, build).provider("default"),
        model="stale-startup-model",
        max_tokens=0,
    )
    result = await service.generate(prompt="dynamic")
    assert concrete.chat.await_args.kwargs["model"] == "model-a"
    assert concrete.chat.await_args.kwargs["max_tokens"] == 777
    assert result.model_binding["runtime_id"] == "a"

    await service.generate(
        prompt="override",
        model="provider-model-alias",
        max_tokens=128,
    )
    assert concrete.chat.await_args.kwargs["model"] == "provider-model-alias"
    assert concrete.chat.await_args.kwargs["max_tokens"] == 128
