from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent.config_models import Config, ModelRuntimeConfig
from agent.model_runtime.registry import (
    ModelGeneration,
    ModelRegistry,
    current_model_binding,
    model_config_digest,
)
from agent.model_runtime.catalog.litellm_registry import resolve_catalog_capabilities
from agent.model_runtime.catalog.litellm_registry import resolve_catalog_provider_id
from agent.model_runtime.types import LLMResponse
from agent.model_runtime.usage import normalize_provider_usage
from agent.provider import _normalize_chat_usage


class _Provider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[str] = []
        self.last_kwargs: dict[str, object] = {}

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.last_kwargs = dict(kwargs)
        self.calls.append(str(kwargs["model"]))
        return LLMResponse(content=self.name)


def _config(default: str = "a", fast: str = "fast-a") -> Config:
    runtimes = {
        runtime_id: ModelRuntimeConfig(
            runtime_id=runtime_id,
            provider="openai",
            model=f"model-{runtime_id}",
            max_output_tokens=4096 if runtime_id == "a" else 8192,
        )
        for runtime_id in ("a", "b", "fast-a", "fast-b")
    }
    return Config(
        provider="openai",
        model=runtimes[default].model,
        api_key="test",
        system_prompt="test",
        runtime_id=default,
        model_runtimes=runtimes,
        fast_runtime_id=fast,
    )


def _builder(config: Config, generation_id: int) -> ModelGeneration:
    providers = {
        runtime_id: _Provider(f"generation-{generation_id}:{runtime_id}")
        for runtime_id in config.model_runtimes
    }
    return ModelGeneration(
        generation_id=generation_id,
        config_digest=model_config_digest(config),
        runtimes=dict(config.model_runtimes),
        providers=providers,
        role_runtime_ids={
            "default": config.runtime_id,
            "fast": config.fast_runtime_id or config.runtime_id,
            "agent": config.agent_runtime_id or config.runtime_id,
            "vision": config.vl_runtime_id or config.runtime_id,
        },
    )


@pytest.mark.asyncio
async def test_running_execution_keeps_generation_and_next_execution_uses_reload() -> None:
    registry = ModelRegistry(_config(), _builder)
    provider = registry.provider("default")

    async with registry.execution_scope() as binding:
        assert binding.describe()["runtime_id"] == "a"
        first = await provider.chat([], [], "ignored", 0)
        await registry.reload(_config(default="b", fast="fast-b"))
        second = await provider.chat([], [], "ignored", 0)

    third = await provider.chat([], [], "ignored", 0)
    assert [first.content, second.content, third.content] == [
        "generation-1:a",
        "generation-1:a",
        "generation-2:b",
    ]
    assert registry.current.generation_id == 2


@pytest.mark.asyncio
async def test_role_provider_resolves_dynamic_default_limit_and_local_override() -> None:
    registry = ModelRegistry(_config(), _builder)
    provider = registry.provider("default", force_disable_thinking=True)

    await provider.chat([], [], "ignored", 0)
    first_concrete = registry.current.providers["a"]
    assert first_concrete.calls == ["model-a"]
    assert first_concrete.last_kwargs["max_tokens"] == 4096
    assert first_concrete.last_kwargs["disable_thinking"] is True

    await registry.reload(_config(default="b"))
    await provider.chat([], [], "ignored", 0)
    second_concrete = registry.current.providers["b"]
    assert second_concrete.last_kwargs["max_tokens"] == 8192

    await provider.chat([], [], "ignored", 512)
    assert second_concrete.last_kwargs["max_tokens"] == 512


@pytest.mark.asyncio
async def test_execution_not_started_at_reload_uses_new_generation() -> None:
    registry = ModelRegistry(_config(), _builder)
    provider = registry.provider("fast")
    release = asyncio.Event()

    async def queued() -> str | None:
        await release.wait()
        response = await provider.chat([], [], "ignored", 0)
        return response.content

    task = asyncio.create_task(queued())
    await registry.reload(_config(default="b", fast="fast-b"))
    release.set()
    assert await task == "generation-2:fast-b"


@pytest.mark.asyncio
async def test_explicit_session_runtime_survives_default_reload() -> None:
    registry = ModelRegistry(_config(), _builder)
    provider = registry.provider("agent")

    async with registry.execution_scope("b") as binding:
        assert current_model_binding() is binding
        await registry.reload(_config(default="b"))
        response = await provider.chat([], [], "ignored", 0)
        assert response.content == "generation-1:b"
    assert current_model_binding() is None

    response = await provider.chat([], [], "ignored", 0)
    assert response.content == "generation-2:b"


@pytest.mark.asyncio
async def test_explicit_session_effort_is_scoped_to_selected_chat_model() -> None:
    config = _config()
    runtimes = dict(config.model_runtimes)
    runtimes["b"] = replace(
        runtimes["b"],
        reasoning_effort="medium",
        supported_reasoning_efforts=("low", "medium", "high"),
    )
    registry = ModelRegistry(replace(config, model_runtimes=runtimes), _builder)
    agent = registry.provider("agent")
    fast = registry.provider("fast")

    async with registry.execution_scope("b", "high"):
        await agent.chat([], [], "ignored", 0)
        await fast.chat([], [], "ignored", 0)

    assert registry.current.providers["b"].last_kwargs["extra_body"] == {
        "reasoning_effort": "high"
    }
    assert registry.current.providers["fast-a"].last_kwargs["extra_body"] == {}


@pytest.mark.asyncio
async def test_explicit_session_effort_rejects_unsupported_value() -> None:
    config = _config()
    runtimes = dict(config.model_runtimes)
    runtimes["b"] = replace(
        runtimes["b"],
        supported_reasoning_efforts=("low", "high"),
    )
    registry = ModelRegistry(replace(config, model_runtimes=runtimes), _builder)

    with pytest.raises(ValueError, match="不支持推理强度"):
        async with registry.execution_scope("b", "medium"):
            pass


@pytest.mark.asyncio
async def test_invalid_candidate_does_not_replace_current_generation() -> None:
    def builder(config: Config, generation_id: int) -> ModelGeneration:
        if config.system_prompt == "broken":
            raise ValueError("candidate failed")
        return _builder(config, generation_id)

    registry = ModelRegistry(_config(), builder)
    with pytest.raises(ValueError, match="candidate failed"):
        await registry.reload(replace(_config(default="b"), system_prompt="broken"))
    assert registry.current.generation_id == 1
    assert registry.current.role_runtime_ids["default"] == "a"


@pytest.mark.asyncio
async def test_nested_scope_reuses_binding_and_rejects_conflict() -> None:
    registry = ModelRegistry(_config(), _builder)
    async with registry.execution_scope("a") as outer:
        async with registry.execution_scope() as inner:
            assert inner is outer
        with pytest.raises(RuntimeError, match="显式模型绑定冲突"):
            async with registry.execution_scope("b"):
                pass

    with pytest.raises(ValueError, match="模型 runtime 不存在"):
        async with registry.execution_scope("missing"):
            pass


def test_litellm_registry_resolves_caps_and_keeps_unknown_explicit() -> None:
    known = resolve_catalog_capabilities("openai", "gpt-5.2-pro")
    assert known is not None
    assert known.context_window == 400_000
    assert known.max_output_tokens == 128_000
    assert known.input_modalities == ("text", "image")
    assert known.input_modalities_known is True
    assert known.supported_reasoning_efforts == (
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    )
    assert known.source == "litellm"
    assert resolve_catalog_capabilities("unknown", "future-model") is None
    assert (
        resolve_catalog_provider_id(
            "openai",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
        )
        == "deepseek"
    )
    assert (
        resolve_catalog_provider_id(
            "unknown",
            model="future-model",
            base_url="https://api.deepseek.com.evil.invalid/v1",
        )
        == ""
    )


def test_genai_prices_normalizes_cache_read_and_write_fields() -> None:
    usage = normalize_provider_usage(
        {
            "model": "openai/gpt-4.1",
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_write_tokens": 12,
                },
            },
        },
        provider_id="openrouter",
        provider_api_url="https://openrouter.ai/api/v1",
        api_flavor="chat",
    )
    assert usage is not None
    assert usage.input_tokens == 120
    assert usage.cache_write_input_tokens == 12
    assert usage.cached_input_tokens == 80
    assert usage.output_tokens == 30
    assert usage.coverage.value == "exact"


def test_genai_prices_unknown_provider_does_not_fake_usage() -> None:
    usage = normalize_provider_usage(
        {"usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        provider_id="private-gateway",
        provider_api_url="https://llm.internal.invalid/v1",
        api_flavor="chat",
    )
    assert usage is None


def test_usage_normalizer_preserves_deepseek_cache_hit_extension() -> None:
    usage = _normalize_chat_usage(
        {
            "model": "deepseek-chat",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_cache_hit_tokens": 70,
                "prompt_cache_miss_tokens": 30,
            },
        },
        provider_id="deepseek",
        provider_api_url="https://api.deepseek.com/v1",
    )
    assert usage is not None
    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 70
    assert usage.output_tokens == 20
    assert usage.coverage.value == "exact"


def test_usage_aggregate_preserves_partial_requests() -> None:
    from agent.model_runtime.types import ModelUsage, UsageCoverage
    from agent.model_runtime.usage import aggregate_usage

    usage = aggregate_usage(
        [
            ModelUsage(
                input_tokens=10,
                coverage=UsageCoverage.PARTIAL,
                request_count=1,
            )
        ]
    )
    assert usage.input_tokens == 10
    assert usage.covered_request_count == 0
    assert usage.coverage is UsageCoverage.PARTIAL
