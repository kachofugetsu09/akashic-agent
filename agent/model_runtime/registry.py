from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

from agent.config_models import Config, ModelRuntimeConfig
from agent.model_runtime.store import ModelRegistryStore
from agent.provider import LLMProvider


@dataclass
class ModelGeneration:
    """Hold one immutable runtime/provider publication and its active leases."""

    generation_id: int
    config_digest: str
    runtimes: Mapping[str, ModelRuntimeConfig]
    providers: Mapping[str, Any]
    role_runtime_ids: Mapping[str, str]
    registry_revision: int = 0
    role_providers: Mapping[str, Any] = field(default_factory=dict)
    lease_count: int = 0
    retired: bool = False

    def resolve(
        self,
        role: str,
        explicit_runtime_id: str | None,
    ) -> tuple[str, ModelRuntimeConfig, Any]:
        runtime_id = (
            explicit_runtime_id
            if explicit_runtime_id and role in {"default", "agent"}
            else self.role_runtime_ids[role]
        )
        runtime = self.runtimes[runtime_id]
        provider = self.role_providers.get(role) or self.providers[runtime_id]
        if explicit_runtime_id and role in {"default", "agent"}:
            provider = self.providers[runtime_id]
        return runtime_id, runtime, provider


@dataclass(frozen=True)
class ModelExecutionBinding:
    registry: ModelRegistry
    generation: ModelGeneration
    explicit_runtime_id: str | None
    explicit_reasoning_effort: str = ""

    def describe(self, role: str = "agent") -> dict[str, object]:
        runtime_id, runtime, _provider = self.generation.resolve(
            role,
            self.explicit_runtime_id,
        )
        return {
            "generation_id": self.generation.generation_id,
            "config_digest": self.generation.config_digest,
            "role": role,
            "runtime_id": runtime_id,
            "provider": runtime.provider,
            "model": runtime.model,
            "reasoning_effort": self.reasoning_effort_for(role, runtime),
        }

    def reasoning_effort_for(
        self,
        role: str,
        runtime: ModelRuntimeConfig,
    ) -> str:
        """Resolve session effort only for the explicitly selected chat model."""

        if self.explicit_runtime_id and role in {"default", "agent"}:
            return self.explicit_reasoning_effort or runtime.reasoning_effort
        return runtime.reasoning_effort


_CURRENT_BINDING: ContextVar[ModelExecutionBinding | None] = ContextVar(
    "model_execution_binding",
    default=None,
)


GenerationBuilder = Callable[[Config, int], ModelGeneration]


class ModelRegistry:
    """Build, publish, and lease immutable model generations."""

    def __init__(self, config: Config, builder: GenerationBuilder) -> None:
        self._builder = builder
        self._lock = asyncio.Lock()
        self._reload_lock = asyncio.Lock()
        self._next_generation_id = 1
        self._current = builder(config, self._next_generation_id)
        self._retired: dict[int, ModelGeneration] = {}
        self._config_path = config.config_path
        self._workspace_path = config.workspace_path
        self._store = ModelRegistryStore.for_workspace(self._workspace_path)

    @property
    def current(self) -> ModelGeneration:
        return self._current

    def provider(
        self,
        role: str,
        *,
        force_disable_thinking: bool = False,
        honor_session_selection: bool = True,
    ) -> RoleBoundProvider:
        if role not in self._current.role_runtime_ids:
            raise KeyError(f"未知模型角色: {role}")
        return RoleBoundProvider(
            self,
            role,
            force_disable_thinking=force_disable_thinking,
            honor_session_selection=honor_session_selection,
        )

    def has_runtime(self, runtime_id: str) -> bool:
        return runtime_id in self._current.runtimes and not runtime_id.startswith(
            "__role_"
        )

    def list_runtimes(self) -> list[dict[str, object]]:
        generation = self._current
        roles_by_runtime: dict[str, list[str]] = {}
        for role, runtime_id in generation.role_runtime_ids.items():
            roles_by_runtime.setdefault(runtime_id, []).append(role)
        return [
            {
                "id": runtime.runtime_id,
                "provider": runtime.provider,
                "catalogProvider": runtime.catalog_provider_id or runtime.provider,
                "model": runtime.model,
                "reasoningEffort": runtime.reasoning_effort,
                "supportedReasoningEfforts": list(
                    runtime.supported_reasoning_efforts
                ),
                "sourceId": runtime.source_id,
                "sourceName": runtime.source_name or runtime.provider,
                "contextWindow": runtime.context_window,
                "maxOutputTokens": runtime.max_output_tokens,
                "inputModalities": list(runtime.input_modalities),
                "capabilitySource": runtime.capability_source,
                "capabilitySources": {
                    "contextWindow": runtime.context_window_source,
                    "maxOutputTokens": runtime.max_output_tokens_source,
                    "inputModalities": runtime.input_modalities_source,
                },
                "roles": sorted(roles_by_runtime.get(runtime.runtime_id, [])),
            }
            for runtime in generation.runtimes.values()
            if not runtime.runtime_id.startswith("__role_")
        ]

    async def reload(self, config: Config) -> ModelGeneration:
        """Build a complete candidate, then atomically publish it."""

        # 1. Serialize reloads so every published generation has a unique id.
        async with self._reload_lock:
            generation_id = self._next_generation_id + 1
            candidate = self._builder(config, generation_id)

            # 2. A short critical section publishes one complete generation.
            async with self._lock:
                previous = self._current
                self._current = candidate
                self._next_generation_id = generation_id
                previous.retired = True
                if previous.lease_count:
                    self._retired[previous.generation_id] = previous
        return candidate

    async def _refresh_from_store_if_changed(self) -> None:
        """Publish the latest committed database revision before a new scope."""

        # 1. The revision check is cheap and never mutates the current generation.
        revision = self._store.revision()
        if revision == 0 or revision == self._current.registry_revision:
            return

        # 2. Serialize candidate construction and recheck after waiting for peers.
        async with self._reload_lock:
            revision = self._store.revision()
            if revision == self._current.registry_revision:
                return
            config = Config.load(
                self._config_path,
                workspace=self._workspace_path,
            )
            generation_id = self._next_generation_id + 1
            candidate = self._builder(config, generation_id)

            # 3. Publish one complete generation; active leases retain the old one.
            async with self._lock:
                previous = self._current
                self._current = candidate
                self._next_generation_id = generation_id
                previous.retired = True
                if previous.lease_count:
                    self._retired[previous.generation_id] = previous

    async def refresh(self) -> ModelGeneration:
        """Refresh from canonical storage and return the visible generation."""

        await self._refresh_from_store_if_changed()
        return self._current

    @asynccontextmanager
    async def execution_scope(
        self,
        explicit_runtime_id: str | None = None,
        explicit_reasoning_effort: str = "",
    ) -> AsyncIterator[ModelExecutionBinding]:
        """Freeze one generation for a complete execution unit."""

        existing = _CURRENT_BINDING.get()
        if existing is not None:
            if existing.registry is not self:
                raise RuntimeError("同一执行不能绑定两个 ModelRegistry")
            if (
                explicit_runtime_id is not None
                and explicit_runtime_id != existing.explicit_runtime_id
            ):
                raise RuntimeError("嵌套执行的显式模型绑定冲突")
            if (
                explicit_reasoning_effort
                and explicit_reasoning_effort != existing.explicit_reasoning_effort
            ):
                raise RuntimeError("嵌套执行的推理强度绑定冲突")
            yield existing
            return

        # 1. A new execution observes the latest committed database revision.
        await self._refresh_from_store_if_changed()

        # 2. Snapshot and validate before publishing the context-local binding.
        generation = self._current
        if explicit_runtime_id and explicit_runtime_id not in generation.runtimes:
            raise ValueError(f"模型 runtime 不存在: {explicit_runtime_id}")
        if explicit_reasoning_effort and not explicit_runtime_id:
            raise ValueError("显式推理强度必须绑定显式模型 runtime")
        if explicit_reasoning_effort and explicit_runtime_id:
            runtime = generation.runtimes[explicit_runtime_id]
            supported = runtime.supported_reasoning_efforts
            if supported and explicit_reasoning_effort not in supported:
                raise ValueError(
                    f"模型 {runtime.model} 不支持推理强度: {explicit_reasoning_effort}"
                )
        generation.lease_count += 1
        binding = ModelExecutionBinding(
            self,
            generation,
            explicit_runtime_id,
            explicit_reasoning_effort,
        )
        token = _CURRENT_BINDING.set(binding)
        try:
            yield binding
        finally:
            # 3. Release only this scope's lease; retired generations drain naturally.
            _CURRENT_BINDING.reset(token)
            generation.lease_count -= 1
            if generation.lease_count < 0:
                raise RuntimeError("模型 generation lease 计数为负")
            if generation.retired and generation.lease_count == 0:
                self._retired.pop(generation.generation_id, None)

    def resolve(
        self,
        role: str,
        *,
        honor_session_selection: bool = True,
    ) -> tuple[ModelRuntimeConfig, Any, ModelExecutionBinding | None]:
        binding = _CURRENT_BINDING.get()
        if binding is not None and binding.registry is self:
            _runtime_id, runtime, provider = binding.generation.resolve(
                role,
                binding.explicit_runtime_id if honor_session_selection else None,
            )
            return runtime, provider, binding
        _runtime_id, runtime, provider = self._current.resolve(role, None)
        return runtime, provider, None


class RoleBoundProvider(LLMProvider):
    """Preserve the LLMProvider surface while resolving through a registry role."""

    def __init__(
        self,
        registry: ModelRegistry,
        role: str,
        *,
        force_disable_thinking: bool = False,
        honor_session_selection: bool = True,
    ) -> None:
        self.registry = registry
        self.role = role
        self.force_disable_thinking = force_disable_thinking
        self.honor_session_selection = honor_session_selection

    def _resolved_runtime(self) -> ModelRuntimeConfig:
        runtime, _provider, _binding = self.registry.resolve(
            self.role,
            honor_session_selection=self.honor_session_selection,
        )
        return runtime

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        max_tokens: int,
        tool_choice: str | dict = "auto",
        extra_body: dict | None = None,
        disable_thinking: bool = False,
        on_content_delta: Callable[[Any], Awaitable[None]] | None = None,
        cache_namespace: str = "",
    ) -> Any:
        async with self.registry.execution_scope():
            runtime, provider, binding = self.registry.resolve(
                self.role,
                honor_session_selection=self.honor_session_selection,
            )
            request_extra = dict(extra_body or {})
            if binding is not None:
                effort = (
                    binding.reasoning_effort_for(self.role, runtime)
                    if self.honor_session_selection
                    else runtime.reasoning_effort
                )
                if effort and not self.force_disable_thinking and not disable_thinking:
                    request_extra["reasoning_effort"] = effort
            return await provider.chat(
                messages=messages,
                tools=tools,
                model=runtime.model,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                extra_body=request_extra,
                disable_thinking=self.force_disable_thinking or disable_thinking,
                on_content_delta=on_content_delta,
                cache_namespace=cache_namespace,
            )

    @property
    def context_window(self) -> int:
        return self._resolved_runtime().context_window

    @property
    def runtime_id(self) -> str:
        """Return the runtime selected by the current frozen generation."""

        return self._resolved_runtime().runtime_id

    @property
    def model(self) -> str:
        """Return the model selected by the current frozen generation."""

        return self._resolved_runtime().model

    @property
    def max_output_tokens(self) -> int:
        """Return the configured output budget for the current runtime."""

        return self._resolved_runtime().max_output_tokens

    def estimate_context_tokens(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        _runtime, provider, _binding = self.registry.resolve(
            self.role,
            honor_session_selection=self.honor_session_selection,
        )
        return int(provider.estimate_context_tokens(messages, tools))

    def estimate_appended_message_tokens(self, messages: list[dict]) -> int:
        _runtime, provider, _binding = self.registry.resolve(
            self.role,
            honor_session_selection=self.honor_session_selection,
        )
        return int(provider.estimate_appended_message_tokens(messages))

    def __getattr__(self, name: str) -> Any:
        _runtime, provider, _binding = self.registry.resolve(
            self.role,
            honor_session_selection=self.honor_session_selection,
        )
        try:
            return getattr(provider, name)
        except AttributeError:
            primary = getattr(provider, "primary", None)
            if primary is None:
                raise
            return getattr(primary, name)


@asynccontextmanager
async def model_execution_scope(
    provider: object,
    explicit_runtime_id: str | None = None,
    explicit_reasoning_effort: str = "",
) -> AsyncIterator[ModelExecutionBinding | None]:
    """Lease a model generation when the provider is registry-backed."""

    if not isinstance(provider, RoleBoundProvider):
        yield None
        return
    async with provider.registry.execution_scope(
        explicit_runtime_id,
        explicit_reasoning_effort,
    ) as binding:
        yield binding


def current_model_binding() -> ModelExecutionBinding | None:
    return _CURRENT_BINDING.get()


def model_config_digest(config: Config) -> str:
    """Return a secret-free digest of runtime and role semantics."""

    payload = {
        "runtimes": {
            runtime_id: {
                "provider": runtime.provider,
                "model": runtime.model,
                "source_id": runtime.source_id,
                "source_name": runtime.source_name,
                "catalog_provider_id": runtime.catalog_provider_id,
                "auth": runtime.auth,
                "base_url": runtime.base_url,
                "reasoning_effort": runtime.reasoning_effort,
                "supported_reasoning_efforts": list(
                    runtime.supported_reasoning_efforts
                ),
                "context_window": runtime.context_window,
                "max_output_tokens": runtime.max_output_tokens,
                "input_modalities": list(runtime.input_modalities),
                "capability_source": runtime.capability_source,
                "context_window_source": runtime.context_window_source,
                "max_output_tokens_source": runtime.max_output_tokens_source,
                "input_modalities_source": runtime.input_modalities_source,
            }
            for runtime_id, runtime in sorted(config.model_runtimes.items())
        },
        "roles": {
            "default": config.runtime_id,
            "fast": config.fast_runtime_id or config.runtime_id,
            "agent": config.agent_runtime_id or config.runtime_id,
            "vision": config.vl_runtime_id or config.runtime_id,
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
