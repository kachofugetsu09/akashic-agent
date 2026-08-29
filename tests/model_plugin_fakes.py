from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.plugin_composition import (
    BoundModelDescriptor,
    CapabilitySources,
    ContextLengthError,
    LLMResponse,
    ModelCapabilities,
    ModelContinuation,
    ModelRequest,
    ModelRole,
    ModelUsage,
    ToolCall,
    UsageCoverage,
)
from agent.plugin_composition import CHAT_MODELS, MODEL_CATALOG
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from agent.provider import ContextLengthError as LegacyContextLengthError


_MODEL_PROVIDERS: dict[Path, object] = {}


class BoundChatModelFake:
    """Expose an existing test provider through the public chat-model contract."""

    def __init__(
        self,
        provider: object,
        *,
        model_id: str | None = None,
        model: str | None = None,
        role: ModelRole = ModelRole.AGENT,
    ) -> None:
        self.provider = provider
        wire_model = model or str(getattr(provider, "model", "test-model"))
        runtime_id = model_id or str(getattr(provider, "runtime_id", "test-runtime"))
        context_window = int(getattr(provider, "context_window", 0))
        max_output_tokens = getattr(provider, "max_output_tokens", None)
        input_modalities = tuple(
            getattr(provider, "input_modalities", ("text",))
        )
        self._descriptor = BoundModelDescriptor(
            binding_id=f"test-binding:{id(provider)}:{runtime_id}:{wire_model}",
            plugin_snapshot_id="test-plugin-snapshot",
            model_revision=1,
            model_id=runtime_id,
            connection_id="test-connection",
            driver_id="test-driver",
            driver_contract_version="1",
            auth_identity="test",
            model=wire_model,
            role=role,
            reasoning_effort=None,
            capabilities=ModelCapabilities(
                context_window=context_window or None,
                max_output_tokens=(
                    max_output_tokens if isinstance(max_output_tokens, int) else None
                ),
                input_modalities=input_modalities,
            ),
            capability_sources=CapabilitySources(),
            capability_digest="test-capabilities",
        )

    @property
    def descriptor(self) -> BoundModelDescriptor:
        return self._descriptor

    @property
    def max_tool_schemas(self) -> int | None:
        value = getattr(self.provider, "max_tool_schemas", None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        return int(self.provider.estimate_context_tokens(list(messages), list(tools)))

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return int(self.provider.estimate_appended_message_tokens(list(messages)))

    async def complete(self, request: ModelRequest) -> LLMResponse:
        kwargs = {
            "messages": list(request.messages),
            "tools": list(request.tools),
            "model": self.descriptor.model,
            "max_tokens": request.max_output_tokens,
            "tool_choice": request.tool_choice,
            "disable_thinking": request.disable_reasoning,
            "on_content_delta": request.on_delta,
            "cache_namespace": request.prompt_cache_key,
        }
        if request.continuation is not None:
            kwargs["model_state"] = dict(request.continuation.payload)
        try:
            response = await self.provider.chat(**kwargs)
        except LegacyContextLengthError as exc:
            raise ContextLengthError(str(exc)) from exc
        continuation = None
        provider_fields = getattr(response, "provider_fields", {})
        state = provider_fields.get("model_state") if isinstance(provider_fields, dict) else None
        if isinstance(state, dict):
            continuation = ModelContinuation(
                binding_id=self.descriptor.binding_id,
                payload=state,
            )
        usage = _public_usage(getattr(response, "usage", None))
        return LLMResponse(
            content=response.content,
            tool_calls=[
                ToolCall(item.id, item.name, dict(item.arguments))
                for item in response.tool_calls
            ],
            thinking=response.thinking,
            finish_reason=response.finish_reason,
            continuation=continuation,
            cache_prompt_tokens=response.cache_prompt_tokens,
            cache_hit_tokens=response.cache_hit_tokens,
            usage=usage,
        )


def _public_usage(value: object) -> ModelUsage | None:
    if value is None:
        return None
    coverage = UsageCoverage(str(getattr(value, "coverage", "unavailable")))
    return ModelUsage(
        input_tokens=getattr(value, "input_tokens", None),
        cache_write_input_tokens=getattr(value, "cache_write_input_tokens", None),
        cached_input_tokens=getattr(value, "cached_input_tokens", None),
        output_tokens=getattr(value, "output_tokens", None),
        reasoning_output_tokens=getattr(value, "reasoning_output_tokens", None),
        request_count=int(getattr(value, "request_count", 1)),
        covered_request_count=int(getattr(value, "covered_request_count", 0)),
        coverage=coverage,
    )


class _TestModelCatalog:
    def validate_chat_selection(self, selection: object) -> object:
        return selection


class _TestModelExecution:
    def __init__(self, provider: object) -> None:
        self.agent = BoundChatModelFake(provider)
        self.default = BoundChatModelFake(provider, role=ModelRole.DEFAULT)

    def chat(self, role: ModelRole) -> BoundChatModelFake:
        return self.default if role is ModelRole.DEFAULT else self.agent


class _TestChatModels:
    def __init__(self, provider: object) -> None:
        self.provider = provider
        self.execution_calls = 0

    @asynccontextmanager
    async def execution(self, **_selection: object):
        self.execution_calls += 1
        yield _TestModelExecution(self.provider)


def build_test_chat_models(provider: object) -> object:
    return _TestChatModels(provider)


def register_test_model_provider(workspace: Path, provider: object) -> None:
    """Register one deterministic provider for an ordinary test plugin Root."""

    _MODEL_PROVIDERS[workspace.resolve()] = provider


def unregister_test_model_provider(workspace: Path) -> None:
    _MODEL_PROVIDERS.pop(workspace.resolve(), None)


async def provide_test_model_services(ctx: object) -> None:
    """Publish public model facades from a test plugin's own apply hook."""

    workspace = Path(ctx.runtime.workspace).resolve()  # type: ignore[attr-defined]
    provider = _MODEL_PROVIDERS.get(workspace)
    if provider is None:
        raise RuntimeError(f"test model provider 未注册: {workspace}")
    _ = await ctx.provide(CHAT_MODELS, _TestChatModels(provider))  # type: ignore[attr-defined]
    _ = await ctx.provide(MODEL_CATALOG, _TestModelCatalog())  # type: ignore[attr-defined]


@contextmanager
def bind_test_model_snapshot(
    provider: object,
    *,
    chat_models: object | None = None,
) -> Iterator[None]:
    """Bind the two public model services for AgentLoop contract tests."""

    snapshot = build_test_model_snapshot(provider, chat_models=chat_models)
    lease = SimpleNamespace(
        active=True,
        snapshot=snapshot,
        validation_candidate_plugin_ids=frozenset(),
    )
    token = bind_runtime_snapshot(lease)
    try:
        yield
    finally:
        reset_runtime_snapshot(token)


def build_test_model_snapshot(
    provider: object,
    *,
    chat_models: object | None = None,
) -> object:
    services = {
        CHAT_MODELS: chat_models or _TestChatModels(provider),
        MODEL_CATALOG: _TestModelCatalog(),
    }

    async def serial(*_args: object, **_kwargs: object) -> None:
        return None

    async def observe(*_args: object, **_kwargs: object) -> None:
        return None

    context = SimpleNamespace(
        require=lambda key: services[key],
        serial=serial,
        observe=observe,
        emit=lambda *_args, **_kwargs: None,
    )
    return SimpleNamespace(
        snapshot_id="test-plugin-snapshot",
        composition_root=SimpleNamespace(context=context),
        command_registry=None,
        tool_registry=None,
        plugin_skill_index=None,
    )


class _TestSnapshotLease:
    def __init__(self, snapshot: object) -> None:
        self.active = True
        self.snapshot = snapshot
        self.validation_candidate_plugin_ids = frozenset()

    async def __aenter__(self) -> object:
        return self.snapshot

    async def __aexit__(self, *_exc: object) -> None:
        self.active = False

    def fork(self) -> _TestSnapshotLease:
        return _TestSnapshotLease(self.snapshot)

    async def release(self) -> None:
        self.active = False


def build_test_model_store(
    provider: object,
    *,
    chat_models: object | None = None,
) -> object:
    snapshot = build_test_model_snapshot(provider, chat_models=chat_models)

    async def acquire(*_args: object, **_kwargs: object) -> _TestSnapshotLease:
        return _TestSnapshotLease(snapshot)

    return SimpleNamespace(current=snapshot, acquire=acquire)
