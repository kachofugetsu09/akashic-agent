from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from agent.plugin_composition import (
    BoundModelDescriptor,
    BoundChatModel,
    CapabilitySources,
    ChatModels,
    CompositionRoot,
    LLMResponse,
    ModelCapabilities,
    ModelContinuation,
    ModelExecution,
    ModelRequest,
    ModelRole,
    ModelUnavailableError,
    ModelUsage,
    ServiceKey,
    ToolCall,
    UsageCoverage,
)
from agent.plugin_composition import CHAT_MODELS, MODEL_CATALOG
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)

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
        input_modalities = tuple(getattr(provider, "input_modalities", ("text",)))
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
        continuation = request.continuation
        if (
            continuation is not None
            and continuation.binding_id != self.descriptor.binding_id
        ):
            raise ModelUnavailableError("continuation 不属于当前 model binding")
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
        if continuation is not None:
            kwargs["model_state"] = dict(continuation.payload)
        response = await self.provider.chat(**kwargs)
        continuation = None
        response_continuation = getattr(response, "continuation", None)
        if isinstance(response_continuation, ModelContinuation):
            continuation = ModelContinuation(
                binding_id=self.descriptor.binding_id,
                payload=response_continuation.payload,
            )
        provider_fields = getattr(response, "provider_fields", {})
        state = (
            provider_fields.get("model_state")
            if isinstance(provider_fields, dict)
            else None
        )
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
        self._chat = {
            role: BoundChatModelFake(provider, role=role) for role in ModelRole
        }

    def chat(self, role: ModelRole) -> BoundChatModel:
        return self._chat[role]


class _TestChatModels:
    def __init__(self, provider: object) -> None:
        self.provider = provider
        self.execution_calls = 0

    @asynccontextmanager
    async def execution(
        self,
        *,
        model_id: str | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[ModelExecution]:
        del model_id, reasoning_effort
        self.execution_calls += 1
        yield _TestModelExecution(self.provider)


def build_test_chat_models(provider: object) -> _TestChatModels:
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


@asynccontextmanager
async def bind_test_model_snapshot(
    provider: object,
    *,
    chat_models: object | None = None,
) -> AsyncIterator[None]:
    """Bind the two public model services for AgentLoop contract tests."""

    store = build_test_model_store(provider, chat_models=chat_models)
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        yield
    finally:
        reset_runtime_snapshot(token)
        await lease.release()


def build_test_model_snapshot(
    provider: object,
    *,
    chat_models: object | None = None,
) -> RuntimeSnapshot:
    root = _TestCompositionRoot("test-model-snapshot")
    root.provide_test_service(
        CHAT_MODELS,
        chat_models or _TestChatModels(provider),
    )
    root.provide_test_service(MODEL_CATALOG, _TestModelCatalog())
    return RuntimeSnapshotCompiler().compile(
        {},
        snapshot_revision="test-model-snapshot",
        composition_root=root,
    )


class _TestCompositionRoot(CompositionRoot):
    """Build a real snapshot root with deterministic model services."""

    def provide_test_service(
        self,
        key: ServiceKey[object],
        value: object,
    ) -> None:
        self._register_provider(key, value, self.root_fiber)


def build_test_model_store(
    provider: object,
    *,
    chat_models: object | None = None,
) -> RuntimeSnapshotStore:
    snapshot = build_test_model_snapshot(provider, chat_models=chat_models)
    store = RuntimeSnapshotStore()
    store.install(snapshot)
    return store
