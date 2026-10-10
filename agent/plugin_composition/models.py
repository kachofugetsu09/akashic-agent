from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from agent.plugin_contracts.message import freeze_json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, AsyncContextManager, Literal, Protocol, TypeAlias, cast

from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey

if TYPE_CHECKING:
    from agent.plugin_composition.context import Context
    from agent.plugin_composition.bindings import Bindings


ModelKind: TypeAlias = Literal["chat", "embedding"]
ModelAvailability: TypeAlias = Literal["available", "disabled", "driver_unavailable"]
UsageCoverage: TypeAlias = Literal["exact", "partial", "unavailable"]


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    request_count: int = 1
    covered_request_count: int = 0
    coverage: UsageCoverage = "unavailable"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze_json_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class ModelContinuation:
    """Opaque driver state bound to one exact BoundModelDescriptor.binding_id."""

    binding_id: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_json_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str | None
    tool_calls: Sequence[ToolCall] = ()
    thinking: str | None = None
    finish_reason: str | None = None
    continuation: ModelContinuation | None = None
    usage: ModelUsage | None = None
    call_record_id: str | None = None
    # 原生响应部件由调用账本保存，随同一 binding 的消息重放。
    provider_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """冻结一次响应，多个请求等待者不能相互改变已结算事实。"""
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.provider_metadata is not None:
            if not isinstance(self.provider_metadata, Mapping):
                raise TypeError("响应协议 metadata 必须是 JSON 对象")
            object.__setattr__(self, "provider_metadata", _freeze_json_mapping(self.provider_metadata))


StreamCallback: TypeAlias = Callable[[dict[str, str]], Awaitable[None]]


def read_content_refs(value: object) -> tuple[tuple[str, int], ...]:
    """在请求与持久记录边界校验同一种消息内容位置编码。"""
    if not isinstance(value, (tuple, list)):
        raise ValueError("内容位置必须是数组")
    refs: list[tuple[str, int]] = []
    for ref in value:
        if (not isinstance(ref, (tuple, list)) or len(ref) != 2
                or not isinstance(ref[0], str) or not ref[0]
                or type(ref[1]) is not int or ref[1] < 0):
            raise ValueError("内容位置需要消息 ID 和非负整数索引")
        refs.append((ref[0], ref[1]))
    if len(set(refs)) != len(refs):
        raise ValueError("请求内容位置不能重复")
    return tuple(refs)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: Sequence[Mapping[str, Any]]
    tools: Sequence[Mapping[str, Any]] = ()
    max_output_tokens: int = 0
    system_prompt: str = ""
    tool_choice: str | Mapping[str, Any] = "auto"
    prompt_cache_key: str | None = None
    on_delta: StreamCallback | None = None
    continuation: ModelContinuation | None = None
    disable_reasoning: bool = False
    request_key: str | None = None
    # 本次完整展示、尚无成功展示回执的消息内容位置；不发送给 provider。
    content_refs: tuple[tuple[str, int], ...] = ()
    content_transformed: bool = False

    def __post_init__(self) -> None:
        """在唯一调用边界冻结请求，adapter 和并行调用不能改写彼此输入。"""
        object.__setattr__(
            self, "messages", _freeze_json_rows(self.messages)
        )
        object.__setattr__(
            self, "tools", _freeze_json_rows(self.tools)
        )
        object.__setattr__(self, "content_refs", read_content_refs(self.content_refs))
        if type(self.content_transformed) is not bool:
            raise ValueError("内容投影标记必须是 bool")
        if isinstance(self.tool_choice, Mapping):
            object.__setattr__(
                self, "tool_choice", _freeze_json_mapping(self.tool_choice)
            )


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    context_window: int | None = None
    max_output_tokens: int | None = None
    input_modalities: tuple[str, ...] = ("text",)
    supports_tool_calls: bool | None = None
    supports_parallel_tool_calls: bool | None = None
    supported_reasoning_efforts: tuple[str, ...] = ()
    embedding_dimensions: int | None = None
    embedding_normalization: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilitySources:
    context_window: str = "unknown"
    max_output_tokens: str = "unknown"
    input_modalities: str = "unknown"
    tool_calls: str = "unknown"
    parallel_tool_calls: str = "unknown"
    reasoning_efforts: str = "unknown"
    embedding_dimensions: str = "unknown"
    embedding_normalization: str = "unknown"


@dataclass(frozen=True, slots=True)
class BoundModelDescriptor:
    binding_id: str
    plugin_snapshot_id: str
    model_revision: int
    model_id: str
    connection_id: str
    driver_id: str
    driver_contract_version: str
    auth_identity: str
    model: str
    role: str
    reasoning_effort: str | None
    capabilities: ModelCapabilities
    capability_sources: CapabilitySources
    capability_digest: str


@dataclass(frozen=True, slots=True)
class EmbeddingSpaceDescriptor:
    plugin_snapshot_id: str
    model_revision: int
    model_id: str
    connection_id: str
    driver_id: str
    driver_contract_version: str
    auth_identity: str
    connection_fingerprint: str
    model: str
    dimensions: int
    normalization: str
    capability_digest: str
    schema_version: int = 1

    @property
    def identity(self) -> str:
        return ":".join(
            (
                self.driver_id,
                self.driver_contract_version,
                self.connection_id,
                self.auth_identity,
                self.connection_fingerprint,
                self.model_id,
                str(self.dimensions),
                self.normalization,
                self.capability_digest,
                str(self.schema_version),
            )
        )


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    usage: ModelUsage | None = None


class DriverChatModel(Protocol):
    async def complete(self, request: ModelRequest) -> LLMResponse: ...

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int: ...

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int: ...

    @property
    def max_tool_schemas(self) -> int | None: ...


class DriverEmbeddingModel(Protocol):
    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...


@dataclass(frozen=True, slots=True)
class ConnectionDescriptor:
    connection_id: str
    name: str
    driver_id: str
    auth_identity: str
    availability: ModelAvailability


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    model_id: str
    connection_id: str
    kind: ModelKind
    model: str
    default_reasoning_effort: str | None
    capabilities: ModelCapabilities
    capability_sources: CapabilitySources
    availability: ModelAvailability


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """服务返回的模型候选；None 表示尚无用途证据，不能直接持久化。"""

    kind: ModelKind | None
    model: str
    capabilities: ModelCapabilities
    capability_sources: CapabilitySources
    default_reasoning_effort: str | None = None
    driver_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "driver_config", _freeze_json_mapping(self.driver_config)
        )


@dataclass(frozen=True, slots=True)
class ChatModelSelection:
    model_id: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class ModelCatalogSnapshot:
    revision: int
    connections: tuple[ConnectionDescriptor, ...]
    models: tuple[ModelDescriptor, ...]
    role_bindings: Mapping[str, str]
    default_embedding_model_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "role_bindings",
            MappingProxyType(dict(self.role_bindings)),
        )

    def connection(self, connection_id: str) -> ConnectionDescriptor:
        """Return one connection from this exact catalog revision."""

        for connection in self.connections:
            if connection.connection_id == connection_id:
                return connection
        raise KeyError(connection_id)

    def model(self, model_id: str) -> ModelDescriptor:
        """Return one model from this exact catalog revision."""

        for model in self.models:
            if model.model_id == model_id:
                return model
        raise KeyError(model_id)


class CredentialHandle(Protocol):
    @property
    def connection_id(self) -> str: ...

    @property
    def auth_identity(self) -> str: ...

    async def read(self) -> Mapping[str, str]: ...

    async def refresh(self, payload: Mapping[str, str]) -> None: ...

    def exclusive(self) -> AsyncContextManager[None]: ...


@dataclass(frozen=True, slots=True)
class DriverConnectionDescriptor:
    connection_id: str
    name: str
    driver_id: str
    endpoint: str
    auth_identity: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _freeze_json_mapping(self.config))


@dataclass(frozen=True, slots=True)
class DriverConnection:
    bind_chat: Callable[
        [BoundModelDescriptor, Mapping[str, Any]],
        DriverChatModel,
    ]
    bind_embedding: Callable[
        [EmbeddingSpaceDescriptor, Mapping[str, Any]],
        DriverEmbeddingModel,
    ]

    close: Callable[[], Awaitable[None]] | None = None

    async def aclose(self) -> None:
        """等待资源关闭完成，再向调用方传回取消。"""
        if self.close is not None:
            # 1. 独立任务保护关闭过程，避免重复取消截断底层资源释放。
            task = asyncio.ensure_future(self.close())
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
            # 2. 关闭失败必须可见；关闭成功后恢复调用方的取消。
            task.result()
            if cancelled:
                raise asyncio.CancelledError


def _freeze_json_rows(value: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """接纳普通 Sequence；内部深冻结数组不再逐行遍历。"""
    rows = value if isinstance(value, (list, tuple)) else tuple(value)
    return cast(tuple[Mapping[str, Any], ...], freeze_json(rows))


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """模型请求与消息共用同一 JSON 冻结边界。"""
    return cast(Mapping[str, Any], freeze_json(value))


__all__ = [
    "BoundModelDescriptor",
    "CapabilitySources",
    "ChatModelSelection",
    "ConnectionDescriptor",
    "CredentialHandle",
    "DiscoveredModel",
    "DriverConnection",
    "DriverConnectionDescriptor",
    "DriverChatModel",
    "DriverEmbeddingModel",
    "EmbeddingResult",
    "EmbeddingSpaceDescriptor",
    "LLMResponse",
    "ModelAvailability",
    "ModelCapabilities",
    "ModelCatalogSnapshot",
    "ModelContinuation",
    "ModelDescriptor",
    "ModelKind",
    "ModelRequest",
    "ModelUsage",
    "StreamCallback",
    "ToolCall",
    "UsageCoverage",
]
