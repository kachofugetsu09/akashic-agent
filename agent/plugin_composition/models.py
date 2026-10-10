from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pydantic import BaseModel, ConfigDict, Field
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





class BoundEmbeddingModel(Protocol):
    @property
    def descriptor(self) -> EmbeddingSpaceDescriptor: ...

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...


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








class SavedEmbedding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    model_id: str = Field(min_length=1)
    space_identity: str = Field(min_length=1)
    dimensions: int = Field(gt=0)


def read_embedding_binding(bindings: Bindings, identity: str) -> SavedEmbedding:
    return SavedEmbedding.model_validate(dict(bindings.describe(identity, EMBEDDINGS)))


@asynccontextmanager
async def open_embedding(bindings: Bindings, identity: str) -> AsyncGenerator[BoundEmbeddingModel]:
    """在归档 Root 核对当前同名模型，配置漂移时在远程调用前拒绝。"""
    saved = read_embedding_binding(bindings, identity)
    async with bindings.open(identity, EMBEDDINGS) as (embeddings, _metadata):
        # 1. driver.open 可能联网，先拒绝已经变化的 endpoint、身份或空间。
        descriptor = embeddings.describe(model_id=saved.model_id)
        if (descriptor.identity, descriptor.dimensions) != (saved.space_identity, saved.dimensions):
            raise ModelUnavailableError("已保存 embedding 配置已变化，不能替换原调用").exception()
        async with embeddings.bind(model_id=saved.model_id) as model:
            # 2. open 的 await 期间设置仍可能变化；以真正取得的模型再核对一次。
            if (model.descriptor.identity, model.descriptor.dimensions) != (saved.space_identity, saved.dimensions):
                raise ModelUnavailableError("打开期间 embedding 配置已变化，不能替换原调用").exception()
            yield model


class Embeddings(Protocol):
    def save_binding(self, bindings: Bindings, *, model_id: str | None = None) -> str:
        """固定所选模型、空间与实际 driver 代码，不归档凭据。"""
        ...

    def describe(
        self,
        *,
        model_id: str | None = None,
    ) -> EmbeddingSpaceDescriptor:
        """描述一个稳定向量空间，不打开远程连接。"""

        ...

    def bind(
        self,
        *,
        model_id: str | None = None,
    ) -> AsyncContextManager[BoundEmbeddingModel]: ...


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















EMBEDDINGS = ServiceKey[Embeddings]("models.embeddings.v1")




@dataclass(frozen=True, slots=True)
class ModelError:
    """模型失败的冻结事实；普通异常只负责把它传过调用栈。"""

    message: str
    retryable: bool = False
    retry_at: float | None = None
    # rejected 和 unsent 是发送边界的正面证据；None 保留远端效果未知。
    send_evidence: str | None = None
    response_delta_seen: bool = False
    retry_safe: bool = False
    retry_after: float | None = None

    def __str__(self) -> str:
        return self.message

    def exception(self) -> Exception:
        """用标准异常传递冻结值，超时仍保留标准 TimeoutError 语义。"""
        return TimeoutError(self) if isinstance(self, ModelTimeoutError) else RuntimeError(self)

    @classmethod
    def read(cls, error: BaseException, *kinds: type[ModelError]) -> ModelError | None:
        """只读取明确的模型失败载荷；未知程序错误不能取得恢复语义。"""
        if not isinstance(error, (RuntimeError, TimeoutError)) or len(error.args) != 1:
            return None
        value = error.args[0]
        return value if isinstance(value, cls) and (not kinds or isinstance(value, kinds)) else None

    @classmethod
    def matches(cls, error: BaseException, *kinds: type[ModelError]) -> bool:
        return cls.read(error, *kinds) is not None

    @classmethod
    def change(cls, error: BaseException, **changes: Any) -> Exception:
        """构造补充边界事实的新失败，不改变原错误与已结算值。"""
        value = cls.read(error)
        if value is None:
            raise TypeError("异常不是模型失败")
        return replace(value, **changes).exception()


@dataclass(frozen=True, slots=True)
class AuthenticationError(ModelError): ...


@dataclass(frozen=True, slots=True)
class RateLimitError(ModelError):
    retryable: bool = True


@dataclass(frozen=True, slots=True)
class QuotaError(ModelError): ...


@dataclass(frozen=True, slots=True)
class InvalidRequestError(ModelError): ...


@dataclass(frozen=True, slots=True)
class ContextLengthError(ModelError): ...


@dataclass(frozen=True, slots=True)
class ContentSafetyError(ModelError): ...


@dataclass(frozen=True, slots=True)
class ModelTimeoutError(ModelError):
    retryable: bool = True


@dataclass(frozen=True, slots=True)
class TransportError(ModelError):
    retryable: bool = True


@dataclass(frozen=True, slots=True)
class EmptyResponseError(ModelError):
    """模型调用成功，但没有可提交的正文或工具调用。"""

    retryable: bool = True


@dataclass(frozen=True, slots=True)
class OutputLengthError(ModelError):
    """模型达到生成长度限制；正文或工具参数可能不完整。"""


@dataclass(frozen=True, slots=True)
class DriverUnavailableError(ModelError): ...


@dataclass(frozen=True, slots=True)
class ModelUnavailableError(ModelError): ...


@dataclass(frozen=True, slots=True)
class ModelControlUnavailable(ModelError):
    """本次服务作用域没有模型管理能力。"""


@dataclass(frozen=True, slots=True)
class RevisionConflictError(ModelError): ...


def _freeze_json_rows(value: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """接纳普通 Sequence；内部深冻结数组不再逐行遍历。"""
    rows = value if isinstance(value, (list, tuple)) else tuple(value)
    return cast(tuple[Mapping[str, Any], ...], freeze_json(rows))


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """模型请求与消息共用同一 JSON 冻结边界。"""
    return cast(Mapping[str, Any], freeze_json(value))


__all__ = [
    "AuthenticationError",
    "BoundEmbeddingModel",
    "BoundModelDescriptor",
    "CapabilitySources",
    "ChatModelSelection",
    "ConnectionDescriptor",
    "ContentSafetyError",
    "ContextLengthError",
    "CredentialHandle",
    "DiscoveredModel",
    "DriverConnection",
    "DriverConnectionDescriptor",
    "DriverChatModel",
    "DriverEmbeddingModel",
    "DriverUnavailableError",
    "EMBEDDINGS",
    "EmbeddingResult",
    "SavedEmbedding",
    "read_embedding_binding",
    "open_embedding",
    "Embeddings",
    "EmbeddingSpaceDescriptor",
    "LLMResponse",
    "ModelAvailability",
    "ModelCapabilities",
    "ModelCatalogSnapshot",
    "ModelContinuation",
    "ModelDescriptor",
    "ModelError",
    "ModelKind",
    "ModelRequest",
    "ModelTimeoutError",
    "ModelUnavailableError",
    "ModelUsage",
    "InvalidRequestError",
    "QuotaError",
    "RateLimitError",
    "RevisionConflictError",
    "StreamCallback",
    "ToolCall",
    "TransportError",
    "UsageCoverage",
]
