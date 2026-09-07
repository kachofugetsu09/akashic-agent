from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass, field
from enum import StrEnum
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, AsyncContextManager, Literal, Protocol, TypeAlias

from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey

if TYPE_CHECKING:
    from agent.plugin_composition.context import Context
    from agent.plugin_composition.bindings import Bindings


class ModelRole(StrEnum):
    DEFAULT = "default"
    FAST = "fast"
    AGENT = "agent"
    VISION = "vision"


class ModelKind(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"


class ModelAvailability(StrEnum):
    AVAILABLE = "available"
    DISABLED = "disabled"
    DRIVER_UNAVAILABLE = "driver_unavailable"


class UsageCoverage(StrEnum):
    EXACT = "exact"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    request_count: int = 1
    covered_request_count: int = 0
    coverage: UsageCoverage = UsageCoverage.UNAVAILABLE


@dataclass(frozen=True, slots=True)
class ModelCallStats:
    """调用的公开统计；不含凭据、请求正文或 provider continuation。"""

    call_record_id: str
    model: str
    state: Literal["started", "success", "unknown"]
    first_token_ms: float | None
    duration_ms: float | None
    usage: ModelUsage | None


MODEL_CALL_STATS = ServiceKey[Callable[[str], ModelCallStats]]("models.call-stats.v1")


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelContinuation:
    """Opaque driver state bound to one exact BoundModelDescriptor.binding_id."""

    binding_id: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_json_mapping(self.payload))


@dataclass(slots=True)
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    finish_reason: str | None = None
    continuation: ModelContinuation | None = None
    usage: ModelUsage | None = None
    call_record_id: str | None = None


StreamCallback: TypeAlias = Callable[[dict[str, str]], Awaitable[None]]


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

    def __post_init__(self) -> None:
        """在唯一调用边界冻结请求，adapter 和并行调用不能改写彼此输入。"""
        object.__setattr__(
            self, "messages", tuple(_freeze_json_mapping(row) for row in self.messages)
        )
        object.__setattr__(
            self, "tools", tuple(_freeze_json_mapping(row) for row in self.tools)
        )
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
    role: ModelRole
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


class BoundChatModel(Protocol):
    @property
    def descriptor(self) -> BoundModelDescriptor: ...

    async def complete(self, request: ModelRequest) -> LLMResponse:
        """Reject a mismatched continuation before starting external I/O."""

        ...

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


class ModelExecution(Protocol):
    def chat(self, role: ModelRole) -> BoundChatModel: ...


class ChatModels(Protocol):
    def execution(
        self,
        *,
        model_id: str | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncContextManager[ModelExecution]: ...

    def independent_execution(
        self,
        *,
        model_id: str | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncContextManager[ModelExecution]:
        """Open a model execution without a parent task's model binding."""

        ...


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
            raise ModelUnavailableError("已保存 embedding 配置已变化，不能替换原调用")
        async with embeddings.bind(model_id=saved.model_id) as model:
            # 2. open 的 await 期间设置仍可能变化；以真正取得的模型再核对一次。
            if (model.descriptor.identity, model.descriptor.dimensions) != (saved.space_identity, saved.dimensions):
                raise ModelUnavailableError("打开期间 embedding 配置已变化，不能替换原调用")
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
    """Provider evidence that the models plugin may persist under its own ID."""

    kind: ModelKind
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
    role_bindings: Mapping[ModelRole, str]
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


class ModelCatalog(Protocol):
    def snapshot(self) -> ModelCatalogSnapshot: ...

    def validate_chat_selection(
        self,
        selection: ChatModelSelection,
    ) -> ChatModelSelection: ...


@dataclass(frozen=True, slots=True)
class AddConnection:
    expected_revision: int
    connection_id: str
    name: str
    driver_id: str
    endpoint: str
    auth_identity: str
    credential: Mapping[str, str]
    driver_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UpdateConnection:
    expected_revision: int
    connection_id: str
    name: str
    auth_identity: str
    endpoint: str | None = None
    credential: Mapping[str, str] | None = None
    driver_config: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DisableConnection:
    expected_revision: int
    connection_id: str


@dataclass(frozen=True, slots=True)
class AddModel:
    expected_revision: int
    model_id: str
    connection_id: str
    kind: ModelKind
    model: str
    capabilities: ModelCapabilities
    capability_sources: CapabilitySources
    default_reasoning_effort: str | None = None
    driver_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SetDefaultModel:
    expected_revision: int
    role: ModelRole | None
    model_id: str


@dataclass(frozen=True, slots=True)
class SyncModels:
    expected_revision: int
    connection_id: str


@dataclass(frozen=True, slots=True)
class StartConnectionAuth:
    driver_id: str
    connection_id: str
    input: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FinishConnectionAuth:
    expected_revision: int
    attempt_id: str


@dataclass(frozen=True, slots=True)
class CancelConnectionAuth:
    attempt_id: str


@dataclass(frozen=True, slots=True)
class CreateConnectionWithModel:
    """Probe and commit one new connection and its first model atomically."""

    connection: AddConnection
    model: AddModel


ModelChange: TypeAlias = (
    AddConnection
    | UpdateConnection
    | DisableConnection
    | AddModel
    | SetDefaultModel
    | SyncModels
    | StartConnectionAuth
    | FinishConnectionAuth
    | CancelConnectionAuth
    | CreateConnectionWithModel
)


@dataclass(frozen=True, slots=True)
class SettingsReceipt:
    revision: int
    status: str
    attempt_id: str | None = None
    challenge: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.challenge is not None:
            object.__setattr__(
                self,
                "challenge",
                _freeze_json_mapping(self.challenge),
            )


class ModelSettings(Protocol):
    async def discover(self, connection: AddConnection) -> tuple[DiscoveredModel, ...]: ...

    async def apply(self, command: ModelChange) -> SettingsReceipt: ...


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


DriverOpen: TypeAlias = Callable[
    [DriverConnectionDescriptor, CredentialHandle],
    Awaitable[DriverConnection],
]
DriverDiscover: TypeAlias = Callable[
    [DriverConnectionDescriptor, CredentialHandle],
    Awaitable[tuple[DiscoveredModel, ...]],
]
DriverProbe: TypeAlias = Callable[
    [DriverConnectionDescriptor, CredentialHandle],
    Awaitable[None],
]
DriverAuthHandler: TypeAlias = Callable[
    [Mapping[str, Any]],
    Awaitable[Mapping[str, Any]],
]


@dataclass(frozen=True, slots=True)
class ModelDriverDefinition:
    driver_id: str
    contract_version: str
    open: DriverOpen
    discover: DriverDiscover | None = None
    probe: DriverProbe | None = None
    start_auth: DriverAuthHandler | None = None
    finish_auth: DriverAuthHandler | None = None
    cancel_auth: DriverAuthHandler | None = None


class ModelDrivers(Protocol):
    async def register(
        self,
        ctx: Context,
        definition: ModelDriverDefinition,
    ) -> Effect: ...


CHAT_MODELS = ServiceKey[ChatModels]("models.chat.v1")
EMBEDDINGS = ServiceKey[Embeddings]("models.embeddings.v1")
MODEL_CATALOG = ServiceKey[ModelCatalog]("models.catalog.v1")
MODEL_SETTINGS = ServiceKey[ModelSettings]("models.settings.v1")
MODEL_DRIVERS = ServiceKey[ModelDrivers]("models.drivers.v1")


class ModelError(RuntimeError):
    retryable = False


class AuthenticationError(ModelError): ...


class RateLimitError(ModelError):
    retryable = True


class QuotaError(ModelError): ...


class InvalidRequestError(ModelError): ...


class ContextLengthError(ModelError): ...


class ContentSafetyError(ModelError): ...


class ModelTimeoutError(ModelError, TimeoutError):
    retryable = True


class TransportError(ModelError):
    retryable = True


class DriverUnavailableError(ModelError): ...


class ModelUnavailableError(ModelError): ...


class RevisionConflictError(ModelError): ...


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy JSON data into immutable mappings and tuples."""

    active: set[int] = set()

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise ValueError("JSON value 不允许循环引用")
            active.add(identity)
            try:
                frozen: dict[str, Any] = {}
                for key, nested in item.items():
                    if not isinstance(key, str):
                        raise TypeError("JSON object key 必须是字符串")
                    frozen[key] = freeze(nested)
                return MappingProxyType(frozen)
            finally:
                active.remove(identity)
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError("JSON value 不允许循环引用")
            active.add(identity)
            try:
                return tuple(freeze(nested) for nested in item)
            finally:
                active.remove(identity)
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("JSON number 必须是有限值")
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"不支持的 JSON value: {type(item).__name__}")

    return freeze(value)


__all__ = [
    "AddConnection",
    "AddModel",
    "AuthenticationError",
    "BoundChatModel",
    "BoundEmbeddingModel",
    "BoundModelDescriptor",
    "CancelConnectionAuth",
    "CapabilitySources",
    "CHAT_MODELS",
    "ChatModels",
    "ChatModelSelection",
    "ConnectionDescriptor",
    "ContentSafetyError",
    "ContextLengthError",
    "CredentialHandle",
    "DisableConnection",
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
    "FinishConnectionAuth",
    "LLMResponse",
    "MODEL_CATALOG",
    "MODEL_CALL_STATS",
    "ModelCallStats",
    "MODEL_DRIVERS",
    "MODEL_SETTINGS",
    "ModelAvailability",
    "ModelCapabilities",
    "ModelCatalog",
    "ModelCatalogSnapshot",
    "ModelChange",
    "ModelContinuation",
    "ModelDescriptor",
    "ModelDriverDefinition",
    "ModelDrivers",
    "ModelError",
    "ModelExecution",
    "ModelKind",
    "ModelRequest",
    "ModelRole",
    "ModelSettings",
    "ModelTimeoutError",
    "ModelUnavailableError",
    "ModelUsage",
    "InvalidRequestError",
    "QuotaError",
    "RateLimitError",
    "RevisionConflictError",
    "SetDefaultModel",
    "SettingsReceipt",
    "StartConnectionAuth",
    "SyncModels",
    "StreamCallback",
    "ToolCall",
    "TransportError",
    "UpdateConnection",
    "UsageCoverage",
]
