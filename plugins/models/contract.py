"""Models 的请求、响应、失败、选择、投影和 driver 公共合同。"""
from __future__ import annotations

import asyncio
import base64
import io
import socket
import ssl

import httpx
from PIL import Image, ImageOps
from plugins.ledger.contract import detect_supported_image_mime
from core.common.frozen_json import freeze_json
from dataclasses import (
    field,
    replace,
    dataclass,
)
from types import MappingProxyType
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import (
    asynccontextmanager,
    AbstractAsyncContextManager,
)
from typing import (
    cast,
    AsyncContextManager,
    Literal,
    TypeAlias,
    Any,
    Protocol,
)
from plugins.ledger.contract import Bindings
from agent.plugin_composition import (
    Context,
    Effect,
)
from plugins.ledger.contract import ArtifactRead
from plugins.channels.contract import AttachmentRef
from agent.plugin_composition.model import ServiceKey
from plugins.ledger.contract import (
    ContentPart,
    ContentReferences,
    Message,
    ToolCall as MessageToolCall,
)


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
        object.__setattr__(self, "arguments", cast(Mapping[str, Any], freeze_json(self.arguments)))


@dataclass(frozen=True, slots=True)
class ModelContinuation:
    """Opaque driver state bound to one exact BoundModelDescriptor.binding_id."""

    binding_id: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", cast(Mapping[str, Any], freeze_json(self.payload)))


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
            object.__setattr__(self, "provider_metadata", cast(Mapping[str, Any], freeze_json(self.provider_metadata)))


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
    request_key: str | None = None
    # 本次完整展示、尚无成功展示回执的消息内容位置；不发送给 provider。
    content_refs: tuple[tuple[str, int], ...] = ()
    content_transformed: bool = False

    @staticmethod
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

    def __post_init__(self) -> None:
        """在唯一调用边界冻结请求，adapter 和并行调用不能改写彼此输入。"""
        object.__setattr__(
            self, "messages", cast(tuple[Mapping[str, Any], ...], freeze_json(
                self.messages if isinstance(self.messages, (list, tuple)) else tuple(self.messages)
            ))
        )
        object.__setattr__(
            self, "tools", cast(tuple[Mapping[str, Any], ...], freeze_json(
                self.tools if isinstance(self.tools, (list, tuple)) else tuple(self.tools)
            ))
        )
        object.__setattr__(self, "content_refs", self.read_content_refs(self.content_refs))
        if type(self.content_transformed) is not bool:
            raise ValueError("内容投影标记必须是 bool")
        if isinstance(self.tool_choice, Mapping):
            object.__setattr__(
                self, "tool_choice", cast(Mapping[str, Any], freeze_json(self.tool_choice))
            )


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    usage: ModelUsage | None = None


ModelKind: TypeAlias = Literal["chat", "embedding"]


ModelAvailability: TypeAlias = Literal["available", "disabled", "driver_unavailable"]


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
            self, "driver_config", cast(Mapping[str, Any], freeze_json(self.driver_config))
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
        object.__setattr__(self, "config", cast(Mapping[str, Any], freeze_json(self.config)))


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


ContentRenderer = Callable[[ContentPart], Sequence[Mapping[str, Any]]]


CallReader = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class RenderedContent:
    """一个内容块的模型视图；complete 只在完整表达该块时为真。"""

    blocks: tuple[Mapping[str, Any], ...]
    complete: bool = False


ContentTransform = Callable[[Message, int], RenderedContent | None]


PrepareContent = Callable[[tuple[Message, ...], str, frozenset[str], frozenset[tuple[str, int]]], ContentTransform]


class ContentViews(Protocol):
    """纯内容投影注册；不授予消息、模型调用或工具执行权限。"""

    async def register(self, ctx: Context, *, name: str, prepare: PrepareContent,
                       dynamic_kinds: frozenset[str] | None = None) -> Effect: ...
    def bind(self) -> AbstractAsyncContextManager[Any]: ...


CONTENT_VIEWS = ServiceKey[ContentViews]("models.content-views.v1")


class ContextModel(Protocol):
    """Model 的只读请求投影；这里没有 complete 或工具执行权。"""

    @property
    def context_window(self) -> int | None: ...
    @property
    def max_tool_schemas(self) -> int | None: ...
    def estimate(self, request: ModelRequest) -> int: ...
    def render(
        self,
        messages: Sequence[Message],
        *,
        after_seq: int,
        summary_reference: str | None = None,
        fresh: bool = False,
        current_reminder: str | None = None,
        current_reminder_input_id: str | None = None,
        current_context: str | None = None,
    ) -> ModelRequest:
        """接收完整事实；after_seq 是摘要覆盖末尾，-1 表示没有覆盖。

        实现方接受任意 Sequence，包括存储签发的 MessageSnapshot；不得修改输入。

        fresh 明确从选定近期窗口开始新请求，不接续旧 opaque 状态。
        summary_reference 明确要求从这份摘要开始新请求；只有同一摘要下的
        后续成功响应才接续 opaque state。只给 after_seq 不授权丢弃 replay。
        current_reminder 与 current_reminder_input_id 成对声明本次材料；同一 Input
        的相同材料保留首次使用位置，新材料追加到本次请求末尾。
        current_context 是不进入历史的实时材料，随当前 reminder 放置；没有 reminder
        时放在本来源最新 Input 后。每次使用当前值，不从旧 Output 恢复。
        """
        ...


class MessageProjection(ContextModel, Protocol):
    def facts(
        self,
        response: LLMResponse,
        call_indices: Sequence[int],
        *,
        reminder: str | None = None,
        reminder_input_id: str | None = None,
        actual_calls: Sequence[MessageToolCall | ContentPart] | None = None,
        content_refs: tuple[tuple[str, int], ...] = (),
        content_transformed: bool = False,
    ) -> ContentPart: ...


class ModelSelection(Protocol):
    def read_saved(self, metadata: Mapping[str, object]) -> ChatModelSelection: ...
    def read(self, messages: Sequence[Message]) -> ChatModelSelection | None: ...
    def check(self, part: ContentPart) -> ContentReferences: ...
    def write_saved(
        self, metadata: MutableMapping[str, object], selection: ChatModelSelection
    ) -> None: ...


class ModelContent(Protocol):
    def render(
        self,
        part: ContentPart,
        *,
        artifacts: Mapping[str, tuple[Mapping[str, Any], ...]],
        read_message: Callable[[str], Message | None] | None = None,
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def load_artifacts(
        self,
        reader: ArtifactRead,
        refs: Sequence[AttachmentRef],
        *,
        accepts_images: bool,
        current_artifact_ids: frozenset[str],
    ) -> Mapping[str, tuple[Mapping[str, Any], ...]]: ...


class ModelChecks(Protocol):
    def check_facts(self, part: ContentPart) -> ContentReferences: ...
    def check_tool_rejection(self, part: ContentPart) -> ContentReferences: ...


class ModelProjections(Protocol):
    def create(
        self,
        model: BoundChatModel,
        *,
        source: str,
        render_content: ContentRenderer,
        tool_name: Callable[[str], str],
        read_call: CallReader,
        check_summary: Callable[[ContentPart], ContentReferences],
        keep_input_ids: tuple[str, ...] = (),
        prepare_content: PrepareContent | None = None,
        tool_names: frozenset[str] = frozenset(),
        dynamic_content_kinds: frozenset[str] | None = frozenset(),
    ) -> MessageProjection: ...


MODEL_SELECTION = ServiceKey[ModelSelection]("models.selection.v1")


MODEL_CONTENT = ServiceKey[ModelContent]("models.content.v3")


MODEL_CHECKS = ServiceKey[ModelChecks]("models.message-checks.v1")


MODEL_PROJECTION = ServiceKey[ModelProjections]("models.projection.v1")


MODEL_CALLS = ServiceKey[CallReader]("models.calls.v1")


@dataclass(frozen=True, slots=True)
class ModelCallStats:
    """调用的公开统计；不含凭据、请求正文或 provider continuation。"""

    call_record_id: str
    model: str
    state: Literal["started", "success", "error"]
    first_token_ms: float | None
    duration_ms: float | None
    usage: ModelUsage | None


MODEL_CALL_STATS = ServiceKey[Callable[[str], ModelCallStats]]("models.call-stats.v1")


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

    def key_recovery(self, request_key: str) -> str:
        """该 request key 最近耐久记录的恢复裁决（Models 独占分类）：

        - "open"：无终结结算（无记录、成功、在途或仍有耐久退避额度）；
        - "rejected"：provider 明确容量拒绝——本代可续跑有界缩减，
          真实 resume 也可开新准备；
        - "answered"：其他可证明失败——真实 resume 后允许新准备如实付费；
        - "uncertain"：旧记录或没有恢复安排的未知失败；自动重入不重发，
          新 Input 或显式 resume 可授权新的模型准备，工具效果仍按原回执恢复。

        非 "open" 即终结：终结 key 不因重启/重调获得新预算。"""
        ...


class ModelExecution(Protocol):
    def chat(self, role: str) -> BoundChatModel: ...


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


class ModelCatalog(Protocol):
    def snapshot(self) -> ModelCatalogSnapshot: ...

    def validate_chat_selection(
        self,
        selection: ChatModelSelection,
    ) -> ChatModelSelection: ...


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
    probe_embedding: Callable[[DriverConnectionDescriptor, CredentialHandle, str], Awaitable[DiscoveredModel]] | None = None
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


MODEL_CATALOG = ServiceKey[ModelCatalog]("models.catalog.v1")


MODEL_DRIVERS = ServiceKey[ModelDrivers]("models.drivers.v1")


class BoundEmbeddingModel(Protocol):
    @property
    def descriptor(self) -> EmbeddingSpaceDescriptor: ...

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...


@dataclass(frozen=True, slots=True)
class SavedEmbedding:
    """已归档的模型与空间；恢复不能重新选择另一个空间。"""

    model_id: str
    space_identity: str
    dimensions: int

    def __post_init__(self) -> None:
        if (not isinstance(self.model_id, str) or not self.model_id
                or not isinstance(self.space_identity, str) or not self.space_identity
                or type(self.dimensions) is not int or self.dimensions <= 0):
            raise ValueError("保存的 embedding 需要模型、空间与正整数维度")

    @classmethod
    def read(cls, bindings: Bindings, identity: str) -> SavedEmbedding:
        """在持久表示边界校验原三个字段，返回同一种冻结值。"""
        saved = bindings.describe(identity, EMBEDDINGS)
        if set(saved) != {"model_id", "space_identity", "dimensions"}:
            raise ValueError("保存的 embedding 字段不完整或包含未知字段")
        return cls(cast(str, saved["model_id"]), cast(str, saved["space_identity"]),
                   cast(int, saved["dimensions"]))

    @classmethod
    @asynccontextmanager
    async def open(cls, bindings: Bindings, identity: str) -> AsyncGenerator[BoundEmbeddingModel]:
        """先核对归档空间，再借用实际 driver；打开期间漂移也明确拒绝。"""
        saved = cls.read(bindings, identity)
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

MAX_IMAGE_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_TOTAL_BYTES = 40 * 1024 * 1024
MAX_IMAGE_DATA_URI_BYTES = 8 * 1024 * 1024
MAX_IMAGE_DATA_URI_TOTAL_BYTES = 16 * 1024 * 1024
MAX_IMAGE_EDGE = 4096
MAX_IMAGE_PIXELS = 40_000_000


def validate_image_attachment_budget(sizes: list[int]) -> None:
    """Bound raw image bytes across a model request, including retained history."""

    oversized = next((size for size in sizes if size > MAX_IMAGE_FILE_BYTES), None)
    if oversized is not None:
        raise ValueError(
            f"单张图片不能超过 {MAX_IMAGE_FILE_BYTES // 1024 // 1024}MB"
            f"（当前 {oversized / 1024 / 1024:.1f}MB）。"
        )
    total = sum(sizes)
    if total > MAX_IMAGE_TOTAL_BYTES:
        raise ValueError(
            f"模型请求中的图片（含历史消息）合计不能超过 "
            f"{MAX_IMAGE_TOTAL_BYTES // 1024 // 1024}MB"
            f"（当前 {total / 1024 / 1024:.1f}MB）。"
        )


def encode_image_bytes(raw: bytes) -> str:
    """从 Artifact 的已核验只读 bytes 构造有界请求图，不重新打开来源路径。"""
    validate_image_attachment_budget([len(raw)])
    mime = detect_supported_image_mime(raw[:4096])
    if mime is None:
        raise ValueError("不支持的图片格式。仅支持 PNG、JPEG、GIF、BMP、WebP。")

    try:
        with Image.open(io.BytesIO(raw)) as image:
            _validate_image_pixels(image.width, image.height)
            image.verify()

        with Image.open(io.BytesIO(raw)) as image:
            _validate_image_pixels(image.width, image.height)
            image = ImageOps.exif_transpose(image)
            if image.mode not in ("RGB", "L"):
                canvas = Image.new("RGB", image.size, (255, 255, 255))
                alpha = image.getchannel("A") if "A" in image.getbands() else None
                canvas.paste(image.convert("RGB"), mask=alpha)
                image = canvas
            elif image.mode == "L":
                image = image.convert("RGB")

            raw_b64_len = _base64_encoded_size(len(raw))
            if max(image.size) > MAX_IMAGE_EDGE or raw_b64_len > MAX_IMAGE_DATA_URI_BYTES:
                image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))

            if raw_b64_len <= MAX_IMAGE_DATA_URI_BYTES and max(image.size) <= MAX_IMAGE_EDGE:
                buf = io.BytesIO()
                if mime == "image/jpeg":
                    image.save(buf, format="JPEG", quality=95, optimize=True)
                    clean_mime = "image/jpeg"
                else:
                    image.save(buf, format="PNG", optimize=True)
                    clean_mime = "image/png"
                clean_bytes = buf.getvalue()
                if _base64_encoded_size(len(clean_bytes)) <= MAX_IMAGE_DATA_URI_BYTES:
                    clean_b64 = base64.b64encode(clean_bytes).decode()
                    return f"data:{clean_mime};base64,{clean_b64}"

            compressed_b64_len = 0
            for quality in (85, 75, 65, 55, 45):
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=quality, optimize=True)
                candidate_bytes = buf.getvalue()
                compressed_b64_len = _base64_encoded_size(len(candidate_bytes))
                if compressed_b64_len <= MAX_IMAGE_DATA_URI_BYTES:
                    candidate_b64 = base64.b64encode(candidate_bytes).decode()
                    return f"data:image/jpeg;base64,{candidate_b64}"
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("图片文件无法解码或已损坏。请确认这是有效图片。") from exc

    raise ValueError(
        f"图片压缩后仍然过大（{compressed_b64_len / 1024 / 1024:.1f}MB base64），"
        f"上限为 {MAX_IMAGE_DATA_URI_BYTES / 1024 / 1024:.0f}MB。"
        "请继续压缩图片或裁剪到只包含需要分析的区域。"
    )


def _base64_encoded_size(size: int) -> int:
    """每 3 个输入字节编码为 4 个字符，末组不足时补齐。"""
    return ((size + 2) // 3) * 4


def _validate_image_pixels(width: int, height: int) -> None:
    """在像素解码前拒绝会显著放大内存的图片。"""

    pixels = width * height
    if pixels > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"图片像素过多（{width}×{height}），"
            f"上限为 {MAX_IMAGE_PIXELS // 1_000_000} 百万像素。"
            "请缩小图片或裁剪后重试。"
        )

def describe_transport_error(error: Exception) -> str:
    """展示网络阶段与已知底层原因；不包含请求正文、URL 路径或凭据。"""
    # 1. 异常类型说明失败阶段，未发送证据仍由 driver 单独给出。
    if isinstance(error, httpx.ConnectTimeout):
        reason = "连接模型服务超时，请检查网络、代理或节点"
    elif isinstance(error, httpx.ConnectError):
        reason = "无法连接模型服务，请检查网络、代理或节点"
    elif isinstance(error, httpx.ReadTimeout):
        reason = "等待模型服务响应超时"
    elif isinstance(error, httpx.RemoteProtocolError):
        reason = "模型服务提前关闭连接或返回了无效 HTTP 响应"
    elif isinstance(error, httpx.ReadError):
        reason = "读取模型响应失败，连接已中断"
    elif isinstance(error, httpx.WriteError):
        reason = "发送模型请求失败，连接已中断"
    elif isinstance(error, httpx.WriteTimeout):
        reason = "向模型服务发送请求超时"
    elif isinstance(error, httpx.PoolTimeout):
        reason = "等待可用 HTTP 连接超时"
    elif isinstance(error, (httpx.TimeoutException, TimeoutError)):
        reason = "模型请求超时"
    else:
        reason = "与模型服务的传输中断"
    # 2. 只透传系统网络诊断，不直接打印可能含 token 的异常文本。
    details = [type(error).__name__]
    cause: BaseException | None = error
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, ssl.SSLCertVerificationError):
            details.append(f"TLS 证书校验失败：{cause.verify_message}")
        elif isinstance(cause, socket.gaierror):
            details.append(f"DNS 解析失败：{cause.strerror}")
        elif isinstance(cause, OSError) and cause.strerror:
            details.append(f"{type(cause).__name__}: {cause.strerror}")
        cause = cause.__cause__ or cause.__context__
    return f"{reason}（{'；'.join(dict.fromkeys(details))}）"
