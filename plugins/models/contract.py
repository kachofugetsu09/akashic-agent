"""模型选择、内容和投影的公共合同；不绑定默认模型插件。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import cast
from agent.plugin_composition.bindings import Bindings

from agent.plugin_composition.models import (
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
)

from collections.abc import Awaitable
from typing import (
    AsyncContextManager,
    Literal,
    TypeAlias,
)

from agent.plugin_composition.models import (
    BoundModelDescriptor,
    CredentialHandle,
    DiscoveredModel,
    DriverConnection,
    DriverConnectionDescriptor,
    ModelCatalogSnapshot,
    ModelUsage,
)

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from agent.plugin_composition import Context, Effect
from agent.plugin_composition.artifacts import ArtifactRead
from agent.plugin_composition.channels import AttachmentRef
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.models import (
    ChatModelSelection,
    LLMResponse,
    ModelRequest,
)
from agent.plugin_contracts import ContentPart, ContentReferences, Message, ToolCall

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
        actual_calls: Sequence[ToolCall | ContentPart] | None = None,
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
