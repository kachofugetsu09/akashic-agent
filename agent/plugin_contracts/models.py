"""模型选择、内容和投影的公共合同；不绑定默认模型插件。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from agent.plugin_composition import Context, Effect
from agent.plugin_composition.channels import AttachmentRef
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.models import (
    BoundChatModel,
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

    async def register(self, ctx: Context, *, name: str, prepare: PrepareContent) -> Effect: ...
    def bind(self) -> AbstractAsyncContextManager[PrepareContent]: ...


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
        messages: tuple[Message, ...],
        *,
        after_seq: int,
        summary_reference: str | None = None,
        fresh: bool = False,
        current_reminder: str | None = None,
        current_reminder_input_id: str | None = None,
    ) -> ModelRequest:
        """接收完整事实；after_seq 是摘要覆盖末尾，-1 表示没有覆盖。

        fresh 明确从选定近期窗口开始新请求，不接续旧 opaque 状态。
        summary_reference 明确要求从这份摘要开始新请求；只有同一摘要下的
        后续成功响应才接续 opaque state。只给 after_seq 不授权丢弃 replay。
        current_reminder 与 current_reminder_input_id 成对声明本次尾部材料；投影
        只折叠同一 Input、同一材料身份的已保存 reminder，其他历史事实照常重放。
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

    def describe_artifacts(
        self,
        refs: Sequence[AttachmentRef],
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
    ) -> MessageProjection: ...


MODEL_SELECTION = ServiceKey[ModelSelection]("models.selection.v1")
MODEL_CONTENT = ServiceKey[ModelContent]("models.content.v2")
MODEL_CHECKS = ServiceKey[ModelChecks]("models.message-checks.v1")
MODEL_PROJECTION = ServiceKey[ModelProjections]("models.projection.v1")
MODEL_CALLS = ServiceKey[CallReader]("models.calls.v1")
