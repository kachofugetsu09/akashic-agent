"""上下文构造与材料贡献合同；持久摘要仍由摘要插件拥有。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from plugins.models.contract import ModelRequest
from plugins.models.contract import BoundChatModel
from plugins.ledger.contract import ContentPart, ContentReferences, Message
from plugins.models.contract import ContextModel

MaterialKind = Literal["context", "recall", "profile"]
MaterialData = Mapping[str, object]
SummaryData = Mapping[str, object]
ReductionStatus = Callable[[str, bool], Awaitable[None]]
# 回调第二个参数声明诊断是否随本次 Output 保留；False 只更新临时进度。
Prepare = Callable[[tuple[Message, ...], str], Awaitable[MaterialData]]


class SummaryReducer(Protocol):
    """摘要 owner 先持久发布再返回；None 表示保留已有摘要，不做缩减。

    summary_limit 限定只有 snapshot 前若干条能进摘要；之后的消息仍属于请求视图，原文保留。
    """

    async def __call__(
        self,
        snapshot: tuple[Message, ...],
        materials: MaterialData,
        request: ModelRequest,
        model: BoundChatModel,
        projection: ContextModel,
        *,
        source: str,
        force: bool,
        on_status: ReductionStatus | None = None,
        summary_limit: int | None = None,
    ) -> SummaryData | None: ...


class ContextBuilder(Protocol):
    def check_summary(self, part: ContentPart) -> ContentReferences: ...
    def check_notice(self, part: ContentPart) -> ContentReferences: ...
    def summary_range(
        self, snapshot: tuple[Message, ...], source_message_ids: tuple[str, ...]
    ) -> range: ...
    def settled_prefixes(self, messages: tuple[Message, ...]) -> tuple[int, ...]: ...
    def reminder_content(self, materials: MaterialData) -> str | None: ...
    def build(
        self,
        snapshot: Sequence[Message],
        *,
        materials: MaterialData,
        model: ContextModel,
        tools: Sequence[Mapping[str, Any]] = (),
        max_output_tokens: int,
        window_start: str | None = None,
        current_reminder_input_id: str | None = None,
    ) -> ModelRequest: ...
    def build_attempt(
        self,
        snapshot: Sequence[Message],
        *,
        materials: MaterialData,
        model: ContextModel,
        tools: Sequence[Mapping[str, Any]] = (),
        max_output_tokens: int,
        window_start: str | None = None,
        current_reminder_input_id: str | None = None,
    ) -> tuple[ModelRequest, str | None]: ...


class MaterialView(Protocol):
    async def prepare(
        self,
        snapshot: tuple[Message, ...],
        source: str,
        *,
        caller: Context | None = None,
        reminders: tuple[Mapping[str, object], ...] = (),
    ) -> MaterialData: ...
    async def reduce(
        self,
        snapshot: tuple[Message, ...],
        materials: MaterialData,
        request: ModelRequest,
        model: BoundChatModel,
        projection: ContextModel,
        *,
        source: str,
        force: bool,
        on_status: ReductionStatus | None = None,
        summary_limit: int | None = None,
    ) -> SummaryData | None: ...


class ContextMaterialsV4(Protocol):
    """注册 Effect 归贡献者；bind 固定并保护本次实际使用的材料。"""

    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        prepare: Prepare,
        kind: MaterialKind,
        priority: int = 0,
        prompt: bool = False,
        reduce: SummaryReducer | None = None,
    ) -> Effect: ...
    def bind(
        self, *, exclude_kinds: frozenset[MaterialKind] = frozenset()
    ) -> AbstractAsyncContextManager[MaterialView]: ...


# 参考 Pi 的 reserveTokens：context_window 是输入上限，容量判断只为本次回复留有限余量，
# 不按模型最大输出整段扣除（272K 窗口扣 128K 会让可用输入只剩一半）；小窗口按比例留。
OUTPUT_RESERVE_TOKENS = 16_384


def output_reserve(max_output_tokens: int, context_window: int) -> int:
    return min(max_output_tokens or OUTPUT_RESERVE_TOKENS, OUTPUT_RESERVE_TOKENS, context_window // 4)


CONTEXT = ServiceKey[ContextBuilder]("context.v2")
MATERIALS_V4 = ServiceKey[ContextMaterialsV4]("context.materials.v4")
# UI.message_display 按 Message 内容 kind 拼接 message.display:<kind> 后借用 renderer。
CONTEXT_NOTICE_DISPLAY = ServiceKey[Callable[[ContentPart], Mapping[str, object]]]("message.display:context.notice")
