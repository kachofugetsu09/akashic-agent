"""上下文构造与材料贡献合同；持久摘要仍由摘要插件拥有。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from plugins.models.contract import ModelRequest
from plugins.models.contract import BoundChatModel
from plugins.ledger.contract import ContentPart, ContentReferences, HistoryWindow, Input, Message, Output
from plugins.models.contract import ContextModel

MaterialKind = Literal["context", "recall", "profile"]
MaterialData = Mapping[str, object]
SummaryData = Mapping[str, object]
ReductionStatus = Callable[[str, bool], Awaitable[None]]
# 回调第二个参数声明诊断是否随本次 Output 保留；False 只更新临时进度。
Prepare = Callable[[tuple[Message, ...], str], Awaitable[MaterialData]]
# 摘要 owner 证明已连续覆盖的前缀（绑定账本前缀版本）；请求历史可只读其后的窗口，没有可证明的覆盖时为 None。
HistoryStart = Callable[[str], Awaitable[HistoryWindow | None]]


class SummaryReducer(Protocol):
    """摘要 owner 先持久发布再返回；None 表示保留已有摘要，不做缩减。

    summary_limit 限定只有 snapshot 前若干条能进摘要；之后的消息仍属于请求视图，原文保留。
    force bypasses the soft trigger. required=False allows a failed optimization to
    keep the original request only while it still fits the hard capacity.
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
        required: bool = True,
        on_status: ReductionStatus | None = None,
        summary_limit: int | None = None,
    ) -> SummaryData | None: ...


# 摘要 owner 的可选到期判断：只看内存状态，在换完整读面之前调用。snapshot 必须是过滤后的请求视图。
ReductionDue = Callable[[tuple[Message, ...], int, BoundChatModel], bool]


class ContextBuilder(Protocol):
    def check_summary(self, part: ContentPart) -> ContentReferences: ...
    def check_notice(self, part: ContentPart) -> ContentReferences: ...
    def summary_indexes(
        self, snapshot: tuple[Message, ...], source_message_ids: tuple[str, ...]
    ) -> tuple[int, ...]: ...
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
        history_after: int = -1,
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
        history_after: int = -1,
    ) -> tuple[ModelRequest, str | None]: ...


class MaterialView(Protocol):
    """Each prepare call shares the context provider's deadline across contributors.

    The provider defaults to 900 seconds, set by prepare_timeout_seconds, leaving
    room for graph replay, queued learning, and slow disk work. Timeout names the
    material, drains its owner work, and fails without retry. Providers
    must honor cancellation; cleanup may finish after the deadline. Synchronous
    context building and summary reduction keep their own execution contracts.
    """

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
        required: bool = True,
        on_status: ReductionStatus | None = None,
        summary_limit: int | None = None,
    ) -> SummaryData | None: ...
    def reduce_due(self, snapshot: tuple[Message, ...], estimated: int, model: BoundChatModel) -> bool: ...


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
        history_start: HistoryStart | None = None,
        reduce_due: ReductionDue | None = None,
    ) -> Effect: ...
    def bind(
        self, *, exclude_kinds: frozenset[MaterialKind] = frozenset()
    ) -> AbstractAsyncContextManager[MaterialView]: ...
    async def history_start(self, session_id: str) -> HistoryWindow | None: ...


# 参考 Pi 的 reserveTokens：context_window 是输入上限，容量判断只为本次回复留有限余量，
# 不按模型最大输出整段扣除（272K 窗口扣 128K 会让可用输入只剩一半）；小窗口按比例留。
OUTPUT_RESERVE_TOKENS = 16_384


def output_reserve(max_output_tokens: int, context_window: int) -> int:
    return min(max_output_tokens or OUTPUT_RESERVE_TOKENS, OUTPUT_RESERVE_TOKENS, context_window // 4)


# 请求估算达到窗口的这一比例（或 provider 已拒绝）时摘要 owner 才会压缩；低于它的缩减是空操作。
SUMMARY_SOFT_LIMIT = 0.74
# 参考 Claude Code timeBasedMCConfig (gapThresholdMinutes=60)：闲置达到 1 小时后服务端 KV 缓存已过期，
# 在新请求发送前对已超过空闲容量阈值的上下文执行压缩，避免重复冷启动重读长前缀。
IDLE_COMPACT_MINUTES = 60
IDLE_COMPACT_TOKENS = 200_000


# 空闲缺口的两端：上一条非 Input 消息与末尾连续 Input 的第一条。
# 只在末尾是新 Input（本轮尚未产生 Output 或 ToolResult）且前驱不是未完成 Output(continue) 时存在；
# 进入工具循环后末尾为 ToolResult，没有缺口，保证不会在热缓存回合中途触发空闲压缩。
def _idle_gap(snapshot: Sequence[Message]) -> tuple[Message, Message] | None:
    if len(snapshot) < 2 or not isinstance(snapshot[-1].body, Input):
        return None
    index = len(snapshot) - 1
    while index > 0 and isinstance(snapshot[index - 1].body, Input):
        index -= 1
    if index == 0:
        return None
    previous = snapshot[index - 1]
    if isinstance(previous.body, Output) and previous.body.finish == "continue":
        return None
    return previous, snapshot[index]


def idle_gap_minutes(snapshot: Sequence[Message]) -> float | None:
    gap = _idle_gap(snapshot)
    if gap is None:
        return None
    minutes = (gap[1].recorded_at - gap[0].recorded_at).total_seconds() / 60.0
    return minutes if minutes >= 0.0 else 0.0


# 缺口的稳定身份：缺口前最后一条消息。同一缺口上再来的 Input 身份不变。
def idle_gap_anchor(snapshot: Sequence[Message]) -> str | None:
    gap = _idle_gap(snapshot)
    return None if gap is None else gap[0].message_id


def idle_summary_due(
    estimated: int,
    snapshot: Sequence[Message] = (),
    *,
    idle_minutes: int = IDLE_COMPACT_MINUTES,
    idle_tokens: int = IDLE_COMPACT_TOKENS,
) -> bool:
    if idle_minutes <= 0 or idle_tokens <= 0 or estimated < idle_tokens:
        return False
    gap = idle_gap_minutes(snapshot)
    return gap is not None and gap >= idle_minutes


def background_idle_anchor(
    snapshot: Sequence[Message],
    *,
    source: str = "conversation",
    boundary_source: str = "compaction",
) -> str | None:
    """无进行中回合且尚未为当前静默期写入压缩边界时，返回末条会话消息 ID。"""
    if not snapshot:
        return None
    latest_boundary_seq = max(
        (message.seq for message in snapshot if message.source == boundary_source),
        default=-1,
    )
    last_conv = next(
        (message for message in reversed(snapshot) if message.source == source),
        None,
    )
    if last_conv is None or last_conv.seq <= latest_boundary_seq:
        return None
    if isinstance(last_conv.body, Input):
        return None
    if isinstance(last_conv.body, Output) and last_conv.body.finish == "continue":
        return None
    return last_conv.message_id


def background_idle_due(
    estimated: int,
    snapshot: Sequence[Message],
    *,
    now: object,
    idle_minutes: int = IDLE_COMPACT_MINUTES,
    idle_tokens: int = IDLE_COMPACT_TOKENS,
    source: str = "conversation",
    boundary_source: str = "compaction",
) -> bool:
    """后台空闲扫盘判定：对话静默已达阈值且体积达标（未等新消息进来即可先压缩）。"""
    from datetime import datetime

    if idle_minutes <= 0 or idle_tokens <= 0 or estimated < idle_tokens:
        return False
    anchor = background_idle_anchor(snapshot, source=source, boundary_source=boundary_source)
    if anchor is None:
        return False
    last_conv = next((message for message in reversed(snapshot) if message.message_id == anchor), None)
    if last_conv is None or not isinstance(now, datetime):
        return False
    minutes = (now - last_conv.recorded_at).total_seconds() / 60.0
    return minutes >= idle_minutes


def summary_due(
    estimated: int,
    context_window: int | None,
    *,
    force: bool,
    snapshot: Sequence[Message] = (),
    idle_minutes: int = 0,
    idle_tokens: int = 0,
) -> bool:
    if context_window is None:
        return False
    if force or estimated >= int(context_window * SUMMARY_SOFT_LIMIT):
        return True
    return idle_summary_due(
        estimated, snapshot, idle_minutes=idle_minutes, idle_tokens=idle_tokens,
    )


CONTEXT = ServiceKey[ContextBuilder]("context.v2")
MATERIALS_V4 = ServiceKey[ContextMaterialsV4]("context.materials.v4")
# UI.message_display 按 Message 内容 kind 拼接 message.display:<kind> 后借用 renderer。
CONTEXT_NOTICE_DISPLAY = ServiceKey[Callable[[ContentPart], Mapping[str, object]]]("message.display:context.notice")
