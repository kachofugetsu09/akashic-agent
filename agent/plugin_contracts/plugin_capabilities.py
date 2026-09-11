"""插件之间共享能力的公开结构合同（key + 消费者可见的 Protocol）。

这些能力由某个插件提供、其它插件消费。消费者不应 import 提供方的实现模块，
因此 key 与「消费者实际调用的方法子集」由合同层拥有，实现留在各插件。

约定：Protocol 只声明**真实调用过**的方法。新增消费者方法时，必须同时在这里
补声明，否则类型检查会失败——这是刻意的，避免合同层脱离实际使用面膨胀。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.events import EmitEventKey
from agent.plugin_contracts.context import Reminder
from agent.plugin_contracts.delivery_api import Receipt
from agent.plugin_contracts.message import Message

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager, AbstractContextManager

    from agent.plugin_composition.models import StreamCallback
    from agent.plugin_composition.tasks import Task
    from agent.plugin_composition.messages import MessageReader, MessageWriter
    from agent.plugin_contracts.restart import ExternalRootPermit


# ── 回复完成策略 ───────────────────────────────────────────────
@runtime_checkable
class CompletionPort(Protocol):
    """可选策略覆盖一次回复的完成阶段；Reply 本身不取得发送能力。"""

    def activity(
        self, reader: MessageReader, source: str
    ) -> "AbstractContextManager[None]":
        """进入「该来源正在活动」的作用域。"""
        ...

    def __call__(
        self,
        reader: MessageReader,
        source: str,
        *,
        child_permit: Callable[[], ExternalRootPermit] | None = None,
    ) -> "AbstractAsyncContextManager[None]":
        """进入一次完成阶段。"""
        ...


REPLY_COMPLETION = ServiceKey[CompletionPort]("reply.completion.v1")


# ── 回复程序与命令 ─────────────────────────────────────────────
REPLY_PROGRAM = ServiceKey[
    Callable[["Task", "MessageReader", str, Sequence[Reminder]], Awaitable[Message]]
]("reply.program.v1")

CONVERSATION_COMMANDS = ServiceKey[
    Callable[["Task", "MessageReader", str], Awaitable[Message | None]]
]("conversation.commands.v1")


# ── 语义兴趣 ───────────────────────────────────────────────────
@runtime_checkable
class SemanticInterestPort(Protocol):
    """用已完成对话的固定向量衡量候选兴趣；不写学习图。"""

    async def score(self, texts: Sequence[str], *, cutoff: str) -> tuple[float, ...]:
        """按截止时间前的证据打分。"""
        ...


SEMANTIC_INTEREST = ServiceKey[SemanticInterestPort]("akasha.semantic-interest.v1")


# ── 投递历史（只读） ───────────────────────────────────────────
@runtime_checkable
class DeliveryHistoryPort(Protocol):
    """跨来源只读真实送达历史；查询不会发送或改写回执。"""

    def recent(
        self,
        *,
        since: datetime,
        until: datetime,
        limit: int,
        excluded_sources: frozenset[str] = frozenset(),
        visibility: Literal["listed", "internal"] | None = None,
    ) -> tuple[Any, ...]:
        """按首次本地确认时间倒序返回区间内的不同消息。"""
        ...

    def status(self, message_id: str, sink: str) -> dict[str, object] | None:
        """读取一条真实送达的阶段；不取得发送权。"""
        ...


DELIVERY_READ = ServiceKey[DeliveryHistoryPort]("delivery.read.v1")


# ── 独立投递执行 ───────────────────────────────────────────────
@runtime_checkable
class DeliveriesPort(Protocol):
    """独立发送已 prepared 的消息；重试只使用原绑定、地址和幂等键。"""

    async def send(self, message_id: str, sink: str) -> Receipt:
        """向一个目的地发送并返回回执。"""
        ...


# ── 摘要查询（只读） ───────────────────────────────────────────
@runtime_checkable
class SummaryLookupPort(Protocol):
    """归档提供的窄读取口；不能发布摘要、推进 head 或启动模型。"""

    def head(self, session_id: str) -> object | None:
        """读取当前已发布摘要。"""
        ...

    def resolve(self, metadata: Mapping[str, object], *, session_id: str) -> object:
        """只解析 binding 固定的原始记录，并核对完整父链。"""
        ...


# ── Drift 提案与事件 ──────────────────────────────────────────
@runtime_checkable
class DriftProposalServicesPort(Protocol):
    """Drift 提案侧的服务面；消费者只按 Protocol 使用。"""

    def __getattr__(self, name: str) -> Any:
        """由实现定义具体方法；消费者按名调用。"""
        ...


DRIFT_PROPOSALS = ServiceKey[DriftProposalServicesPort]("drift.proposals.v1")
DRIFT_CHANGED = EmitEventKey[None]("drift.changed")
