"""Akasha 只依赖外部插件发布的窄能力和结构值。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_contracts import CallRef, ContentPart, Message


Outcome = Literal["success", "denied", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class CallSource:
    """实际工具调用的不可变消息前缀；不携带 reader 或写权限。"""

    call_ref: CallRef
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class Result:
    """工具 provider 可消费的结构结果。"""

    outcome: Outcome
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "denied", "error", "interrupted"}:
            raise ValueError("工具结果状态无效")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("工具结果必须是内容块")
        object.__setattr__(self, "parts", parts)


class ToolRef(Protocol):
    """tools owner 返回的不可变工具描述。"""

    name: str
    description: Mapping[str, object]


class ToolView(Protocol):
    """被授予 Akasha 的工具引用集合。"""

    refs: tuple[ToolRef, ...]


class ToolCatalog(Protocol):
    async def declare_group(
        self, ctx: Context, *, always_on: bool = False, description: str,
    ) -> object: ...

    async def register(
        self, ctx: Context, *, name: str, description: str,
        parameters: Mapping[str, object], open: object, capture: object | None = None,
        idempotent: bool = False, risk: str = "read-write",
    ) -> ToolRef: ...

    def view(self, *refs: ToolRef) -> ToolView: ...


TOOLS = ServiceKey[ToolCatalog]("tools.v1")


class ContentCapability(Protocol):
    """content owner 发布的注册与历史资格窄口。"""

    async def register(
        self, ctx: Context, definition: Mapping[str, object], *, prepare: object | None = None,
    ) -> object: ...

    def is_user_input(self, message: Message) -> bool: ...

    def legacy_post_commit_effect(self, message: Message) -> str | None: ...


CONTENT = ServiceKey[ContentCapability]("content.v2")


class Turn(Protocol):
    """turn projection 只返回消息身份和区间，不复制消息正文。"""

    @property
    def ending_message_id(self) -> str | None: ...

    @property
    def status(self) -> Literal["open", "complete", "quiet", "abandoned"]: ...

    @property
    def message_ids(self) -> tuple[str, ...]: ...

    @property
    def observations(self) -> tuple[tuple[CallRef, str], ...]: ...

    @property
    def through_seq(self) -> int: ...


class TurnProjection(Protocol):
    def project(self, messages: tuple[Message, ...] | list[Message], source: str) -> tuple[Turn, ...]: ...


TURN_PROJECTION = ServiceKey[TurnProjection]("turn.projection.v1")


PostCommitReader = Callable[[Message], str | None]
