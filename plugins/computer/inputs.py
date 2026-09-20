"""Computer 只读取工具调用前缀和 Turn 身份。"""
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_contracts import CallRef, ContentPart, Message


@dataclass(frozen=True)
class Result:
    outcome: Literal["success", "error"]
    parts: tuple[ContentPart, ...]


class CallSource(Protocol):
    @property
    def call_ref(self) -> CallRef: ...
    @property
    def messages(self) -> tuple[Message, ...]: ...


class Turn(Protocol):
    @property
    def status(self) -> str: ...
    @property
    def message_ids(self) -> tuple[str, ...]: ...


class TurnProjection(Protocol):
    def project(self, messages: Sequence[Message], source: str) -> tuple[Turn, ...]: ...


class ToolCatalog(Protocol):
    async def declare_group(self, ctx: Context, *, description: str) -> Effect: ...
    async def register(
        self, ctx: Context, *, name: str, description: str, parameters: Mapping[str, object],
        open: Callable[[Mapping[str, object]], AbstractAsyncContextManager[object]],
        capture: Callable[[Mapping[str, object]], Mapping[str, object]],
        public: bool, idempotent: bool, risk: Literal["external-side-effect"],
    ) -> object: ...


TOOLS = ServiceKey[ToolCatalog]("tools.v1")
TURN_PROJECTION = ServiceKey[TurnProjection]("turn.projection.v1")
