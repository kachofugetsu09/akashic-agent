from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from agent.tools.base import Tool


ToolSource = str


@dataclass(frozen=True, slots=True)
class ToolGrant:
    """Freeze the tool names one Turn may expose and execute."""

    names: frozenset[str] | None = None
    denied: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.names is not None and any(not name for name in self.names):
            raise ValueError("Tool grant name 不能为空")
        if any(not name for name in self.denied):
            raise ValueError("Tool deny name 不能为空")
        if self.names is not None and self.names.intersection(self.denied):
            raise ValueError("Tool grant 与 deny 不能包含同一名称")

    @classmethod
    def only(cls, names: Sequence[str]) -> ToolGrant:
        return cls(frozenset(names))

    @classmethod
    def except_names(cls, names: Sequence[str]) -> ToolGrant:
        return cls(None, frozenset(names))

    def allows(self, name: str) -> bool:
        return name not in self.denied and (self.names is None or name in self.names)

    def visible(self, available: Sequence[str]) -> tuple[str, ...]:
        return tuple(name for name in available if self.allows(name))


@dataclass(frozen=True, slots=True)
class TurnExecutionScope:
    """Freeze transient Prompt, memory, and Tool rights for one Turn."""

    prompt_hints: tuple[str, ...] = ()
    tool_grant: ToolGrant = ToolGrant()
    tool_overrides: Mapping[str, "Tool"] = MappingProxyType({})
    memory_read: bool = True
    memory_write: bool = True
    stateless: bool = False
    tool_source: ToolSource = "passive"

    def __post_init__(self) -> None:
        if any(not hint.strip() for hint in self.prompt_hints):
            raise ValueError("Turn scope prompt hint 不能为空")
        overrides = dict(self.tool_overrides)
        if any(not name or tool.name != name for name, tool in overrides.items()):
            raise ValueError("Turn scope tool override 名称必须与 Tool 一致")
        if any(not self.tool_grant.allows(name) for name in overrides):
            raise ValueError("Turn scope tool override 必须已由 Tool grant 授权")
        if not self.tool_source or self.tool_source.strip() != self.tool_source:
            raise ValueError("Turn scope tool source 必须非空且无首尾空白")
        object.__setattr__(self, "tool_overrides", MappingProxyType(overrides))


_CURRENT_TURN_SCOPE: ContextVar[TurnExecutionScope | None] = ContextVar(
    "akashic_current_turn_execution_scope",
    default=None,
)


def get_current_turn_scope() -> TurnExecutionScope | None:
    """Return the transient scope bound to the current execution task."""

    return _CURRENT_TURN_SCOPE.get()


def bind_turn_scope(
    scope: TurnExecutionScope,
) -> Token[TurnExecutionScope | None]:
    return _CURRENT_TURN_SCOPE.set(scope)


def reset_turn_scope(token: Token[TurnExecutionScope | None]) -> None:
    _CURRENT_TURN_SCOPE.reset(token)
