from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from agent.control.turn_scope import ToolGrant

from agent.plugin_composition.events import (
    ObserveEventKey,
    SerialEventKey,
    TransformEventKey,
)

ToolStatus = Literal["success", "denied", "error"]


@dataclass
class ToolExecutionRequest:
    """Describe one Core-owned tool execution admission request."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    source: str
    session_key: str = ""
    channel: str = ""
    chat_id: str = ""
    tool_batch: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    tool_batch_index: int = 0
    grant: ToolGrant = ToolGrant()


@dataclass
class ToolExecutionResult:
    """Record the settled result of one Core-owned tool execution."""

    status: ToolStatus
    output: Any
    final_arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ToolInputIdentity:
    call_id: str
    tool_name: str
    source: str
    session_key: str
    channel: str
    chat_id: str
    tool_batch: tuple[Mapping[str, Any], ...]
    tool_batch_index: int


@dataclass(frozen=True, slots=True)
class ToolInput:
    """Expose one immutable call identity with explicitly replaceable arguments."""

    _identity: _ToolInputIdentity
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))

    @classmethod
    def from_request(
        cls,
        request: ToolExecutionRequest,
        arguments: Mapping[str, Any],
    ) -> ToolInput:
        identity = _ToolInputIdentity(
            call_id=request.call_id,
            tool_name=request.tool_name,
            source=request.source,
            session_key=request.session_key,
            channel=request.channel,
            chat_id=request.chat_id,
            tool_batch=tuple(
                _freeze_mapping(item) for item in request.tool_batch
            ),
            tool_batch_index=request.tool_batch_index,
        )
        return cls(identity, arguments)

    @property
    def call_id(self) -> str:
        return self._identity.call_id

    @property
    def tool_name(self) -> str:
        return self._identity.tool_name

    @property
    def source(self) -> str:
        return self._identity.source

    @property
    def session_key(self) -> str:
        return self._identity.session_key

    @property
    def channel(self) -> str:
        return self._identity.channel

    @property
    def chat_id(self) -> str:
        return self._identity.chat_id

    @property
    def tool_batch(self) -> tuple[Mapping[str, Any], ...]:
        return self._identity.tool_batch

    @property
    def tool_batch_index(self) -> int:
        return self._identity.tool_batch_index

    def with_arguments(self, arguments: Mapping[str, Any]) -> ToolInput:
        return ToolInput(self._identity, arguments)

    def same_call(self, other: ToolInput) -> bool:
        return self._identity is other._identity

    def mutable_arguments(self) -> dict[str, Any]:
        return {
            key: _thaw_value(value)
            for key, value in self.arguments.items()
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Freeze the settled tool fact observed after the invoker returns."""

    input: ToolInput
    status: ToolStatus
    result: str

    @classmethod
    def from_execution(
        cls,
        request: ToolExecutionRequest,
        execution: ToolExecutionResult,
    ) -> ToolResult:
        return cls(
            input=ToolInput.from_request(request, execution.final_arguments),
            status=execution.status,
            result=_safe_result_text(execution.output),
        )

    @property
    def call_id(self) -> str:
        return self.input.call_id

    @property
    def tool_name(self) -> str:
        return self.input.tool_name

    @property
    def arguments(self) -> Mapping[str, Any]:
        return self.input.arguments

    @property
    def source(self) -> str:
        return self.input.source

    @property
    def session_key(self) -> str:
        return self.input.session_key

    @property
    def channel(self) -> str:
        return self.input.channel

    @property
    def chat_id(self) -> str:
        return self.input.chat_id


TOOL_INPUT_PREPARE = TransformEventKey(
    "tool.input.prepare",
    ToolInput,
    "akashic.tool-input.v1",
)
TOOL_EXECUTION_AUTHORIZE = SerialEventKey[ToolInput, str](
    "tool.execution.authorize",
    str,
    "akashic.tool-deny-reason.v1",
)
TOOL_RESULT = ObserveEventKey[ToolResult]("tool.result")


def _safe_result_text(output: object) -> str:
    try:
        return str(output)
    except BaseException:
        return f"<unprintable {type(output).__name__}>"


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


def _freeze_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise TypeError("工具参数对象的 key 必须是字符串")
        return _freeze_mapping(
            {
                cast(str, key): item
                for key, item in mapping.items()
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in cast(list[object], value))
    if isinstance(value, tuple):
        return tuple(
            _freeze_value(item) for item in cast(tuple[object, ...], value)
        )
    raise TypeError(f"工具参数必须是 JSON 值，实际为 {type(value).__name__}")


def _thaw_value(value: object) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise TypeError("冻结工具参数对象的 key 必须是字符串")
        return {
            cast(str, key): _thaw_value(item)
            for key, item in mapping.items()
        }
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in cast(tuple[object, ...], value)]
    return value
