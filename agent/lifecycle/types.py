from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent.prompting.assembler import PromptSectionRender
from bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from agent.core.response_parser import ResponseMetadata
    from agent.core.runtime_support import SessionLike, TurnRunResult


# 1. 工厂函数：给 dataclass field(default_factory=...) 提供显式类型签名，消除 pyright Unknown 推断。
def _empty_str_list() -> list[str]:
    return []


def _empty_metadata() -> dict[str, Any]:
    return {}


def _empty_prompt_sections() -> list[PromptSectionRender]:
    return []


@dataclass
class TurnState:
    msg: InboundMessage
    session_key: str
    dispatch_outbound: bool
    session: SessionLike | None = None
    retrieval_raw: object | None = None


@dataclass
class BeforeTurnCtx:
    # read-only by convention
    session_key: str
    channel: str
    chat_id: str
    content: str
    timestamp: datetime
    retrieved_memory_block: str
    retrieval_trace_raw: object | None
    history_messages: tuple[Any, ...]
    # writable
    skill_names: list[str] = field(default_factory=_empty_str_list)
    abort: bool = False
    abort_reply: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True)
class BeforeReasoningInput:
    state: TurnState
    before_turn: BeforeTurnCtx


@dataclass
class BeforeReasoningCtx:
    # read-only by convention
    session_key: str
    channel: str
    chat_id: str
    content: str
    timestamp: datetime
    # writable
    skill_names: list[str]
    retrieved_memory_block: str
    extra_hints: list[str] = field(default_factory=_empty_str_list)
    abort: bool = False
    abort_reply: str = ""


@dataclass(frozen=True)
class PromptRenderInput:
    session_key: str
    channel: str
    chat_id: str
    content: str
    media: list[str] | None
    timestamp: datetime
    history: list[dict[str, Any]]
    skill_names: list[str] | None
    retrieved_memory_block: str
    disabled_sections: set[str]
    turn_injection_prompt: str
    extra_hints: list[str] | None = None


@dataclass
class PromptRenderCtx:
    # read-only by convention
    session_key: str
    channel: str
    chat_id: str
    content: str
    media: list[str] | None
    timestamp: datetime
    history: list[dict[str, Any]]
    skill_names: list[str] | None
    retrieved_memory_block: str
    disabled_sections: set[str]
    turn_injection_prompt: str
    extra_hints: list[str] = field(default_factory=_empty_str_list)
    # writable
    system_sections_top: list[PromptSectionRender] = field(
        default_factory=_empty_prompt_sections
    )
    system_sections_bottom: list[PromptSectionRender] = field(
        default_factory=_empty_prompt_sections
    )


@dataclass(frozen=True)
class PromptRenderResult:
    messages: list[dict[str, Any]]


@dataclass(frozen=True)
class BeforeStepInput:
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    messages: list[dict[str, Any]]
    visible_names: set[str] | None


@dataclass
class BeforeStepCtx:
    # read-only by convention
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    input_tokens_estimate: int
    visible_tool_names: frozenset[str] | None
    # writable
    extra_hints: list[str] = field(default_factory=_empty_str_list)
    early_stop: bool = False
    early_stop_reply: str = ""


@dataclass(frozen=True)
class AfterStepCtx:
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    tools_called: tuple[str, ...]
    partial_reply: str
    tools_used_so_far: tuple[str, ...]
    tool_chain_partial: tuple[dict[str, Any], ...]
    partial_thinking: str | None
    has_more: bool


@dataclass(frozen=True)
class AfterReasoningInput:
    state: TurnState
    turn_result: TurnRunResult


@dataclass
class AfterReasoningCtx:
    # read-only by convention
    session_key: str
    channel: str
    chat_id: str
    tools_used: tuple[str, ...]
    thinking: str | None
    response_metadata: ResponseMetadata
    streamed: bool
    tool_chain: tuple[dict[str, Any], ...]
    context_retry: dict[str, object]
    # writable
    reply: str
    media: list[str] = field(default_factory=_empty_str_list)
    meme_tag: str | None = None
    outbound_metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True)
class AfterReasoningResult:
    ctx: AfterReasoningCtx
    outbound: OutboundMessage


@dataclass
class TurnSnapshot:
    state: TurnState
    outbound: OutboundMessage
    ctx: AfterReasoningCtx


@dataclass(frozen=True)
class AfterTurnCtx:
    session_key: str
    channel: str
    chat_id: str
    reply: str
    tools_used: tuple[str, ...]
    thinking: str | None
    # pre-dispatch intent flag: dispatch has NOT happened yet when Tap handlers run
    will_dispatch: bool


@dataclass(frozen=True)
class BeforeToolCallCtx:
    session_key: str
    channel: str
    chat_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AfterToolResultCtx:
    session_key: str
    channel: str
    chat_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: str
    status: str


@dataclass
class PreToolCtx:
    """pre-tool hook 上下文 — mutable，handler 返回 dict 表示新 arguments"""
    session_key: str
    channel: str
    chat_id: str
    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""
    source: str = ""
    request_text: str = ""
    tool_batch: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    tool_batch_index: int = 0
