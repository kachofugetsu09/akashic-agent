from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeAlias

from agent.prompting.assembler import AssembledTurnInput, PromptSectionRender
from agent.plugin_composition.turn_lifecycle import BeforeTurnCtx
from bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from agent.core.response_parser import ResponseMetadata
    from agent.core.runtime_support import SessionLike, TurnRunResult


# 插件阶段接口：现有插件直接构造或读写下列上下文。新核心可以用薄层转换，
# 但迁移插件前不得删除字段、改变可写范围或调整上下文出现的阶段。
@dataclass
class TurnPersistencePolicy:
    persist_user: bool = True
    persist_assistant: bool = True


@dataclass
class TurnState:
    msg: InboundMessage
    session_key: str
    dispatch_outbound: bool
    session: SessionLike | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    persistence: TurnPersistencePolicy = field(default_factory=TurnPersistencePolicy)


@dataclass(frozen=True)
class BeforeReasoningInput:
    state: TurnState
    before_turn: BeforeTurnCtx


@dataclass
class BeforeReasoningCtx:
    # before-* ctx 走 GATE 链，插件可直接改写字段影响后续阶段。
    # 按约定只读
    session_key: str
    channel: str
    chat_id: str
    content: str
    timestamp: datetime
    # 可写
    skill_names: list[str]
    extra_hints: list[str] = field(default_factory=list[str])
    abort: bool = False
    abort_reply: str = ""


@dataclass(frozen=True)
class PromptRenderInput:
    session_key: str
    channel: str
    chat_id: str
    content: str
    multimodal: bool
    media: list[str] | None
    timestamp: datetime
    history: list[dict[str, Any]]
    skill_names: list[str] | None
    disabled_sections: set[str]
    turn_injection_prompt: str
    extra_hints: list[str] | None = None


@dataclass
class PromptRenderCtx:
    # render/before-step ctx 走 GATE 链，插件可直接改写字段影响后续阶段。
    # 按约定只读
    session_key: str
    channel: str
    chat_id: str
    content: str
    media: list[str] | None
    timestamp: datetime
    history: list[dict[str, Any]]
    skill_names: list[str] | None
    disabled_sections: set[str]
    turn_injection_prompt: str
    extra_hints: list[str] = field(default_factory=list[str])
    # 可写
    system_sections_top: list[PromptSectionRender] = field(
        default_factory=list[PromptSectionRender]
    )
    system_sections_bottom: list[PromptSectionRender] = field(
        default_factory=list[PromptSectionRender]
    )


# 保留旧导入名；运行时结果由 ContextBuilder 的唯一组装结果承载。
PromptRenderResult: TypeAlias = AssembledTurnInput


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
    # before-* ctx 走 GATE 链，插件可直接改写字段影响后续阶段。
    # 按约定只读
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    input_tokens_estimate: int
    visible_tool_names: frozenset[str] | None
    # 可写
    extra_hints: list[str] = field(default_factory=list[str])
    early_stop: bool = False
    early_stop_reply: str = ""


@dataclass(frozen=True)
class AfterStepCtx:
    # after-* fanout ctx 是观察快照；需要补充 metadata 时由 PhaseModule replace 新实例。
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    context_tokens_estimate: int
    tools_called: tuple[str, ...]
    partial_reply: str
    tools_used_so_far: tuple[str, ...]
    tool_chain_partial: tuple[dict[str, Any], ...]
    partial_thinking: str | None
    has_more: bool
    early_stop: bool = False
    early_stop_reason: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class AfterReasoningInput:
    state: TurnState
    turn_result: TurnRunResult


@dataclass
class AfterReasoningCtx:
    # after_reasoning 仍是 GATE 链，插件可改写 reply/media/outbound_metadata。
    # 按约定只读
    session_key: str
    channel: str
    chat_id: str
    tools_used: tuple[str, ...]
    thinking: str | None
    response_metadata: ResponseMetadata
    streamed: bool
    tool_chain: tuple[dict[str, Any], ...]
    context_retry: dict[str, object]
    # 可写
    reply: str
    media: list[str] = field(default_factory=list[str])
    meme_tag: str | None = None
    persist_user_metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    persist_assistant_metadata: dict[str, Any] = field(
        default_factory=dict[str, Any]
    )
    outbound_metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class TurnSnapshot:
    state: TurnState
    outbound: OutboundMessage
    ctx: AfterReasoningCtx


@dataclass(frozen=True)
class AfterTurnCtx:
    # after-* fanout ctx 是观察快照；需要补充 metadata 时由 PhaseModule replace 新实例。
    session_key: str
    channel: str
    chat_id: str
    reply: str
    tools_used: tuple[str, ...]
    thinking: str | None
    # dispatch 前意图标记：Tap handler 运行时尚未发生 dispatch
    will_dispatch: bool
    extra_metadata: dict[str, Any] = field(default_factory=dict[str, Any])


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
