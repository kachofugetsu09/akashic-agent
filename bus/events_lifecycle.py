from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agent.core.types import ToolCallGroup


# 插件事件接口：现有插件订阅下列事件并读取其字段。核心重构可以转换事件，
# 但迁移插件前不得删除字段、改名或改变事件相对持久化和发送的时机。
@dataclass(frozen=True)
class TurnStarted:
    session_key: str
    channel: str
    chat_id: str
    content: str
    timestamp: datetime
    turn_id: str = ""
    control_turn_id: str = ""
    client_message_id: str = ""


@dataclass(frozen=True)
class StreamDeltaReady:
    session_key: str
    channel: str
    chat_id: str
    turn_id: str = ""
    content_delta: str = ""
    thinking_delta: str = ""


@dataclass(frozen=True)
class TurnOutputCompleted:
    """provider 已无更多可见输出、Stop 已无意义的展示层信号。

    只表示输出完成，不宣称 Akasha / turn 已权威终结；权威终态仍由
    TurnCommitted 与 message.final / turn.completed 承担。
    """

    session_key: str
    channel: str
    chat_id: str
    turn_id: str = ""
    client_message_id: str = ""


@dataclass(frozen=True)
class TurnCommitted:
    session_key: str
    channel: str
    chat_id: str
    input_message: str
    persisted_user_message: str | None
    assistant_response: str
    tools_used: list[str]
    turn_id: str = ""
    client_message_id: str = ""
    persisted_user_message_id: str | None = None
    persisted_user_message_ids: tuple[str, ...] = ()
    assistant_message_id: str | None = None
    thinking: str | None = None
    raw_reply: str | None = None
    meme_tag: str | None = None
    meme_media_count: int | None = None
    tool_chain_raw: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    tool_call_groups: list["ToolCallGroup"] = field(
        default_factory=list["ToolCallGroup"]
    )
    timestamp: datetime | None = None
    post_reply_budget: dict[str, int] = field(default_factory=dict[str, int])
    react_stats: dict[str, int] = field(default_factory=dict[str, int])
    extra: dict[str, Any] = field(default_factory=dict[str, Any])
    model_usage: dict[str, Any] = field(default_factory=dict[str, Any])
    model_binding: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class ProactiveFinished:
    session_key: str
    tick_id: str
    mode: Literal["proactive", "drift"]
    terminal_action: str | None
    gate_exit: str | None
    skip_reason: str
    steps_taken: int
    alert_count: int
    content_count: int
    context_count: int
    final_message: str
    llm_call_count: int
    cache_prompt_tokens: int | None = None
    cache_hit_tokens: int | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True)
class DriftFinished:
    session_key: str
    skill_name: str
    status: str
    briefing: str
    message_result: str
    timestamp: datetime


@dataclass(frozen=True)
class ToolCallStarted:
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    turn_id: str = ""


@dataclass(frozen=True)
class ToolCallCompleted:
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    final_arguments: dict[str, Any]
    status: str
    result_preview: str
    runtime_provenance: dict[str, str] = field(default_factory=dict)
    turn_id: str = ""
