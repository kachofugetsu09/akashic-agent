"""
被动 AgentLoop 中断机制 — 数据结构与协议。

Channel 层识别 /stop 命令后，通过 InterruptController.request_interrupt()
走控制面打断当前正在执行的 turn，不经过 MessageBus 数据面。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


_DEFAULT_TTL_S = 1800  # 30 分钟


@dataclass
class TurnInterruptState:
    """一个被中断的 turn 的快照，纯内存态，不落库。"""

    session_key: str
    original_user_message: str
    original_metadata: dict = field(default_factory=dict)
    partial_reply: str = ""
    partial_thinking: str | None = None
    tools_used: list[str] = field(default_factory=list)
    tool_chain_partial: list[dict] = field(default_factory=list)
    interrupted_by: str = "/stop"
    interrupted_at: float = field(default_factory=time.monotonic)
    ttl_seconds: int = _DEFAULT_TTL_S

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.interrupted_at) > self.ttl_seconds


@dataclass
class InterruptResult:
    """request_interrupt() 的返回值。"""

    status: str  # "interrupted" | "idle"
    session_key: str = ""
    message: str = ""


class InterruptController(Protocol):
    """Channel 层调用的中断协议，由 AgentLoop 实现。"""

    def request_interrupt(
        self,
        session_key: str,
        sender: str = "",
        command: str = "/stop",
    ) -> InterruptResult: ...
