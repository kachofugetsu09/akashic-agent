from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from agent.core.types import HistoryMessage, RetrievalTrace


@dataclass
class RetrievalRequest:
    message: str
    session_key: str
    channel: str
    chat_id: str
    history: list[HistoryMessage]  # 完整会话历史，无截窗。pipeline 实现负责自行决定使用范围。
    # DefaultMemoryRetrievalPipeline 内部截取末尾 MemoryConfig.window 条后使用。
    session_metadata: dict[str, object]
    timestamp: datetime | None = None
    extra: dict[str, object] = field(default_factory=dict[str, object])


@dataclass
class RetrievalResult:
    block: str
    trace: RetrievalTrace | None = None
    metadata: dict[str, object] = field(default_factory=dict[str, object])


@runtime_checkable
class MemoryRetrievalPipeline(Protocol):
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...
