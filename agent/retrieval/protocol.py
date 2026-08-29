from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class RetrievalRequest:
    message: str
    session_key: str
    channel: str
    chat_id: str
    history: list[dict[str, Any]]  # 完整会话历史，无截窗。consumer 自行决定使用范围。
    # retrieval pipeline 自己决定是否需要检索投影；这里始终接收完整 session history。
    session_metadata: dict[str, object]
    turn_id: str = ""
    timestamp: datetime | None = None
