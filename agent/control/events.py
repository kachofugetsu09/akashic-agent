from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TurnEvent:
    """描述一个可稳定投影到协议的 turn 生命周期事件。"""

    method: str
    thread_id: str
    turn_id: str
    data: dict[str, Any]
    timestamp: datetime

    @classmethod
    def create(
        cls,
        method: str,
        thread_id: str,
        turn_id: str,
        **data: Any,
    ) -> TurnEvent:
        return cls(method, thread_id, turn_id, data, datetime.now(timezone.utc))

    def to_notification(self) -> dict[str, object]:
        params: dict[str, object] = {
            "threadId": self.thread_id,
            "turnId": self.turn_id,
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
        }
        params.update(self.data)
        return {"jsonrpc": "2.0", "method": self.method, "params": params}
