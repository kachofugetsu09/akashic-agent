from __future__ import annotations

from typing import Protocol


class EventPublisher(Protocol):
    async def fanout(self, event: object) -> None: ...
