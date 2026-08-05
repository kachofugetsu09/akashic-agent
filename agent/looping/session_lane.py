from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class _SessionLaneState:
    lock: asyncio.Lock
    users: int = 0


class SessionLaneRegistry:
    """Serialize complete turns per session while allowing unrelated sessions to run."""

    def __init__(self) -> None:
        self._states: dict[str, _SessionLaneState] = {}

    @asynccontextmanager
    async def hold(self, session_key: str) -> AsyncIterator[None]:
        """Acquire one session lane and reclaim it after the final waiter leaves."""

        # 1. Register holders and waiters before yielding to the event loop.
        key = session_key.strip()
        if not key:
            raise ValueError("session_key 不能为空")
        state = self._states.get(key)
        if state is None:
            state = _SessionLaneState(lock=asyncio.Lock())
            self._states[key] = state
        state.users += 1

        try:
            # 2. Hold the lane for the complete turn lifecycle.
            async with state.lock:
                yield
        finally:
            # 3. Remove idle lane state without disturbing newer owners.
            state.users -= 1
            if state.users == 0 and self._states.get(key) is state:
                del self._states[key]
