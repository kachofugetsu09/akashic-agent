from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, cast

from agent.lifecycle.types import PromptRenderInput, PromptRenderResult

logger = logging.getLogger("agent.tool_discovery")

from bus.events import InboundMessage


@dataclass
class MemoryConfig:
    window: int = 40


@dataclass
class LLMServices:
    provider: object
    light_provider: object


@dataclass
class MemoryServices:
    engine: object


@dataclass
class ToolDiscoveryState:
    _unlocked: dict[str, OrderedDict[str, None]] = field(default_factory=dict)
    capacity: int = 5
    session_capacity: int = 1024
    _session_lru: OrderedDict[str, None] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )

    def get_preloaded(self, session_key: str) -> set[str]:
        self._touch_session(session_key)
        return set(self._unlocked.get(session_key, {}).keys())

    def get_preloaded_ordered(self, session_key: str) -> list[str]:
        self._touch_session(session_key)
        return list(self._unlocked.get(session_key, {}).keys())

    def unlock_names_from_result(self, result_json: str) -> list[str]:
        """从工具搜索结果中提取有效且不重复的工具名。"""

        try:
            parsed: object = json.loads(result_json)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, dict):
            return []
        data = cast(dict[str, object], parsed)

        raw_unlocked = data.get("unlocked")
        raw_names: list[object]
        if isinstance(raw_unlocked, list):
            raw_names = cast(list[object], raw_unlocked)
        else:
            matched = data.get("matched", [])
            if not isinstance(matched, list):
                return []
            raw_names = [
                cast(dict[str, object], item).get("name")
                for item in cast(list[object], matched)
                if isinstance(item, dict)
            ]

        names: list[str] = []
        seen: set[str] = set()
        for item in raw_names:
            if isinstance(item, str) and item and item not in seen:
                names.append(item)
                seen.add(item)
        return names

    def unlock_from_result(self, result_json: str) -> set[str]:
        """从工具搜索结果中提取工具名集合。"""

        return set(self.unlock_names_from_result(result_json))

    def update(self, session_key: str, tools_used: list[str], always_on: set[str]) -> None:
        skip = always_on | {"tool_search"}
        cacheable = [name for name in tools_used if name not in skip]
        if not cacheable:
            self._touch_session(session_key)
            return
        lru: OrderedDict[str, None] = self._unlocked.setdefault(
            session_key,
            OrderedDict(),
        )
        self._touch_session(session_key)
        newly_added: list[str] = []
        for name in cacheable:
            if name in lru:
                lru.move_to_end(name)
            else:
                lru[name] = None
                newly_added.append(name)
            while len(lru) > self.capacity:
                evicted, _ = lru.popitem(last=False)
                logger.info("[LRU驱逐] session=%s 移除最旧工具: %s", session_key, evicted)
        if newly_added:
            logger.info(
                "[LRU更新] session=%s 新增工具: %s，当前LRU: %s",
                session_key,
                newly_added,
                list(lru.keys()),
            )

    def _touch_session(self, session_key: str) -> None:
        if session_key not in self._unlocked:
            return
        self._session_lru[session_key] = None
        self._session_lru.move_to_end(session_key)
        while self._session_lru and len(self._session_lru) > self.session_capacity:
            evicted, _ = self._session_lru.popitem(last=False)
            _ = self._unlocked.pop(evicted)
            logger.info("[LRU驱逐] 移除最旧会话工具缓存: %s", evicted)


class SessionLike(Protocol):
    key: str
    messages: list[dict]
    metadata: dict[str, object]
    last_consolidated: int

    def get_history(
        self,
        max_messages: int = 500,
        *,
        start_index: int | None = None,
    ) -> list[dict]: ...
    def add_message(self, role: str, content: str, media=None, **kwargs) -> None: ...

@dataclass
class TurnRunResult:
    reply: str | None
    tools_used: list[str] = field(default_factory=list)
    tool_chain: list[dict] = field(default_factory=list)
    media: list[str] = field(default_factory=list)
    thinking: str | None = None
    streamed: bool = False
    context_retry: dict[str, object] = field(default_factory=dict)


class AgentLoopRunner(Protocol):
    async def __call__(
        self,
        initial_messages: list[dict],
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
    ) -> tuple[str, list[str], list[dict], set[str] | None, str | None]:
        ...


class PromptRenderRunner(Protocol):
    async def __call__(
        self,
        input: PromptRenderInput,
    ) -> PromptRenderResult:
        ...
