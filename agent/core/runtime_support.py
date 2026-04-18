from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable, Protocol

logger = logging.getLogger("agent.tool_discovery")

from bus.events import InboundMessage


@dataclass
class MemoryConfig:
    window: int = 40
    top_k_procedure: int = 4
    top_k_history: int = 8
    route_intention_enabled: bool = False
    procedure_guard_enabled: bool = True
    gate_llm_timeout_ms: int = 800
    gate_max_tokens: int = 96
    hyde_enabled: bool = False
    hyde_timeout_ms: int = 2000


@dataclass
class LLMServices:
    provider: object
    light_provider: object


@dataclass
class MemoryServices:
    port: object
    query_rewriter: object | None = None
    hyde_enhancer: object | None = None
    sufficiency_checker: object | None = None


@dataclass
class ToolDiscoveryState:
    _unlocked: dict[str, OrderedDict[str, None]] = field(default_factory=dict)
    capacity: int = 5

    def get_preloaded(self, session_key: str) -> set[str]:
        return set(self._unlocked.get(session_key, {}).keys())

    def unlock_from_result(self, result_json: str) -> set[str]:
        """Parse a tool_search JSON result and return the tool names in 'matched'.

        Replaces the module-level _unlock_from_tool_search() that previously
        lived in agent/core/reasoner.py. Pure parsing — no mutation of external
        state; caller decides what to do with the returned names.
        """
        try:
            data = json.loads(result_json)
            return {
                item["name"]
                for item in data.get("matched", [])
                if isinstance(item.get("name"), str) and item["name"]
            }
        except Exception:
            return set()

    def update(self, session_key: str, tools_used: list[str], always_on: set[str]) -> None:
        skip = always_on | {"tool_search"}
        lru: OrderedDict[str, None] = self._unlocked.setdefault(
            session_key,
            OrderedDict(),
        )
        newly_added: list[str] = []
        for name in tools_used:
            if name in skip:
                continue
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


class SessionLike(Protocol):
    key: str
    messages: list[dict]
    metadata: dict[str, object]
    last_consolidated: int

    def get_history(self, max_messages: int = 500) -> list[dict]: ...
    def add_message(self, role: str, content: str, media=None, **kwargs) -> None: ...


ConsolidationRunner = Callable[[SessionLike], Awaitable[None]]


@dataclass
class TurnRunResult:
    reply: str | None
    tools_used: list[str] = field(default_factory=list)
    tool_chain: list[dict] = field(default_factory=list)
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
