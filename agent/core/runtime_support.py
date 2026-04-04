from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

from bus.events import InboundMessage


@dataclass
class MemoryConfig:
    window: int = 40
    top_k_procedure: int = 4
    top_k_history: int = 8
    route_intention_enabled: bool = False
    sop_guard_enabled: bool = True
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

    def update(self, session_key: str, tools_used: list[str], always_on: set[str]) -> None:
        skip = always_on | {"tool_search"}
        lru: OrderedDict[str, None] = self._unlocked.setdefault(
            session_key,
            OrderedDict(),
        )
        for name in tools_used:
            if name in skip:
                continue
            if name in lru:
                lru.move_to_end(name)
            else:
                lru[name] = None
            while len(lru) > self.capacity:
                lru.popitem(last=False)


class SessionLike(Protocol):
    key: str
    messages: list[dict]
    metadata: dict[str, object]
    last_consolidated: int

    def get_history(self, max_messages: int = 500) -> list[dict]: ...
    def add_message(self, role: str, content: str, media=None, **kwargs) -> None: ...


ConsolidationRunner = Callable[[SessionLike], Awaitable[None]]


class TurnRunner(Protocol):
    async def run(
        self,
        msg: InboundMessage,
        session: SessionLike,
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        retrieved_memory_block: str = "",
    ) -> tuple[str, list[str], list[dict], str | None]:
        ...


class AgentLoopRunner(Protocol):
    async def __call__(
        self,
        initial_messages: list[dict],
        request_time: object | None = None,
        preloaded_tools: set[str] | None = None,
        preflight_injected: bool = False,
    ) -> tuple[str, list[str], list[dict], set[str] | None, str | None]:
        ...
