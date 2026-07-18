from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, cast

from agent.lifecycle.types import PromptRenderInput, PromptRenderResult

logger = logging.getLogger("agent.tool_discovery")


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
    _unlocked: dict[str, OrderedDict[str, None]] = field(
        default_factory=dict[str, OrderedDict[str, None]]
    )
    capacity: int = 5
    session_capacity: int = 1024
    _session_lru: OrderedDict[str, None] = field(
        default_factory=OrderedDict[str, None],
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
        """解析工具搜索结果，并返回可解锁的唯一工具名。"""

        # 1. 校验外部 JSON 根节点。
        try:
            parsed: object = json.loads(result_json)
        except json.JSONDecodeError as exc:
            raise ValueError("tool_search 返回非法 JSON") from exc
        if not isinstance(parsed, dict):
            raise TypeError("tool_search 结果必须是 object")
        result = cast(dict[str, object], parsed)

        # 2. 优先消费 tool_search 当前协议的 unlocked，兼容旧 matched 结果。
        if "unlocked" in result:
            return self._parse_unlocked_names(result["unlocked"])
        return self._parse_matched_names(result.get("matched"))

    @staticmethod
    def _parse_unlocked_names(raw_names: object) -> list[str]:
        if not isinstance(raw_names, list):
            raise TypeError("tool_search.unlocked 必须是字符串数组")
        names: list[str] = []
        for index, item in enumerate(cast(list[object], raw_names)):
            if not isinstance(item, str) or not item or item != item.strip():
                raise TypeError(f"tool_search.unlocked[{index}] 必须是非空工具名")
            names.append(item)
        return list(dict.fromkeys(names))

    @staticmethod
    def _parse_matched_names(raw_matches: object) -> list[str]:
        if not isinstance(raw_matches, list):
            raise TypeError("tool_search.matched 必须是数组")
        names: list[str] = []
        for index, item in enumerate(cast(list[object], raw_matches)):
            if not isinstance(item, dict):
                raise TypeError(f"tool_search.matched[{index}] 必须是 object")
            match = cast(dict[str, object], item)
            name = match.get("name")
            if not isinstance(name, str) or not name or name != name.strip():
                raise TypeError(f"tool_search.matched[{index}].name 必须是非空工具名")
            names.append(name)
        return list(dict.fromkeys(names))

    def unlock_from_result(self, result_json: str) -> set[str]:
        """从工具搜索结果中提取工具名集合。"""

        return set(self.unlock_names_from_result(result_json))

    def update(
        self,
        session_key: str,
        tools_used: list[str],
        always_on: set[str],
        non_preloadable: set[str] | None = None,
    ) -> None:
        """更新单个 session 的工具 LRU，并跳过常驻工具。"""

        # 1. 过滤不应缓存的工具，避免创建空 session 项。
        skip = always_on | {"tool_search"} | (non_preloadable or set())
        cacheable = [name for name in tools_used if name not in skip]
        if not cacheable:
            self._touch_session(session_key)
            return

        # 2. 触碰 session LRU，再更新该 session 内的工具顺序。
        lru: OrderedDict[str, None] = self._unlocked.setdefault(
            session_key,
            OrderedDict(),
        )
        self._touch_session(session_key)

        # 3. 写入工具并淘汰最久未使用项。
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
        """刷新 session LRU，并淘汰超出全局容量的旧缓存。"""

        # 1. 没有工具缓存的 session 不进入 session LRU。
        if session_key not in self._unlocked:
            return
        self._session_lru[session_key] = None
        self._session_lru.move_to_end(session_key)

        # 2. 同步淘汰工具缓存，保持两级 LRU 的键集合一致。
        while self._session_lru and len(self._session_lru) > self.session_capacity:
            evicted, _ = self._session_lru.popitem(last=False)
            _ = self._unlocked.pop(evicted)
            logger.info("[LRU驱逐] 移除最旧会话工具缓存: %s", evicted)


class SessionLike(Protocol):
    key: str
    messages: list[dict[str, object]]
    metadata: dict[str, object]
    last_consolidated: int

    def get_history(
        self,
        max_messages: int = 500,
        *,
        start_index: int | None = None,
    ) -> list[dict[str, object]]: ...
    def add_message(
        self,
        role: str,
        content: str,
        media: list[str] | None = None,
        **kwargs: object,
    ) -> dict[str, object]: ...


@dataclass
class TurnRunResult:
    reply: str | None
    tools_used: list[str] = field(default_factory=list[str])
    tool_chain: list[dict[str, object]] = field(
        default_factory=list[dict[str, object]]
    )
    media: list[str] = field(default_factory=list[str])
    thinking: str | None = None
    streamed: bool = False
    context_retry: dict[str, object] = field(default_factory=dict[str, object])
    model_state: dict[str, object] | None = None
    mobile_attention: Literal["confirmation"] | None = None


class AgentLoopRunner(Protocol):
    async def __call__(
        self,
        initial_messages: list[dict[str, object]],
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
    ) -> tuple[
        str,
        list[str],
        list[dict[str, object]],
        set[str] | None,
        str | None,
    ]:
        ...


class PromptRenderRunner(Protocol):
    async def __call__(
        self,
        input: PromptRenderInput,
    ) -> PromptRenderResult:
        ...
