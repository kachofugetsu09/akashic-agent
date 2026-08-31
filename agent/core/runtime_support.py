from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, cast

logger = logging.getLogger("agent.tool_discovery")


@dataclass
class ToolDiscoveryState:
    _unlocked: OrderedDict[str, OrderedDict[str, None]] = field(
        default_factory=OrderedDict[str, OrderedDict[str, None]]
    )
    capacity: int = 5
    session_capacity: int = 1024

    def get_preloaded_ordered(self, session_key: str) -> list[str]:
        self._touch_session(session_key)
        return list(self._unlocked.get(session_key, {}).keys())

    def unlock_names_from_result(self, result_json: str) -> list[str]:
        """解析工具搜索结果，并返回可解锁的唯一工具名。"""

        # 1. 校验工具边界返回的 JSON 根节点。
        try:
            parsed: object = json.loads(result_json)
        except json.JSONDecodeError as exc:
            raise ValueError("tool_search 返回非法 JSON") from exc
        if not isinstance(parsed, dict):
            raise TypeError("tool_search 结果必须是 object")
        result = cast(dict[str, object], parsed)

        # 2. unlocked 是运行时解锁事实；matched 仅供模型阅读。
        if "unlocked" not in result:
            raise ValueError("tool_search 结果缺少 unlocked")
        raw_names = result["unlocked"]
        if not isinstance(raw_names, list):
            raise TypeError("tool_search.unlocked 必须是字符串数组")
        names: list[str] = []
        for index, item in enumerate(cast(list[object], raw_names)):
            if not isinstance(item, str) or not item or item != item.strip():
                raise TypeError(f"tool_search.unlocked[{index}] 必须是非空工具名")
            names.append(item)
        return list(dict.fromkeys(names))

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

        if session_key not in self._unlocked:
            return
        self._unlocked.move_to_end(session_key)
        while len(self._unlocked) > self.session_capacity:
            evicted, _ = self._unlocked.popitem(last=False)
            logger.info("[LRU驱逐] 移除最旧会话工具缓存: %s", evicted)


class SessionLike(Protocol):
    key: str
    created_at: datetime
    messages: list[dict[str, object]]
    metadata: dict[str, object]
    last_consolidated: int

    def get_history(self, max_messages: int = 500) -> list[dict[str, object]]: ...
    def issue_projection_grant(self, turn_id: str) -> object: ...

    def revoke_projection_grant(self, grant: object) -> None: ...
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
