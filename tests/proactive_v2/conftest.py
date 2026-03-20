"""
共享测试设施：FakeStateStore、FakeRng、FakeLLM、make_agent_tick、cfg_with
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from proactive.config import ProactiveConfig
from proactive_v2.tools import ToolDeps


# ── FakeStateStore ────────────────────────────────────────────────────────

class FakeStateStore:
    """ProactiveStateStore 的最小 fake，只实现 AgentTick 需要的接口。"""

    def __init__(self):
        self._delivery_count: int = 0
        self._is_dup: bool = False
        self._last_context_only_at: datetime | None = None
        self._ctx_only_count: int = 0
        self.context_only_send_marked: bool = False
        self._deliveries: list[str] = []

    # pre-gate
    def count_deliveries_in_window(self, session_key: str, window_hours: int) -> int:
        return self._delivery_count

    # post-guard
    def is_delivery_duplicate(self, session_key: str, delivery_key: str, window_hours: int) -> bool:
        return self._is_dup

    def mark_delivery(self, session_key: str, delivery_key: str) -> None:
        self._deliveries.append(delivery_key)

    # context gate
    def get_last_context_only_at(self, session_key: str) -> datetime | None:
        return self._last_context_only_at

    def count_context_only_in_window(self, session_key: str, window_hours: int) -> int:
        return self._ctx_only_count

    def mark_context_only_send(self, session_key: str) -> None:
        self.context_only_send_marked = True

    # helpers
    def set_delivery_count(self, n: int) -> None:
        self._delivery_count = n

    def set_is_duplicate(self, v: bool) -> None:
        self._is_dup = v

    def set_last_context_only_at(self, dt: datetime | None) -> None:
        self._last_context_only_at = dt

    def set_context_only_count(self, n: int) -> None:
        self._ctx_only_count = n


# ── FakeRng ───────────────────────────────────────────────────────────────

class FakeRng:
    def __init__(self, value: float = 0.5):
        self._value = value

    def random(self) -> float:
        return self._value


# ── FakeAckSink ──────────────────────────────────────────────────────────

class FakeAckSink:
    """记录所有 ACK 调用的 (compound_key, ttl_hours) 对。"""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, compound_key: str, ttl_hours: int) -> None:
        self.calls.append((compound_key, ttl_hours))

    def acked(self, key: str, ttl: int) -> bool:
        return (key, ttl) in self.calls

    def ttls_for(self, key: str) -> list[int]:
        return [ttl for k, ttl in self.calls if k == key]

    def not_acked(self, key: str) -> bool:
        return all(k != key for k, _ in self.calls)

    def all_keys(self) -> set[str]:
        return {k for k, _ in self.calls}


# ── FakeAlertAckSink ─────────────────────────────────────────────────────

class FakeAlertAckSink:
    """记录 alert_ack_fn 调用的 compound_key 列表（无 TTL）。"""

    def __init__(self):
        self.keys: list[str] = []

    async def __call__(self, compound_key: str) -> None:
        self.keys.append(compound_key)

    def all_keys(self) -> set[str]:
        return set(self.keys)

    def called_with(self, key: str) -> bool:
        return key in self.keys


# ── FakeLLM ──────────────────────────────────────────────────────────────

class FakeLLM:
    """预定义工具调用序列。序列耗尽后返回 None（loop 自然结束）。"""

    def __init__(self, sequence: list[tuple[str, dict]]):
        self._sequence = list(sequence)
        self._index = 0
        self.calls: list[list[dict]] = []   # 每次 llm 调用收到的 messages

    async def __call__(self, messages: list[dict], schemas: list[dict]) -> dict | None:
        self.calls.append(list(messages))
        if self._index >= len(self._sequence):
            return None
        name, args = self._sequence[self._index]
        self._index += 1
        return {"name": name, "input": args}


# ── cfg_with ──────────────────────────────────────────────────────────────

def cfg_with(**kwargs) -> ProactiveConfig:
    """从默认 ProactiveConfig 创建，只覆盖指定字段。"""
    return ProactiveConfig(**kwargs)


# ── make_agent_tick ───────────────────────────────────────────────────────

def make_agent_tick(
    *,
    cfg: ProactiveConfig | None = None,
    session_key: str = "test_session",
    state_store: FakeStateStore | None = None,
    any_action_gate: Any = None,
    last_user_at_fn: Any = None,
    passive_busy_fn: Any = None,
    sender: Any = None,
    deduper: Any = None,
    tool_deps: ToolDeps | None = None,
    llm_fn: Any = None,
    rng: Any = None,
    recent_proactive_fn: Any = None,
):
    from proactive_v2.agent_tick import AgentTick

    # 合理的默认值：所有 gate 都放行
    if state_store is None:
        state_store = FakeStateStore()
        state_store.set_delivery_count(0)

    if any_action_gate is None:
        gate = MagicMock()
        gate.should_act.return_value = (True, {})
        any_action_gate = gate

    if sender is None:
        sender = AsyncMock()
        sender.send.return_value = True

    if deduper is None:
        deduper = AsyncMock()
        deduper.is_duplicate = AsyncMock(return_value=(False, ""))

    if tool_deps is None:
        tool_deps = ToolDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[]),
            context_fn=AsyncMock(return_value=[]),
            recent_chat_fn=AsyncMock(return_value=[]),
        )

    if rng is None:
        rng = FakeRng(value=1.0)  # random() > context_prob → gate 关

    return AgentTick(
        cfg=cfg or ProactiveConfig(),
        session_key=session_key,
        state_store=state_store,
        any_action_gate=any_action_gate,
        last_user_at_fn=last_user_at_fn or (lambda: None),
        passive_busy_fn=passive_busy_fn,
        sender=sender,
        deduper=deduper,
        tool_deps=tool_deps,
        llm_fn=llm_fn,
        rng=rng,
        recent_proactive_fn=recent_proactive_fn,
    )
