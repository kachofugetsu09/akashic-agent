"""
bus/processing.py — 会话级被动处理信号

AgentLoop 在处理每条入站消息时调用 enter/exit；
ProactiveLoop/AgentTick 在发送前调用 is_busy(session_key) 检查目标会话，
避免被动回复正在运行时同时发出主动消息。

设计约束：
- 只在单一 asyncio 事件循环中使用，enter/exit 之间无 await，操作是原子的。
- session_key 作用域：A 会话 busy 不影响 B 会话的 proactive 判断。
"""

from __future__ import annotations


class ProcessingState:
    """会话级被动处理计数器。"""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def enter(self, session_key: str) -> None:
        """标记 session_key 开始处理被动消息。"""
        self._counts[session_key] = self._counts.get(session_key, 0) + 1

    def exit(self, session_key: str) -> None:
        """标记 session_key 完成处理被动消息。"""
        self._counts[session_key] = max(0, self._counts.get(session_key, 0) - 1)

    def is_busy(self, session_key: str) -> bool:
        """返回目标会话当前是否正在处理被动回复。"""
        return self._counts.get(session_key, 0) > 0
