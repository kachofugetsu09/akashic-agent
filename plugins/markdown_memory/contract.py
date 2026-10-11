"""Markdown 记忆公开已提交写入的只读历史，供 Fleet Observe 展示。"""
from collections.abc import Callable
from agent.plugin_composition import ServiceKey


MEMORY_WRITES = ServiceKey[
    Callable[[tuple[str, str] | None, int], tuple[dict[str, object], ...]]
]("markdown-memory.writes.v1")
