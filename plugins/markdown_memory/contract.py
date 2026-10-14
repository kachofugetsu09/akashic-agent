"""Markdown 记忆公开已提交写入的只读历史；读者不取得写入权限。"""
from collections.abc import Callable
from agent.plugin_composition import ServiceKey


MEMORY_WRITES = ServiceKey[
    Callable[[tuple[str, str] | None, int], tuple[dict[str, object], ...]]
]("markdown-memory.writes.v1")
