from __future__ import annotations

from agent.plugins import Plugin


class MemoryRollup(Plugin):
    name = "memory_rollup"
    version = "0.1.0"
    desc = "人工 review memory2 的长期记忆候选"
