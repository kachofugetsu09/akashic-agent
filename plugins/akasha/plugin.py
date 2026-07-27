"""Plugin discovery exports for Akasha V2."""

from agent.plugins import Plugin

from .memory_plugin import MemoryPlugin


class AkashaPlugin(Plugin):
    """Register the V2 package without legacy inspector contributions."""

    name = "akasha"


__all__ = ["AkashaPlugin", "MemoryPlugin"]
