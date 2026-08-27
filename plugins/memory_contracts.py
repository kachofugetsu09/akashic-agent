"""Shared service keys for memory plugins and their consumers."""

from agent.plugin_composition import ServiceKey

MEMORY_RECALL = ServiceKey[object]("memory.recall.v1")

__all__ = ["MEMORY_RECALL"]
