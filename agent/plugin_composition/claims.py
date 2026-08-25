from __future__ import annotations

from agent.plugin_composition.model import ServiceKey


# Marker only: vector-backed providers declare a mutually exclusive role.
EMBEDDING_MEMORY_PLUGIN = ServiceKey[object]("plugin.claim.embedding_memory")
