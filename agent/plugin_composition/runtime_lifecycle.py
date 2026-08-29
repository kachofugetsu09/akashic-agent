from __future__ import annotations

from dataclasses import dataclass

from agent.plugin_composition.events import SerialEventKey


@dataclass(frozen=True, slots=True)
class RuntimeStarted:
    """Signal that formal external services are ready for plugin-owned work."""


@dataclass(frozen=True, slots=True)
class RuntimeStopping:
    """Signal that plugin-owned work must settle before external services stop."""


@dataclass(frozen=True, slots=True)
class SnapshotSealing:
    """Signal that a ready candidate must freeze its private registries."""


RUNTIME_STARTED = SerialEventKey[RuntimeStarted, object]("runtime.started")
RUNTIME_STOPPING = SerialEventKey[RuntimeStopping, object]("runtime.stopping")
SNAPSHOT_SEALING = SerialEventKey[SnapshotSealing, object]("snapshot.sealing")
