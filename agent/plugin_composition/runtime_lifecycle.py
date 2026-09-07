from __future__ import annotations

from dataclasses import dataclass

from agent.plugin_composition.events import EmitEventKey, SerialEventKey


@dataclass(frozen=True, slots=True)
class RuntimeStarting:
    """在正式接纳开放前同步恢复临时资源；不启动外部工作。"""


@dataclass(frozen=True, slots=True)
class RuntimeStarted:
    """Signal that formal external services are ready for plugin-owned work."""


@dataclass(frozen=True, slots=True)
class RuntimeStopping:
    """Signal that plugin-owned work must settle before external services stop."""


@dataclass(frozen=True, slots=True)
class SnapshotSealing:
    """Signal that a ready candidate must freeze its private registries."""


RUNTIME_STARTING = EmitEventKey[RuntimeStarting]("runtime.starting")
RUNTIME_STARTED = SerialEventKey[RuntimeStarted, object]("runtime.started")
RUNTIME_STOPPING = SerialEventKey[RuntimeStopping, object]("runtime.stopping")
SNAPSHOT_SEALING = SerialEventKey[SnapshotSealing, object]("snapshot.sealing")
