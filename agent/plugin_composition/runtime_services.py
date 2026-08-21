from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

from agent.plugin_composition.model import ServiceKey
from core.memory.plugin import ActiveRecallView, MemoryTurnRuntimeApi


@dataclass(frozen=True, slots=True)
class MemoryRuntimeInfo:
    """Describe the selected Core Memory runtime without exposing its engine."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("MemoryRuntimeInfo.name 必须是非空且无首尾空白的字符串")


MEMORY_RUNTIME = ServiceKey[MemoryRuntimeInfo]("core.memory.runtime")


class MemoryTurnRuntime:
    """向 exact Root 暴露所选 Memory engine 的窄 Turn 能力。"""

    def __init__(self, runtime: MemoryTurnRuntimeApi | None) -> None:
        self._runtime = runtime

    @classmethod
    def candidate_validation(cls) -> MemoryTurnRuntime:
        """保留候选拓扑，但拒绝读取或消费正式 Memory runtime。"""

        return cls(None)

    @property
    def formal(self) -> bool:
        """Return whether this service owns the selected formal Turn runtime."""

        return self._runtime is not None

    def take_user_metadata(self, turn_id: str) -> MappingProxyType[str, object]:
        """消费一个正式 Turn 的插件 metadata，并断开可变对象引用。"""

        runtime = self._require_formal()
        metadata = dict(runtime.take_turn_user_metadata(turn_id))
        if any(not isinstance(key, str) or not key for key in metadata):
            raise RuntimeError("Memory Turn metadata key 必须是非空字符串")
        return MappingProxyType(deepcopy(metadata))

    def wait_active_recall(
        self,
        session_key: str,
        turn_id: str,
    ) -> ActiveRecallView | None:
        """读取一个正式 Turn 的有界 active recall 快照。"""

        return self._require_formal().wait_active_recall(session_key, turn_id)

    def _require_formal(self) -> MemoryTurnRuntimeApi:
        if self._runtime is None:
            raise RuntimeError("candidate 验证期禁止访问正式 Memory Turn runtime")
        return self._runtime


MEMORY_TURN_RUNTIME = ServiceKey[MemoryTurnRuntime]("core.memory.turn_runtime")
