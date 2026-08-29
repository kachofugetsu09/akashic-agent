from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast

from agent.plugin_composition.model import ServiceKey


class _SessionState(Protocol):
    messages: list[dict[str, object]]
    last_consolidated: int


class _CompactionState(Protocol):
    generation: int
    consolidated_through_seq: int


ExistingSessionLookup = Callable[
    [str], tuple[_SessionState, _CompactionState | None]
]


@dataclass(frozen=True, slots=True)
class SessionReadSnapshot:
    session_key: str
    messages: tuple[Mapping[str, object], ...]
    compaction_generation: int | None
    consolidated_through_seq: int | None


class SessionReadService:
    """读取既有 Session 的脱离快照，不暴露持久化 owner。"""

    def __init__(self, lookup_existing: ExistingSessionLookup | None) -> None:
        self._lookup_existing = lookup_existing

    @classmethod
    def candidate_validation(cls) -> SessionReadService:
        """创建只保留拓扑、拒绝读取正式 Session 的候选服务。"""

        return cls(None)

    def read(self, session_key: str) -> SessionReadSnapshot | None:
        """返回一个既有 Session 快照，不创建持久状态。"""

        # 1. candidate 只验证能力接线，禁止读取正式 Session
        if self._lookup_existing is None:
            raise RuntimeError("candidate 验证期禁止读取正式 Session")

        # 2. Session 不存在是查询结果，其余持久化错误继续暴露
        try:
            session, compaction = self._lookup_existing(session_key)
        except KeyError:
            return None

        expected_generation = 0 if compaction is None else compaction.generation
        if session.last_consolidated != expected_generation:
            raise RuntimeError(
                "Session 与 active compaction generation 不一致: "
                f"{session_key}:{session.last_consolidated}!={expected_generation}"
            )

        # 3. 复制可变消息，避免插件获得持久化 owner 的对象引用
        messages = tuple(
            MappingProxyType(cast(dict[str, object], deepcopy(message)))
            for message in session.messages
        )
        return SessionReadSnapshot(
            session_key=session_key,
            messages=messages,
            compaction_generation=(
                None if compaction is None else compaction.generation
            ),
            consolidated_through_seq=(
                None if compaction is None else compaction.consolidated_through_seq
            ),
        )


SESSION_READ = ServiceKey[SessionReadService]("core.session_read")
