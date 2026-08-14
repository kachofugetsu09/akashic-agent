from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, cast

from agent.plugin_composition.model import ServiceKey


class _SessionState(Protocol):
    messages: list[dict[str, Any]]
    last_consolidated: int


ExistingSessionLookup = Callable[[str], _SessionState]


@dataclass(frozen=True, slots=True)
class SessionReadSnapshot:
    session_key: str
    messages: tuple[Mapping[str, object], ...]
    last_consolidated: int


class SessionReadService:
    """读取既有 Session 的脱离快照，不暴露持久化 owner。"""

    def __init__(self, lookup_existing: ExistingSessionLookup) -> None:
        self._lookup_existing = lookup_existing

    def read(self, session_key: str) -> SessionReadSnapshot | None:
        """返回一个既有 Session 快照，不创建持久状态。"""

        # 1. Session 不存在是查询结果，其余持久化错误继续暴露
        try:
            session = self._lookup_existing(session_key)
        except KeyError:
            return None

        # 2. 复制可变消息，避免插件获得持久化 owner 的对象引用
        messages = tuple(
            MappingProxyType(cast(dict[str, object], deepcopy(message)))
            for message in session.messages
        )
        return SessionReadSnapshot(
            session_key=session_key,
            messages=messages,
            last_consolidated=session.last_consolidated,
        )


SESSION_READ = ServiceKey[SessionReadService]("core.session_read")
