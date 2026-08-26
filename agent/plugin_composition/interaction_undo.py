from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias

from agent.plugin_composition.model import ServiceKey


@dataclass(frozen=True, slots=True)
class InteractionUndoResult:
    """记录一次已提交 interaction 删除及其派生收敛状态。"""

    control_turn_id: str
    session_key: str
    message_ids: tuple[str, ...]
    backup_path: str
    reconciliation_pending: bool
    old_last_consolidated: int
    new_last_consolidated: int


SourceDelete: TypeAlias = Callable[[], object | None]
SourceMutationFence: TypeAlias = Callable[[str, SourceDelete], Awaitable[object | None]]
UndoLatest: TypeAlias = Callable[
    [str, SourceMutationFence | None], Awaitable[InteractionUndoResult | None]
]


class InteractionUndoService:
    """暴露 Core-owned interaction 撤销，并允许派生插件安装一致性围栏。"""

    def __init__(self, undo_latest: UndoLatest | None) -> None:
        self._undo_latest = undo_latest
        self._source_fence: SourceMutationFence | None = None

    @classmethod
    def candidate_validation(cls) -> InteractionUndoService:
        """创建只有拓扑身份、没有 destructive owner 的候选服务。"""

        return cls(None)

    def bind_source_fence(
        self,
        fence: SourceMutationFence,
    ) -> Callable[[], None]:
        """绑定唯一派生状态围栏，并返回精确解绑动作。"""

        if self._source_fence is not None:
            raise RuntimeError("interaction undo source fence 已有 owner")
        self._source_fence = fence

        def cleanup() -> None:
            if self._source_fence is not fence:
                raise RuntimeError("interaction undo source fence owner 已漂移")
            self._source_fence = None

        return cleanup

    async def undo_latest(self, session_key: str) -> InteractionUndoResult | None:
        """撤销一个既有 Session 最后的 completed interaction。"""

        if self._undo_latest is None:
            raise RuntimeError("candidate 验证期禁止撤销正式 interaction")
        return await self._undo_latest(session_key, self._source_fence)


INTERACTION_UNDO = ServiceKey[InteractionUndoService]("core.interaction_undo")


__all__ = [
    "INTERACTION_UNDO",
    "InteractionUndoResult",
    "InteractionUndoService",
    "SourceDelete",
    "SourceMutationFence",
]
