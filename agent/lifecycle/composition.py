from __future__ import annotations

from typing import TypeVar

from agent.plugin_composition import CompositionError, EmitEventKey, SerialEventKey
from agent.plugins.snapshot import get_current_runtime_snapshot

P = TypeVar("P")


def emit_turn_event(
    key: EmitEventKey[P],
    payload: P,
) -> None:
    """从请求冻结的 composition Root 分发一个同步 Turn 事件。"""

    # 1. 没有 composition Root 的 bootstrap 和 legacy snapshot 保持原行为
    snapshot = get_current_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        return

    # 2. 同步事件沿 generation 内注册顺序执行并立即传播失败
    snapshot.composition_root.context.emit(key, payload)


async def run_turn_stage_event(
    key: SerialEventKey[P, object],
    payload: P,
) -> None:
    """从请求冻结的 composition Root 分发一个 Turn 阶段事件。"""

    # 1. 没有 composition Root 的 bootstrap 和 legacy snapshot 保持原行为
    snapshot = get_current_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        return

    # 2. 阶段事件只允许有序转换，不能终止 Core Turn
    result = await snapshot.composition_root.context.serial(key, payload)
    if result is not None:
        raise CompositionError(
            "TURN_STAGE_BAIL_NOT_ALLOWED",
            f"Turn 阶段事件不接受 Bail: {key.name}",
        )
