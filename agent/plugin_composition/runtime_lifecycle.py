from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypeVar

from agent.plugin_composition.events import EmitEventKey, SerialEventKey


@dataclass(frozen=True, slots=True)
class RuntimeStarting:
    """本 owner activation 的启动前事件：每次 activation 由内核按准确

    owner Fiber 派发一次，listener 在 LOADING 中先于 required health
    检查执行；不接受 Bail。"""


@dataclass(frozen=True, slots=True)
class RuntimeStarted:
    """本 owner activation 的启动事件：真实启动代码在此执行，随后才

    做 required health 检查并置 ACTIVE；每次 activation 只发一次，
    不接受 Bail。"""


@dataclass(frozen=True, slots=True)
class RuntimeStopping:
    """本 owner activation 的停止事件：在本 owner 排空与 Effect 逆序

    清理之间派发；成功且非 Bail 后该 activation 记 stopping_completed，
    后续清理失败的重试不重发；不接受 Bail。"""


RUNTIME_STARTING = SerialEventKey[RuntimeStarting, object]("runtime.starting")
RUNTIME_STARTED = SerialEventKey[RuntimeStarted, object]("runtime.started")
RUNTIME_STOPPING = SerialEventKey[RuntimeStopping, object]("runtime.stopping")

_T = TypeVar("_T")


# 停止时收尾 ctx.spawn 出来的任务：spawn 早已把任务自己结束时的失败上报给就绪状态，
# 停止只取消仍在运行的任务；早先的业务失败不能再被当成停止失败，否则关停清理记为失败，下次开机要人工结算。
async def stop_spawned(task: asyncio.Task[_T] | None) -> None:
    if task is None:
        return
    cancel_requested = task.cancel() if not task.done() else False
    try:
        await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
    except Exception:
        if cancel_requested:
            raise
