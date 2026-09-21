from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(eq=False)
class ManagerOperation:
    """保留实际操作任务及其一次性的提交许可，不记录资源阶段。"""

    deadline: float
    task: asyncio.Task[Any] = field(init=False)
    revoked: bool = False
    committed: object | None = None
    candidate_shared: bool = False

    def revoke(self, *, cancel: bool = True) -> None:
        """先永久撤销许可；最多向操作任务发送一次取消。"""
        if self.revoked:
            return
        self.revoked = True
        if cancel and not self.task.done():
            self.task.cancel()


current_operation: ContextVar[ManagerOperation | None] = ContextVar(
    "plugin_manager_operation", default=None,
)


def revoke_on_cancel(error: BaseException) -> None:
    """插件自己抛出的取消也不能被恢复路径转换成新的提交许可。"""
    if isinstance(error, asyncio.CancelledError) or (
        isinstance(error, BaseExceptionGroup) and error.subgroup(asyncio.CancelledError) is not None
    ):
        operation = current_operation.get()
        if operation is not None:
            operation.revoke(cancel=False)


class OperationBusyError(RuntimeError):
    """另一操作仍持有任务或资源，普通调用不能排队越过它。"""


class OperationTimeoutError(TimeoutError):
    def __init__(self, operation: ManagerOperation) -> None:
        self.operation = operation
        super().__init__(
            "插件操作超过截止时间；"
            + ("已提交，运行恢复仍需处理" if operation.committed else "提交许可已永久撤销")
            + "；未结束的任务和资源仍由 Manager 持有"
        )


async def run_operation(operation: ManagerOperation, work: Callable[[], Awaitable[T]]) -> T:
    """内部调用及资源关闭继承同一许可，公开入口不重新取得 owner。"""
    token = current_operation.set(operation)
    try:
        return await work()
    except BaseException as error:
        revoke_on_cancel(error)
        raise
    finally:
        current_operation.reset(token)


class _OperationWaiter(asyncio.Future[None]):
    """Task.cancel 同步到达这个实际等待句柄，不能等下一次调度才撤销许可。"""

    def __init__(self, operation: ManagerOperation, *, cancel_worker: bool) -> None:
        super().__init__()
        self.operation = operation
        self.cancel_worker = cancel_worker

    def cancel(self, msg: object = None) -> bool:
        self.operation.revoke(cancel=self.cancel_worker)
        return super().cancel(msg)


async def observe_operation(
    operation: ManagerOperation, *, deadline: float, cancel: bool = True,
) -> Any:
    """有限观察实际任务；超时不等待吞取消的任务退出，也不移除其 owner。"""
    waiter = _OperationWaiter(operation, cancel_worker=cancel)

    def finished(_task: asyncio.Task[Any]) -> None:
        if not waiter.done():
            waiter.set_result(None)

    def expired() -> None:
        operation.revoke(cancel=cancel)
        if not waiter.done():
            waiter.set_result(None)

    operation.task.add_done_callback(finished)
    timer = asyncio.get_running_loop().call_at(deadline, expired)
    try:
        await waiter
    except asyncio.CancelledError as error:
        operation.revoke(cancel=cancel)
        if operation.committed:
            error.add_note("插件操作已经提交；取消不撤销该提交")
        if operation.task.done() and not operation.task.cancelled():
            failure = operation.task.exception()
            if failure is not None:
                raise BaseExceptionGroup("调用取消且操作失败", [error, failure]) from None
        raise
    finally:
        timer.cancel()
        operation.task.remove_done_callback(finished)
    if not operation.task.done() or asyncio.get_running_loop().time() >= deadline:
        operation.revoke(cancel=cancel)
        error = OperationTimeoutError(operation)
        if operation.task.done() and not operation.task.cancelled():
            failure = operation.task.exception()
            if isinstance(failure, OperationTimeoutError) and failure.operation is operation:
                raise failure
            if failure is not None:
                raise BaseExceptionGroup("操作超时且资源操作失败", [error, failure]) from None
        raise error
    try:
        return operation.task.result()
    except asyncio.CancelledError as error:
        if operation.committed:
            error.add_note("插件操作已经提交；取消不撤销该提交")
        raise


async def complete_critical(awaitable: Awaitable[T]) -> tuple[T, bool]:
    """实际 owner 等待已开始的关闭或线程；公开调用的截止由观察者负责。"""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            # wait 不向被等待任务传播取消；线程结束前不能释放它仍在写的目录。
            await asyncio.wait((task,))
        except asyncio.CancelledError:
            cancelled = True
            operation = current_operation.get()
            if operation is not None:
                operation.revoke(cancel=False)
    try:
        return task.result(), cancelled
    except BaseException as error:
        revoke_on_cancel(error)
        if cancelled and isinstance(error, Exception):
            raise BaseExceptionGroup("操作取消且资源清理失败", [asyncio.CancelledError(), error]) from None
        raise
