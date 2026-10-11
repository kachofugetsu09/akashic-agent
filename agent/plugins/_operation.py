from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict, TypeVar

from agent.plugin_composition.plugin_updates import UpdateStatus

T = TypeVar("T")
AcceptedFuture = asyncio.Future[UpdateStatus] | asyncio.Future[dict[str, object]]


class OperationStatus(TypedDict):
    kind: str
    plugin_id: str | None
    update_id: str | None
    state: str
    error: str | None
    revoked: bool
    committed: bool
    accepted: dict[str, object] | None


@dataclass(eq=False)
class ManagerOperation:
    """保留实际操作任务及其一次性的提交许可，不记录资源阶段。"""

    deadline: float
    update_id: str | None = None
    accepted: AcceptedFuture | None = None
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

    def settle_accepted(self, error: BaseException) -> None:
        """结束调用方的 accepted 等待，但保留操作任务和其资源 owner。"""
        if self.accepted is None or self.accepted.done():
            return
        self.accepted.set_exception(error)
        # A disconnected caller cannot retrieve this future; mark the error
        # observed now while the future remains awaitable for a live caller.
        _ = self.accepted.exception()

    def status(self) -> OperationStatus:
        """从实际任务和 accepted 结果读取状态，不记录第二份进度。"""
        # 1. 任务是否结束与选择是否 accepted 是两个独立事实。
        if self.task.cancelled():
            task_state = "cancelled"
            task_error = None
        elif not self.task.done():
            task_state = "running"
            task_error = None
        else:
            task_error_value = self.task.exception()
            task_state = "error" if task_error_value is not None else "done"
            task_error = (
                None
                if task_error_value is None
                else str(task_error_value) or type(task_error_value).__name__
            )

        # 2. 只从已返回的结果识别插件，pending 请求不猜测目标。
        view: OperationStatus = {
            "kind": "install" if self.update_id is not None else "unknown",
            "plugin_id": None,
            "update_id": self.update_id,
            "state": task_state,
            "error": task_error,
            "revoked": self.revoked,
            "committed": self.committed is not None,
            "accepted": None,
        }
        self._add_accepted_status(view)
        return view

    def _add_accepted_status(self, view: OperationStatus) -> None:
        """把 accepted 等待或结果补入本次诊断快照。"""
        if self.accepted is None:
            return
        if self.accepted.cancelled():
            view["accepted"] = {"state": "cancelled"}
            return
        if not self.accepted.done():
            view["accepted"] = {"state": "pending"}
            return
        try:
            result = self.accepted.result()
        except asyncio.CancelledError:
            view["accepted"] = {"state": "cancelled"}
        except Exception as error:
            view["accepted"] = {"state": "error", "error": str(error) or type(error).__name__}
        else:
            if isinstance(result, UpdateStatus):
                view["kind"] = "install"
                view["plugin_id"] = result.plugin_id
                view["accepted"] = {
                    "kind": "install",
                    "state": "accepted",
                    "result": asdict(result),
                }
            else:
                # 卸载结果也交给调用方，读取时仍需校验这个可变字典。
                plugin_id = result.get("plugin_id")
                state = result.get("state")
                if not isinstance(plugin_id, str):
                    raise TypeError("ManagerOperation.accepted uninstall 结果缺少 plugin_id")
                if not isinstance(state, str):
                    raise TypeError("ManagerOperation.accepted uninstall 结果缺少 state")
                view["kind"] = "uninstall"
                view["plugin_id"] = plugin_id
                view["accepted"] = {
                    "kind": "uninstall",
                    "state": state,
                    "result": dict(result),
                }


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
