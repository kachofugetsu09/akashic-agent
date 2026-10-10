"""把实际 Controller 请求固定到贡献 Context 的 owner 与请求身份。"""
from __future__ import annotations

import secrets
from typing import Literal
from agent.plugin_composition.context import Context
from agent.plugin_composition.execution import EXECUTION
from plugins.host_execution.contract import WorkloadLease, WorkloadStartRequest
from .controller_client import WorkloadController


async def cleanup_workloads_for_boot(
    controller: WorkloadController | None, workspace_id: str,
) -> None:
    """执行 provider 启动时清理旧候选；未确认容器和挂载释放时阻止启动。"""
    if controller is None:
        return
    # 1. 沿当前启动 operation 等待宿主；不另起任务，也不重放 start。
    receipts = await controller.cleanup_candidates(workspace_id)
    # 2. 每份回执必须属于本 workspace 的候选，且两项释放都已确认。
    incomplete = tuple(receipt for receipt in receipts if (
        receipt.lease.workspace_id != workspace_id
        or receipt.lease.mode != "candidate"
        or not receipt.container_absent
        or not receipt.mounts_released
    ))
    if incomplete:
        raise RuntimeError(
            f"Workload boot cleanup 未确认: workspace={workspace_id} receipts={incomplete!r}"
        )


class ControllerAccess:
    def __init__(self, ctx: Context, controller: WorkloadController | None, workspace_id: str):
        self._ctx = ctx
        self._controller = controller
        self._workspace_id = workspace_id

    def bind(self, ctx: Context) -> ControllerGrant:
        """每份授权固定 owner 与请求身份，不向插件交出原始 Controller。"""
        execution = self._ctx.require(EXECUTION).bind(ctx)
        if self._controller is None:
            raise RuntimeError("所选资源 provider 需要宿主 Workload Controller")
        return ControllerGrant(self._controller, ctx.runtime.plugin_id, execution.mode, self._workspace_id)


class ControllerGrant:
    def __init__(self, controller: WorkloadController, owner: str, mode: Literal["candidate", "formal"], workspace_id: str):
        self._controller = controller
        self._owner = owner
        self._mode: Literal["candidate", "formal"] = mode
        self._workspace_id = workspace_id
        self._identity = "resource-" + secrets.token_hex(16)

    @property
    def mode(self) -> Literal["candidate", "formal"]:
        return self._mode

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    @property
    def identity(self) -> str:
        return self._identity

    def _check(self, value: WorkloadStartRequest | WorkloadLease) -> None:
        if (value.workspace_id, value.plugin_id, value.mode, value.transaction_id, value.generation_id) != (
            self._workspace_id, self._owner, self._mode, self._identity, self._identity,
        ):
            raise PermissionError("Controller 请求超出当前 Context 的授权")

    async def start(self, request: WorkloadStartRequest):
        self._check(request)
        return await self._controller.start(request)

    async def stop(self, lease: WorkloadLease):
        self._check(lease)
        return await self._controller.stop(lease)

