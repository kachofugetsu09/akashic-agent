"""把宿主执行权限绑定到真实 Root/Context；这不是同 UID Python 沙箱。"""
from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent.plugin_composition.context import Context
from agent.plugin_composition.execution import (
    EXECUTION,
    WORKLOAD_CONTROLLER,
    ChildProcess,
    PreparedProcess,
    WorkloadLease,
    WorkloadStartRequest,
)
from agent.workloads.client import WorkloadController
from utils.process_group import (
    OwnedProcessGroup,
    owned_process_env,
    process_group_spawn_kwargs,
)


async def cleanup_workloads_for_boot(
    controller: WorkloadController | None, workspace_id: str,
) -> None:
    """真实 Core boot 清理旧候选；未确认容器和挂载释放时阻止启动。"""
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


# 名称层永久保留的执行环境键：宿主继承来源 + 数据根/工作区合同键。
_RESERVED_ENV_NAMES = frozenset({
    "PATH", "PYTHONPATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ",
    "AKASHIC_BOOT_ID", "AKASHIC_SUPERVISED",
    "HOME", "AKA_PLUGIN_DATA_DIR", "AKASHIC_PLUGIN_DATA_DIR", "AKASHIC_WORKSPACE",
})


@dataclass(frozen=True)
class CodeOwner:
    generation_id: str
    code_dir: Path
    command: Callable[[tuple[str, ...], str], tuple[str, ...]]


class ExecutionAccess:
    def __init__(self, root_token: object, owners: Mapping[object, CodeOwner], *, candidate: bool):
        self._root_token = root_token
        self._owners: dict[tuple[str, str], CodeOwner] = {}
        for key, owner in owners.items():
            if isinstance(key, tuple):
                self._owners[key] = owner
            else:
                self._owners[(str(key), owner.generation_id)] = owner
        self._mode: Literal["candidate", "formal"] = "candidate" if candidate else "formal"
        self._environment = {key: os.environ[key] for key in (
            "PATH", "PYTHONPATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ",
            "AKASHIC_BOOT_ID", "AKASHIC_SUPERVISED",
        ) if key in os.environ}

    def add_owner(self, plugin_id: str, generation_id: str, owner: CodeOwner) -> None:
        """Add one exact generation owner without rebuilding the Root facade."""
        key = (plugin_id, generation_id)
        if key in self._owners:
            raise RuntimeError(f"执行 owner 已存在: {plugin_id}/{generation_id}")
        self._owners[key] = owner

    def remove_owner(self, plugin_id: str, generation_id: str) -> None:
        """Remove only the generation whose Fiber and resources are already closed."""
        self._owners.pop((plugin_id, generation_id), None)

    def bind(self, ctx: Context) -> ExecutionGrant:
        """验证实际代码 owner，不接受自报插件名、模式或数据根。"""
        if ctx.root_instance_token is not self._root_token or ctx.require(EXECUTION) is not self:
            raise PermissionError("执行授权不能跨 Root")
        runtime = ctx.runtime
        owner = self._owners.get((runtime.plugin_id, runtime.generation_id))
        if owner is None or owner.code_dir.resolve() != runtime.plugin_dir.resolve():
            raise PermissionError("执行 Context 不属于固定代码制品")
        return ExecutionGrant(ctx, owner, self._mode, self._environment)


class ExecutionGrant:
    def __init__(self, ctx: Context, owner: CodeOwner, mode: Literal["candidate", "formal"], environment: Mapping[str, str]):
        self._ctx = ctx
        self._owner = owner
        self._mode: Literal["candidate", "formal"] = mode
        self._environment = dict(environment)
        self._issue_token = object()

    @property
    def mode(self) -> Literal["candidate", "formal"]:
        return self._mode

    def command(self, command: tuple[str, ...], cwd: str) -> tuple[str, ...]:
        self.cwd(cwd)
        resolved = self._owner.command(command, cwd)
        if not resolved or not Path(resolved[0]).is_absolute() or not Path(resolved[0]).is_file():
            raise RuntimeError("执行命令没有固定的可执行制品")
        return resolved

    def cwd(self, relative: str) -> Path:
        root = self._owner.code_dir.resolve(strict=True)
        value = (root / relative).resolve(strict=True)
        if not value.is_relative_to(root) or not value.is_dir():
            raise PermissionError("执行 cwd 越出固定代码制品")
        return value

    def environment(self, values: Mapping[str, str], candidate_values: Mapping[str, str]) -> dict[str, str]:
        """候选只取显式候选输入；宿主固定键由本授权钉住，调用方不得覆盖。"""
        runtime = self._ctx.runtime
        fixed = dict(self._environment)
        fixed.update({
            "HOME": str(runtime.data_dir),
            "AKA_PLUGIN_DATA_DIR": str(runtime.data_dir),
            "AKASHIC_PLUGIN_DATA_DIR": str(runtime.data_dir),
            "AKASHIC_WORKSPACE": str(runtime.workspace),
        })
        # 固定键在名称层永久保留：宿主继承键与数据根/身份合同键一律
        # 由本授权钉住，调用方在 env/candidate_env 提供同名键即拒绝
        # （含宿主未设置时的引入），候选模式忽略 formal 输入不等于默许。
        conflicts = sorted({
            key
            for source in (values, candidate_values)
            for key in source
            if key in _RESERVED_ENV_NAMES
        })
        if conflicts:
            raise PermissionError(
                "执行环境键由宿主固定，调用方不得提供: " + ", ".join(conflicts)
            )
        result = dict(candidate_values if self._mode == "candidate" else values)
        result.update(fixed)
        return result

    def prepare_process(
        self,
        command: tuple[str, ...],
        cwd: str,
        env: Mapping[str, str],
        candidate_env: Mapping[str, str] = {},
    ) -> PreparedProcess:
        """一次完成 command/cwd/environment 三项校验并签发冻结制品；
        provider 须先备好端口/endpoint/scope 等运行期材料并入 env 输入。"""
        return PreparedProcess(
            self._issue_token,
            command=self.command(command, cwd),
            cwd=str(self.cwd(cwd)),
            env=self.environment(env, candidate_env),
        )

    async def spawn(
        self,
        prepared: PreparedProcess,
        *,
        stdin: object = None,
        stdout: object = None,
        stderr: object = None,
        limit: int | None = None,
    ) -> tuple[ChildProcess, bool]:
        """只消费本授权签发的 PreparedProcess；取消仍先接住实际回执。"""
        if type(prepared) is not PreparedProcess or not prepared._issued_by(self._issue_token):
            raise PermissionError("spawn 只接受本授权签发的 PreparedProcess")
        return await _spawn_child(
            prepared.command, cwd=prepared.cwd, env=prepared.env,
            stdin=stdin, stdout=stdout, stderr=stderr, limit=limit,
        )


async def _spawn_child(
    command: tuple[str, ...],
    *,
    cwd: str | None,
    env: Mapping[str, str] | None,
    stdin: object,
    stdout: object,
    stderr: object,
    limit: int | None,
) -> tuple["HostedChildProcess", bool]:
    options: dict[str, object] = {
        "stdin": stdin,
        "stdout": stdout,
        "stderr": stderr,
        "cwd": cwd,
        # 宿主已 scrub 全部父环境：签发的冻结 env 是唯一环境来源，
        # 仅 Supervisor 身份标记由 owned_process_env 重新钉住。
        "env": owned_process_env(
            dict(env or {}),
            scrub_keys=frozenset(os.environ),
        ),
        **process_group_spawn_kwargs(),
    }
    if limit is not None:
        options["limit"] = limit
    process, spawn_cancelled = await spawn_process(*command, **options)
    return HostedChildProcess(OwnedProcessGroup.from_process(process)), spawn_cancelled


class HostedChildProcess:
    """同一授权下的子进程句柄：暴露 stdio，进程组终止走宿主实现。"""

    def __init__(self, group: OwnedProcessGroup) -> None:
        self._group = group
        self.process: asyncio.subprocess.Process = group.process

    @property
    def group_id(self) -> int | None:
        return self._group.group_id

    async def terminate(self, *, timeout_s: float) -> None:
        await self._group.terminate(timeout_s=timeout_s)

    async def kill(self, *, timeout_s: float) -> None:
        await self._group.kill(timeout_s=timeout_s)


class ControllerAccess:
    def __init__(self, execution: ExecutionAccess, controller: WorkloadController | None, workspace_id: str):
        self._execution = execution
        self._controller = controller
        self._workspace_id = workspace_id

    def bind(self, ctx: Context) -> ControllerGrant:
        """每份授权固定 owner 与请求身份，不向插件交出原始 Controller。"""
        execution = self._execution.bind(ctx)
        if ctx.require(WORKLOAD_CONTROLLER) is not self:
            raise PermissionError("Controller 授权不属于当前 Root")
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


async def spawn_process(*command, **options):
    """取消等待也先接住 spawn 回执，调用方登记 process 后再传播取消。"""
    task = asyncio.create_task(asyncio.create_subprocess_exec(*command, **options))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return task.result(), cancelled
