"""普通进程 provider 取得实际进程，由贡献 Scope 负责关闭。"""
from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from agent.plugin_composition import Context
from agent.plugin_composition.model import FiberState
from agent.plugin_composition.execution import EXECUTION
from plugins.managed_processes.contract import MANAGED_PROCESSES, ManagedProcessDefinition
from .definitions import _normalize_definition
from .host import ManagedProcessGenerationHost

api_version = 3
name = "managed_processes"
version = "1.0.0"
desc = "取得并持有 Context 所属的进程"
inject = (EXECUTION,)


class ManagedProcessHandle:
    def __init__(self, provider, ctx, definition, grant, health):
        self._provider, self._ctx = provider, ctx
        self._token = ctx.fiber.activation_token
        self._definition, self._grant = definition, grant
        self._id = "process-" + secrets.token_hex(16)
        self._lock, self._closed = asyncio.Lock(), False
        self._closing, self._borrowers = False, 0
        self._drained = asyncio.Event()
        self._drained.set()
        self._starting: asyncio.Task[None] | None = None
        self._host = ManagedProcessGenerationHost(
            grant,
            on_health=lambda generation_id, process_name, healthy, reason: health.recover() if healthy else health.degrade(reason),
            on_incident=lambda generation_id, process_name, kind, message: ctx.report_incident(kind, message))

    async def start(self):
        if _starts_in_background(self._definition, self._grant):
            # 失败原因由 host 记为 Incident，并在 borrow 时交给借用方。
            self._starting = asyncio.create_task(
                self._start_generation(), name="managed-process-start:" + self._definition.name)
            self._starting.add_done_callback(lambda task: task.cancelled() or task.exception())
            return
        await self._start_generation()

    async def _start_generation(self):
        async with self._lock:
            if self._closed:
                raise RuntimeError("进程 owner 已关闭")
            value = self._definition
            # 声明值原样交给 host；command/cwd/env 的授权在 spawn 边界内由
            # ExecutionGrant.prepare_process 一次完成并签发冻结制品。
            await self._host.start_generation(self._id, {value.name: value}, mode=self._grant.mode)

    def check(self, ctx):
        self._provider.check(ctx)
        if (ctx.runtime.plugin_id != self._ctx.runtime.plugin_id
            or ctx.fiber.activation_token is not self._token or self._closed or self._closing):
            raise PermissionError("进程句柄不属于当前 Context activation")

    def port(self, ctx):
        self.check(ctx)
        return self._host.endpoint(self._id, self._definition.name).port

    @asynccontextmanager
    async def borrow(self, ctx):
        self.check(ctx)
        if self._starting is not None:
            # 共享的后台启动不随单个借用方取消。
            await asyncio.shield(self._starting)
        port = self.port(ctx)
        self._borrowers += 1
        self._drained.clear()
        try:
            yield port
        finally:
            self._borrowers -= 1
            if not self._borrowers:
                self._drained.set()

    async def aclose(self):
        """关闭先停止借用准入；最后一个借用释放后才终止进程。"""
        self._closing = True
        starting = self._starting
        if starting is not None:
            _ = starting.cancel()
            _ = await asyncio.gather(starting, return_exceptions=True)
        async with self._lock:
            if self._closed:
                return
            await self._drained.wait()
            await self._host.stop_generation(self._id)
            self._closed = True
            del self._provider._entries[(self._ctx.runtime.plugin_id, self._definition.name)]


class ManagedProcesses:
    def __init__(self, ctx):
        self._ctx, self._entries = ctx, {}

    def check(self, ctx):
        if ctx.root_instance_token is not self._ctx.root_instance_token or ctx.require(MANAGED_PROCESSES) is not self:
            raise PermissionError("进程 provider 不能跨 Root")

    async def register(self, ctx: Context, definition: ManagedProcessDefinition):
        self.check(ctx)
        if ctx.fiber.state is not FiberState.LOADING:
            raise RuntimeError("资源注册只允许在 apply 初始化中执行")
        definition = _normalize_definition(ctx.runtime.plugin_dir, definition)
        key = (ctx.runtime.plugin_id, definition.name)
        if key in self._entries:
            raise ValueError("进程名称已被本 owner 使用")
        grant = self._ctx.require(EXECUTION).bind(ctx)
        health = await ctx.health(
            "process:" + definition.name, required=not _starts_in_background(definition, grant))
        health.degrade("starting")
        owner = ManagedProcessHandle(self, ctx, definition, grant, health)
        def setup():
            self._entries[key] = owner
            return owner.aclose
        await ctx.effect(setup, label="process:" + definition.name)
        await owner.start()
        return owner


def _starts_in_background(definition: ManagedProcessDefinition, grant) -> bool:
    return definition.background_ready and grant.mode == "formal"


async def apply(ctx: Context):
    await ctx.provide(MANAGED_PROCESSES, ManagedProcesses(ctx))
