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
        self._health = health
        self._host = ManagedProcessGenerationHost(
            grant,
            on_health=lambda generation_id, process_name, healthy, reason: health.recover() if healthy else health.degrade(reason),
            on_incident=lambda generation_id, process_name, kind, message: ctx.report_incident(kind, message))

    async def start(self):
        if _starts_in_background(self._definition, self._grant):
            # 失败原因由 host 记为 Incident，并在 borrow 时交给借用方。
            self._starting = asyncio.create_task(
                self._start_in_background(), name="managed-process-start:" + self._definition.name)
            self._starting.add_done_callback(lambda task: task.cancelled() or task.exception())
            return
        await self._start_generation()

    # 后台启动失败后以新 generation 退避重试：宿主开机时 CPU 争用可让就绪超过时限，
    # 借用方只启动一次，不重试就会让整个来源停摆到下次重启。
    async def _start_in_background(self):
        for attempt, delay in enumerate((*_RETRY_DELAYS_S, None), start=1):
            try:
                await self._start_generation()
                return
            except Exception as error:
                # host 收尾写的是 stopped；改成失败原因，状态页能与正常停止区分。
                self._health.degrade(f"start failed ({attempt}): {type(error).__name__}: {error}")
                # 失败 generation 未清理干净时不换 id：关闭仍要能停掉并暴露残留的旧进程。
                cleaned = self._host.get(self._id) is None and self._host.tombstone(self._id) is None
                if delay is None or self._closing or self._closed or not cleaned:
                    raise
            await asyncio.sleep(delay)
            async with self._lock:
                if self._closing or self._closed:
                    raise RuntimeError("进程 owner 已关闭")
                self._id = "process-" + secrets.token_hex(16)

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
        starting = self._starting
        # 共享的后台启动不随单个借用方取消；owner 撤销 activation 后借用方立即退出，
        # 卸载不必等启动超时。
        while starting is not None and not starting.done():
            _ = await asyncio.wait((starting,), timeout=0.1)
            self.check(ctx)
        failure = None if starting is None or starting.cancelled() else starting.exception()
        if failure is not None:
            raise failure
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


# 后台启动失败后的重试间隔；耗尽后把最后一次失败交给借用方。
_RETRY_DELAYS_S = (10.0, 30.0, 60.0)


def _starts_in_background(definition: ManagedProcessDefinition, grant) -> bool:
    return definition.background_ready and grant.mode == "formal"


async def apply(ctx: Context):
    await ctx.provide(MANAGED_PROCESSES, ManagedProcesses(ctx))
