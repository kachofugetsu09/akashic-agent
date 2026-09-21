"""普通进程 provider 取得实际进程，由贡献 Scope 负责关闭。"""
from __future__ import annotations

import asyncio
import secrets
from dataclasses import replace
from contextlib import asynccontextmanager
from agent.plugin_composition import Context
from agent.plugin_composition.model import FiberState
from agent.plugin_composition.execution import EXECUTION
from agent.plugin_composition.process_slots import MANAGED_PROCESSES, ManagedProcessDefinition
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
        self._host = ManagedProcessGenerationHost(
            grant,
            on_health=lambda _id, _name, ready, reason: health.recover() if ready else health.degrade(reason),
            on_incident=lambda _id, _name, kind, reason: ctx.report_incident(kind, reason))

    async def start(self):
        async with self._lock:
            if self._closed:
                raise RuntimeError("进程 owner 已关闭")
            value = self._definition
            materialized = replace(value, command=self._grant.command(value.command, value.cwd),
                cwd=str(self._grant.cwd(value.cwd)), env=self._grant.environment(value.env, value.candidate_env))
            await self._host.start_generation(self._id, {value.name: materialized}, mode=self._grant.mode)

    def port(self, ctx):
        self._provider.check(ctx)
        if (ctx.runtime.plugin_id != self._ctx.runtime.plugin_id
            or ctx.fiber.activation_token is not self._token or self._closed or self._closing):
            raise PermissionError("进程句柄不属于当前 Context activation")
        return self._host.endpoint(self._id, self._definition.name).port

    @asynccontextmanager
    async def borrow(self, ctx):
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
        health = await ctx.health("process:" + definition.name)
        health.degrade("starting")
        owner = ManagedProcessHandle(self, ctx, definition, grant, health)
        def setup():
            self._entries[key] = owner
            return owner.aclose
        await ctx.effect(setup, label="process:" + definition.name)
        await owner.start()
        return owner


async def apply(ctx: Context):
    await ctx.provide(MANAGED_PROCESSES, ManagedProcesses(ctx))
