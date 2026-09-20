"""普通 Workload provider 在 apply 中取得实际 lease。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from agent.plugin_composition import Context
from agent.plugin_composition.model import FiberState
from agent.plugin_composition.execution import WORKLOAD_CONTROLLER
from agent.plugin_composition.workload_slots import WORKLOADS, Workload
from .definitions import WorkloadBinding, _descriptor, _normalize_workload
from .host import WorkloadGenerationHost

api_version = 3
name = "workloads"
version = "1.0.0"
desc = "取得并持有 Context 所属的 Workload"
inject = (WORKLOAD_CONTROLLER,)


class WorkloadHandle:
    def __init__(self, provider, ctx, definition, grant, health):
        self._provider, self._ctx = provider, ctx
        self._token = ctx.fiber.activation_token
        self._definition, self._grant = definition, grant
        self._lock, self._closed = asyncio.Lock(), False
        self._closing = False
        self._binding = WorkloadBinding(_descriptor(ctx.runtime.plugin_id, definition), health, ctx.fiber, self._token, ctx.report_incident)
        self._host = WorkloadGenerationHost(grant, workspace_id=grant.workspace_id,
            on_health=lambda _id, _name, ready, reason: health.recover() if ready else health.degrade(reason),
            on_incident=lambda _id, _name, kind, reason: ctx.report_incident(kind, reason))

    async def start(self):
        async with self._lock:
            if self._closed:
                raise RuntimeError("Workload owner 已关闭")
            await self._host.start_generation(self._grant.identity, self._ctx.runtime.plugin_id,
                {self._definition.name: self._binding}, mode=self._grant.mode)

    def _check(self, ctx):
        self._provider.check(ctx)
        if (ctx.runtime.plugin_id != self._ctx.runtime.plugin_id
            or ctx.fiber.activation_token is not self._token or self._closed or self._closing):
            raise PermissionError("Workload 句柄不属于当前 Context activation")

    def url(self, ctx, port):
        self._check(ctx)
        current = self._host.get(self._grant.identity)
        if current is None or self._host.tombstone(self._grant.identity) is not None:
            raise RuntimeError("Workload 没有可用的实际 lease")
        return current.endpoints[(self._definition.name, port)]

    @asynccontextmanager
    async def borrow(self, ctx):
        self._check(ctx)
        async with self._host.borrow(self._binding.descriptor) as endpoints:
            yield endpoints

    async def aclose(self):
        """失败保留 host、pending 请求和 Effect，成功才解除注册。"""
        self._closing = True
        async with self._lock:
            if self._closed:
                return
            await self._host.stop_generation(self._grant.identity)
            self._closed = True
            del self._provider._entries[(self._ctx.runtime.plugin_id, self._definition.name)]


class Workloads:
    def __init__(self, ctx):
        self._ctx, self._entries = ctx, {}

    @property
    def root_instance_token(self):
        return self._ctx.root_instance_token

    def check(self, ctx):
        if ctx.root_instance_token is not self.root_instance_token or ctx.require(WORKLOADS) is not self:
            raise PermissionError("Workload provider 不能跨 Root")

    async def register(self, ctx: Context, workload: Workload):
        """先登记关闭责任，再等待 Controller，不存在延迟启动阶段。"""
        self.check(ctx)
        if ctx.fiber.state is not FiberState.LOADING:
            raise RuntimeError("资源注册只允许在 apply 初始化中执行")
        definition = _normalize_workload(workload)
        key = (ctx.runtime.plugin_id, definition.name)
        if key in self._entries:
            raise ValueError("Workload 名称已被本 owner 使用")
        grant = self._ctx.require(WORKLOAD_CONTROLLER).bind(ctx)
        health = await ctx.health("workload:" + definition.name)
        health.degrade("starting")
        owner = WorkloadHandle(self, ctx, definition, grant, health)
        def setup():
            self._entries[key] = owner
            return owner.aclose
        await ctx.effect(setup, label="workload:" + definition.name)
        await owner.start()
        return owner

    def urls(self, ctx):
        self.check(ctx)
        return {(name, port.name): handle.url(ctx, port.name)
            for (owner, name), handle in self._entries.items()
            if owner == ctx.runtime.plugin_id for port in handle._definition.ports}


async def apply(ctx: Context):
    await ctx.provide(WORKLOADS, Workloads(ctx))
