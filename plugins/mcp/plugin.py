"""普通 MCP provider；每次 open 取得独立会话，不重放工具调用。"""
from __future__ import annotations

import asyncio
import secrets
from collections.abc import Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from agent.plugin_composition import Context
from agent.plugin_composition.model import FiberState
from agent.plugin_composition.execution import EXECUTION
from agent.plugin_composition.mcp_slots import MCP_SERVERS, McpServerDefinition, McpSessionFailure
from .definitions import McpServerBinding, _descriptor, _normalize_definition, _ENV_NAME, _RESERVED_ENV
from .host import McpGenerationHost, McpMaterializedCommand

api_version = 3
name = "mcp"
version = "1.0.0"
desc = "按调用取得 MCP 会话并持有关闭责任"
inject = (EXECUTION,)


@dataclass
class Registration:
    ctx: Context
    token: object
    definition: McpServerDefinition
    health: object
    grant: object


class Session:
    def __init__(self, provider, registration):
        self._provider, self._entry = provider, registration
        self.identity = "mcp-" + secrets.token_hex(16)
        self._lock = asyncio.Lock()
        self._borrowed = AsyncExitStack()
        self._closed = False
        self._failure = None
        self._reason = None
        self._effect = None
        health, ctx = registration.health, registration.ctx
        self._host = McpGenerationHost(
            registration.grant,
            on_health=self._health,
            on_incident=lambda _id, _name, kind, reason: ctx.report_incident(kind, reason))

    def _health(self, _identity, _name, ready, reason):
        # 正常按调用退出不会使注册目标失效；并发失败会话仍保持降级。
        self._reason = None if ready or reason == "stopped" else reason
        self._provider.refresh_health(self._entry)

    async def start(self):
        """会话先归 Scope，再借实际资源并等待 MCP 握手。"""
        async with self._lock:
            if self._closed:
                raise RuntimeError("MCP 会话 owner 已关闭")
            entry = self._entry
            ctx, definition, grant = entry.ctx, entry.definition, entry.grant
            endpoints = {}
            for ref in definition.workload_env:
                urls = await self._borrowed.enter_async_context(ref.workload.borrow(ctx))
                endpoints[ref.env] = urls[ref.port]
            for ref in definition.endpoint_env:
                endpoints[ref.env] = str(await self._borrowed.enter_async_context(ref.process.borrow(ctx)))
            value = replace(definition, endpoint_env=(), workload_env=(), env={}, candidate_env={})
            binding = McpServerBinding(_descriptor(ctx.runtime.plugin_id, value), value, entry.health,
                ctx.fiber, entry.token, ctx.runtime.plugin_dir, ctx.runtime.data_dir,
                ctx.runtime.workspace, ctx.report_incident)
            # 声明值原样交给 host；command/cwd/env 的授权在 spawn 边界内由
            # ExecutionGrant.prepare_process 一次完成。endpoint/scope 是 provider
            # 运行期材料，经 extra_env 并入签发输入，不绕过授权。
            extra_env = dict(endpoints)
            extra_env["AKASHIC_MCP_SCOPE_ID"] = self.identity
            command = McpMaterializedCommand(
                command=definition.command, cwd=definition.cwd,
                env=dict(definition.env), candidate_env=dict(definition.candidate_env),
                extra_env=extra_env,
            )
            runtime = await self._host.start_generation(self.identity,
                {definition.name: binding},
                {definition.name: command}, mode=grant.mode)
            return runtime.server(definition.name)

    async def aclose(self):
        """MCP 关闭成功才释放借用；失败保留会话和同一 Scope Effect。"""
        async with self._lock:
            if self._closed:
                return
            try:
                await self._host.stop_generation(self.identity)
                await self._borrowed.aclose()
            except BaseException as error:
                self._failure = error
                self._provider.refresh_health(self._entry)
                raise
            self._closed = True
            del self._provider._sessions[self.identity]
            self._provider.refresh_health(self._entry)


class McpServers:
    def __init__(self, ctx):
        self._ctx, self._entries, self._sessions = ctx, {}, {}

    @property
    def root_instance_token(self):
        return self._ctx.root_instance_token

    def check(self, ctx):
        if ctx.root_instance_token is not self.root_instance_token or ctx.require(MCP_SERVERS) is not self:
            raise PermissionError("MCP provider 不能跨 Root")

    async def register(self, ctx: Context, definition: McpServerDefinition):
        """注册按调用打开的目标，立即检查实际依赖句柄的 owner。"""
        self.check(ctx)
        if ctx.fiber.state is not FiberState.LOADING:
            raise RuntimeError("资源注册只允许在 apply 初始化中执行")
        definition = _normalize_definition(ctx.runtime.plugin_dir, definition)
        if definition.name in self._entries:
            raise ValueError("MCP 名称重复")
        for ref in (*definition.workload_env, *definition.endpoint_env):
            if not _ENV_NAME.fullmatch(ref.env) or ref.env in _RESERVED_ENV:
                raise ValueError("MCP 资源环境变量名称无效")
        for ref in definition.workload_env:
            ref.workload.url(ctx, ref.port)
        for ref in definition.endpoint_env:
            ref.process.port(ctx)
        grant = self._ctx.require(EXECUTION).bind(ctx)
        health = await ctx.health("mcp:" + definition.name)
        entry = Registration(ctx, ctx.fiber.activation_token, definition, health, grant)
        def setup():
            self._entries[definition.name] = entry
            def cleanup():
                del self._entries[definition.name]
            return cleanup
        await ctx.effect(setup, label="mcp-target:" + definition.name)

    @asynccontextmanager
    async def open(self, ctx: Context, name: str):
        self.check(ctx)
        async with ctx.runtime_scope():
            ctx.require_runtime_owner(MCP_SERVERS, self)
            entry = self._entries[name]
            if entry.token is not ctx.fiber.activation_token or entry.ctx.runtime.plugin_id != ctx.runtime.plugin_id:
                raise PermissionError("MCP 目标不属于当前 Context activation")
            owner = Session(self, entry)
            def setup():
                self._sessions[owner.identity] = owner
                return owner.aclose
            effect = await ctx.effect(setup, label="mcp-session:" + owner.identity)
            owner._effect = effect
            try:
                yield await owner.start()
            finally:
                await effect.aclose()

    def refresh_health(self, entry):
        """目标健康由仍持有的实际会话共同决定，不复制到 Core 状态。"""
        reasons = [(str(owner._failure) or type(owner._failure).__name__) if owner._failure is not None else owner._reason
            for owner in self._sessions.values() if owner._entry is entry
            and (owner._failure is not None or owner._reason is not None)]
        if reasons:
            entry.health.degrade(reasons[0])
        else:
            entry.health.recover()

    def failures(self):
        return tuple(McpSessionFailure(owner.identity, owner._entry.definition.name,
            str(owner._failure) or type(owner._failure).__name__)
            for owner in self._sessions.values() if owner._failure is not None)

    async def retry_cleanup(self, ctx: Context, identity: str):
        """只重试本 Context 的保留句柄，不重开会话或重放工具。"""
        self.check(ctx)
        owner = self._sessions[identity]
        if owner._entry.ctx is not ctx:
            raise PermissionError("MCP 清理句柄不属于当前 Context")
        if owner._failure is None:
            raise RuntimeError("MCP 会话没有失败的清理责任")
        await owner._effect.aclose()

    def catalog(self):
        """List registered targets without claiming a live tool directory."""
        return [
            {"owner_id": entry.ctx.runtime.plugin_id, "name": name, "status": "declared"}
            for name, entry in sorted(self._entries.items())
        ]

    async def inspect(
        self, caller: Context, reader: object, owner_id: str, name: str,
    ) -> list[dict[str, object]]:
        """Read real tools in one session owned by the original contribution."""
        from agent.plugin_composition.runtime_catalog import RUNTIME_MCP_DETAIL, RuntimeCatalogUnavailable

        if caller.root_instance_token is not self.root_instance_token:
            raise PermissionError("MCP provider 不能跨 Root")
        caller.require_declared_runtime_owner(RUNTIME_MCP_DETAIL, reader)
        entry = self._entries.get(name)
        if entry is None or entry.ctx.runtime.plugin_id != owner_id:
            raise RuntimeCatalogUnavailable("mcp_not_found", f"MCP server 不存在: {owner_id}/{name}")
        # open checks the original activation and holds its Fiber until close.
        async with self.open(entry.ctx, name) as server:
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": _plain_schema(tool.input_schema),
                }
                for tool in server.tools.values()
            ]


def _plain_schema(value):
    """Copy the session's frozen JSON schema before the route expires."""
    if isinstance(value, Mapping):
        return {key: _plain_schema(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_schema(child) for child in value]
    return value


async def apply(ctx: Context):
    await ctx.provide(MCP_SERVERS, McpServers(ctx))
