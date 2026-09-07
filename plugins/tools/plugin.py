from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
import re
from typing import Literal, cast

from agent.plugin_composition import Context, Effect, ServiceKey, RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.bindings import Bindings
from session.message import CallRef, ToolResult, freeze_json
from session.log import MessageReader
from agent.restart import ExternalRootPermit
from plugins.content.plugin import check_text

from plugins.tools.api import Authorize, BoundTool, CallSource, MessageReply, Result, display_name, result_message_id
from plugins.tools.abandon import follow_abandon, reject_start
from plugins.tools.execution import ToolExecution
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE
from agent.plugin_composition.tasks import TASKS

api_version = 3
name = "tools"
version = "1.0.0"
desc = "声明工具并固定实际实现；一次调用的回执独立于会话"
inject = ()

Prepare = Callable[[Mapping[str, object]], Awaitable[Mapping[str, object]]]
BindingAuthorize = Callable[[Mapping[str, object]], Awaitable[None]]
Candidates = Mapping[str, Mapping[str, object]]
OpenTarget = Callable[[Mapping[str, object]], AbstractAsyncContextManager[BoundTool]]
Capture = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class _Registration:
    context: Context
    description: Mapping[str, object]
    open: OpenTarget
    capture: Capture | None


@dataclass(frozen=True, slots=True)
class _Preparation:
    context: Context
    name: str
    prepare: Prepare


@dataclass(frozen=True, slots=True)
class _Authorization:
    context: Context
    name: str
    authorize: BindingAuthorize


class _ToolView:
    """每次打开只访问固定目标，释放后不能保留入口再执行。"""

    def __init__(self, target: BoundTool, preparation: _Preparation | None):
        self._target = target
        self._preparation = preparation
        self._active = True

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("工具 binding scope 已释放")

    @property
    def idempotent(self) -> bool:
        self._check_active()
        return self._target.idempotent

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        """贡献先转换，实际工具一次接纳最终参数；授权在这之后执行。"""
        self._check_active()
        if self._preparation is not None:
            arguments = await self._preparation.prepare(arguments)
            self._check_active()
        result = await self._target.prepare(arguments, source)
        self._check_active()
        return result

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        self._check_active()
        return await self._target.invoke(key, arguments)

    async def query(self, key: str) -> Result | None:
        self._check_active()
        return await self._target.query(key)

    def close(self) -> None:
        self._active = False


class ToolCatalog:
    """普通注册表拥有工具描述、目标与参数准备；不管理消息或循环。"""

    def __init__(self, ctx: Context):
        self._ctx = ctx
        self._tools: dict[str, _Registration] = {}
        self._preparations: dict[str, _Preparation] = {}
        self._authorizations: dict[str, _Authorization] = {}

    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, object],
        open: OpenTarget,
        capture: Capture | None = None,
        discovery: bool = False,
        public: bool = True,
        idempotent: bool = False,
        risk: Literal["read-only", "read-write", "external-side-effect"] = "read-write",
        always_on: bool = False,
        preloadable: bool = True,
        requires_search: bool = False,
        search_hint: str | None = None,
    ) -> Effect:
        """目标自行校验参数 schema；注册表固定发现描述与真实资源入口。"""
        self._check_context(ctx)
        if (
            re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) is None
            or not description.strip()
        ):
            raise ValueError("工具名或描述无效")
        if parameters.get("type") != "object":
            raise ValueError("工具参数必须声明 object schema")
        if risk not in {"read-only", "read-write", "external-side-effect"}:
            raise ValueError("工具风险声明无效")
        if any(
            type(value) is not bool
            for value in (idempotent, always_on, preloadable, requires_search, discovery, public)
        ):
            raise TypeError("工具执行和发现选项必须是 bool")
        if always_on and requires_search:
            raise ValueError("工具不能同时常驻和要求搜索解锁")
        if discovery and (not always_on or requires_search):
            raise ValueError("捕获候选的工具必须直接可见，不能递归等待搜索解锁")
        if discovery and capture is not None:
            raise ValueError("发现工具的 open 参数已由候选集合拥有")
        if capture is not None and not callable(capture):
            raise TypeError("工具 capture 必须是同步回调")
        descriptor = cast(
            Mapping[str, object],
            freeze_json(
                {
                    "name": name,
                    "owner": ctx.runtime.plugin_id,
                    "discovery": discovery,
                    "public": public,
                    "description": description,
                    "parameters": parameters,
                    "idempotent": idempotent,
                    "risk": risk,
                    "always_on": always_on,
                    "preloadable": preloadable,
                    "requires_search": requires_search,
                    "search_hint": search_hint,
                }
            ),
        )

        def setup() -> Callable[[], None]:
            if name in self._tools:
                raise ValueError(f"工具名重复: {name}")
            self._tools[name] = _Registration(ctx, descriptor, open, capture)

            def cleanup() -> None:
                del self._tools[name]

            return cleanup

        return await ctx.effect(setup, label=f"tool:{name}")

    async def register_prepare(
        self, ctx: Context, *, tool: str, name: str, prepare: Prepare
    ) -> Effect:
        """每个工具的参数改写只有一个 owner，不按安装顺序串联未知转换。"""
        self._check_context(ctx)
        if not name or not tool:
            raise ValueError("参数准备必须有工具名与贡献名")

        def setup() -> Callable[[], None]:
            if tool in self._preparations:
                raise ValueError(f"工具参数准备已有 owner: {tool}")
            self._preparations[tool] = _Preparation(ctx, name, prepare)

            def cleanup() -> None:
                del self._preparations[tool]

            return cleanup

        return await ctx.effect(setup, label=f"tool-prepare:{name}")

    async def register_authorize(
        self, ctx: Context, *, tool: str, name: str, authorize: BindingAuthorize
    ) -> Effect:
        """每个工具的独立限制只有一个 owner，并随 binding 固定。"""
        self._check_context(ctx)
        if not name or not tool:
            raise ValueError("工具限制必须有工具名与贡献名")

        def setup() -> Callable[[], None]:
            if tool in self._authorizations:
                raise ValueError(f"工具限制已有 owner: {tool}")
            self._authorizations[tool] = _Authorization(ctx, name, authorize)

            def cleanup() -> None:
                del self._authorizations[tool]

            return cleanup

        return await ctx.effect(setup, label=f"tool-authorize:{name}")

    def _check_context(self, ctx: Context) -> None:
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("工具注册不能跨 composition Root")

    def execution(
        self, authorize: Authorize, *, child_permit: Callable[[], ExternalRootPermit] | None = None,
    ) -> ToolExecution:
        """正式调用取得工具 owner 的回执与任务；归档发现不打开这些能力。"""
        bindings = self._ctx.require(BINDINGS)

        async def authorize_binding(
            binding_id: str, arguments: Mapping[str, object]
        ) -> Mapping[str, object]:
            async with bindings.open(binding_id, TOOLS) as (catalog, metadata):
                await catalog.authorize(metadata, arguments)
            return await authorize(binding_id, arguments)

        return ToolExecution(
            self._ctx.require(OWNER_STATE).open(self._ctx),
            self._ctx.require(TASKS).open(self._ctx),
            lambda identity: open_tool(bindings, identity),
            authorize_binding,
            task_key="effects",
            child_permit=child_permit,
        )

    def descriptions(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._tools[key].description for key in sorted(self._tools)
                     if self._tools[key].description["public"])

    async def drain_calls(self, calls: tuple[CallRef, ...]) -> None:
        """清理 owner 等待原效果退出；终态结果不等于资源已经释放。"""
        from plugins.tools.api import durable_call_key

        tasks = self._ctx.require(TASKS).open(self._ctx)
        for ref in calls:
            task = await tasks.admit(("effects", durable_call_key(ref)), lambda slot: slot.current)
            if task is not None:
                try:
                    _ = await task.join()
                except asyncio.CancelledError:
                    caller = asyncio.current_task()
                    if caller is not None and caller.cancelling():
                        raise

    def bind(
        self, name: str, bindings: Bindings, *, candidates: Candidates | None = None,
        configuration: Mapping[str, object] | None = None,
    ) -> str:
        """从真实注册 Context 固定闭包，不让调用者省略准备贡献或重选目标。"""
        registration = self._tools[name]
        preparation = self._preparations.get(name)
        authorization = self._authorizations.get(name)
        discovery = registration.description["discovery"]
        if discovery and candidates is None:
            raise ValueError("固定发现工具需要来源允许的候选快照")
        if not discovery and candidates is not None:
            raise ValueError("普通工具不接收候选目录")
        if configuration is not None and registration.capture is None:
            raise ValueError("该工具未声明 binding 配置入口")
        if candidates is not None:
            candidates = check_candidates(candidates)
            for item in candidates.values():
                if bindings.describe(cast(str, item["binding_id"]), TOOLS)["tool"] != item["tool"]:
                    raise ValueError("候选描述不属于指定 binding")
        contributors = (
            registration.context,
            *(() if preparation is None else (preparation.context,)),
            *(() if authorization is None else (authorization.context,)),
        )
        state: Mapping[str, object] | None = None
        if registration.capture is not None:
            _ = self._ctx.require_runtime_owner(TOOLS, self)
            options = freeze_json({} if configuration is None else configuration)
            if not isinstance(options, Mapping):
                raise TypeError("工具 binding 配置必须是 JSON 对象")
            captured = freeze_json(registration.capture(cast(Mapping[str, object], options)))
            if not isinstance(captured, Mapping):
                raise TypeError("工具 binding state 必须是 JSON 对象")
            state = cast(Mapping[str, object], captured)
        return bindings.bind(
            TOOLS,
            {
                "tool": registration.description,
                "prepare": None if preparation is None else preparation.name,
                **({"authorize": authorization.name} if authorization is not None else {}),
                **({"candidates": candidates} if discovery else {}),
                **({"state": state} if state is not None else {}),
            },
            contributors=contributors,
        )

    @asynccontextmanager
    async def open(self, metadata: Mapping[str, object]) -> AsyncIterator[BoundTool]:
        """只启动所选目标；资源和环境由实际目标 owner 按归档身份打开。"""
        if not isinstance(metadata.get("tool"), Mapping):
            raise ValueError("工具 binding 描述无效")
        description = cast(Mapping[str, object], metadata["tool"])
        name = description["name"]
        if not isinstance(name, str):
            raise ValueError("工具 binding 缺少工具名")
        registration = self._tools[name]
        preparation = self._preparations.get(name)
        if registration.description != description or metadata["prepare"] != (
            None if preparation is None else preparation.name
        ):
            raise ValueError("归档工具描述或参数准备与 binding 不一致")
        expected: set[str] = {"tool", "prepare"}
        if "authorize" in metadata:
            authorization = self._authorizations.get(name)
            if authorization is None or metadata["authorize"] != authorization.name:
                raise ValueError("归档工具限制与 binding 不一致")
            expected.add("authorize")
        if description["discovery"]:
            expected.add("candidates")
        if registration.capture is not None:
            expected.add("state")
        if set(metadata) != expected:
            raise ValueError("工具 binding 字段无效")
        candidates: Candidates = check_candidates(metadata["candidates"]) if description["discovery"] else {}
        state: Mapping[str, object] = candidates
        if registration.capture is not None:
            captured = metadata["state"]
            if not isinstance(captured, Mapping):
                raise ValueError("工具 binding state 必须是 JSON 对象")
            state = cast(Mapping[str, object], captured)
        async with self._ctx.runtime_scope():
            async with registration.open(state) as target:
                if target.idempotent != description["idempotent"]:
                    raise ValueError("工具幂等协议与固定描述不一致")
                view = _ToolView(target, preparation)
                try:
                    yield view
                finally:
                    view.close()

    async def authorize(
        self, metadata: Mapping[str, object], arguments: Mapping[str, object]
    ) -> None:
        """只执行 binding 固定的独立限制；旧无字段 binding 不追附当前限制。"""
        if "authorize" not in metadata:
            return
        description = metadata.get("tool")
        if not isinstance(description, Mapping) or not isinstance(description.get("name"), str):
            raise ValueError("工具 binding 描述无效")
        authorization = self._authorizations.get(cast(str, description["name"]))
        if authorization is None or metadata["authorize"] != authorization.name:
            raise ValueError("归档工具限制与 binding 不一致")
        async with self._ctx.runtime_scope():
            await authorization.authorize(arguments)


def check_candidates(value: object) -> Candidates:
    """候选边界只有普通工具的公开描述与精确绑定，不接受嵌套发现目录。"""
    if not isinstance(value, Mapping):
        raise ValueError("工具候选必须是对象")
    rows = cast(Mapping[object, object], value)
    for name, raw in rows.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValueError("工具候选格式无效")
        item = cast(Mapping[str, object], raw)
        if set(item) != {"binding_id", "tool"} or not isinstance(item["binding_id"], str) or not item["binding_id"]:
            raise ValueError("工具候选缺少精确绑定")
        if not isinstance(item["tool"], Mapping):
            raise ValueError("工具候选缺少描述")
        tool = cast(Mapping[str, object], item["tool"])
        if tool.get("name") != name or tool.get("discovery") is not False:
            raise ValueError("工具候选名称不一致或包含递归发现")
    return cast(Candidates, freeze_json(rows))


TOOLS = ServiceKey[ToolCatalog]("tools.v1")
TOOL_DISPLAY_NAME = ServiceKey[Callable[[str], str]]("tools.display-name.v1")


@asynccontextmanager
async def open_tool(bindings: Bindings, binding_id: str) -> AsyncIterator[BoundTool]:
    """真实 binding 选出归档注册表，目标 facade 拥有其资源 lease。"""
    async with bindings.open(binding_id, TOOLS) as (catalog, metadata):
        async with catalog.open(metadata) as target:
            yield target


async def apply(ctx: Context, config: object) -> None:
    _ = await ctx.provide(TOOLS, ToolCatalog(ctx))

    def read_name(binding_id: str) -> str:
        """只读原 binding 的名称；诊断消费者不能打开或执行工具。"""
        return display_name(ctx.require(BINDINGS).describe(binding_id, TOOLS))

    _ = await ctx.provide(TOOL_DISPLAY_NAME, read_name)
    watcher: asyncio.Task[None] | None = None

    async def start(_event: object) -> None:
        nonlocal watcher
        async def reply(reader: MessageReader, source: str, ref: CallRef) -> MessageReply:
            async with ctx.runtime_scope():
                writer = ctx.require(MESSAGE_WRITERS).bind(
                    ctx, author="tool", source=source, body_types=(ToolResult,), content={"text": check_text},
                )(reader.session_id, call_ref=ref)
            return MessageReply(result_message_id(ref), ref, reader, writer, reject_start)

        watcher = await ctx.spawn(follow_abandon(
            ctx.require(MESSAGE_CATALOG), ctx.require(OWNER_STATE).open(ctx),
            ctx.require(TASKS).open(ctx), reply, task_key="effects",
            report_incident=ctx.report_incident,
        ), name="tools-abandon")

    async def stop(_event: object) -> None:
        if watcher is not None:
            _ = watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
