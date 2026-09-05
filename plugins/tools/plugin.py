from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
import re
from typing import Literal, cast

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.bindings import Bindings
from session.message import freeze_json

from .execution import BoundTool, Result

api_version = 3
name = "tools"
version = "1.0.0"
desc = "声明工具并固定实际实现；一次调用的回执独立于会话"
inject = ()

Prepare = Callable[[Mapping[str, object]], Awaitable[Mapping[str, object]]]
OpenTarget = Callable[[], AbstractAsyncContextManager[BoundTool]]


@dataclass(frozen=True, slots=True)
class _Registration:
    context: Context
    description: Mapping[str, object]
    open: OpenTarget


@dataclass(frozen=True, slots=True)
class _Preparation:
    context: Context
    name: str
    prepare: Prepare


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

    async def prepare(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """贡献先转换，实际工具一次接纳最终参数；授权在这之后执行。"""
        self._check_active()
        if self._preparation is not None:
            arguments = await self._preparation.prepare(arguments)
            self._check_active()
        result = await self._target.prepare(arguments)
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

    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, object],
        open: OpenTarget,
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
            for value in (idempotent, always_on, preloadable, requires_search)
        ):
            raise TypeError("工具执行和发现选项必须是 bool")
        descriptor = cast(
            Mapping[str, object],
            freeze_json(
                {
                    "name": name,
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
            self._tools[name] = _Registration(ctx, descriptor, open)

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

    def _check_context(self, ctx: Context) -> None:
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("工具注册不能跨 composition Root")

    def descriptions(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._tools[key].description for key in sorted(self._tools))

    def bind(self, name: str, bindings: Bindings) -> str:
        """从真实注册 Context 固定闭包，不让调用者省略准备贡献或重选目标。"""
        registration = self._tools[name]
        preparation = self._preparations.get(name)
        contributors = (registration.context,) + (
            () if preparation is None else (preparation.context,)
        )
        return bindings.bind(
            TOOLS,
            {
                "tool": registration.description,
                "prepare": None if preparation is None else preparation.name,
            },
            contributors=contributors,
        )

    @asynccontextmanager
    async def open(self, metadata: Mapping[str, object]) -> AsyncIterator[BoundTool]:
        """只启动所选目标；资源和环境由实际目标 owner 按归档身份打开。"""
        if set(metadata) != {"tool", "prepare"} or not isinstance(
            metadata["tool"], Mapping
        ):
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
        async with self._ctx.runtime_scope():
            async with registration.open() as target:
                if target.idempotent != description["idempotent"]:
                    raise ValueError("工具幂等协议与固定描述不一致")
                view = _ToolView(target, preparation)
                try:
                    yield view
                finally:
                    view.close()


TOOLS = ServiceKey[ToolCatalog]("tools.v1")


@asynccontextmanager
async def open_tool(bindings: Bindings, binding_id: str) -> AsyncIterator[BoundTool]:
    """真实 binding 选出归档注册表，目标 facade 拥有其资源 lease。"""
    async with bindings.open(binding_id, TOOLS) as (catalog, metadata):
        async with catalog.open(metadata) as target:
            yield target


async def apply(ctx: Context, config: object) -> None:
    _ = await ctx.provide(TOOLS, ToolCatalog(ctx))
