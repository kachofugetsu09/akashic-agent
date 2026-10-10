"""UI 定义、目录封存和 Dashboard 资源由普通插件拥有。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import FunctionType, ModuleType
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import (
    RUNTIME_STARTING,
    Context,
    Effect,
    FiberState,
    ServiceKey,
)
from agent.plugin_composition.requests import RequestContext
from agent.plugin_composition.host import HOST_INFO
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.ui.contract import (
    UI,
    WEB_UI,
    Configuration,
    DashboardBinding,
    WebModuleDescriptor,
    WebUiCatalog,
)
from plugins.ui.contract import UI_SLOTS
from plugins.workloads.contract import WORKLOADS

from .dashboard import DashboardResources, _server_routes, _require_routes_available
from .dashboard_app import create_dashboard_app
from .dashboard_server import start_dashboard_server
from .dashboard_host import LiveDashboardMiddleware
from .plugin_ui import PluginUiSlots
from .queries import LivePluginUiProvider
from agent.plugin_contracts.ui import PLUGIN_UI, MESSAGE_DISPLAY
from .message_display import project_message_rows
from .web import build_web_ui_catalog, encode_web_ui_bootstrap, resolve_web_module

api_version = 3
name = "ui"
version = "1.0.0"
desc = "注册并封存插件的 Web 与 Dashboard UI"
inject = (HOST_INFO, RUNTIME_CATALOG)
entrypoints = {"dashboard": "dashboard_server.main"}

@dataclass
class Registration:
    ctx: Context
    web: WebModuleDescriptor | None
    resources: DashboardResources | None
    registration_uuid: str
    effect: Effect | None = None
    lifecycle_effect: Effect | None = None
    binding: DashboardBinding | None = None
    initialized: bool = False


class ConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1, max_length=128)
    expected_input: str = Field(pattern=r"^[0-9a-f]{64}$")
    values: dict[str, object]


class Ui:
    """一个 Root 的唯一 UI 注册表；关闭不删除代码或插件数据。"""

    def __init__(self, ctx: Context, routes: tuple[object, ...]) -> None:
        self._ctx = ctx
        self._routes = routes
        self._entries: dict[str, Registration] = {}

    def register_configuration(
        self, app: FastAPI, context: RequestContext,
        key: ServiceKey[Configuration], prefix: str,
    ) -> None:
        """HTTP 边界只校验请求信封；插件解释字段、处理业务错误。"""
        @app.get(prefix)
        async def read():
            return await context.require(key).read()

        @app.post(prefix)
        async def save(request: ConfigRequest):
            try:
                return await context.require(key).save(request.request_id, request.expected_input, request.values)
            except ValidationError as error:
                raise HTTPException(422, error.errors(include_input=False, include_url=False, include_context=False)) from error
            except ValueError as error:
                raise HTTPException(422, str(error)) from error

        @app.get(prefix + "/receipts/{request_id}")
        async def receipt(request_id: str):
            try:
                return context.require(key).receipt(request_id)
            except KeyError as error:
                raise HTTPException(404, "配置回执不存在") from error

    @property
    def root_instance_token(self) -> object:
        return self._ctx.root_instance_token

    async def register(
        self, ctx: Context, *, web: str | None = None, dashboard: Callable[[], ModuleType] | None = None,
        requires: tuple[str, ...] = (), provides: tuple[str, ...] = (),
        contract_digests: Mapping[str, str] | None = None,
    ) -> Effect:
        """从实际贡献 Context 固定身份和代码路径，再取得注册 Effect。"""
        # 1. provider 只接受同一 Root 的贡献方，不接受自报 owner 或 generation。
        if ctx.root_instance_token is not self._ctx.root_instance_token or ctx.require(UI) is not self:
            raise ValueError("UI 注册不能跨 composition Root")
        owner = ctx.runtime.plugin_id
        if owner in self._entries:
            raise ValueError(f"UI owner 重复注册: {owner}")
        if web is None and dashboard is None:
            raise ValueError("UI 注册必须包含 Web 或 Dashboard")
        requires = _contracts(requires)
        provides = _contracts(provides)
        digests = {} if contract_digests is None else dict(contract_digests)
        registration_uuid = uuid4().hex
        if any(not isinstance(key, str) or not isinstance(value, str)
               or re.fullmatch(r"[0-9a-f]{64}", value) is None for key, value in digests.items()):
            raise ValueError("Web contract digest 必须是 SHA-256")
        if set(digests) - (set(requires) | set(provides)):
            raise ValueError("Web contract digest 没有对应声明")
        if web is None and (requires or provides or digests):
            raise ValueError("Web contract 必须属于 Web 模块")

        # 2. JS/CSS 与合同由 provider 校验；Dashboard 资源在 owner 启动时建立。
        root = ctx.runtime.plugin_dir.resolve(strict=True)
        module = None
        if web is not None:
            _path(root, web, ".js")
            asset = resolve_web_module(root, web, requires=requires, provides=provides,
                                       contract_digests=tuple(sorted(digests.items())))
            assert asset is not None
            module = WebModuleDescriptor(
                owner, registration_uuid, ctx.runtime.generation_id, asset,
            )
        if dashboard is not None:
            if not isinstance(dashboard, FunctionType):
                raise ValueError("Dashboard loader 必须是贡献插件的函数")
            loader_path = Path(dashboard.__code__.co_filename).resolve(strict=True)
            if not loader_path.is_relative_to(root):
                raise ValueError("Dashboard loader 不属于贡献方代码制品")
        resources = None if dashboard is None else DashboardResources(
            ctx, dashboard, has_web=module is not None, registry=self,
        )
        registration = Registration(ctx, module, resources, registration_uuid)

        async def initialize(_event: object = None) -> None:
            """Build this registration in its real owner lifecycle scope."""

            if self._entries.get(owner) is not registration or registration.initialized:
                return
            modules = tuple(
                entry.web for entry in self._entries.values()
                if entry.web is not None
            )
            build_web_ui_catalog(modules)
            if resources is not None:
                occupied = list(_server_routes(self._routes))
                for entry in self._entries.values():
                    if entry.binding is None:
                        continue
                    _require_routes_available(entry.binding, occupied)
                    occupied.extend(entry.binding.routes)
                with self._ctx.borrow(WORKLOADS) as workloads:
                    registration.binding = resources.build(
                        occupied=occupied,
                        workload_urls={} if workloads is None else workloads.urls(ctx),
                        validation=self._ctx.require(HOST_INFO).validation,
                    )
            registration.initialized = True

        async def setup():
            self._entries[owner] = registration
            try:
                registration.lifecycle_effect = await ctx.on(
                    RUNTIME_STARTING, initialize,
                )
            except BaseException:
                del self._entries[owner]
                raise

            async def cleanup() -> None:
                if registration.lifecycle_effect is not None:
                    await registration.lifecycle_effect.aclose()
                if resources is not None:
                    await resources.aclose()
                if self._entries.get(owner) is registration:
                    del self._entries[owner]

            return cleanup

        registration.effect = await ctx.effect(setup, label="ui")
        if ctx.fiber.state is FiberState.ACTIVE:
            async with ctx.runtime_scope():
                await initialize()
        return registration.effect

    def catalog(self) -> WebUiCatalog:
        return build_web_ui_catalog(tuple(
            entry.web for _, entry in sorted(self._entries.items())
            if entry.web is not None and entry.initialized
            and entry.ctx.fiber.state is FiberState.ACTIVE
        ))

    def bindings(self) -> tuple[DashboardBinding, ...]:
        return tuple(
            entry.binding for _, entry in sorted(self._entries.items())
            if entry.binding is not None and entry.initialized
            and entry.ctx.fiber.state is FiberState.ACTIVE
        )

    def contributors(self) -> tuple[Context, ...]:
        return tuple(entry.ctx for _, entry in sorted(self._entries.items()))

    async def bootstrap(self) -> bytes:
        async with self._ctx.runtime_scope():
            self._ctx.require_runtime_owner(WEB_UI, self)
            # catalog 只含已激活并初始化的贡献；无关管理操作不关闭读取。
            return encode_web_ui_bootstrap(self.catalog(), self._ctx.generation_id)

    async def state(self) -> dict[str, str | bool]:
        async with self._ctx.runtime_scope():
            self._ctx.require_runtime_owner(WEB_UI, self)
            return {"snapshotId": self._ctx.generation_id, "catalogId": self.catalog().identity,
                    "updating": bool(self._ctx.require(RUNTIME_CATALOG)(self._ctx)["updating"])}


def _contracts(value: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(not isinstance(item, str) or not item or item.strip() != item for item in value):
        raise ValueError("Web contracts 必须是不重复的非空字符串 tuple")
    if len(value) != len(set(value)):
        raise ValueError("Web contracts 不得重复")
    return value


def _path(root: Path, relative_path: str, suffix: str) -> Path:
    """拒绝绝对路径和跨制品链接，路径只能来自贡献方固定代码。"""
    if not isinstance(relative_path, str) or not relative_path or relative_path.strip() != relative_path:
        raise ValueError("UI 模块必须是相对文件路径")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in relative_path:
        raise ValueError("UI 模块路径越过贡献方代码制品")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.suffix != suffix or not path.is_file():
        raise ValueError("UI 模块不属于贡献方代码制品或类型不符")
    return path


async def apply(ctx: Context) -> None:
    app = create_dashboard_app(ctx.runtime.plugin_dir / "static/dashboard")
    registry = Ui(ctx, tuple(app.routes))
    app.add_middleware(LiveDashboardMiddleware, context=ctx, registry=registry)
    await ctx.provide(UI, registry, binding_contributors=registry.contributors)
    await ctx.provide(WEB_UI, registry, binding_contributors=registry.contributors)
    slots = PluginUiSlots(ctx)
    await ctx.provide(UI_SLOTS, slots, binding_contributors=slots.contributors)
    queries = LivePluginUiProvider(ctx, slots)
    await ctx.effect(lambda: queries.aclose, label="ui.queries")
    await ctx.provide(PLUGIN_UI, queries)

    async def display(page, *, display_only: bool):
        return await project_message_rows(ctx, page, display_only=display_only)

    await ctx.provide(MESSAGE_DISPLAY, ctx.entrypoint(display))

    if not ctx.require(HOST_INFO).validation:
        await start_dashboard_server(ctx, app)
