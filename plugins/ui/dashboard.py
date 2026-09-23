"""Dashboard provider 拥有模块、路由校验与关闭失败的资源句柄。"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast
from types import MappingProxyType, ModuleType

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.convertors import FloatConvertor, IntegerConvertor, PathConvertor, StringConvertor, UUIDConvertor
from starlette.routing import WebSocketRoute

from agent.plugin_composition import Context, DashboardContext
from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.host import HOST_INFO
from agent.plugin_composition.model import CompositionError, ServiceKey, resolve_declared_workspace_file, resolve_declared_workspace_root
from agent.plugin_composition.ui import UI, DashboardBinding, DashboardRoute


class DashboardImportError(RuntimeError):
    pass


class Closeable(Protocol):
    def close(self) -> object: ...


class DashboardResources:
    """保留实际模块和关闭顺序；失败时不先解除 owner。"""

    def __init__(
        self,
        ctx: Context,
        loader: Callable[[], ModuleType],
        *,
        has_web: bool,
        registry: object,
    ) -> None:
        self.ctx = ctx
        self.loader = loader
        self.has_web = has_web
        self._registry = registry
        self.closeables: list[Closeable] = []
        self._started = False

    async def aclose(self) -> None:
        while self.closeables:
            value = self.closeables[-1].close()
            if inspect.isawaitable(value):
                await value
            self.closeables.pop()

    def build(
        self, *, occupied: list[DashboardRoute],
        workload_urls: Mapping[tuple[str, str], str],
    ) -> DashboardBinding:
        """延迟加载原包模块，校验域路由，并保留实际返回的资源。"""
        # 1. 同一次资源取得只能执行一次；失败由原 Effect 清理。
        if self._started:
            raise RuntimeError("Dashboard 初始化已执行；先关闭原注册，不能重放资源取得")
        self._started = True
        ctx = self.ctx
        runtime = ctx.runtime
        workspace = runtime.workspace.resolve(strict=False)
        data_root = runtime.data_dir.resolve(strict=False)
        workspace_roots = runtime.workspace_roots
        workspace_files = runtime.workspace_files
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        try:
            with plugin_entrypoint(
                plugin_id=runtime.plugin_id,
                generation_id=runtime.generation_id,
                fiber=runtime.plugin_id,
                operation="dashboard.module_load",
            ):
                module = self.loader()
        except Exception as error:
            raise DashboardImportError(str(error)) from error
        source_path = getattr(module, "__file__", None)
        if not isinstance(module, ModuleType) or not isinstance(source_path, str):
            raise RuntimeError("Dashboard loader 必须返回实际 Python 模块")
        module_path = Path(source_path).resolve(strict=True)
        if (not module_path.is_relative_to(runtime.plugin_dir.resolve(strict=True))
                or module_path.suffix != ".py" or not module_path.is_file()):
            raise RuntimeError("Dashboard 模块不属于贡献方代码制品")
        # 2. 模块 ABI 和权限只在 provider 边界解释。
        register = getattr(module, "register", None)
        if not callable(register):
            raise RuntimeError(f"dashboard module 缺少 register: {module_path}")
        enabled = getattr(module, "plugin_enabled", None)
        if enabled is not None and not callable(enabled):
            raise RuntimeError("v3 dashboard plugin_enabled 必须是可调用对象")
        dependencies = getattr(module, "inject", ())
        if (not isinstance(dependencies, tuple)
                or any(not isinstance(key, ServiceKey) for key in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            raise ValueError("Dashboard inject 必须是不重复的 ServiceKey tuple")

        def resolve(key: ServiceKey[object]) -> object:
            """只在路由实际租约内解析声明能力，旧 Dashboard 不能借新 generation。"""
            if key not in dependencies:
                raise CompositionError("SERVICE_UNDECLARED", f"Dashboard 未声明能力: {key.name}")
            # 旧 generation 的 Fiber 已退役：scope 校验不能先经 ctx.require
            # 取服务（INACTIVE_SERVICE 会吞掉 runtime scope 拒绝）。直接以
            # 注册时的实际 Ui 实例证明 owner 归属。
            ctx.require_runtime_owner(UI, self._registry)
            return ctx.require(key)

        dashboard_context = DashboardContext(
            plugin_id=runtime.plugin_id,
            plugin_dir=module_path.parent,
            data_root=data_root,
            validation=ctx.require(HOST_INFO).validation,
            _resolve=resolve,
            _workspace_roots=tuple(
                (name, resolve_declared_workspace_root(workspace, name))
                for name in workspace_roots
            ),
            _workspace_files=tuple(
                (name, resolve_declared_workspace_file(workspace, name))
                for name in workspace_files
            ),
            _workload_urls=MappingProxyType(
                dict(workload_urls)
            ),
        )
        enabled_result = True
        if callable(enabled):
            with plugin_entrypoint(
                plugin_id=runtime.plugin_id,
                generation_id=runtime.generation_id,
                fiber=runtime.plugin_id,
                operation="dashboard.plugin_enabled",
            ):
                enabled_result = enabled(dashboard_context)
                _reject_dashboard_awaitable(
                    enabled_result,
                    operation="plugin_enabled",
                )
                if not isinstance(enabled_result, bool):
                    raise RuntimeError("v3 dashboard plugin_enabled 必须返回 bool")
        if enabled_result:
            with plugin_entrypoint(
                plugin_id=runtime.plugin_id,
                generation_id=runtime.generation_id,
                fiber=runtime.plugin_id,
                operation="dashboard.register",
            ):
                registered = register(app, dashboard_context)
                _reject_dashboard_awaitable(
                    registered,
                    operation="register",
                )
                _take_closeables(registered, self.closeables)
        # 3. 先保留关闭句柄，再校验路由；拒绝发布也不能丢失资源。
        if app.router.on_startup or app.router.on_shutdown:
            raise RuntimeError("dashboard module 不支持 startup/shutdown hook")
        routes = _plugin_routes(app.routes)
        binding = DashboardBinding(
            plugin_id=runtime.plugin_id,
            app=app,
            routes=routes,
            context=ctx,
            runtime_workspace=workspace,
            runtime_data_root=data_root,
            module_name=module.__name__,
            generation_id=runtime.generation_id,
            has_web=self.has_web,
        )
        _require_routes_available(binding, occupied)
        return binding


def _reject_dashboard_awaitable(value: object, *, operation: str) -> None:
    """关闭不受支持的 awaitable，并让 v3 Dashboard ABI 错误显式失败。"""

    if not inspect.isawaitable(value):
        return
    close = getattr(value, "close", None)
    if callable(close):
        try:
            close()
        except Exception as error:
            raise RuntimeError(
                f"v3 dashboard {operation} 不支持 async，且 awaitable 关闭失败"
            ) from error
    raise RuntimeError(f"v3 dashboard {operation} 不支持 async")


def _take_closeables(value: object, owned: list[Closeable]) -> None:
    """即使返回列表有坏项，所有实际关闭句柄也先归原注册 Effect。"""
    if value is None:
        return
    values = value if isinstance(value, (list, tuple)) else (value,)
    invalid = []
    for index, item in enumerate(values):
        if callable(getattr(item, "close", None)):
            owned.append(cast(Closeable, item))
        else:
            invalid.append(index)
    if invalid:
        raise RuntimeError(f"v3 dashboard register 返回值不是 closeable: indexes={invalid}")


def _plugin_routes(routes: Sequence[object]) -> tuple[DashboardRoute, ...]:
    if any(not isinstance(route, (APIRoute, WebSocketRoute)) for route in routes):
        raise RuntimeError("dashboard module 只支持 HTTP API 或 WebSocket route")
    typed = tuple(
        route for route in routes if isinstance(route, (APIRoute, WebSocketRoute))
    )
    builtin_convertor_types = {
        StringConvertor,
        PathConvertor,
        IntegerConvertor,
        FloatConvertor,
        UUIDConvertor,
    }
    if any(
        type(convertor) not in builtin_convertor_types
        for route in typed
        for convertor in route.param_convertors.values()
    ):
        raise RuntimeError("dashboard route 只支持内建 path converter")
    return typed


def _core_routes(routes: tuple[object, ...]) -> tuple[DashboardRoute, ...]:
    return tuple(
        route for route in routes if isinstance(route, (APIRoute, WebSocketRoute))
    )


def _require_routes_available(
    binding: DashboardBinding,
    occupied: list[DashboardRoute],
) -> None:
    conflicts: list[str] = []
    for index, route in enumerate(binding.routes):
        for other in occupied:
            methods = _overlapping_methods(route, other)
            if methods and _route_paths_overlap(route, other):
                conflicts.append(f"{','.join(methods)} {route.path} <> {other.path}")
        for other in binding.routes[:index]:
            methods = _overlapping_methods(route, other)
            if (
                methods
                and _route_paths_overlap(route, other)
                and not _ordered_specific_route_wins(other, route)
            ):
                conflicts.append(f"{','.join(methods)} {route.path} <> {other.path}")
    if conflicts:
        raise RuntimeError(f"dashboard route 冲突: {', '.join(conflicts)}")


def _route_paths_overlap(first: DashboardRoute, second: DashboardRoute) -> bool:
    first_sample = _sample_route_path(first)
    second_sample = _sample_route_path(second)
    return bool(
        first.path_regex.fullmatch(second_sample)
        or second.path_regex.fullmatch(first_sample)
    )


def _overlapping_methods(first: DashboardRoute, second: DashboardRoute) -> list[str]:
    if isinstance(first, APIRoute) != isinstance(second, APIRoute):
        return []
    if isinstance(first, WebSocketRoute):
        return ["WEBSOCKET"]
    assert isinstance(second, APIRoute)
    if not first.methods and not second.methods:
        return ["*"]
    if not first.methods:
        return sorted(second.methods or ())
    if not second.methods:
        return sorted(first.methods)
    return sorted(first.methods.intersection(second.methods))


def _ordered_specific_route_wins(
    first: DashboardRoute,
    second: DashboardRoute,
) -> bool:
    """Allow an earlier narrow route that cannot shadow the later broad route."""

    first_sample = _sample_route_path(first)
    second_sample = _sample_route_path(second)
    return bool(
        second.path_regex.fullmatch(first_sample)
        and not first.path_regex.fullmatch(second_sample)
    )


def _sample_route_path(route: DashboardRoute) -> str:
    def replace(match: re.Match[str]) -> str:
        convertor = route.param_convertors[match.group(1)]
        regex = re.compile(f"^(?:{convertor.regex})$")
        for candidate in (
            "x",
            "1",
            "1.0",
            "00000000-0000-0000-0000-000000000000",
            "x/y",
        ):
            if regex.fullmatch(candidate):
                return candidate
        raise RuntimeError(f"dashboard route convertor 不受支持: {route.path}")

    return re.sub(r"\{([^}:]+)(?::[^}]+)?\}", replace, route.path)
