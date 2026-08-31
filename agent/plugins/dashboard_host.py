from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.convertors import (
    FloatConvertor,
    IntegerConvertor,
    PathConvertor,
    StringConvertor,
    UUIDConvertor,
)
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import Match, WebSocketRoute

from agent.plugin_composition import DashboardContext
from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.model import (
    resolve_declared_workspace_file,
    resolve_declared_workspace_root,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.generation import PluginGeneration
from agent.plugins.scope import PluginScope
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)

logger = logging.getLogger(__name__)

DashboardRoute = APIRoute | WebSocketRoute


class _DashboardImportError(RuntimeError):
    pass


@dataclass
class DashboardBinding:
    plugin_id: str
    app: FastAPI
    routes: tuple[DashboardRoute, ...]
    runtime_workspace: Path | None = None
    runtime_data_root: Path | None = None
    validation: bool = False
    module_name: str = ""
    _scope: PluginScope | None = field(default=None, repr=False)

    def matches(self, scope: dict[str, Any]) -> bool:
        return any(route.matches(scope)[0] is Match.FULL for route in self.routes)


class PluginDashboardHost:
    def __init__(
        self,
        *,
        core_routes: tuple[object, ...],
        workload_urls: Callable[[str], Mapping[tuple[str, str], str]] | None = None,
    ) -> None:
        self._core_routes = _core_routes(core_routes)
        self._workload_urls = workload_urls or (lambda _generation_id: {})
        self._bindings: dict[tuple[str, Path], DashboardBinding] = {}
        self._unavailable: set[str] = set()

    def prepare_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self._prepare_snapshot(snapshot, tolerate_failures=False)

    def prepare_initial_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self._prepare_snapshot(snapshot, tolerate_failures=True)

    def _prepare_snapshot(
        self,
        snapshot: RuntimeSnapshot,
        *,
        tolerate_failures: bool,
    ) -> None:
        bindings: list[DashboardBinding] = []
        occupied = list(self._core_routes)
        active_generations = {
            generation.plugin_id for generation in snapshot.active_generations()
        }
        for generation in snapshot.generations.values():
            if not isinstance(generation.instance, ComposablePlugin):
                raise RuntimeError(
                    f"Dashboard 只接受 v3 generation: {generation.plugin_id}"
                )
            if generation.plugin_id not in active_generations:
                continue
            module_path = generation.contributions.dashboard_module
            generation_id = generation.generation_id
            if module_path is None or generation_id in self._unavailable:
                continue
            root = snapshot.composition_root
            if root is None:
                raise RuntimeError(
                    f"v3 Dashboard 缺少 composition Root: {generation.plugin_id}"
                )
            runtime = root.plugin_runtime(generation.plugin_id)
            runtime_workspace = runtime.workspace.resolve(strict=False)
            data_root = runtime.data_dir.resolve(strict=False)
            workspace_roots = runtime.workspace_roots
            workspace_files = runtime.workspace_files
            validation = data_root != generation.data_dir.resolve(strict=False)
            if validation:
                runtime_workspace.mkdir(parents=True, exist_ok=True)
            binding_key = (generation_id, runtime_workspace)
            binding = self._bindings.get(binding_key)
            if binding is None:
                binding_scope = generation.scope
                if validation:
                    binding_scope = PluginScope(
                        f"{generation.plugin_id}:dashboard-validation",
                        generation_id=generation.generation_id,
                        diagnostic_plugin_id=generation.plugin_id,
                    )
                    generation.scope.defer(
                        "validation_dashboard",
                        lambda binding_scope=binding_scope: (
                            _close_dashboard_scope(binding_scope)
                        ),
                    )
                try:
                    binding = self._build_binding(
                        generation,
                        module_path,
                        occupied=occupied,
                        workspace=runtime_workspace,
                        data_root=data_root,
                        workspace_roots=workspace_roots,
                        workspace_files=workspace_files,
                        scope=binding_scope,
                        validation=validation,
                    )
                except Exception as error:
                    if (
                        not tolerate_failures
                        or not isinstance(error, _DashboardImportError)
                        or generation.contributions.web_module is not None
                    ):
                        raise
                    self._unavailable.add(generation_id)

                    def remove_unavailable(
                        generation_id: str = generation_id,
                    ) -> None:
                        self._unavailable.discard(generation_id)

                    generation.scope.defer(
                        "dashboard_unavailable",
                        remove_unavailable,
                    )
                    logger.warning(
                        "初始插件 dashboard 挂载失败 (%s): %s",
                        generation.plugin_id,
                        error,
                    )
                    continue
                self._bindings[binding_key] = binding

                def remove_binding(
                    binding_key: tuple[str, Path] = binding_key,
                ) -> None:
                    _ = self._bindings.pop(binding_key, None)

                binding_scope.defer(
                    "dashboard",
                    remove_binding,
                )
            else:
                _require_routes_available(binding, occupied)
            bindings.append(binding)
            occupied.extend(binding.routes)
        snapshot.dashboard_bindings = tuple(bindings)

    async def release_validation(self, snapshot: RuntimeSnapshot) -> None:
        """Close candidate-only dashboard resources before formal rebuild."""

        # 1. Candidate bindings own a child scope so promotion can retire them early.
        bindings = tuple(
            binding
            for binding in snapshot.dashboard_bindings
            if isinstance(binding, DashboardBinding) and binding.validation
        )
        failures = []
        for binding in reversed(bindings):
            scope = binding._scope
            if scope is None:
                raise RuntimeError(
                    f"candidate dashboard 缺少隔离 scope: {binding.plugin_id}"
                )
            failures.extend(await scope.aclose())

        # 2. A formal snapshot must never retain a validation-workspace binding.
        snapshot.dashboard_bindings = tuple(
            binding
            for binding in snapshot.dashboard_bindings
            if not isinstance(binding, DashboardBinding) or not binding.validation
        )
        if failures:
            details = ", ".join(
                f"{failure.resource}: {failure.error}" for failure in failures
            )
            raise RuntimeError(f"candidate dashboard 清理失败: {details}")

    def _build_binding(
        self,
        generation: PluginGeneration,
        module_path: Path,
        *,
        occupied: list[DashboardRoute],
        workspace: Path,
        data_root: Path,
        workspace_roots: tuple[str, ...],
        workspace_files: tuple[str, ...],
        scope: PluginScope,
        validation: bool,
    ) -> DashboardBinding:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        suffix = (
            ""
            if not validation
            else "_validation_"
            + hashlib.sha256(str(workspace).encode()).hexdigest()[:12]
        )
        name = f"{generation.module_path}.dashboard{suffix}"
        module = ModuleType(name)
        module.__file__ = str(module_path)
        module.__package__ = generation.module_path
        sys.modules[name] = module
        try:
            source = module_path.read_text(encoding="utf-8")
            try:
                with plugin_entrypoint(
                    plugin_id=generation.plugin_id,
                    generation_id=generation.generation_id,
                    fiber=generation.plugin_id,
                    operation="dashboard.module_load",
                ):
                    exec(compile(source, str(module_path), "exec"), module.__dict__)
            except Exception as error:
                raise _DashboardImportError(str(error)) from error
            register = getattr(module, "register", None)
            if not callable(register):
                raise RuntimeError(f"dashboard module 缺少 register: {module_path}")
            enabled = getattr(module, "plugin_enabled", None)
            if enabled is not None and not callable(enabled):
                raise RuntimeError("v3 dashboard plugin_enabled 必须是可调用对象")
            dashboard_context = DashboardContext(
                plugin_id=generation.plugin_id,
                plugin_dir=module_path.parent,
                data_root=data_root,
                validation=validation,
                _workspace_roots=tuple(
                    (name, resolve_declared_workspace_root(workspace, name))
                    for name in workspace_roots
                ),
                _workspace_files=tuple(
                    (name, resolve_declared_workspace_file(workspace, name))
                    for name in workspace_files
                ),
                _workload_urls=MappingProxyType(
                    dict(self._workload_urls(generation.generation_id))
                ),
            )
            enabled_result = True
            if callable(enabled):
                with plugin_entrypoint(
                    plugin_id=generation.plugin_id,
                    generation_id=generation.generation_id,
                    fiber=generation.plugin_id,
                    operation="dashboard.plugin_enabled",
                ):
                    enabled_result = enabled(dashboard_context)
                    _reject_dashboard_awaitable(
                        enabled_result,
                        operation="plugin_enabled",
                    )
                    if not isinstance(enabled_result, bool):
                        raise RuntimeError("v3 dashboard plugin_enabled 必须返回 bool")
            registered = None
            if enabled_result:
                with plugin_entrypoint(
                    plugin_id=generation.plugin_id,
                    generation_id=generation.generation_id,
                    fiber=generation.plugin_id,
                    operation="dashboard.register",
                ):
                    registered = register(app, dashboard_context)
                    _reject_dashboard_awaitable(
                        registered,
                        operation="register",
                    )
                    closeables = _dashboard_closeables(registered)
            else:
                closeables = []
            for index, closeable in enumerate(closeables):
                scope.defer(
                    f"dashboard_closeable:{index}",
                    getattr(closeable, "close"),
                )
            if app.router.on_startup or app.router.on_shutdown:
                raise RuntimeError("dashboard module 不支持 startup/shutdown hook")
            routes = _plugin_routes(app.routes)
            binding = DashboardBinding(
                plugin_id=generation.plugin_id,
                app=app,
                routes=routes,
                runtime_workspace=workspace,
                runtime_data_root=data_root,
                validation=validation,
                module_name=name,
                _scope=scope,
            )
            _require_routes_available(binding, occupied)
        except BaseException:
            _ = sys.modules.pop(name, None)
            raise

        def remove_module() -> None:
            _ = sys.modules.pop(name, None)

        scope.defer("dashboard_module", remove_module)
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


def _dashboard_closeables(value: object) -> list[object]:
    """严格归一化 v3 register 返回的受 scope 管理资源。"""

    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else (value,)
    closeables = list(values)
    for index, item in enumerate(closeables):
        if not callable(getattr(item, "close", None)):
            raise RuntimeError(
                f"v3 dashboard register 返回值不是 closeable: index={index}"
            )
    return closeables


class SnapshotDashboardMiddleware:
    def __init__(self, app: object, snapshot_store: RuntimeSnapshotStore) -> None:
        self._app = app
        self._snapshot_store = snapshot_store

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type in {"http", "websocket"}:
            headers = Headers(scope=scope)
            try:
                web_identity = (
                    _websocket_request_identity(scope)
                    if scope_type == "websocket"
                    else _web_request_identity(headers)
                )
            except RuntimeError as error:
                logger.warning(
                    "Web UI WebSocket 身份解析失败: path=%s error=%s",
                    scope.get("path"),
                    error,
                )
                await _reject_web_request(409, "stale_catalog", scope, receive, send)
                return
            dashboard_path = str(scope.get("path", "")).startswith("/api/dashboard/")
            if scope_type == "websocket" and web_identity is None:
                if dashboard_path:
                    logger.warning(
                        "Web UI WebSocket 缺少 generation 身份: path=%s",
                        scope.get("path"),
                    )
                    await _reject_web_request(
                        403, "forbidden_contract", scope, receive, send
                    )
                    return
                await self._app(scope, receive, send)  # type: ignore[operator]
                return
            if (
                scope_type == "websocket"
                and web_identity is not None
                and not _same_origin_websocket(headers)
            ):
                logger.warning(
                    "Web UI WebSocket 来源不匹配: path=%s origin=%s host=%s",
                    scope.get("path"),
                    headers.get("origin"),
                    headers.get("host"),
                )
                await _reject_web_request(
                    403, "forbidden_contract", scope, receive, send
                )
                return
            if (
                web_identity is None
                and dashboard_path
                and headers.get("sec-fetch-site") in {"same-origin", "same-site"}
            ):
                await _reject_web_request(
                    403, "forbidden_contract", scope, receive, send
                )
                return
            if self._snapshot_store.current is None:
                if web_identity is not None:
                    await _reject_web_request(
                        409, "stale_catalog", scope, receive, send
                    )
                    return
                await self._app(scope, receive, send)  # type: ignore[operator]
                return
            try:
                lease = (
                    self._snapshot_store.lease(web_identity[0])
                    if web_identity is not None
                    else await self._snapshot_store.acquire()
                )
            except RuntimeError as error:
                logger.warning(
                    "Web UI snapshot 不可租用: reason=%s",
                    type(error).__name__,
                )
                await _reject_web_request(409, "stale_catalog", scope, receive, send)
                return
            async with lease:
                if web_identity is not None and not _web_request_matches(
                    lease.snapshot,
                    web_identity,
                ):
                    logger.warning(
                        "Web UI WebSocket generation 已过期",
                    )
                    await _reject_web_request(
                        409, "stale_catalog", scope, receive, send
                    )
                    return
                token = bind_runtime_snapshot(lease)
                try:
                    for raw_binding in lease.snapshot.dashboard_bindings:
                        binding = raw_binding
                        if isinstance(binding, DashboardBinding) and binding.matches(
                            scope
                        ):
                            generation = lease.snapshot.generations[binding.plugin_id]
                            if (
                                generation.contributions.web_module is not None
                                and web_identity is None
                            ):
                                await _reject_web_request(
                                    403,
                                    "forbidden_contract",
                                    scope,
                                    receive,
                                    send,
                                )
                                return
                            if (
                                web_identity is not None
                                and binding.plugin_id != web_identity[2]
                            ):
                                logger.warning(
                                    "Web UI WebSocket plugin 身份不匹配",
                                )
                                await _reject_web_request(
                                    403,
                                    "forbidden_contract",
                                    scope,
                                    receive,
                                    send,
                                )
                                return
                            route = next(
                                route
                                for route in binding.routes
                                if route.matches(scope)[0] is Match.FULL
                            )
                            with plugin_entrypoint(
                                plugin_id=binding.plugin_id,
                                generation_id=generation.generation_id,
                                fiber=binding.plugin_id,
                                operation=f"dashboard.{scope_type}",
                                entrypoint=route.path,
                            ):
                                if scope_type == "websocket":
                                    await _run_dashboard_websocket(
                                        binding.app,
                                        scope,
                                        receive,
                                        send,
                                        lease.snapshot,
                                    )
                                else:
                                    await binding.app(scope, receive, send)
                            return
                    if web_identity is not None:
                        logger.warning(
                            "Web UI WebSocket 路由不存在",
                        )
                        await _reject_web_request(
                            403,
                            "forbidden_contract",
                            scope,
                            receive,
                            send,
                        )
                        return
                    await self._app(scope, receive, send)  # type: ignore[operator]
                    return
                finally:
                    reset_runtime_snapshot(token)
            return
        await self._app(scope, receive, send)  # type: ignore[operator]


async def _run_dashboard_websocket(
    app: object,
    scope: dict[str, Any],
    receive: Any,
    send: Any,
    snapshot: RuntimeSnapshot,
) -> None:
    """Close one live plugin socket before its exact snapshot can drain."""

    closed = False

    async def tracked_send(message: dict[str, Any]) -> None:
        nonlocal closed
        if message.get("type") == "websocket.close":
            closed = True
        await send(message)

    task = asyncio.create_task(app(scope, receive, tracked_send))  # type: ignore[operator]
    try:
        while not task.done() and snapshot.accepting_leases:
            await asyncio.wait((task,), timeout=0.2)
        if task.done():
            await task
            return
        if not closed:
            await tracked_send(
                {
                    "type": "websocket.close",
                    "code": 1012,
                    "reason": "plugin generation changed",
                }
            )
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def _web_request_identity(headers: Headers) -> tuple[str, str, str, str] | None:
    values = (
        headers.get("x-akashic-web-snapshot", ""),
        headers.get("x-akashic-web-catalog", ""),
        headers.get("x-akashic-web-module", ""),
        headers.get("x-akashic-web-generation", ""),
    )
    if not any(values):
        return None
    if not all(values):
        raise RuntimeError("Web UI 请求身份不完整")
    return values


def _websocket_request_identity(
    scope: dict[str, Any],
) -> tuple[str, str, str, str] | None:
    try:
        query = parse_qs(
            bytes(scope.get("query_string", b"")).decode("ascii"),
            keep_blank_values=True,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("Web UI WebSocket 请求身份无效") from error
    names = (
        "__akashic_web_snapshot",
        "__akashic_web_catalog",
        "__akashic_web_module",
        "__akashic_web_generation",
    )
    values = tuple(query.get(name, []) for name in names)
    if not any(values):
        return None
    if any(len(value) != 1 or not value[0] or len(value[0]) > 256 for value in values):
        raise RuntimeError("Web UI WebSocket 请求身份不完整")
    return tuple(value[0] for value in values)  # type: ignore[return-value]


def _same_origin_websocket(headers: Headers) -> bool:
    origin = headers.get("origin")
    host = headers.get("host")
    if not origin or not host:
        return False
    parsed = urlsplit(origin)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.username is None
        and parsed.password is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
        and parsed.netloc.casefold() == host.casefold()
    )


def _web_request_matches(
    snapshot: RuntimeSnapshot,
    identity: tuple[str, str, str, str],
) -> bool:
    snapshot_id, catalog_id, plugin_id, generation_id = identity
    catalog = snapshot.web_ui_catalog
    return (
        snapshot.snapshot_id == snapshot_id
        and catalog is not None
        and catalog.identity == catalog_id
        and any(
            item.plugin_id == plugin_id and item.generation_id == generation_id
            for item in catalog.modules
        )
    )


async def _web_error(
    status: int,
    code: str,
    scope: dict[str, Any],
    receive: Any,
    send: Any,
) -> None:
    headers = {"X-Akashic-Web-Stale": "1"} if code == "stale_catalog" else None
    await JSONResponse(
        {"code": code},
        status_code=status,
        headers=headers,
    )(scope, receive, send)


async def _reject_web_request(
    status: int,
    code: str,
    scope: dict[str, Any],
    receive: Any,
    send: Any,
) -> None:
    if scope.get("type") == "websocket":
        await send(
            {
                "type": "websocket.close",
                "code": 4409 if code == "stale_catalog" else 4403,
                "reason": code,
            }
        )
        return
    await _web_error(status, code, scope, receive, send)


async def _close_dashboard_scope(scope: PluginScope) -> None:
    failures = await scope.aclose()
    if failures:
        details = ", ".join(
            f"{failure.resource}: {failure.error}" for failure in failures
        )
        raise RuntimeError(f"candidate dashboard 清理失败: {details}")


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
