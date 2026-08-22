from __future__ import annotations

import hashlib
import inspect
import logging
import re
import sys
from dataclasses import dataclass, field
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.convertors import (
    FloatConvertor,
    IntegerConvertor,
    PathConvertor,
    StringConvertor,
    UUIDConvertor,
)
from starlette.routing import Match

from agent.plugin_composition import DashboardContext
from agent.plugin_composition.model import resolve_declared_workspace_root
from agent.plugins.composable import ComposablePlugin
from agent.plugins.generation import PluginGeneration
from agent.plugins.private_proactive import PrivateFamily
from agent.plugins.scope import PluginScope
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)

logger = logging.getLogger(__name__)


class _DashboardImportError(RuntimeError):
    pass


@dataclass
class DashboardBinding:
    plugin_id: str
    app: FastAPI
    routes: tuple[APIRoute, ...]
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
    ) -> None:
        self._core_routes = _core_routes(core_routes)
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
            validation = data_root != generation.data_dir.resolve(strict=False)
            if validation:
                runtime_workspace.mkdir(parents=True, exist_ok=True)
            binding_key = (generation_id, runtime_workspace)
            binding = self._bindings.get(binding_key)
            if binding is None:
                binding_scope = generation.scope
                if validation:
                    binding_scope = PluginScope(
                        f"{generation.plugin_id}:dashboard-validation"
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
                        scope=binding_scope,
                        validation=validation,
                    )
                except Exception as error:
                    if not tolerate_failures or not isinstance(
                        error,
                        _DashboardImportError,
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
        private_binding = self._prepare_private_proactive_dashboard(
            snapshot,
            occupied=occupied,
        )
        if private_binding is not None:
            bindings.append(private_binding)
        snapshot.dashboard_bindings = tuple(bindings)

    def _prepare_private_proactive_dashboard(
        self,
        snapshot: RuntimeSnapshot,
        *,
        occupied: list[APIRoute],
    ) -> DashboardBinding | None:
        """把 exact Default/Wake reader 投影到当前 snapshot。"""

        # 1. 从私有 catalog 确定唯一 family 与 exact primary generation。
        catalog = snapshot.private_proactive_catalog
        if catalog is None:
            return None
        available_families: tuple[PrivateFamily, ...] = ("default", "wake")
        families: list[PrivateFamily] = []
        for family in available_families:
            if catalog.family(family):
                families.append(family)
        if len(families) != 1:
            raise RuntimeError("private proactive Dashboard family 必须唯一")
        family = families[0]
        primary = catalog.family(family)[0]
        generation = snapshot.generations.get(primary.member)
        if generation is None or generation.generation_id != primary.generation_id:
            raise RuntimeError("private proactive Dashboard generation 不匹配")
        root = snapshot.composition_root
        if root is None:
            raise RuntimeError("private proactive Dashboard 缺少 composition Root")
        runtime = root.plugin_runtime(primary.member)
        workspace = runtime.workspace.resolve(strict=False)
        validation = runtime.data_dir.resolve(strict=False) != (
            generation.data_dir.resolve(strict=False)
        )
        if validation:
            workspace.mkdir(parents=True, exist_ok=True)

        # 2. 复用 generation scope，使 reader 与 snapshot 一起 drain/close。
        binding_key = (f"private-dashboard:{generation.generation_id}", workspace)
        binding = self._bindings.get(binding_key)
        if binding is not None:
            _require_routes_available(binding, occupied)
            return binding
        scope = generation.scope
        if validation:
            scope = PluginScope(
                f"{generation.plugin_id}:private-dashboard-validation"
            )
            generation.scope.defer(
                "private_validation_dashboard",
                lambda scope=scope: _close_dashboard_scope(scope),
            )
        binding = self._build_private_proactive_binding(
            family,
            generation=generation,
            workspace=workspace,
            scope=scope,
            validation=validation,
            occupied=occupied,
        )
        self._bindings[binding_key] = binding

        def remove_binding() -> None:
            _ = self._bindings.pop(binding_key, None)

        scope.defer("private_dashboard", remove_binding)
        return binding

    def _build_private_proactive_binding(
        self,
        family: PrivateFamily,
        *,
        generation: PluginGeneration,
        workspace: Path,
        scope: PluginScope,
        validation: bool,
        occupied: list[APIRoute],
    ) -> DashboardBinding:
        """注册一个 Core-private proactive Dashboard binding。"""

        # 1. 调用 family 固定的 Core reader，不解析外部 module/callable。
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        if family == "wake":
            from plugins.wake_proactive.dashboard import register_private_dashboard
        else:
            from plugins.default_proactive.dashboard import register_private_dashboard
        registered = register_private_dashboard(app, workspace)
        for index, closeable in enumerate(_dashboard_closeables(registered)):
            scope.defer(
                f"private_dashboard_closeable:{index}",
                getattr(closeable, "close"),
            )
        if app.router.on_startup or app.router.on_shutdown:
            raise RuntimeError("private proactive dashboard 不支持 startup/shutdown hook")

        # 2. 沿用 Dashboard host 的路由冲突与 snapshot dispatch 合同。
        routes = _plugin_routes(app.routes)
        binding = DashboardBinding(
            plugin_id=f"{family}-proactive",
            app=app,
            routes=routes,
            runtime_workspace=workspace,
            runtime_data_root=generation.data_dir.resolve(strict=False),
            validation=validation,
            module_name=f"core.private_proactive.dashboard.{family}",
            _scope=scope,
        )
        _require_routes_available(binding, occupied)
        return binding

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
        occupied: list[APIRoute],
        workspace: Path,
        data_root: Path,
        workspace_roots: tuple[str, ...],
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
            )
            enabled_result = True
            if callable(enabled):
                enabled_result = enabled(dashboard_context)
                _reject_dashboard_awaitable(
                    enabled_result,
                    operation="plugin_enabled",
                )
            if not isinstance(enabled_result, bool):
                raise RuntimeError("v3 dashboard plugin_enabled 必须返回 bool")
            registered = None
            if enabled_result:
                registered = register(app, dashboard_context)
                _reject_dashboard_awaitable(
                    registered,
                    operation="register",
                )
            closeables: list[object] = []
            if enabled_result:
                closeables = _dashboard_closeables(registered)
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
        if scope.get("type") == "http" and self._snapshot_store.current is not None:
            lease = await self._snapshot_store.acquire()
            async with lease:
                token = bind_runtime_snapshot(lease)
                try:
                    for raw_binding in lease.snapshot.dashboard_bindings:
                        binding = raw_binding
                        if isinstance(binding, DashboardBinding) and binding.matches(scope):
                            await binding.app(scope, receive, send)
                            return
                    await self._app(scope, receive, send)  # type: ignore[operator]
                    return
                finally:
                    reset_runtime_snapshot(token)
        await self._app(scope, receive, send)  # type: ignore[operator]


async def _close_dashboard_scope(scope: PluginScope) -> None:
    failures = await scope.aclose()
    if failures:
        details = ", ".join(
            f"{failure.resource}: {failure.error}" for failure in failures
        )
        raise RuntimeError(f"candidate dashboard 清理失败: {details}")


def _plugin_routes(routes: Sequence[object]) -> tuple[APIRoute, ...]:
    if any(not isinstance(route, APIRoute) for route in routes):
        raise RuntimeError("dashboard module 只支持 HTTP API route")
    typed = tuple(route for route in routes if isinstance(route, APIRoute))
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


def _core_routes(routes: tuple[object, ...]) -> tuple[APIRoute, ...]:
    return tuple(route for route in routes if isinstance(route, APIRoute))


def _require_routes_available(
    binding: DashboardBinding,
    occupied: list[APIRoute],
) -> None:
    conflicts: list[str] = []
    for index, route in enumerate(binding.routes):
        for other in occupied:
            methods = _overlapping_methods(route, other)
            if methods and _route_paths_overlap(route, other):
                conflicts.append(
                    f"{','.join(methods)} {route.path} <> {other.path}"
                )
        for other in binding.routes[:index]:
            methods = _overlapping_methods(route, other)
            if (
                methods
                and _route_paths_overlap(route, other)
                and not _ordered_static_route_wins(other, route)
            ):
                conflicts.append(
                    f"{','.join(methods)} {route.path} <> {other.path}"
                )
    if conflicts:
        raise RuntimeError(f"dashboard route 冲突: {', '.join(conflicts)}")


def _route_paths_overlap(first: APIRoute, second: APIRoute) -> bool:
    first_sample = _sample_route_path(first)
    second_sample = _sample_route_path(second)
    return bool(
        first.path_regex.fullmatch(second_sample)
        or second.path_regex.fullmatch(first_sample)
    )


def _overlapping_methods(first: APIRoute, second: APIRoute) -> list[str]:
    if not first.methods and not second.methods:
        return ["*"]
    if not first.methods:
        return sorted(second.methods or ())
    if not second.methods:
        return sorted(first.methods)
    return sorted(first.methods.intersection(second.methods))


def _ordered_static_route_wins(first: APIRoute, second: APIRoute) -> bool:
    return not first.param_convertors and bool(second.param_convertors)


def _sample_route_path(route: APIRoute) -> str:
    def replace(match: re.Match[str]) -> str:
        convertor = route.param_convertors[match.group(1)]
        regex = re.compile(f"^(?:{convertor.regex})$")
        for candidate in ("x", "1", "1.0", "00000000-0000-0000-0000-000000000000", "x/y"):
            if regex.fullmatch(candidate):
                return candidate
        raise RuntimeError(f"dashboard route convertor 不受支持: {route.path}")

    return re.sub(r"\{([^}:]+)(?::[^}]+)?\}", replace, route.path)
