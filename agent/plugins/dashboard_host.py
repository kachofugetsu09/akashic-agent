from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
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

from agent.plugins.generation import PluginGeneration
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

    def matches(self, scope: dict[str, Any]) -> bool:
        return any(route.matches(scope)[0] is Match.FULL for route in self.routes)


class PluginDashboardHost:
    def __init__(
        self,
        *,
        workspace: Path,
        memory_admin: object,
        memory_store: object,
        core_routes: tuple[object, ...],
    ) -> None:
        self._workspace = workspace
        self._memory_admin = memory_admin
        self._memory_store = memory_store
        self._core_routes = _core_routes(core_routes)
        self._bindings: dict[str, DashboardBinding] = {}
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
        for generation in snapshot.generations.values():
            module_path = generation.contributions.dashboard_module
            generation_id = generation.generation_id
            if module_path is None or generation_id in self._unavailable:
                continue
            binding = self._bindings.get(generation_id)
            if binding is None:
                try:
                    binding = self._build_binding(
                        generation,
                        module_path,
                        occupied=occupied,
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
                self._bindings[generation_id] = binding

                def remove_binding(
                    generation_id: str = generation_id,
                ) -> None:
                    _ = self._bindings.pop(generation_id, None)

                generation.scope.defer(
                    "dashboard",
                    remove_binding,
                )
            else:
                _require_routes_available(binding, occupied)
            bindings.append(binding)
            occupied.extend(binding.routes)
        snapshot.dashboard_bindings = tuple(bindings)

    def _build_binding(
        self,
        generation: PluginGeneration,
        module_path: Path,
        *,
        occupied: list[APIRoute],
    ) -> DashboardBinding:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        app.state.memory_admin = self._memory_admin
        app.state.memory_store = self._memory_store
        name = f"{generation.module_path}.dashboard"
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
            closeables = (
                []
                if callable(enabled) and not enabled(app)
                else _closeables(register(app, module_path.parent, self._workspace))
            )
            for index, closeable in enumerate(closeables):
                generation.scope.defer(
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
            )
            _require_routes_available(binding, occupied)
        except BaseException:
            _ = sys.modules.pop(name, None)
            raise

        def remove_module() -> None:
            _ = sys.modules.pop(name, None)

        generation.scope.defer("dashboard_module", remove_module)
        return binding


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


def _closeables(value: object) -> list[object]:
    values = value if isinstance(value, list) else [value]
    return [item for item in values if callable(getattr(item, "close", None))]


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
