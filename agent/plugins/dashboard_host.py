from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import Match, WebSocketRoute

from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.ui import UI, UiRegistry
from agent.plugins.snapshot import RuntimeSnapshot, RuntimeSnapshotStore, bind_runtime_snapshot, reset_runtime_snapshot

logger = logging.getLogger(__name__)


class PluginDashboardHost:
    """把 HTTP 宿主事实交给实际 Root 的 UI provider。"""

    def __init__(
        self, *, core_routes: tuple[object, ...],
        workload_urls: Callable[[str], Mapping[tuple[str, str], str]] | None = None,
    ) -> None:
        self._core_routes = core_routes
        self._workload_urls = workload_urls or (lambda _generation_id: {})

    def prepare_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self._prepare_snapshot(snapshot, tolerate_failures=False)

    def prepare_initial_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self._prepare_snapshot(snapshot, tolerate_failures=True)

    def _prepare_snapshot(self, snapshot: RuntimeSnapshot, *, tolerate_failures: bool) -> None:
        registry = _ui_registry(snapshot)
        if registry is None:
            return
        root = snapshot.composition_root
        assert root is not None
        validation = frozenset(
            generation.plugin_id for generation in snapshot.active_generations()
            if root.plugin_runtime(generation.plugin_id).data_dir.resolve()
            != generation.data_dir.resolve()
        )
        registry.prepare_dashboard(
            core_routes=self._core_routes, workload_urls=self._workload_urls,
            validation_owners=validation, tolerate_failures=tolerate_failures,
        )

    async def release_validation(self, snapshot: RuntimeSnapshot) -> None:
        registry = _ui_registry(snapshot)
        if registry is not None:
            await registry.release_validation()


def _ui_registry(snapshot: RuntimeSnapshot) -> UiRegistry | None:
    root = snapshot.composition_root
    registry = None if root is None else root.context.get(UI)
    if registry is not None:
        assert root is not None
        if registry.root_instance_token is not root.instance_token:
            raise RuntimeError("UI provider 不属于所选 snapshot 的实际 Root")
    return registry


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
                    registry = _ui_registry(lease.snapshot)
                    for binding in (() if registry is None else registry.bindings()):
                        if binding.matches(scope):
                            if (
                                binding.has_web
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
                                generation_id=binding.generation_id,
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
    registry = _ui_registry(snapshot)
    catalog = None if registry is None else registry.catalog()
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
