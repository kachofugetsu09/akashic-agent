from __future__ import annotations

import asyncio
import logging
from collections.abc import MutableMapping
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import Match

from agent.plugin_composition import CompositionError, CompositionRoot, Context, FiberState
from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.ui import UI, DashboardBinding, UiRegistry, WebUiCatalog

logger = logging.getLogger(__name__)


class LiveRootHost(Protocol):
    @property
    def live_root(self) -> CompositionRoot | None: ...


class PluginDashboardHost:
    """Read the actual live Root provider and its current UI projection."""

    def __init__(self, plugin_manager: LiveRootHost) -> None:
        self._plugin_manager = plugin_manager

    def current(self) -> tuple[CompositionRoot, Context, UiRegistry] | None:
        """Return the live Root and exact UI provider relation."""

        root = self._plugin_manager.live_root
        if root is None:
            return None
        if not isinstance(root, CompositionRoot):
            raise TypeError("PluginManager.live_root 必须是 CompositionRoot")
        registry = root.service_value(UI)
        if registry is None:
            return None
        provider_context, registered = root._service_provider(UI)  # pyright: ignore[reportPrivateUsage]
        if registered is not registry:
            raise RuntimeError("UI provider 不是当前 live Root 的 active provider")
        if provider_context.fiber.state is not FiberState.ACTIVE:
            return None
        if registry.root_instance_token is not root.instance_token:
            raise RuntimeError("UI provider 不属于正式 live Root")
        return root, provider_context, registry


class LiveDashboardMiddleware:
    """Route each request through the live Root and original owner Context."""

    def __init__(self, app: object, plugin_manager: LiveRootHost) -> None:
        self._app = app
        self._host = PluginDashboardHost(plugin_manager)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type not in {"http", "websocket"}:
            await self._app(scope, receive, send)  # type: ignore[operator]
            return

        headers = Headers(scope=scope)
        try:
            identity = (
                _websocket_request_identity(scope)
                if scope_type == "websocket"
                else _web_request_identity(headers)
            )
        except RuntimeError as error:
            logger.warning("Web UI 请求身份解析失败: path=%s error=%s", scope.get("path"), error)
            await _reject_web_request(409, "stale_catalog", scope, receive, send)
            return

        dashboard_path = str(scope.get("path", "")).startswith("/api/dashboard/")
        if scope_type == "websocket" and identity is None:
            if dashboard_path:
                await _reject_web_request(403, "forbidden_contract", scope, receive, send)
                return
            await self._app(scope, receive, send)  # type: ignore[operator]
            return
        if (
            scope_type == "websocket"
            and identity is not None
            and not _same_origin_websocket(headers)
        ):
            await _reject_web_request(403, "forbidden_contract", scope, receive, send)
            return
        if (
            identity is None
            and dashboard_path
            and headers.get("sec-fetch-site") in {"same-origin", "same-site"}
        ):
            await _reject_web_request(403, "forbidden_contract", scope, receive, send)
            return

        current = self._host.current()
        if current is None:
            if identity is not None:
                await _reject_web_request(409, "stale_catalog", scope, receive, send)
                return
            await self._app(scope, receive, send)  # type: ignore[operator]
            return

        root, ui_context, registry = current
        captured = None
        binding = None
        route = None
        core_passthrough = False
        rejection: tuple[int, str] | None = None
        try:
            async with ui_context.runtime_scope():
                catalog = registry.catalog()
                if identity is not None and not _web_request_matches(root, catalog, identity):
                    rejection = (409, "stale_catalog")
                else:
                    binding = _matching_binding(registry.bindings(), scope)
                    if binding is None:
                        if identity is not None:
                            rejection = (403, "forbidden_contract")
                        else:
                            core_passthrough = True
                    elif binding.has_web and identity is None:
                        rejection = (403, "forbidden_contract")
                    elif identity is not None and binding.plugin_id != identity[2]:
                        rejection = (403, "forbidden_contract")
                    else:
                        route = next(
                            route for route in binding.routes
                            if route.matches(scope)[0] is Match.FULL
                        )
                        if scope_type == "websocket":
                            async with binding.context.runtime_scope():
                                captured = binding.context.capture_runtime_scope()
        except CompositionError as error:
            if error.code in {"OWNER_UNAVAILABLE", "STALE_ACTIVATION", "INACTIVE_SERVICE"}:
                rejection = (409, "stale_catalog")
            else:
                raise
        if rejection is not None:
            await _reject_web_request(*rejection, scope, receive, send)
            return
        if core_passthrough:
            await self._app(scope, receive, send)  # type: ignore[operator]
            return
        if scope_type == "websocket":
            assert captured is not None
            assert binding is not None
            await _run_dashboard_websocket(binding, scope, receive, send, captured)
            return
        assert binding is not None
        assert route is not None
        async with binding.context.runtime_scope():
            with plugin_entrypoint(
                plugin_id=binding.plugin_id,
                generation_id=binding.generation_id,
                fiber=binding.plugin_id,
                operation=f"dashboard.{scope_type}",
                entrypoint=route.path,
            ):
                await binding.app(scope, receive, send)


async def _run_dashboard_websocket(
    binding: DashboardBinding,
    scope: dict[str, Any],
    receive: Any,
    send: Any,
    captured: Any,
) -> None:
    """Keep one captured owner scope until the socket application exits."""

    closed = False
    app_scope_entered = False
    owner_withdrawn = False
    app_cancelled = False
    monitor_cancelled = False
    caller_cancelled = False
    errors: list[BaseException] = []

    async def tracked_send(message: MutableMapping[str, Any]) -> None:
        nonlocal closed
        if message.get("type") == "websocket.close":
            closed = True
        await send(message)

    async def run_app() -> None:
        nonlocal app_scope_entered
        async with captured:
            app_scope_entered = True
            with plugin_entrypoint(
                plugin_id=binding.plugin_id,
                generation_id=binding.generation_id,
                fiber=binding.plugin_id,
                operation="dashboard.websocket",
            ):
                await binding.app(scope, receive, tracked_send)

    app_coro = run_app()
    app_task: asyncio.Task[None] | None = None
    monitor_task: asyncio.Task[None] | None = None

    async def monitor() -> None:
        nonlocal owner_withdrawn
        await captured.wait_admission_closed()
        owner_withdrawn = True
        cancel_app_once()

    def cancel_app_once() -> None:
        nonlocal app_cancelled
        if app_task is not None and not app_task.done() and not app_cancelled:
            app_cancelled = True
            app_task.cancel()

    def cancel_monitor_once() -> None:
        nonlocal monitor_cancelled
        if monitor_task is not None and not monitor_task.done() and not monitor_cancelled:
            monitor_cancelled = True
            monitor_task.cancel()

    try:
        monitor_coro = monitor()
        try:
            app_task = asyncio.create_task(app_coro, name="dashboard-websocket")
        except BaseException:
            app_coro.close()
            monitor_coro.close()
            raise
        try:
            monitor_task = asyncio.create_task(monitor_coro, name="dashboard-websocket-drain")
        except BaseException:
            monitor_coro.close()
            cancel_app_once()
            raise
        try:
            done, _ = await asyncio.wait(
                (app_task, monitor_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if app_task in done:
                cancel_monitor_once()
            else:
                cancel_app_once()
        except asyncio.CancelledError:
            caller_cancelled = True
            cancel_monitor_once()
            cancel_app_once()
        except BaseException as error:
            errors.append(error)
    except asyncio.CancelledError:
        caller_cancelled = True
        cancel_monitor_once()
        cancel_app_once()
    except BaseException as error:
        errors.append(error)
    finally:
        cancel_monitor_once()
        cancel_app_once()
        if monitor_task is not None:
            interrupted, error = await _consume_task(monitor_task)
            caller_cancelled = caller_cancelled or interrupted
            if error is not None:
                errors.append(error)
        if app_task is not None:
            interrupted, error = await _consume_task(app_task)
            caller_cancelled = caller_cancelled or interrupted
            if error is not None:
                errors.append(error)
        if not app_scope_entered:
            try:
                await captured.close()
            except asyncio.CancelledError:
                caller_cancelled = True
            except BaseException as error:
                errors.append(error)
        if owner_withdrawn and not closed:
            wire_coro = tracked_send({
                "type": "websocket.close",
                "code": 1012,
                "reason": "plugin generation changed",
            })
            try:
                close_task = asyncio.create_task(
                    wire_coro, name="dashboard-websocket-wire-close",
                )
            except asyncio.CancelledError:
                wire_coro.close()
                caller_cancelled = True
            except BaseException as error:
                wire_coro.close()
                errors.append(error)
            else:
                interrupted, error = await _consume_task(close_task)
                caller_cancelled = caller_cancelled or interrupted
                if error is not None:
                    errors.append(error)
    if errors:
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("Dashboard WebSocket 子任务失败", errors)
    if caller_cancelled:
        raise asyncio.CancelledError


async def _consume_task(task: asyncio.Task[object]) -> tuple[bool, BaseException | None]:
    """Join one child and retain caller cancellation plus child failure."""

    interrupted = False
    current = asyncio.current_task()
    if current is None:
        raise RuntimeError("Dashboard WebSocket cleanup 必须运行在 Task 中")
    cancellation_count = current.cancelling()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            next_count = current.cancelling()
            if next_count > cancellation_count:
                interrupted = True
                cancellation_count = next_count
            continue
        except BaseException:
            if not task.done():
                raise
    if task.cancelled():
        return interrupted, None
    return interrupted, task.exception()


def _matching_binding(
    bindings: tuple[DashboardBinding, ...], scope: dict[str, Any],
) -> DashboardBinding | None:
    for binding in bindings:
        if binding.matches(scope):
            return binding
    return None


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
    root: object,
    catalog: WebUiCatalog,
    identity: tuple[str, str, str, str],
) -> bool:
    snapshot_id, catalog_id, plugin_id, generation_id = identity
    return (
        root.generation_id == snapshot_id
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
        {"code": code}, status_code=status, headers=headers,
    )(scope, receive, send)


async def _reject_web_request(
    status: int,
    code: str,
    scope: dict[str, Any],
    receive: Any,
    send: Any,
) -> None:
    if scope.get("type") == "websocket":
        await send({
            "type": "websocket.close",
            "code": 4409 if code == "stale_catalog" else 4403,
            "reason": code,
        })
        return
    await _web_error(status, code, scope, receive, send)
