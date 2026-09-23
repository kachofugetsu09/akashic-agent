from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections.abc import Coroutine, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any, Protocol, cast

from agent.control.context import running_turn_id
from agent.plugin_composition import (
    CompositionError,
    CompositionRoot,
    Context,
    FiberState,
    MobileUiAsset,
    MobileUiBinding,
    MobileUiPluginUnavailable,
    MobileUiQueryOverloaded,
    MobileUiQueryTimeout,
    MobileUiRpcExecutionError,
    MobileUiRpcInvalidRequest,
    MobileUiStaleRevision,
    RuntimeScope,
    UI_SLOTS,
    UiSlots,
)
from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugins._operation import complete_critical
from core.error_context import current_session_key

MOBILE_UI_QUERY_TIMEOUT_SECONDS = 20.0
MOBILE_UI_QUERY_WORKERS = 8
MOBILE_UI_QUERY_QUEUE_LIMIT = 16
logger = logging.getLogger(__name__)


class MobileUiProvider(Protocol):
    async def catalog(self) -> dict[str, object]: ...

    async def asset(
        self,
        plugin_id: str,
        plugin_revision: str,
        kind: str,
        sha256: str,
    ) -> dict[str, object]: ...

    async def query(
        self,
        plugin_id: str,
        plugin_revision: str,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]: ...


class PluginMobileUiProvider:
    """Project live Mobile registrations from one CompositionRoot."""

    def __init__(self, root: CompositionRoot) -> None:
        self._root = root
        self._executor = ThreadPoolExecutor(
            max_workers=MOBILE_UI_QUERY_WORKERS,
            thread_name_prefix="mobile-plugin-ui",
        )
        self._draining_queries: set[asyncio.Task[dict[str, object]]] = set()
        self._admission_lock = asyncio.Lock()
        self._queries_idle = asyncio.Event()
        self._queries_idle.set()
        self._admitted_queries = 0
        self._accepting = True
        self._executor_closed = False

    async def aclose(self) -> None:
        """Close admission, settle physical queries, then shut down workers."""

        async with self._admission_lock:
            self._accepting = False
        cancelled = False
        _result, phase_cancelled = await complete_critical(self._wait_for_queries())
        cancelled |= phase_cancelled
        if not self._executor_closed:
            try:
                _result, phase_cancelled = await complete_critical(
                    asyncio.to_thread(
                        self._executor.shutdown,
                        wait=True,
                        cancel_futures=True,
                    )
                )
            except BaseException as error:
                if cancelled and not (
                    isinstance(error, BaseExceptionGroup)
                    and error.subgroup(asyncio.CancelledError) is not None
                ):
                    raise BaseExceptionGroup(
                        "插件 mobile UI 关闭被取消且 executor shutdown 失败",
                        [asyncio.CancelledError(), error],
                    ) from None
                raise
            cancelled |= phase_cancelled
            self._executor_closed = True
        if cancelled:
            raise asyncio.CancelledError

    async def catalog(self) -> dict[str, object]:
        """Read each current ACTIVE registration inside its exact Context scope."""

        items: list[dict[str, object]] = []
        ui_context, slots = self._ui_slots()
        ui_scope = ui_context.runtime_scope()
        try:
            await ui_scope.__aenter__()
        except CompositionError as error:
            if error.code in {"STALE_ACTIVATION", "OWNER_UNAVAILABLE"}:
                raise MobileUiPluginUnavailable("当前 Root 没有 Mobile UI provider") from error
            raise
        try:
            for binding in slots.bindings():
                if not self._binding_is_active(binding):
                    continue
                target_scope = binding.context.runtime_scope()
                try:
                    await target_scope.__aenter__()
                except CompositionError as error:
                    if error.code in {"STALE_ACTIVATION", "OWNER_UNAVAILABLE"}:
                        continue
                    raise
                try:
                    with plugin_entrypoint(
                        plugin_id=binding.descriptor.owner,
                        generation_id=binding.context.runtime.generation_id,
                        fiber=binding.context.fiber.name,
                        operation="mobile_ui.available",
                    ):
                        if not binding.available():
                            continue
                    items.append(self._catalog_item(binding))
                finally:
                    await target_scope.__aexit__(None, None, None)
        finally:
            await ui_scope.__aexit__(None, None, None)
        encoded = json.dumps(
            items,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        return {
            "catalog_revision": hashlib.sha256(encoded).hexdigest(),
            "items": items,
        }

    async def asset(
        self,
        plugin_id: str,
        plugin_revision: str,
        kind: str,
        sha256: str,
    ) -> dict[str, object]:
        """Read one fixed asset while holding its target registration scope."""

        binding = await self._select_binding(plugin_id, plugin_revision)
        scope = binding.context.runtime_scope()
        try:
            await scope.__aenter__()
        except CompositionError as error:
            if error.code in {"STALE_ACTIVATION", "OWNER_UNAVAILABLE"}:
                raise MobileUiPluginUnavailable(plugin_id) from error
            raise
        try:
            with plugin_entrypoint(
                plugin_id=binding.descriptor.owner,
                generation_id=binding.context.runtime.generation_id,
                fiber=binding.context.fiber.name,
                operation="mobile_ui.available",
            ):
                if not binding.available():
                    raise MobileUiPluginUnavailable(plugin_id)
            content, expected_sha256 = _asset_content(binding.asset, kind)
            if expected_sha256 != sha256:
                raise MobileUiStaleRevision(plugin_id)
            return {
                "plugin_id": plugin_id,
                "plugin_revision": plugin_revision,
                "kind": kind,
                "sha256": expected_sha256,
                "content": content,
            }
        finally:
            await scope.__aexit__(None, None, None)

    async def query(
        self,
        plugin_id: str,
        plugin_revision: str,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """Admit a fixed target, then drain its physical thread independently."""

        await self._reserve_query_slot()
        captured_scope: RuntimeScope | None = None
        task: asyncio.Task[dict[str, object]] | None = None
        try:
            binding, captured_scope = await self._capture_query_scope(
                plugin_id, plugin_revision,
            )
            coroutine = self._run_query(
                binding,
                method,
                payload,
                captured_scope=captured_scope,
                session_id=session_id,
                turn_id=turn_id,
            )
            task = await self._publish_query(coroutine)
            task.add_done_callback(
                lambda completed: self._query_done(completed, captured_scope)
            )
            try:
                async with asyncio.timeout(MOBILE_UI_QUERY_TIMEOUT_SECONDS):
                    return await asyncio.shield(task)
            except TimeoutError as error:
                raise MobileUiQueryTimeout(
                    f"插件 mobile UI query 超时: {plugin_id}.{method}"
                ) from error
            except asyncio.CancelledError:
                raise
        except BaseException:
            if task is None:
                if captured_scope is not None and not captured_scope._closed:  # pyright: ignore[reportPrivateUsage]
                    if captured_scope._entered_task is None:  # pyright: ignore[reportPrivateUsage]
                        captured_scope._close()  # pyright: ignore[reportPrivateUsage]
                await self._release_query_slot()
            raise

    async def _wait_for_queries(self) -> None:
        """Wait for reservations and physical child tasks to settle."""

        while True:
            async with self._admission_lock:
                tasks = tuple(self._draining_queries)
                if self._admitted_queries == 0 and not tasks:
                    return
            if tasks:
                _ = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                await self._queries_idle.wait()

    async def _reserve_query_slot(self) -> None:
        """Reserve bounded admission before selecting or capturing a target."""

        async with self._admission_lock:
            if not self._accepting:
                raise MobileUiPluginUnavailable("mobile UI provider 已关闭")
            limit = MOBILE_UI_QUERY_WORKERS + MOBILE_UI_QUERY_QUEUE_LIMIT
            if self._admitted_queries >= limit:
                raise MobileUiQueryOverloaded("插件 mobile UI query 队列已满")
            self._admitted_queries += 1
            self._queries_idle.clear()

    async def _release_query_slot(self) -> None:
        async with self._admission_lock:
            self._admitted_queries -= 1
            if self._admitted_queries < 0:
                raise RuntimeError("插件 mobile UI query admission 计数失衡")
            if self._admitted_queries == 0:
                self._queries_idle.set()

    async def _publish_query(
        self,
        coroutine: Coroutine[Any, Any, dict[str, object]],
    ) -> asyncio.Task[dict[str, object]]:
        """Publish a child only while the provider admission lock is held."""

        async with self._admission_lock:
            if not self._accepting:
                coroutine.close()
                raise MobileUiPluginUnavailable("mobile UI provider 正在关闭")
            try:
                task = asyncio.create_task(coroutine)
            except BaseException:
                coroutine.close()
                raise
            self._draining_queries.add(task)
            return task

    def _query_done(
        self,
        completed: asyncio.Task[dict[str, object]],
        captured_scope: RuntimeScope,
    ) -> None:
        """Settle one child, including a child cancelled before its first line."""

        self._draining_queries.discard(completed)
        if not captured_scope._closed and captured_scope._entered_task is None:  # pyright: ignore[reportPrivateUsage]
            captured_scope._close()  # pyright: ignore[reportPrivateUsage]
        self._admitted_queries -= 1
        if self._admitted_queries < 0:
            raise RuntimeError("插件 mobile UI query admission 计数失衡")
        if self._admitted_queries == 0:
            self._queries_idle.set()
        if not completed.cancelled():
            _ = completed.exception()

    async def _capture_query_scope(
        self,
        plugin_id: str,
        plugin_revision: str,
    ) -> tuple[MobileUiBinding, RuntimeScope]:
        """Freeze one binding and capture its permit before creating a child."""

        ui_context, slots = self._ui_slots()
        ui_scope = ui_context.runtime_scope()
        try:
            await ui_scope.__aenter__()
        except CompositionError as error:
            if error.code in {"STALE_ACTIVATION", "OWNER_UNAVAILABLE"}:
                raise MobileUiPluginUnavailable("当前 Root 没有 Mobile UI provider") from error
            raise
        try:
            binding = self._find_binding(slots, plugin_id)
            self._check_revision(binding, plugin_revision)
            target_scope = binding.context.runtime_scope()
            try:
                await target_scope.__aenter__()
            except CompositionError as error:
                if error.code in {"STALE_ACTIVATION", "OWNER_UNAVAILABLE"}:
                    raise MobileUiPluginUnavailable(plugin_id) from error
                raise
            try:
                with plugin_entrypoint(
                    plugin_id=binding.descriptor.owner,
                    generation_id=binding.context.runtime.generation_id,
                    fiber=binding.context.fiber.name,
                    operation="mobile_ui.available",
                ):
                    if not binding.available():
                        raise MobileUiPluginUnavailable(plugin_id)
                    captured = binding.context.capture_runtime_scope()
                return binding, captured
            finally:
                await target_scope.__aexit__(None, None, None)
        finally:
            await ui_scope.__aexit__(None, None, None)

    async def _select_binding(
        self,
        plugin_id: str,
        plugin_revision: str,
    ) -> MobileUiBinding:
        """Select one active binding without consulting a later generation."""

        ui_context, slots = self._ui_slots()
        ui_scope = ui_context.runtime_scope()
        try:
            await ui_scope.__aenter__()
        except CompositionError as error:
            if error.code in {"STALE_ACTIVATION", "OWNER_UNAVAILABLE"}:
                raise MobileUiPluginUnavailable("当前 Root 没有 Mobile UI provider") from error
            raise
        try:
            binding = self._find_binding(slots, plugin_id)
            self._check_revision(binding, plugin_revision)
            return binding
        finally:
            await ui_scope.__aexit__(None, None, None)

    def _ui_slots(self) -> tuple[Context, UiSlots]:
        """Resolve the current UI provider from this Root, not from a snapshot."""

        try:
            context, slots = self._root._service_provider(UI_SLOTS)  # pyright: ignore[reportPrivateUsage]
        except (CompositionError, RuntimeError) as error:
            raise MobileUiPluginUnavailable("当前 Root 没有 Mobile UI provider") from error
        if (
            context.root_instance_token is not self._root.instance_token
            or slots.root_instance_token is not self._root.instance_token
        ):
            raise MobileUiPluginUnavailable("Mobile UI provider 不属于当前 Root")
        return context, slots

    @staticmethod
    def _find_binding(slots: UiSlots, plugin_id: str) -> MobileUiBinding:
        for binding in slots.bindings():
            if binding.descriptor.owner == plugin_id:
                return binding
        raise MobileUiPluginUnavailable(plugin_id)

    def _binding_is_active(self, binding: MobileUiBinding) -> bool:
        if binding.context.root_instance_token is not self._root.instance_token:
            raise MobileUiPluginUnavailable("Mobile UI registration 不属于当前 Root")
        return binding.context.fiber.state is FiberState.ACTIVE

    def _check_revision(self, binding: MobileUiBinding, revision: str) -> None:
        if self._plugin_revision(binding) != revision:
            raise MobileUiStaleRevision(binding.descriptor.owner)

    def _plugin_revision(self, binding: MobileUiBinding) -> str:
        encoded = json.dumps(
            (
                "mobile-ui",
                self._root.generation_id,
                binding.descriptor.owner,
                binding.registration_uuid,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _catalog_item(self, binding: MobileUiBinding) -> dict[str, object]:
        asset = binding.asset
        navigation: dict[str, object] | None = None
        if asset.navigation_label is not None:
            navigation = {
                "label": asset.navigation_label,
                "description": asset.navigation_description,
            }
        return {
            "id": binding.descriptor.owner,
            "revision": self._plugin_revision(binding),
            "module_sha256": asset.module_sha256,
            "module_bytes": asset.module_bytes,
            "stylesheet_sha256": asset.stylesheet_sha256,
            "stylesheet_bytes": asset.stylesheet_bytes,
            "navigation": navigation,
            "slots": list(asset.slots),
        }

    async def _run_query(
        self,
        binding: MobileUiBinding,
        method: str,
        payload: dict[str, object],
        *,
        captured_scope: RuntimeScope,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """Run one fixed handler inside its captured scope and physical worker."""

        async with captured_scope:
            plugin_id = binding.descriptor.owner
            session_token = current_session_key.set(session_id)
            turn_token = running_turn_id.set(turn_id or "")
            failure = "执行失败"
            try:
                loop = asyncio.get_running_loop()
                with plugin_entrypoint(
                    plugin_id=plugin_id,
                    generation_id=binding.context.runtime.generation_id,
                    fiber=binding.context.fiber.name,
                    operation="mobile_ui.query",
                ):
                    diagnostic_context = copy_context()
                    future = loop.run_in_executor(
                        self._executor,
                        lambda: diagnostic_context.run(
                            binding.query,
                            method,
                            payload,
                            session_id=session_id,
                            turn_id=turn_id,
                        ),
                    )
                    result, cancelled = await complete_critical(future)
                    if cancelled:
                        raise asyncio.CancelledError
                    failure = "返回无效"
                    normalized = _normalize_rpc_result(
                        result,
                        plugin_id=plugin_id,
                        method=method,
                    )
            except MobileUiRpcInvalidRequest:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception(
                    "插件 mobile UI query %s: %s.%s",
                    failure,
                    plugin_id,
                    method,
                )
                raise MobileUiRpcExecutionError(
                    f"插件 mobile UI query {failure}: {plugin_id}.{method}"
                ) from error
            finally:
                running_turn_id.reset(turn_token)
                current_session_key.reset(session_token)
            return normalized

def _asset_content(asset: MobileUiAsset, kind: str) -> tuple[str, str]:
    if kind == "module":
        return asset.module, asset.module_sha256
    if kind == "stylesheet" and asset.stylesheet_sha256 is not None:
        return asset.stylesheet, asset.stylesheet_sha256
    raise MobileUiPluginUnavailable(f"mobile UI asset 不存在: {kind}")


def _normalize_rpc_result(
    result: object,
    *,
    plugin_id: str,
    method: str,
) -> dict[str, object]:
    """Validate and normalize one strict JSON RPC result."""

    if not isinstance(result, Mapping):
        raise TypeError(f"插件 mobile UI RPC 必须返回对象: {plugin_id}.{method}")
    mapping = cast(Mapping[object, object], result)
    normalized: dict[str, object] = {}
    active_containers: set[int] = set()
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(
                f"插件 mobile UI RPC 返回键必须是字符串: {plugin_id}.{method}"
            )
        _validate_json_value(
            value,
            plugin_id=plugin_id,
            method=method,
            active_containers=active_containers,
        )
        normalized[key] = value
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > 192 * 1024:
        raise ValueError(f"插件 mobile UI RPC 返回超过 192 KiB: {plugin_id}.{method}")
    return normalized


def _validate_json_value(
    value: object,
    *,
    plugin_id: str,
    method: str,
    active_containers: set[int],
) -> None:
    """Reject values outside the finite, recursively JSON-compatible ABI."""

    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise TypeError(
            f"插件 mobile UI RPC 返回浮点数必须有限: {plugin_id}.{method}"
        )
    if not isinstance(value, (list, dict)):
        raise TypeError(
            f"插件 mobile UI RPC 返回值不是严格 JSON 类型: {plugin_id}.{method}"
        )
    container_id = id(value)
    if container_id in active_containers:
        raise TypeError(f"插件 mobile UI RPC 返回值存在循环: {plugin_id}.{method}")
    active_containers.add(container_id)
    try:
        if isinstance(value, list):
            for item in value:
                _validate_json_value(
                    item,
                    plugin_id=plugin_id,
                    method=method,
                    active_containers=active_containers,
                )
            return
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "插件 mobile UI RPC 嵌套对象键必须是字符串: "
                    f"{plugin_id}.{method}"
                )
            _validate_json_value(
                item,
                plugin_id=plugin_id,
                method=method,
                active_containers=active_containers,
            )
    finally:
        active_containers.remove(container_id)
