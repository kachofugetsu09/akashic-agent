from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Protocol, cast

from agent.plugins.base import Plugin
from agent.plugins.generation import PluginGeneration
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import RuntimeSnapshot

MOBILE_UI_RPC_TIMEOUT_SECONDS = 20.0
logger = logging.getLogger(__name__)


class MobileUiProvider(Protocol):
    def catalog(self) -> list[dict[str, object]]: ...

    def asset(self, plugin_id: str) -> dict[str, object]: ...

    async def call(
        self,
        plugin_id: str,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]: ...


class PluginMobileUiProvider:
    """从当前插件快照提供移动 UI 资产与 RPC。"""

    def __init__(self, manager: PluginManager) -> None:
        self._manager = manager

    def catalog(self) -> list[dict[str, object]]:
        """读取当前活动插件的版本化移动 UI 目录。"""

        snapshot = self._manager.current_snapshot
        return [] if snapshot is None else self._catalog(snapshot)

    def asset(self, plugin_id: str) -> dict[str, object]:
        """读取一个当前活动插件的完整移动 UI 资产。"""

        generation = self._active_generation(self._require_snapshot(), plugin_id)
        contribution = generation.contributions
        asset = contribution.mobile_ui_asset
        if asset is None:
            raise MobileUiPluginUnavailable(plugin_id)
        return {
            "id": plugin_id,
            "revision": generation.source_revision,
            "sha256": asset.sha256,
            "module": asset.module,
            "stylesheet": asset.stylesheet,
        }

    async def call(
        self,
        plugin_id: str,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """把受控请求路由到当前活动插件实例。"""

        async with await self._manager.snapshot_store.acquire() as snapshot:
            generation = self._active_generation(snapshot, plugin_id)
            if generation.contributions.mobile_ui_asset is None:
                raise MobileUiPluginUnavailable(plugin_id)
            plugin = cast(Plugin, generation.instance)
            try:
                async with asyncio.timeout(MOBILE_UI_RPC_TIMEOUT_SECONDS):
                    result = await plugin.mobile_ui_call(
                        method,
                        payload,
                        session_id=session_id,
                        turn_id=turn_id,
                    )
            except TimeoutError as error:
                raise MobileUiRpcTimeout(f"插件 mobile UI RPC 超时: {plugin_id}.{method}") from error
            except MobileUiRpcInvalidRequest:
                raise
            except Exception as error:
                logger.exception("插件 mobile UI RPC 执行失败: %s.%s", plugin_id, method)
                raise MobileUiRpcExecutionError(
                    f"插件 mobile UI RPC 执行失败: {plugin_id}.{method}"
                ) from error
            try:
                normalized = _normalize_rpc_result(result, plugin_id=plugin_id, method=method)
            except Exception as error:
                logger.exception("插件 mobile UI RPC 返回无效: %s.%s", plugin_id, method)
                raise MobileUiRpcExecutionError(
                    f"插件 mobile UI RPC 返回无效: {plugin_id}.{method}"
                ) from error
            return normalized

    def _require_snapshot(self) -> RuntimeSnapshot:
        snapshot = self._manager.current_snapshot
        if snapshot is None:
            raise MobileUiPluginUnavailable("runtime snapshot unavailable")
        return snapshot

    @staticmethod
    def _active_generation(
        snapshot: RuntimeSnapshot,
        plugin_id: str,
    ) -> PluginGeneration:
        generation = snapshot.generations.get(plugin_id)
        if generation is None or generation not in snapshot.active_generations():
            raise MobileUiPluginUnavailable(plugin_id)
        return generation

    @staticmethod
    def _catalog(snapshot: RuntimeSnapshot) -> list[dict[str, object]]:
        return [
            {"id": generation.plugin_id, "revision": generation.source_revision, "sha256": asset.sha256}
            for generation in snapshot.active_generations()
            if (asset := generation.contributions.mobile_ui_asset) is not None
        ]


class MobileUiPluginUnavailable(LookupError):
    pass


class MobileUiRpcTimeout(TimeoutError):
    pass


class MobileUiRpcInvalidRequest(ValueError):
    pass


class MobileUiRpcExecutionError(RuntimeError):
    pass


def _normalize_rpc_result(
    result: object,
    *,
    plugin_id: str,
    method: str,
) -> dict[str, object]:
    """校验并规范化插件 RPC 返回对象。"""

    # 1. 校验返回结构和 JSON 可编码性
    if not isinstance(result, Mapping):
        raise TypeError(f"插件 mobile UI RPC 必须返回对象: {plugin_id}.{method}")
    mapping = cast(Mapping[object, object], result)
    if any(not isinstance(key, str) for key in mapping):
        raise TypeError(f"插件 mobile UI RPC 返回键必须是字符串: {plugin_id}.{method}")
    normalized = {cast(str, key): value for key, value in mapping.items()}
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )

    # 2. 限制移动端单次响应体积
    if len(encoded.encode("utf-8")) > 192 * 1024:
        raise ValueError(f"插件 mobile UI RPC 返回超过 192 KiB: {plugin_id}.{method}")
    return normalized
