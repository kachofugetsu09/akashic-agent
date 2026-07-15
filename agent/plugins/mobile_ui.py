from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Protocol, cast

from agent.plugins.base import Plugin
from agent.plugins.generation import PluginGeneration
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import RuntimeSnapshot

MOBILE_UI_RPC_TIMEOUT_SECONDS = 20.0


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
            if not isinstance(result, Mapping):
                raise RuntimeError(f"插件 mobile UI RPC 必须返回对象: {plugin_id}.{method}")
            normalized = cast(dict[str, object], dict(result))
            encoded = json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            if len(encoded.encode("utf-8")) > 192 * 1024:
                raise RuntimeError(f"插件 mobile UI RPC 返回超过 192 KiB: {plugin_id}.{method}")
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
