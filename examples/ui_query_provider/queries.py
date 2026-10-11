"""独立的串行 UI 查询实现；示例不提供默认实现的并行配额与超时策略。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
import hashlib
import inspect
import json
from typing import cast

from agent.plugin_composition import Context, FiberState
from core.common.file_io import run_file_io
from plugins.ui.contract import (
    PluginUiBinding, PluginUiPluginUnavailable, PluginUiStaleRevision, UiSlots,
)


class SerialPluginUiProvider:
    """只使用公开注册合同；每次查询在贡献方许可中串行执行并排空。"""

    def __init__(self, ctx: Context, slots: UiSlots):
        self._ctx, self._slots = ctx, slots
        self._serial = asyncio.Lock()

    def _revision(self, binding: PluginUiBinding) -> str:
        return hashlib.sha256((self._ctx.generation_id + ":" + binding.registration_uuid).encode()).hexdigest()

    @asynccontextmanager
    async def _select(self, plugin_id: str, revision: str) -> AsyncIterator[PluginUiBinding]:
        """选择当前登记，整段调用持有实际 provider 与贡献方许可。"""
        async with self._ctx.runtime_scope():
            binding = next((item for item in self._slots.bindings() if item.descriptor.owner == plugin_id), None)
            if binding is None or binding.context.fiber.state is not FiberState.ACTIVE:
                raise PluginUiPluginUnavailable(plugin_id)
            async with binding.context.runtime_scope():
                if not binding.available():
                    raise PluginUiPluginUnavailable(plugin_id)
                if self._revision(binding) != revision:
                    raise PluginUiStaleRevision(plugin_id)
                yield binding

    async def catalog(self) -> dict[str, object]:
        """发布当前可用登记的资产摘要，不保留实现对象到调用之外。"""
        items = []
        async with self._ctx.runtime_scope():
            for binding in self._slots.bindings():
                if binding.context.fiber.state is not FiberState.ACTIVE:
                    continue
                async with binding.context.runtime_scope():
                    if not binding.available():
                        continue
                    asset = binding.asset
                    items.append({
                        "id": binding.descriptor.owner, "revision": self._revision(binding),
                        "module_sha256": asset.module_sha256, "module_bytes": asset.module_bytes,
                        "stylesheet_sha256": asset.stylesheet_sha256, "stylesheet_bytes": asset.stylesheet_bytes,
                        "navigation": None if asset.navigation_label is None else {
                            "label": asset.navigation_label, "description": asset.navigation_description},
                        "slots": list(asset.slots),
                    })
        return {"items": items, "catalog_revision": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()}

    async def asset(self, plugin_id: str, plugin_revision: str, kind: str, sha256: str) -> dict[str, object]:
        async with self._select(plugin_id, plugin_revision) as binding:
            asset = binding.asset
            if kind == "module":
                content, digest = asset.module, asset.module_sha256
            elif kind == "stylesheet" and asset.stylesheet_sha256 is not None:
                content, digest = asset.stylesheet, asset.stylesheet_sha256
            else:
                raise ValueError("未知 UI 资产类型")
            if digest != sha256:
                raise PluginUiStaleRevision(plugin_id)
            return {"plugin_id": plugin_id, "plugin_revision": plugin_revision,
                    "kind": kind, "sha256": digest, "content": content}

    async def query(self, plugin_id: str, plugin_revision: str, method: str, payload: dict[str, object],
                    *, session_id: str | None, turn_id: str | None) -> dict[str, object]:
        """同步 handler 的物理读取结束后才归还许可；异步 handler 在当前 scope 执行。"""
        async with self._select(plugin_id, plugin_revision) as binding, self._serial:
            def call():
                return binding.query(method, payload, session_id=session_id, turn_id=turn_id)
            if inspect.iscoroutinefunction(binding.query):
                result = await cast(Awaitable[object], call())
            else:
                result = await run_file_io(call)
            if not isinstance(result, dict):
                raise TypeError("UI 查询必须返回 JSON object")
            return json.loads(json.dumps(result, allow_nan=False))
