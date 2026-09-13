"""普通 runtime_inspection 能力的客户端投影。

客户端只依赖这组请求级只读方法；如何从当前 generation 取得它们由宿主
绑定，不把 Core Snapshot、ComposablePlugin 或私有快照对象带进插件。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from agent.plugin_composition.rpc import RpcMethod

from .capabilities import (
    INSPECTION_DOCUMENTS_GET,
    INSPECTION_DOCUMENTS_LIST,
    INSPECTION_JOBS_GET,
    INSPECTION_JOBS_LIST,
    INSPECTION_SKILLS_LIST,
)
from .services import RuntimeInspectionError, RuntimeInspectionService


class ScopedRpcRuntimeInspection:
    """Resolve one inspection RPC inside the exact request scope per call."""

    def __init__(self, open_scope) -> None:
        self._open_scope = open_scope

    async def _call(self, key: object, payload: Mapping[str, object]) -> dict[str, object]:
        async with self._open_scope() as scope:
            method = scope.require(key)
            if not isinstance(method, RpcMethod):
                raise RuntimeInspectionError(
                    "invalid_provider",
                    "runtime inspection provider 类型无效",
                )
            params = method.params.model_validate(dict(payload))
            result = await method.invoke(params, None)
        if not isinstance(result, Mapping):
            raise RuntimeInspectionError(
                "invalid_response",
                "runtime inspection RPC 返回值必须是对象",
            )
        return cast(dict[str, object], dict(result))

    async def list_documents(self) -> dict[str, object]:
        return await self._call(INSPECTION_DOCUMENTS_LIST, {})

    async def get_document(self, document_id: str) -> dict[str, object]:
        return await self._call(
            INSPECTION_DOCUMENTS_GET,
            {"document_id": document_id},
        )

    async def list_jobs(self) -> dict[str, object]:
        return await self._call(INSPECTION_JOBS_LIST, {})

    async def get_job(self, job_id: str) -> dict[str, object]:
        return await self._call(INSPECTION_JOBS_GET, {"job_id": job_id})

    async def list_capabilities(self) -> dict[str, object]:
        return await self._call(INSPECTION_SKILLS_LIST, {})

    async def get_mcp(self, owner_id: str, server_name: str) -> dict[str, object]:
        raise RuntimeInspectionError(
            "mcp_unavailable",
            "当前 runtime inspection RPC 未声明 MCP 读取能力",
        )


__all__ = [
    "ScopedRpcRuntimeInspection",
    "RuntimeInspectionError",
    "RuntimeInspectionService",
]
