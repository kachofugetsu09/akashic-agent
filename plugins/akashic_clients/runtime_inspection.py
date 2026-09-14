"""普通 runtime_inspection 能力的客户端投影。

客户端只依赖这组请求级只读方法；如何从当前 generation 取得它们由宿主
绑定，不把 Core Snapshot、ComposablePlugin 或私有快照对象带进插件。
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import cast

from agent.plugin_composition.rpc import RpcMethod
from .capabilities import (
    INSPECTION_DOCUMENTS_GET,
    INSPECTION_DOCUMENTS_LIST,
    INSPECTION_JOBS_GET,
    INSPECTION_JOBS_LIST,
    INSPECTION_SKILLS_LIST,
    RUNTIME_CATALOG,
)
from .services import RuntimeInspectionError, RuntimeInspectionService


class ScopedRpcRuntimeInspection:
    """Resolve one inspection RPC inside the exact request scope per call."""

    def __init__(self, open_scope) -> None:
        self._open_scope = open_scope

    async def _invoke(
        self,
        scope,
        key: object,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Invoke one declared inspection RPC in the caller's exact scope."""

        method = scope.require(key)
        if not isinstance(method, RpcMethod):
            raise RuntimeInspectionError(
                "invalid_provider",
                "runtime inspection provider 类型无效",
            )
        params = method.params.model_validate(dict(payload))
        result = await method.invoke(params, None)
        if not isinstance(result, Mapping):
            raise RuntimeInspectionError("invalid_response", "runtime inspection RPC 返回值必须是对象")
        return cast(dict[str, object], dict(result))

    async def _call(self, key: object, payload: Mapping[str, object]) -> dict[str, object]:
        async with self._open_scope() as scope:
            result = await self._invoke(scope, key, payload)
        _raise_unavailable(result)
        return result

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
        """Restore the existing Mobile aggregate from one request generation."""

        async with self._open_scope() as scope:
            payload = dict(scope.require(RUNTIME_CATALOG)())
            _raise_unavailable(payload)
            skills = await self._invoke(scope, INSPECTION_SKILLS_LIST, {})
            _raise_unavailable(skills)
            items = skills.get("items")
            if not isinstance(items, list):
                raise RuntimeInspectionError(
                    "invalid_response",
                    "runtime inspection skills 响应缺少 items",
                )
            payload["skills"] = items
            expected = {"snapshot_id", "plugins", "skills", "mcp_servers"}
            if payload.keys() != expected:
                raise RuntimeInspectionError(
                    "invalid_response",
                    "runtime catalog 返回字段与客户端合同不一致",
                )
            return payload

    async def get_mcp(self, owner_id: str, server_name: str) -> dict[str, object]:
        """Render one MCP detail from the same neutral catalog capability."""

        async with self._open_scope() as scope:
            payload = scope.require(RUNTIME_CATALOG)()
            _raise_unavailable(payload)
        servers = payload.get("mcp_servers")
        if not isinstance(servers, list):
            raise RuntimeInspectionError("invalid_response", "runtime catalog 缺少 MCP 列表")
        server = next(
            (
                item
                for item in servers
                if isinstance(item, Mapping)
                and item.get("owner_id") == owner_id
                and item.get("name") == server_name
            ),
            None,
        )
        if server is None:
            raise RuntimeInspectionError(
                "mcp_not_found",
                f"MCP server 不存在: {owner_id}/{server_name}",
            )
        tools = server.get("tools")
        if not isinstance(tools, list):
            raise RuntimeInspectionError("invalid_response", "runtime catalog MCP tools 无效")
        return {
            "owner_id": owner_id,
            "name": server_name,
            "tool_count": len(tools),
            "tools": tools,
            "markdown": _mcp_markdown(owner_id, server_name, tools),
        }


def _raise_unavailable(payload: Mapping[str, object]) -> None:
    """Translate a neutral inspection failure at the client boundary."""

    unavailable = payload.get("unavailable")
    if unavailable is None:
        return
    if not isinstance(unavailable, Mapping):
        raise RuntimeInspectionError(
            "invalid_response",
            "runtime inspection unavailable 响应无效",
        )
    code = unavailable.get("code")
    message = unavailable.get("message")
    if not isinstance(code, str) or not isinstance(message, str):
        raise RuntimeInspectionError(
            "invalid_response",
            "runtime inspection unavailable 响应无效",
        )
    raise RuntimeInspectionError(code, message)


def _mcp_markdown(
    owner_id: str,
    server_name: str,
    tools: list[object],
) -> str:
    """Render the existing read-only MCP detail format."""

    lines = [f"# {server_name}", "", f"归属：`{owner_id}`", "", "## 工具", ""]
    for value in tools:
        if not isinstance(value, Mapping):
            raise RuntimeInspectionError("invalid_response", "runtime catalog MCP tool 无效")
        name = value.get("name")
        description = value.get("description")
        input_schema = value.get("input_schema")
        if not isinstance(name, str) or not isinstance(description, str):
            raise RuntimeInspectionError("invalid_response", "runtime catalog MCP tool 字段无效")
        lines.extend(
            (
                f"### `{name}`",
                "",
                description,
                "",
                "```json",
                json.dumps(input_schema, ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            )
        )
    return "\n".join(lines)


__all__ = [
    "ScopedRpcRuntimeInspection",
    "RuntimeInspectionError",
    "RuntimeInspectionService",
]
