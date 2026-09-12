"""Project core-owned runtime facts into the read-only mobile protocol."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from agent.plugin_composition import TopologyFiberView
from agent.plugin_composition.rpc import rpc_method_key
from agent.plugins.composable import ComposablePlugin
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotStore,
    lease_runtime_snapshot,
)

_MAX_RECENT_PLUGIN_INCIDENTS = 20


class RuntimeInspectionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RuntimeInspectionService:
    """将当前 generation 的插件 RPC 响应转换到既有客户端协议。"""

    def __init__(self, *, workspace: Path, snapshot_store: RuntimeSnapshotStore | None) -> None:
        self._snapshot_store = snapshot_store

    async def list_documents(self) -> dict[str, object]:
        async with self._snapshot() as snapshot:
            response = await _call(snapshot, "inspection/documents.list", {})
            if response is None:
                return {"items": [], "unavailable": [_unavailable("documents", "documents_unavailable")]}
            return _checked_response(response)

    async def get_document(self, document_id: str) -> dict[str, object]:
        return await self._required_call("inspection/documents.get", {"document_id": document_id},
                                         "documents_unavailable", "运行时文档检查服务尚未绑定")

    async def list_jobs(self) -> dict[str, object]:
        return await self._required_call("inspection/jobs.list", {},
                                         "scheduler_unavailable", "调度检查服务尚未绑定")

    async def get_job(self, job_id: str) -> dict[str, object]:
        return await self._required_call("inspection/jobs.get", {"job_id": job_id},
                                         "scheduler_unavailable", "调度检查服务尚未绑定")

    async def _required_call(self, method: str, params: dict[str, object],
                             code: str, message: str) -> dict[str, object]:
        """在同一个任务 lease 内解析、校验参数并完成插件调用。"""
        if self._snapshot_store is None or self._snapshot_store.current is None:
            raise RuntimeInspectionError(code, message)
        async with self._snapshot() as snapshot:
            response = await _call(snapshot, method, params)
            if response is None:
                raise RuntimeInspectionError(code, message)
            return _checked_response(response)

    async def list_capabilities(self) -> dict[str, object]:
        async with self._snapshot() as snapshot:
            response = await _call(snapshot, "inspection/skills.list", {})
            payload: dict[str, object] = {
                "snapshot_id": snapshot.snapshot_id,
                "plugins": _plugin_items(snapshot),
                "skills": [],
                "mcp_servers": _mcp_items(snapshot),
            }
            if response is None or "unavailable" in response:
                payload["unavailable"] = [_unavailable("skills", "skills_unavailable")]
            else:
                payload["skills"] = response["items"]
            return payload

    async def get_mcp(self, owner_id: str, server_name: str) -> dict[str, object]:
        async with self._snapshot() as snapshot:
            server = _find_mcp_item(snapshot, owner_id, server_name)
            if server is None:
                raise RuntimeInspectionError("mcp_not_found", f"MCP server 不存在: {owner_id}/{server_name}")
            tools = cast(list[dict[str, object]], server["tools"])
            return {"owner_id": owner_id, "name": server_name, "tool_count": len(tools),
                    "tools": tools, "markdown": _mcp_markdown(owner_id, server_name, tools)}

    @asynccontextmanager
    async def _snapshot(self) -> AsyncIterator[RuntimeSnapshot]:
        """引用计数和当前任务绑定共同覆盖完整读取。"""
        store = self._snapshot_store
        if store is None or store.current is None:
            raise RuntimeInspectionError("runtime_snapshot_unavailable", "运行时能力快照尚未就绪")
        async with lease_runtime_snapshot(store) as snapshot:
            yield snapshot


async def _call(snapshot: RuntimeSnapshot, method: str, params: dict[str, object]) -> dict[str, object] | None:
    """复用正式 RpcMethod 合同；插件拥有具体参数模型与业务行为。"""
    root = snapshot.composition_root
    if root is None:
        return None
    operation = root.context.get(rpc_method_key(method))
    if operation is None:
        return None
    result = await operation.invoke(operation.params.model_validate(params), None)
    if not isinstance(result, Mapping):
        raise TypeError(f"{method} response 必须是对象")
    return dict(result)


def _checked_response(response: dict[str, object]) -> dict[str, object]:
    unavailable = response.get("unavailable")
    if unavailable is not None:
        if not isinstance(unavailable, Mapping):
            raise TypeError("inspection unavailable 必须是对象")
        code, message = unavailable["code"], unavailable["message"]
        if not isinstance(code, str) or not isinstance(message, str):
            raise TypeError("inspection unavailable 的 code/message 必须是字符串")
        raise RuntimeInspectionError(code, message)
    return response


def _unavailable(kind: str, code: str) -> dict[str, str]:
    return {"kind": kind, "code": code, "status": "unavailable"}


def _plugin_items(snapshot: RuntimeSnapshot) -> list[dict[str, object]]:
    """Project generation and composition facts from one leased snapshot."""

    composition = _plugin_composition_items(snapshot)
    items: list[dict[str, object]] = []
    for generation in sorted(
        snapshot.active_generations(),
        key=lambda item: item.plugin_id,
    ):
        api_version = cast(ComposablePlugin, generation.instance).api_version
        current = composition.get(generation.plugin_id)
        if current is None:
            raise RuntimeError(
                f"stable v3 插件缺少 composition inspection: {generation.plugin_id}"
            )
        items.append(
            {
                "id": generation.plugin_id,
                "revision": generation.source_revision,
                "generation_id": generation.generation_id,
                "api_version": api_version,
                "composition": current,
            }
        )
    return items


def _plugin_composition_items(
    snapshot: RuntimeSnapshot,
) -> dict[str, dict[str, object]]:
    """Group current Root facts by the top-level plugin Fiber owner."""

    root = snapshot.composition_root
    topology = snapshot.composition_topology
    if root is None and topology is None:
        return {}
    if root is None or topology is None:
        raise RuntimeError(
            "stable snapshot 的 composition Root 与 Topology 必须成对存在"
        )

    # 1. Frozen parent edges assign every nested Fiber to one top-level plugin.
    all_v3_plugin_ids = set(snapshot.generations)
    active_v3_plugin_ids = {
        generation.plugin_id for generation in snapshot.active_generations()
    }
    all_owners = _top_level_plugin_owners(topology.fibers, all_v3_plugin_ids)
    receipt = root.receipt()
    current_fibers = {fiber.name: fiber for fiber in receipt.fibers}
    if current_fibers.keys() != all_owners.keys():
        raise RuntimeError("stable snapshot 的 current Fiber 与冻结 Topology 不一致")
    owner_by_fiber = {
        name: owner
        for name, owner in all_owners.items()
        if owner in active_v3_plugin_ids
    }
    incident_counts = dict(receipt.incident_counts)

    # 2. Current Health/Incident state is bounded; cumulative counts remain exact.
    result: dict[str, dict[str, object]] = {}
    for plugin_id in sorted(active_v3_plugin_ids):
        owned_names = {
            name for name, owner in owner_by_fiber.items() if owner == plugin_id
        }
        topology_fibers = tuple(
            fiber for fiber in topology.fibers if fiber.name in owned_names
        )
        health = tuple(item for item in receipt.health if item.owner in owned_names)
        recent_incidents = tuple(
            item for item in receipt.incidents if item.owner in owned_names
        )[-_MAX_RECENT_PLUGIN_INCIDENTS:]
        fibers = tuple(
            current_fibers[fiber.name]
            for fiber in topology_fibers
            if fiber.name in current_fibers
        )
        ready = all(
            not fiber.required_for_readiness or fiber.state.value == "active"
            for fiber in fibers
        ) and all(not item.required or item.healthy for item in health)
        result[plugin_id] = {
            "ready": ready,
            "topology_identity": topology.identity,
            "composition_revision": topology.composition_revision,
            "fibers": [
                {
                    "name": topology_fiber.name,
                    "parent": topology_fiber.parent,
                    "state": current_fibers[topology_fiber.name].state.value,
                    "required": topology_fiber.required_for_readiness,
                    "static_active": topology_fiber.static_active,
                    "dependencies": list(topology_fiber.dependencies),
                    "missing_services": list(
                        current_fibers[topology_fiber.name].missing_services
                    ),
                    "error": current_fibers[topology_fiber.name].error,
                }
                for topology_fiber in topology_fibers
            ],
            "health": [
                {
                    "owner": item.owner,
                    "name": item.name,
                    "required": item.required,
                    "healthy": item.healthy,
                    "reason": item.reason,
                }
                for item in health
            ],
            "incident_count": sum(incident_counts.get(name, 0) for name in owned_names),
            "recent_incidents": [
                {
                    "sequence": item.sequence,
                    "owner": item.owner,
                    "kind": item.kind,
                    "message": item.message,
                    "error_type": item.error_type,
                }
                for item in recent_incidents
            ],
            "incident_overflowed": receipt.incident_overflowed,
        }
    return result


def _top_level_plugin_owners(
    fibers: tuple[TopologyFiberView, ...],
    plugin_ids: set[str],
) -> dict[str, str]:
    """Resolve each frozen Fiber name to its top-level plugin Fiber."""

    parent_by_name = {fiber.name: fiber.parent for fiber in fibers}
    owners: dict[str, str] = {}
    for name in parent_by_name:
        current = name
        seen: set[str] = set()
        while parent_by_name[current] is not None:
            if current in seen:
                raise RuntimeError(f"composition parent edge 构成循环: {name}")
            seen.add(current)
            parent = parent_by_name[current]
            if not isinstance(parent, str) or parent not in parent_by_name:
                raise RuntimeError(f"composition parent edge 缺失: {name} -> {parent}")
            current = parent
        if current not in plugin_ids:
            raise RuntimeError(
                f"composition 顶层 Fiber 不属于 active v3 插件: {current}"
            )
        owners[name] = current
    return owners


def _mcp_items(snapshot: RuntimeSnapshot) -> list[dict[str, object]]:
    """Project exact v3 MCP servers from the stable Root registry."""

    # 1. v3 declarations provide owner identity; ToolRegistry provides live schemas.
    items: list[dict[str, object]] = []
    registry = snapshot.mcp_server_registry
    if registry is not None:
        for descriptor in registry.descriptors:
            tools = _mcp_tools_from_registry(snapshot, descriptor.name)
            items.append(
                {
                    "owner_id": descriptor.owner,
                    "name": descriptor.name,
                    "tool_count": len(tools),
                    "tools": tools,
                }
            )

    return sorted(items, key=lambda item: (str(item["owner_id"]), str(item["name"])))


def _mcp_tools_from_registry(
    snapshot: RuntimeSnapshot,
    server_name: str,
) -> list[dict[str, object]]:
    """Read one exact live MCP server projection from the frozen ToolRegistry."""

    registry = snapshot.tool_registry
    if registry is None:
        # 按需 MCP 尚无快照工具目录；查询失败不能中断同一连接上的聊天。
        raise RuntimeInspectionError(
            "mcp_catalog_unavailable",
            "MCP 工具目录暂不可用，声明的服务按需启动",
        )
    prefix = f"mcp_{server_name}__"
    tools: list[dict[str, object]] = []
    for name in registry.get_registered_order(
        registry.get_source_tool_names("mcp", server_name)
    ):
        tool = registry.get_tool(name)
        if tool is None:
            raise RuntimeError(f"stable MCP ToolRegistry 缺少已登记工具: {name}")
        remote_name = name.removeprefix(prefix)
        description = tool.description.removeprefix(f"[MCP:{server_name}] ")
        tools.append(
            {
                "name": remote_name,
                "description": description,
                "input_schema": tool.parameters or {},
            }
        )
    return tools


def _find_mcp_item(
    snapshot: RuntimeSnapshot,
    owner_id: str,
    server_name: str,
) -> dict[str, object] | None:
    return next(
        (
            item
            for item in _mcp_items(snapshot)
            if item["owner_id"] == owner_id and item["name"] == server_name
        ),
        None,
    )


def _mcp_markdown(
    owner_id: str,
    server_name: str,
    tools: list[dict[str, object]],
) -> str:
    lines = [
        f"# {server_name}",
        "",
        f"归属：`{owner_id}`",
        "",
        "## 工具",
        "",
    ]
    for tool in tools:
        lines.extend(
            (
                f"### `{tool['name']}`",
                "",
                str(tool["description"]),
                "",
                "```json",
                json.dumps(
                    tool["input_schema"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
            )
        )
    return "\n".join(lines)
