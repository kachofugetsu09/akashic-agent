"""Expose a narrow read-only catalog for the current runtime scope."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from agent.plugin_composition.model import ServiceKey, TopologyFiberView

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshot


RuntimeCatalogReader = Callable[[], dict[str, object]]
RUNTIME_CATALOG = ServiceKey[RuntimeCatalogReader]("core.runtime_catalog.v1")


class RuntimeCatalogUnavailable(RuntimeError):
    """Report a catalog section that cannot be projected yet."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def build_runtime_catalog(snapshot: RuntimeSnapshot) -> dict[str, object]:
    """Project neutral plugin and MCP facts from one leased snapshot."""

    try:
        mcp_servers = _mcp_items(snapshot)
    except RuntimeCatalogUnavailable as error:
        return {
            "unavailable": {
                "code": error.code,
                "message": str(error),
            }
        }
    return {
        "snapshot_id": snapshot.snapshot_id,
        "plugins": _plugin_items(snapshot),
        "mcp_servers": mcp_servers,
    }


def _plugin_items(snapshot: RuntimeSnapshot) -> list[dict[str, object]]:
    """Project generation and composition facts from one leased snapshot."""

    from agent.plugins.composable import ComposablePlugin

    composition = _plugin_composition_items(snapshot)
    items: list[dict[str, object]] = []
    for generation in sorted(snapshot.active_generations(), key=lambda item: item.plugin_id):
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
                "api_version": cast(ComposablePlugin, generation.instance).api_version,
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
        raise RuntimeError("stable snapshot 的 composition Root 与 Topology 必须成对存在")

    # 1. Frozen parent edges assign every nested Fiber to one top-level plugin.
    all_plugin_ids = set(snapshot.generations)
    active_plugin_ids = {
        generation.plugin_id for generation in snapshot.active_generations()
    }
    all_owners = _top_level_plugin_owners(topology.fibers, all_plugin_ids)
    receipt = root.receipt()
    current_fibers = {fiber.name: fiber for fiber in receipt.fibers}
    if current_fibers.keys() != all_owners.keys():
        raise RuntimeError("stable snapshot 的 current Fiber 与冻结 Topology 不一致")
    owner_by_fiber = {
        name: owner
        for name, owner in all_owners.items()
        if owner in active_plugin_ids
    }
    incident_counts = dict(receipt.incident_counts)

    # 2. Current health details stay bounded while cumulative counts remain exact.
    result: dict[str, dict[str, object]] = {}
    for plugin_id in sorted(active_plugin_ids):
        owned_names = {
            name for name, owner in owner_by_fiber.items() if owner == plugin_id
        }
        topology_fibers = tuple(
            fiber for fiber in topology.fibers if fiber.name in owned_names
        )
        health = tuple(item for item in receipt.health if item.owner in owned_names)
        recent_incidents = tuple(
            item for item in receipt.incidents if item.owner in owned_names
        )[-20:]
        fibers = tuple(current_fibers[fiber.name] for fiber in topology_fibers)
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
            "incident_count": sum(
                incident_counts.get(name, 0) for name in owned_names
            ),
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
            raise RuntimeError(f"composition 顶层 Fiber 不属于 active v3 插件: {current}")
        owners[name] = current
    return owners


def _mcp_items(snapshot: RuntimeSnapshot) -> list[dict[str, object]]:
    """声明不能冒充已取得的工具目录；读取能力仍待 MCP owner 提供。"""
    from agent.plugin_composition.mcp_slots import MCP_SERVERS
    root = snapshot.composition_root
    service = None if root is None else root.context.get(MCP_SERVERS)
    if service is None:
        return []
    if service.root_instance_token is not root.instance_token:
        raise RuntimeError("MCP provider 不属于所选 Root")
    return service.catalog()



__all__ = [
    "RUNTIME_CATALOG",
    "RuntimeCatalogReader",
    "RuntimeCatalogUnavailable",
    "build_runtime_catalog",
]
