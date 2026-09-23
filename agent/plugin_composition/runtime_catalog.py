"""Expose a narrow, read-only projection of the live composition Root."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from agent.plugin_composition.model import HealthView, IncidentView, ServiceKey

if TYPE_CHECKING:
    from agent.plugin_composition.context import CompositionRoot, Context, Fiber
    from agent.plugins.manager import PluginGeneration


RuntimeCatalogReader = Callable[["Context"], dict[str, object]]
RUNTIME_CATALOG = ServiceKey[RuntimeCatalogReader]("core.runtime_catalog.v1")


class RuntimeCatalogUnavailable(RuntimeError):
    """Report a catalog section that cannot be projected yet."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def build_runtime_catalog(
    root: CompositionRoot,
    active_generations: Mapping[str, PluginGeneration],
    draining_generations: Mapping[str, Sequence[PluginGeneration]] | None = None,
) -> dict[str, object]:
    """Read current Fibers, health, incidents, and MCP state from one live Root."""

    revision = root._composition_revision  # pyright: ignore[reportPrivateUsage]
    draining = {} if draining_generations is None else draining_generations
    catalog: dict[str, object] = {
        # Keep the wire name for existing clients. It is now a display identity,
        # not a snapshot lease or a frozen publication token.
        "snapshot_id": f"{root.generation_id}:{revision}",
        "plugins": _plugin_items(root, active_generations, draining, revision),
    }
    try:
        catalog["mcp_servers"] = _mcp_items(root)
    except RuntimeCatalogUnavailable as error:
        catalog["mcp_unavailable"] = {
            "code": error.code,
            "message": str(error),
        }
    return catalog


def _plugin_items(
    root: CompositionRoot,
    active_generations: Mapping[str, PluginGeneration],
    draining_generations: Mapping[str, Sequence[PluginGeneration]],
    composition_revision: int,
) -> list[dict[str, object]]:
    """Project each active generation from current registered Fiber objects."""

    receipt = root.receipt()
    fibers = tuple(
        fiber
        for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
        if fiber.runtime is not None
    )
    current_by_generation: dict[tuple[str, str], list[Fiber]] = {}
    for fiber in fibers:
        assert fiber.runtime is not None
        key = (fiber.runtime.plugin_id, fiber.runtime.generation_id)
        current_by_generation.setdefault(key, []).append(fiber)
    generation_by_fiber_name = {
        fiber.name: (fiber.runtime.plugin_id, fiber.runtime.generation_id)
        for fiber in fibers
        if fiber.runtime is not None
    }
    health_by_generation: dict[tuple[str, str], list[HealthView]] = {}
    for item in receipt.health:
        generation_key = generation_by_fiber_name.get(item.owner)
        if generation_key is not None:
            health_by_generation.setdefault(generation_key, []).append(item)
    generation_by_fiber_id = {
        fiber.fiber_id: (fiber.runtime.plugin_id, fiber.runtime.generation_id)
        for fiber in fibers
        if fiber.runtime is not None
    }
    incident_by_generation: dict[tuple[str, str], list[IncidentView]] = {}
    for item in receipt.incidents:
        generation_key = generation_by_fiber_id.get(item.fiber_id)
        if generation_key is not None:
            incident_by_generation.setdefault(generation_key, []).append(item)
    incident_counts = root._incident_counts  # pyright: ignore[reportPrivateUsage]

    items: list[dict[str, object]] = []
    for plugin_id, generation in sorted(active_generations.items()):
        generation_key = (plugin_id, generation.generation_id)
        plugin_fibers = tuple(
            sorted(current_by_generation.get(generation_key, ()), key=lambda fiber: fiber.name)
        )
        # An active generation without a registered Fiber is not ready. In
        # particular, all(empty) must not turn LOADING/UNLOADING into ready.
        ready = bool(plugin_fibers) and generation.state == "active" and all(
            fiber.state.value == "active" or not fiber.required_for_readiness
            for fiber in plugin_fibers
        )
        health = health_by_generation.get(generation_key, [])
        ready = ready and all(
            not item.required or item.healthy
            for item in health
        )
        manifest = generation.static_manifest
        if manifest is None:
            raise RuntimeError(
                f"generation 缺少已验证 static manifest: {generation.plugin_id}"
            )
        cleanup_pending = any(
            item is generation
            for item in draining_generations.get(plugin_id, ())
        )
        owned_fiber_ids = {fiber.fiber_id for fiber in plugin_fibers}
        incidents = incident_by_generation.get(generation_key, [])
        items.append(
            {
                "id": plugin_id,
                "revision": generation.source_revision,
                "generation_id": generation.generation_id,
                "archive_ref": generation.archive_ref,
                "state": generation.state,
                "api_version": manifest.api_version,
                "load_error": (
                    None
                    if generation.load_error is None
                    else str(generation.load_error)
                    or type(generation.load_error).__name__
                ),
                "cleanup_pending": cleanup_pending,
                "composition": {
                    "ready": ready,
                    "composition_revision": composition_revision,
                    "fibers": [_fiber_item(root, fiber) for fiber in plugin_fibers],
                    "health": [_health_item(item) for item in health],
                    "incident_count": sum(
                        count
                        for (fiber_id, _owner), count in incident_counts.items()
                        if fiber_id in owned_fiber_ids
                    ),
                    "recent_incidents": [_incident_item(item) for item in incidents[-20:]],
                    "incident_overflowed": receipt.incident_overflowed,
                },
            }
        )
    return items


def _fiber_item(root: CompositionRoot, fiber: Fiber) -> dict[str, object]:
    """Serialize one current Fiber without consulting a frozen topology."""

    parent = fiber.parent
    return {
        "name": fiber.name,
        "fiber_id": fiber.fiber_id,
        "parent": None if parent is root.root_fiber else parent.name,
        "state": fiber.state.value,
        "required": fiber.required_for_readiness,
        "dependencies": [key.name for key in fiber.dependencies],
        "missing_services": list(fiber.missing_services),
        "error": None if fiber.error is None else str(fiber.error),
    }


def _health_item(item: HealthView) -> dict[str, object]:
    """Serialize one current health entry."""

    return {
        "owner": item.owner,
        "name": item.name,
        "required": item.required,
        "healthy": item.healthy,
        "reason": item.reason,
    }


def _incident_item(item: IncidentView) -> dict[str, object]:
    """Serialize one bounded recent incident."""

    return {
        "sequence": item.sequence,
        "fiber_id": item.fiber_id,
        "owner": item.owner,
        "kind": item.kind,
        "message": item.message,
        "error_type": item.error_type,
    }


def _mcp_items(root: CompositionRoot) -> list[dict[str, object]]:
    """Read the MCP owner or return an explicit unavailable section."""

    from agent.plugin_composition.mcp_slots import MCP_SERVERS

    service = root.context.get(MCP_SERVERS)
    if service is None:
        raise RuntimeCatalogUnavailable(
            "mcp_provider_unavailable", "MCP provider 尚未在当前 Root 提供"
        )
    if service.root_instance_token is not root.instance_token:
        raise RuntimeError("MCP provider 不属于当前 Root")
    return service.catalog()


__all__ = [
    "RUNTIME_CATALOG",
    "RuntimeCatalogReader",
    "RuntimeCatalogUnavailable",
    "build_runtime_catalog",
]
