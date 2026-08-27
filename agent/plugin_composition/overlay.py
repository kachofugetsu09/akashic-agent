from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TypeVar

from agent.plugin_composition.context import CompositionRoot, Fiber
from agent.plugin_composition.events import (
    Bail,
    EmitEventKey,
    ObserveEventKey,
    ParallelEventKey,
    SerialEventKey,
    TransformEventKey,
)
from agent.plugin_composition.model import (
    CompositionError,
    CompositionReceipt,
    PluginRuntime,
    ServiceKey,
    TopologyView,
)

T = TypeVar("T")
R = TypeVar("R")


class CompositionOverlayContext:
    """Dispatch one immutable plugin selection across stable and candidate Roots."""

    def __init__(self, overlay: CompositionOverlay) -> None:
        self._overlay = overlay

    def get(self, key: ServiceKey[T]) -> T | None:
        value = self._overlay.candidate.service_value(key)
        if value is not None:
            return value
        return self._overlay.stable.service_value(
            key,
            plugin_ids=self._overlay.stable_plugin_ids,
        )

    def require(self, key: ServiceKey[T]) -> T:
        value = self.get(key)
        if value is None:
            raise CompositionError(
                "INACTIVE_SERVICE",
                f"当前快照无法取得 Service: {key.name}",
            )
        return value

    def emit(self, key: EmitEventKey[T], payload: T) -> None:
        self._overlay.candidate._events.emit(  # pyright: ignore[reportPrivateUsage]
            key,
            payload,
            plugin_ids=self._overlay.replaced_plugin_ids,
        )

    async def serial(
        self,
        key: SerialEventKey[T, R],
        payload: T,
    ) -> Bail[R] | None:
        return await self._overlay.candidate._events.serial(  # pyright: ignore[reportPrivateUsage]
            key,
            payload,
            plugin_ids=self._overlay.replaced_plugin_ids,
        )

    async def parallel(self, key: ParallelEventKey[T], payload: T) -> None:
        await self._overlay.candidate._events.parallel(  # pyright: ignore[reportPrivateUsage]
            key,
            payload,
            plugin_ids=self._overlay.replaced_plugin_ids,
        )

    async def transform(self, key: TransformEventKey[T], payload: T) -> T:
        return await self._overlay.candidate._events.transform(  # pyright: ignore[reportPrivateUsage]
            key,
            payload,
            plugin_ids=self._overlay.replaced_plugin_ids,
        )

    async def observe(self, key: ObserveEventKey[T], payload: T) -> None:
        await self._overlay.candidate._events.observe(  # pyright: ignore[reportPrivateUsage]
            key,
            payload,
            plugin_ids=self._overlay.replaced_plugin_ids,
        )


class CompositionOverlay:
    """Select unchanged plugins from stable and changed plugins from candidate."""

    def __init__(
        self,
        stable: CompositionRoot,
        candidate: CompositionRoot,
        *,
        plugin_ids: frozenset[str],
        replaced_plugin_ids: frozenset[str],
    ) -> None:
        if not replaced_plugin_ids or not replaced_plugin_ids <= plugin_ids:
            raise ValueError("composition overlay replaced plugin selection 无效")
        self.stable = stable
        self.candidate = candidate
        self.plugin_ids = plugin_ids
        self.replaced_plugin_ids = replaced_plugin_ids
        self.stable_plugin_ids = plugin_ids - replaced_plugin_ids
        self.generation_id = candidate.generation_id
        self._instance_token = object()
        self.context = CompositionOverlayContext(self)
        self.dispatch_order = tuple(
            (
                candidate if plugin_id in replaced_plugin_ids else stable,
                plugin_id,
            )
            for plugin_id in sorted(plugin_ids)
        )
        self._validate_service_owners()
        self._validate_required_services()

    @property
    def instance_token(self) -> object:
        return self._instance_token

    @property
    def composition_revision(self) -> int:
        return self.topology_view().composition_revision

    @property
    def catalog_root_instance_token(self) -> object:
        """Return the Root that owns candidate-local mutable declarations."""

        return self.candidate.instance_token

    @property
    def catalog_context(self):
        """Return only candidate-local declaration services for snapshot freeze."""

        return self.candidate.context

    @property
    def root_fiber(self) -> Fiber:
        """Expose only the candidate-owned Fiber tree for Core diagnostics."""

        return self.candidate.root_fiber

    def receipt(self) -> CompositionReceipt:
        candidate = self.candidate.receipt()
        services = self.topology_view().services
        return replace(candidate, generation_id=self.generation_id, services=services)

    def topology_view(self) -> TopologyView:
        selected = tuple(
            root.topology_view(plugin_ids=frozenset({plugin_id}))
            for root, plugin_id in self.dispatch_order
        )
        fibers = tuple(
            sorted(
                (fiber for topology in selected for fiber in topology.fibers),
                key=lambda item: item.name,
            )
        )
        services = tuple(
            sorted({service for topology in selected for service in topology.services})
        )
        effects = tuple(
            sorted(effect for topology in selected for effect in topology.effects)
        )
        listener_groups: dict[str, list[str]] = {}
        for root, plugin_id in self.dispatch_order:
            groups = (
                root._events.registration_groups(  # pyright: ignore[reportPrivateUsage]
                    plugin_ids=(plugin_id,)
                )
            )
            for descriptor, owners in groups:
                listener_groups.setdefault(descriptor, []).extend(owners)
        listeners = tuple(
            f"{descriptor}:{owner}"
            for descriptor, owners in listener_groups.items()
            for owner in owners
        )
        payload: dict[str, object] = {
            "fibers": [
                {
                    "name": item.name,
                    "parent": item.parent,
                    "required": item.required_for_readiness,
                    "dependencies": item.dependencies,
                    "static_active": item.static_active,
                }
                for item in fibers
            ],
            "services": services,
            "listeners": listeners,
        }
        identity = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return TopologyView(
            generation_id=self.generation_id,
            identity=identity,
            composition_revision=(
                self.stable.composition_revision + self.candidate.composition_revision
            ),
            fibers=fibers,
            services=services,
            effects=effects,
            listeners=listeners,
        )

    def topology_identity(self) -> str:
        return self.topology_view().identity

    def validation_identity(self) -> str:
        return "|".join(
            (
                self.topology_identity(),
                f"stable:{self.stable.topology_identity()}",
                f"candidate:{self.candidate.validation_identity()}",
            )
        )

    def active_plugin_ids(self) -> frozenset[str]:
        return self.plugin_ids

    def plugin_runtime(self, plugin_id: str) -> PluginRuntime:
        root = self.candidate if plugin_id in self.replaced_plugin_ids else self.stable
        return root.plugin_runtime(plugin_id)

    async def dispose(self) -> None:
        await self.candidate.dispose()

    def _validate_service_owners(self) -> None:
        stable = {
            key: owner
            for key, owner in self.stable.plugin_service_owners().items()
            if owner in self.stable_plugin_ids
        }
        candidate = {
            key: owner
            for key, owner in self.candidate.plugin_service_owners().items()
            if owner in self.replaced_plugin_ids
        }
        collisions = sorted(key.name for key in set(stable) & set(candidate))
        if collisions:
            raise CompositionError(
                "DUPLICATE_SERVICE",
                "candidate 与 stable 重复提供 Service: " + ", ".join(collisions),
            )

    def _validate_required_services(self) -> None:
        """Reject a selection whose required dependency graph is incomplete."""

        topology = self.topology_view()
        available = frozenset(topology.services)
        missing = tuple(
            sorted(
                {
                    dependency
                    for fiber in topology.fibers
                    if fiber.required_for_readiness and fiber.static_active
                    for dependency in fiber.dependencies
                    if dependency not in available
                }
            )
        )
        if missing:
            raise CompositionError(
                "MISSING_SERVICE",
                "candidate 组合缺少 required Service: " + ", ".join(missing),
            )


CompositionSnapshotRoot = CompositionRoot | CompositionOverlay
