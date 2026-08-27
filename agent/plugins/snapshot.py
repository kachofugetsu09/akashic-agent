from __future__ import annotations

import asyncio
import hashlib
from contextvars import ContextVar, Token
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from agent.control.scoped_turn import TurnAdmissionRetiredError

from agent.plugins.generation import PluginGeneration
from agent.tools.registry import ToolRegistry
from agent.skills import SkillIndex
from agent.plugin_composition import (
    CHANNELS,
    COMMANDS,
    BACKGROUND_JOBS,
    TOOL_CATALOG,
    MANAGED_PROCESSES,
    MCP_SERVERS,
    CommandRegistry,
    CommandDefinition,
    CommandDescriptor,
    CompositionError,
    CompositionSnapshotRoot,
    MobileUiRegistry,
    UI_SLOTS,
    TopologyView,
)
from agent.plugin_composition.channels import (
    ChannelFactoryFreezeInput,
    ChannelRegistrySnapshot,
    CommittedChannelCatalog,
    CoreChannelDefinition,
    _freeze_plugin_channels,
    _registry_identity,
    channel_config_revision,
)
from agent.plugin_composition.mcp_slots import (
    McpServerRegistry,
    _freeze_plugin_mcp_servers,
)
from agent.plugin_composition.process_slots import (
    ManagedProcessRegistry,
    _freeze_plugin_managed_processes,
)
from agent.plugin_composition.background_jobs import (
    BackgroundJobCatalog,
    _freeze_plugin_background_jobs,
)
from agent.plugin_composition.tool_catalog import (
    PluginToolCatalog,
    _freeze_plugin_tools,
)

SnapshotState = Literal[
    "compiled",
    "validating",
    "committed",
    "aborted",
    "retired",
]
RuntimeSelector = Literal["stable", "latest"]


@dataclass
class RuntimeSnapshot:
    snapshot_id: str
    generations: Mapping[str, PluginGeneration]
    skill_catalog_generation_id: str | None
    dashboard_bindings: tuple[object, ...] = ()
    mobile_ui_registry: MobileUiRegistry | None = None
    mobile_ui_registry_identity: str | None = None
    channel_registry: ChannelRegistrySnapshot | None = None
    channel_registry_identity: str | None = None
    channel_catalog: CommittedChannelCatalog | None = None
    mcp_server_registry: McpServerRegistry | None = None
    mcp_server_registry_identity: str | None = None
    managed_process_registry: ManagedProcessRegistry | None = None
    managed_process_registry_identity: str | None = None
    background_job_catalog: BackgroundJobCatalog | None = None
    background_job_catalog_identity: str | None = None
    plugin_tool_catalog: PluginToolCatalog | None = None
    plugin_tool_catalog_identity: str | None = None
    tool_registry: ToolRegistry | None = None
    plugin_skill_index: SkillIndex | None = None
    command_registry: CommandRegistry | None = None
    composition_root: CompositionSnapshotRoot | None = None
    composition_topology: TopologyView | None = None
    composition_active_plugin_ids: frozenset[str] | None = None
    composition_validation_identity: str | None = None
    composition_validation_root_token: object | None = field(
        default=None,
        repr=False,
    )
    state: SnapshotState = "compiled"
    lease_count: int = 0
    accepting_leases: bool = True
    _store_token: object | None = field(default=None, repr=False)

    def active_generations(self) -> tuple[PluginGeneration, ...]:
        if self.generations and self.composition_active_plugin_ids is None:
            raise RuntimeError("RuntimeSnapshot 缺少 Root active plugin projection")
        active_plugin_ids = self.composition_active_plugin_ids or frozenset()
        return tuple(
            generation
            for generation in self.generations.values()
            if generation.plugin_id in active_plugin_ids
        )

    def claim(self, store_token: object) -> None:
        if (
            self.state != "compiled"
            or self.lease_count
            or self._store_token is not None
        ):
            raise RuntimeError("RuntimeSnapshot 不是可发布的全新 compiled 快照")
        self._store_token = store_token


@dataclass(frozen=True)
class SnapshotTransaction:
    previous: RuntimeSnapshot | None
    candidate: RuntimeSnapshot


class RuntimeSnapshotCompiler:
    def compile(
        self,
        generations: Mapping[str, PluginGeneration],
        *,
        catalog_generation: PluginGeneration | None = None,
        snapshot_revision: str = "",
        composition_root: CompositionSnapshotRoot | None = None,
        base_snapshot: RuntimeSnapshot | None = None,
        replaced_plugin_ids: frozenset[str] = frozenset(),
        core_channel_definitions: tuple[CoreChannelDefinition, ...] = (),
        require_composition_ready: bool = True,
    ) -> RuntimeSnapshot:
        ordered = [generations[key] for key in sorted(generations)]
        if any(generation.plugin_id != key for key, generation in generations.items()):
            raise RuntimeError("RuntimeSnapshot generation key 与 plugin_id 不一致")
        catalog_owner = catalog_generation or next(
            (
                generation
                for generation in reversed(ordered)
                if generation.skill_catalog
            ),
            None,
        )
        if (
            catalog_owner is not None
            and generations.get(catalog_owner.plugin_id) is not catalog_owner
        ):
            raise RuntimeError("RuntimeSnapshot catalog owner 不属于 generations")
        identity = "|".join(
            f"{generation.plugin_id}:{generation.generation_id}:"
            f"{generation.source_revision}:{generation.config_revision}"
            for generation in ordered
        )
        identity += "|skill:" + (
            catalog_owner.skill_catalog.generation_id
            if catalog_owner is not None and catalog_owner.skill_catalog is not None
            else ""
        )
        identity += f"|snapshot:{snapshot_revision}"
        composition_topology: TopologyView | None = None
        composition_active_plugin_ids: frozenset[str] | None = None
        command_registry: CommandRegistry | None = None
        mobile_ui_registry: MobileUiRegistry | None = None
        channel_registry: ChannelRegistrySnapshot | None = None
        channel_catalog: CommittedChannelCatalog | None = None
        mcp_server_registry: McpServerRegistry | None = None
        managed_process_registry: ManagedProcessRegistry | None = None
        background_job_catalog: BackgroundJobCatalog | None = None
        plugin_tool_catalog: PluginToolCatalog | None = None
        if base_snapshot is None and replaced_plugin_ids:
            raise ValueError("replaced_plugin_ids 需要 base_snapshot")
        if base_snapshot is not None and not replaced_plugin_ids:
            raise ValueError("base_snapshot overlay 需要 replaced_plugin_ids")
        if composition_root is not None:
            catalog_root_token = getattr(
                composition_root,
                "catalog_root_instance_token",
                composition_root.instance_token,
            )
            catalog_context = getattr(
                composition_root,
                "catalog_context",
                composition_root.context,
            )
            receipt = composition_root.receipt()
            if require_composition_ready and not receipt.ready:
                raise RuntimeError(
                    "RuntimeSnapshot 插件组合拓扑未就绪: "
                    f"required_pending={receipt.required_pending}, "
                    f"required_degraded={receipt.required_degraded}, "
                    f"incident_overflowed={receipt.incident_overflowed}, "
                    f"external_effects={receipt.external_effects}"
                )
            composition_topology = composition_root.topology_view()
            composition_active_plugin_ids = composition_root.active_plugin_ids()
            identity += f"|composition:{composition_topology.identity}"
            ui_slots = catalog_context.get(UI_SLOTS)
            if ui_slots is not None:
                freeze = getattr(ui_slots, "freeze", None)
                if not callable(freeze):
                    raise RuntimeError("RuntimeSnapshot UI Slots Service 缺少 freeze")
                frozen_registry = freeze()
                if not isinstance(frozen_registry, MobileUiRegistry):
                    raise RuntimeError("RuntimeSnapshot UI Slots freeze 返回值无效")
                mobile_ui_registry = frozen_registry
                identity += f"|mobile-ui:{mobile_ui_registry.identity}"
            commands = catalog_context.get(COMMANDS)
            if commands is not None:
                command_registry = commands.freeze()
                identity += f"|commands:{command_registry.catalog_digest}"
            channel_declarations = catalog_context.get(CHANNELS)
            if channel_declarations is not None:
                channel_registry = _freeze_plugin_channels(
                    channel_declarations,
                    catalog_root_token,
                    factory_provenance_by_owner={
                        generation.plugin_id: ChannelFactoryFreezeInput(
                            generation_id=generation.generation_id,
                            source_revision=generation.source_revision,
                            config_revision=channel_config_revision(
                                generation.config_projection
                            ),
                        )
                        for generation in ordered
                    },
                )
                if base_snapshot is not None:
                    channel_registry = _merge_channel_registries(
                        base_snapshot.channel_registry,
                        channel_registry,
                        replaced_plugin_ids,
                        composition_root.instance_token,
                    )
                identity += f"|channels-v3:{channel_registry.identity}"
            process_declarations = catalog_context.get(MANAGED_PROCESSES)
            if process_declarations is not None:
                frozen_processes = _freeze_plugin_managed_processes(
                    process_declarations,
                    catalog_root_token,
                )
                for descriptor in frozen_processes.descriptors:
                    if descriptor.owner not in generations:
                        raise RuntimeError(
                            "RuntimeSnapshot managed process owner 不属于 generations: "
                            f"{descriptor.owner}"
                        )
                managed_process_registry = frozen_processes
                identity += f"|managed-process-v3:{frozen_processes.identity}"
            mcp_servers = catalog_context.get(MCP_SERVERS)
            if mcp_servers is not None:
                frozen_mcp = _freeze_plugin_mcp_servers(
                    mcp_servers,
                    catalog_root_token,
                )
                for descriptor in frozen_mcp.descriptors:
                    generation = generations.get(descriptor.owner)
                    if generation is None:
                        raise RuntimeError(
                            "RuntimeSnapshot MCP owner 不属于 generations: "
                            f"{descriptor.owner}"
                        )
                    for endpoint in descriptor.endpoint_env:
                        process = (
                            None
                            if managed_process_registry is None
                            else managed_process_registry.get(endpoint.process)
                        )
                        if (
                            process is None
                            or process.descriptor.owner != descriptor.owner
                        ):
                            raise RuntimeError(
                                "RuntimeSnapshot MCP endpoint 缺少同 owner managed process: "
                                f"{descriptor.owner}:{descriptor.name} -> "
                                f"{endpoint.process}"
                            )
                mcp_server_registry = frozen_mcp
                identity += f"|mcp-v3:{frozen_mcp.identity}"
            background_jobs = catalog_context.get(BACKGROUND_JOBS)
            if background_jobs is not None:
                background_job_catalog = _freeze_plugin_background_jobs(
                    background_jobs,
                    catalog_root_token,
                    {
                        generation.plugin_id: generation.generation_id
                        for generation in ordered
                    },
                )
                self._validate_background_job_catalog(
                    background_job_catalog,
                    generations,
                )
                identity += f"|background-jobs-v3:{background_job_catalog.identity}"
            plugin_tools = catalog_context.get(TOOL_CATALOG)
            if plugin_tools is not None:
                plugin_tool_catalog = _freeze_plugin_tools(
                    plugin_tools,
                    catalog_root_token,
                    {
                        generation.plugin_id: generation.generation_id
                        for generation in ordered
                    },
                )
                self._validate_plugin_tool_catalog(
                    plugin_tool_catalog,
                    generations,
                )
                identity += f"|plugin-tools-v3:{plugin_tool_catalog.identity}"
            if base_snapshot is not None:
                mobile_ui_registry = _merge_owner_mapping_registry(
                    base_snapshot.mobile_ui_registry,
                    mobile_ui_registry,
                    replaced_plugin_ids,
                    MobileUiRegistry,
                    lambda item: item.descriptor.owner,
                )
                command_registry = _merge_command_registries(
                    base_snapshot.command_registry,
                    command_registry,
                    replaced_plugin_ids,
                )
                if channel_registry is None:
                    channel_registry = _merge_channel_registries(
                        base_snapshot.channel_registry,
                        None,
                        replaced_plugin_ids,
                        composition_root.instance_token,
                    )
                managed_process_registry = cast(
                    ManagedProcessRegistry | None,
                    _merge_root_mapping_registry(
                        base_snapshot.managed_process_registry,
                        managed_process_registry,
                        replaced_plugin_ids,
                        ManagedProcessRegistry,
                        composition_root.instance_token,
                        lambda item: getattr(item, "descriptor").owner,
                    ),
                )
                mcp_server_registry = cast(
                    McpServerRegistry | None,
                    _merge_root_mapping_registry(
                        base_snapshot.mcp_server_registry,
                        mcp_server_registry,
                        replaced_plugin_ids,
                        McpServerRegistry,
                        composition_root.instance_token,
                        lambda item: getattr(item, "descriptor").owner,
                    ),
                )
                background_job_catalog = cast(
                    BackgroundJobCatalog | None,
                    _merge_root_mapping_registry(
                        base_snapshot.background_job_catalog,
                        background_job_catalog,
                        replaced_plugin_ids,
                        BackgroundJobCatalog,
                        composition_root.instance_token,
                        lambda item: getattr(item, "plugin_id"),
                    ),
                )
                plugin_tool_catalog = cast(
                    PluginToolCatalog | None,
                    _merge_root_mapping_registry(
                        base_snapshot.plugin_tool_catalog,
                        plugin_tool_catalog,
                        replaced_plugin_ids,
                        PluginToolCatalog,
                        composition_root.instance_token,
                        lambda item: getattr(item, "plugin_id"),
                    ),
                )
                if background_job_catalog is not None:
                    self._validate_background_job_catalog(
                        background_job_catalog,
                        generations,
                    )
                if plugin_tool_catalog is not None:
                    self._validate_plugin_tool_catalog(
                        plugin_tool_catalog,
                        generations,
                    )
                identity += "|overlay-catalogs:" + "|".join(
                    (
                        (
                            ""
                            if mobile_ui_registry is None
                            else mobile_ui_registry.identity
                        ),
                        (
                            ""
                            if command_registry is None
                            else command_registry.catalog_digest
                        ),
                        "" if channel_registry is None else channel_registry.identity,
                        (
                            ""
                            if managed_process_registry is None
                            else managed_process_registry.identity
                        ),
                        (
                            ""
                            if mcp_server_registry is None
                            else mcp_server_registry.identity
                        ),
                        (
                            ""
                            if background_job_catalog is None
                            else background_job_catalog.identity
                        ),
                        (
                            ""
                            if plugin_tool_catalog is None
                            else plugin_tool_catalog.identity
                        ),
                    )
                )
            assert composition_active_plugin_ids is not None
            self._validate_channel_registry(
                channel_registry,
                generations,
                composition_active_plugin_ids,
            )
        if core_channel_definitions:
            channel_catalog = CommittedChannelCatalog(
                plugin_registry=channel_registry,
                core_definitions=tuple(core_channel_definitions),
                root_instance_token=(
                    None
                    if composition_root is None
                    else composition_root.instance_token
                ),
            )
            identity += f"|core-channels-v3:{channel_catalog.identity}"
        canonical_identity = "|".join(
            (
                *(
                    f"{item.plugin_id}:{item.generation_id}:{item.source_revision}:{item.config_revision}"
                    for item in ordered
                ),
                "skill:"
                + (
                    catalog_owner.skill_catalog.generation_id
                    if catalog_owner is not None
                    and catalog_owner.skill_catalog is not None
                    else ""
                ),
                f"snapshot:{snapshot_revision}",
                "composition:"
                + (
                    ""
                    if composition_topology is None
                    else composition_topology.identity
                ),
                "mobile-ui:"
                + ("" if mobile_ui_registry is None else mobile_ui_registry.identity),
                "commands:"
                + ("" if command_registry is None else command_registry.catalog_digest),
                "channels:"
                + ("" if channel_registry is None else channel_registry.identity),
                "processes:"
                + (
                    ""
                    if managed_process_registry is None
                    else managed_process_registry.identity
                ),
                "mcp:"
                + ("" if mcp_server_registry is None else mcp_server_registry.identity),
                "jobs:"
                + (
                    ""
                    if background_job_catalog is None
                    else background_job_catalog.identity
                ),
                "tools:"
                + ("" if plugin_tool_catalog is None else plugin_tool_catalog.identity),
                "channel-catalog:"
                + ("" if channel_catalog is None else channel_catalog.identity),
            )
        )
        snapshot_id = hashlib.sha256(canonical_identity.encode()).hexdigest()[:16]
        return RuntimeSnapshot(
            snapshot_id=snapshot_id,
            generations=MappingProxyType(dict(generations)),
            skill_catalog_generation_id=(
                catalog_owner.skill_catalog.generation_id
                if catalog_owner is not None and catalog_owner.skill_catalog is not None
                else None
            ),
            mobile_ui_registry=mobile_ui_registry,
            mobile_ui_registry_identity=(
                None if mobile_ui_registry is None else mobile_ui_registry.identity
            ),
            channel_registry=channel_registry,
            channel_registry_identity=(
                None if channel_registry is None else channel_registry.identity
            ),
            channel_catalog=channel_catalog,
            mcp_server_registry=mcp_server_registry,
            mcp_server_registry_identity=(
                None if mcp_server_registry is None else mcp_server_registry.identity
            ),
            managed_process_registry=managed_process_registry,
            managed_process_registry_identity=(
                None
                if managed_process_registry is None
                else managed_process_registry.identity
            ),
            background_job_catalog=background_job_catalog,
            background_job_catalog_identity=(
                None
                if background_job_catalog is None
                else background_job_catalog.identity
            ),
            plugin_tool_catalog=plugin_tool_catalog,
            plugin_tool_catalog_identity=(
                None if plugin_tool_catalog is None else plugin_tool_catalog.identity
            ),
            plugin_skill_index=(
                catalog_owner.skill_catalog.normal_plugins
                if catalog_owner is not None and catalog_owner.skill_catalog is not None
                else None
            ),
            command_registry=command_registry,
            composition_root=composition_root,
            composition_topology=composition_topology,
            composition_active_plugin_ids=composition_active_plugin_ids,
        )

    @staticmethod
    def _validate_channel_registry(
        registry: ChannelRegistrySnapshot | None,
        generations: Mapping[str, PluginGeneration],
        active_plugin_ids: frozenset[str],
    ) -> None:
        """Validate the final merged channel catalog against active manifests."""

        frozen_channels: set[tuple[str, str]] = set()
        for descriptor in () if registry is None else registry.descriptors:
            generation = generations.get(descriptor.owner)
            if generation is None:
                raise RuntimeError(
                    "RuntimeSnapshot channel owner 不属于 generations: "
                    f"{descriptor.owner}"
                )
            manifest = generation.static_manifest
            if manifest is None:
                if descriptor.credential_paths:
                    raise RuntimeError(
                        "RuntimeSnapshot channel credential 缺少静态 manifest 声明: "
                        f"{descriptor.owner}:{descriptor.name}"
                    )
            else:
                declared = dict(manifest.channel_credentials).get(
                    descriptor.name,
                    (),
                )
                if declared != descriptor.credential_paths:
                    raise RuntimeError(
                        "RuntimeSnapshot channel credential 声明与静态 manifest 不一致: "
                        f"{descriptor.owner}:{descriptor.name}"
                    )
            frozen_channels.add((descriptor.owner, descriptor.name))

        for generation in generations.values():
            manifest = generation.static_manifest
            if manifest is None or generation.plugin_id not in active_plugin_ids:
                continue
            for channel_name, _paths in manifest.channel_credentials:
                if (generation.plugin_id, channel_name) not in frozen_channels:
                    raise RuntimeError(
                        "RuntimeSnapshot 静态 channel credential 没有对应 Root 声明: "
                        f"{generation.plugin_id}:{channel_name}"
                    )

    @staticmethod
    def _validate_background_job_catalog(
        catalog: BackgroundJobCatalog,
        generations: Mapping[str, PluginGeneration],
    ) -> None:
        """Validate every background job against its exact generation."""

        for binding in catalog.values():
            generation = generations.get(binding.plugin_id)
            if generation is None or generation.generation_id != binding.generation_id:
                raise RuntimeError(
                    "RuntimeSnapshot background job 不属于 exact generation: "
                    f"{binding.plugin_id}:{binding.generation_id}"
                )

    @staticmethod
    def _validate_plugin_tool_catalog(
        catalog: PluginToolCatalog,
        generations: Mapping[str, PluginGeneration],
    ) -> None:
        """Validate every Tool binding against its exact generation."""

        for binding in catalog.values():
            generation = generations.get(binding.plugin_id)
            if generation is None or generation.generation_id != binding.generation_id:
                raise RuntimeError(
                    "RuntimeSnapshot plugin Tool 不属于 exact generation: "
                    f"{binding.plugin_id}:{binding.generation_id}"
                )


def _merge_command_registries(
    base: CommandRegistry | None,
    delta: CommandRegistry | None,
    replaced: frozenset[str],
) -> CommandRegistry | None:
    """Replace command contributions by owner without replaying stable plugins."""

    if base is None and delta is None:
        return None
    commands: dict[str, CommandDefinition] = {}
    owners: dict[str, str] = {}
    generations: dict[str, str] = {}
    fibers: dict[str, str] = {}
    descriptors: list[CommandDescriptor] = []
    for registry in (base, delta):
        if registry is None:
            continue
        for (
            name,
            definition,
        ) in registry._commands.items():  # pyright: ignore[reportPrivateUsage]
            owner = registry._owners[name]  # pyright: ignore[reportPrivateUsage]
            if registry is base and owner in replaced:
                continue
            if name in commands:
                raise CompositionError(
                    "DUPLICATE_COMMAND",
                    f"candidate 与 stable 重复注册 command: {name}",
                )
            commands[name] = definition
            owners[name] = owner
            generation = registry._generations.get(
                name
            )  # pyright: ignore[reportPrivateUsage]
            fiber = registry._fibers.get(name)  # pyright: ignore[reportPrivateUsage]
            if generation is not None:
                generations[name] = generation
            if fiber is not None:
                fibers[name] = fiber
        descriptors.extend(
            item
            for item in registry.descriptors
            if registry is not base or item.owner not in replaced
        )
    return CommandRegistry(
        commands,
        owners,
        tuple(sorted(descriptors, key=lambda item: item.name)),
        generations,
        fibers,
    )


def _merge_owner_mapping_registry(
    base: object | None,
    delta: object | None,
    replaced: frozenset[str],
    registry_type: type[MobileUiRegistry],
    owner_of: Callable[[object], str],
) -> MobileUiRegistry | None:
    """Replace one immutable owner-keyed registry."""

    if base is None and delta is None:
        return None
    bindings: dict[str, object] = {}
    for registry in (base, delta):
        if registry is None:
            continue
        for key in registry:  # type: ignore[union-attr]
            binding = registry[key]  # type: ignore[index]
            owner = owner_of(binding)
            if registry is base and owner in replaced:
                continue
            if key in bindings:
                raise CompositionError(
                    "DUPLICATE_REGISTRATION",
                    f"candidate 与 stable 重复注册: {key}",
                )
            bindings[key] = binding
    return registry_type(cast(Mapping[str, object], bindings))  # type: ignore[arg-type]


def _merge_root_mapping_registry(
    base: object | None,
    delta: object | None,
    replaced: frozenset[str],
    registry_type: type[object],
    root_token: object,
    owner_of: Callable[[object], str],
) -> object | None:
    """Replace one immutable Root-bound registry by contribution owner."""

    if base is None and delta is None:
        return None
    bindings: dict[str, object] = {}
    for registry in (base, delta):
        if registry is None:
            continue
        for key in registry:  # type: ignore[union-attr]
            binding = registry[key]  # type: ignore[index]
            owner = owner_of(binding)
            if registry is base and owner in replaced:
                continue
            if key in bindings:
                raise CompositionError(
                    "DUPLICATE_REGISTRATION",
                    f"candidate 与 stable 重复注册: {key}",
                )
            bindings[key] = binding
    return registry_type(bindings, root_instance_token=root_token)  # type: ignore[call-arg]


def _merge_channel_registries(
    base: ChannelRegistrySnapshot | None,
    delta: ChannelRegistrySnapshot | None,
    replaced: frozenset[str],
    root_token: object,
) -> ChannelRegistrySnapshot | None:
    """Replace immutable channel descriptors and provenance by plugin owner."""

    if base is None and delta is None:
        return None
    descriptors = tuple(
        sorted(
            (
                *(
                    ()
                    if base is None
                    else tuple(
                        item for item in base.descriptors if item.owner not in replaced
                    )
                ),
                *(() if delta is None else delta.descriptors),
            ),
            key=lambda item: item.name,
        )
    )
    factories = tuple(
        sorted(
            (
                *(
                    ()
                    if base is None
                    else tuple(
                        item
                        for item in base.factories
                        if item.plugin_id not in replaced
                    )
                ),
                *(() if delta is None else delta.factories),
            ),
            key=lambda item: (item.plugin_id, item.channel_name),
        )
    )
    return ChannelRegistrySnapshot(
        descriptors=descriptors,
        factories=factories,
        identity=_registry_identity(descriptors, factories),
        root_instance_token=root_token,
    )


# 插件生命周期边界：一个 turn、job、event 或 proactive tick 必须始终使用同一
# snapshot；旧 generation 只有在全部 lease 释放后才能 retire 和清理。
class RuntimeSnapshotLease:
    def __init__(
        self,
        store: RuntimeSnapshotStore,
        snapshot: RuntimeSnapshot,
        validation_candidate_plugin_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._store = store
        self.snapshot = snapshot
        self.validation_candidate_plugin_ids = validation_candidate_plugin_ids
        self._released = False

    @property
    def active(self) -> bool:
        return not self._released

    def fork(self) -> RuntimeSnapshotLease:
        return self._store.fork_lease(self)

    async def __aenter__(self) -> RuntimeSnapshot:
        return self.snapshot

    async def __aexit__(self, *exc_info: object) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._store.release_lease(self.snapshot)


@dataclass(frozen=True)
class _RuntimeSnapshotBinding:
    lease: RuntimeSnapshotLease
    owner_task: asyncio.Task[object] | None


_current_runtime_binding: ContextVar[_RuntimeSnapshotBinding | None] = ContextVar(
    "current_runtime_binding",
    default=None,
)


def bind_runtime_snapshot(
    lease: RuntimeSnapshotLease,
) -> Token[_RuntimeSnapshotBinding | None]:
    return _current_runtime_binding.set(
        _RuntimeSnapshotBinding(
            lease=lease,
            owner_task=asyncio.current_task(),
        )
    )


def reset_runtime_snapshot(token: Token[_RuntimeSnapshotBinding | None]) -> None:
    _current_runtime_binding.reset(token)


def get_current_runtime_snapshot() -> RuntimeSnapshot | None:
    binding = _current_runtime_binding.get()
    if (
        binding is None
        or not binding.lease.active
        or binding.owner_task is not asyncio.current_task()
    ):
        return None
    return binding.lease.snapshot


def get_lifecycle_runtime_snapshot() -> RuntimeSnapshot | None:
    """Resolve a lifecycle snapshot while rejecting inherited or stale bindings."""

    binding = _current_runtime_binding.get()
    if binding is None:
        return None
    if binding.owner_task is not asyncio.current_task():
        raise CompositionError(
            "RUNTIME_SNAPSHOT_BINDING_MISMATCH",
            "lifecycle 必须在绑定 RuntimeSnapshot lease 的 owner task 中运行",
        )
    if not binding.lease.active:
        raise CompositionError(
            "RUNTIME_SNAPSHOT_BINDING_INACTIVE",
            "lifecycle 不能使用已释放的 RuntimeSnapshot lease",
        )
    return binding.lease.snapshot


def lease_current_runtime_snapshot() -> RuntimeSnapshotLease | None:
    lease = get_current_runtime_lease()
    return lease.fork() if lease is not None else None


def get_current_runtime_lease() -> RuntimeSnapshotLease | None:
    binding = _current_runtime_binding.get()
    if (
        binding is None
        or not binding.lease.active
        or binding.owner_task is not asyncio.current_task()
    ):
        return None
    return binding.lease


class RuntimeSnapshotStore:
    def __init__(
        self,
        on_drained: Callable[[RuntimeSnapshot], Awaitable[None]] | None = None,
    ) -> None:
        self._current: RuntimeSnapshot | None = None
        self._latest: RuntimeSnapshot | None = None
        self._snapshots: dict[str, RuntimeSnapshot] = {}
        self._pending: SnapshotTransaction | None = None
        self._provisional: SnapshotTransaction | None = None
        self._on_drained = on_drained
        self._token = object()
        self._condition = asyncio.Condition()
        self._drain_tasks: dict[str, asyncio.Task[None]] = {}
        self._drain_failures: dict[str, BaseException] = {}

    @property
    def current(self) -> RuntimeSnapshot | None:
        return self._current

    @property
    def stable(self) -> RuntimeSnapshot | None:
        return self._current

    async def wait_for_stable_change(
        self,
        current: RuntimeSnapshot,
    ) -> RuntimeSnapshot:
        """Wait until another committed snapshot becomes the stable owner."""

        async with self._condition:
            await self._condition.wait_for(lambda: self._current is not current)
            if self._current is None:
                raise RuntimeError("RuntimeSnapshot stable owner 不可为空")
            return self._current

    async def wait_for_snapshot_drained(self, snapshot: RuntimeSnapshot) -> None:
        """Wait until one retired snapshot finishes its exact drain callback."""

        async with self._condition:
            await self._condition.wait_for(
                lambda: (
                    snapshot.snapshot_id not in self._snapshots
                    or snapshot.snapshot_id in self._drain_failures
                )
            )
            failure = self._drain_failures.get(snapshot.snapshot_id)
            if failure is not None:
                raise failure

    @property
    def latest(self) -> RuntimeSnapshot | None:
        return self._latest or self._current

    @property
    def unpromoted_candidate(self) -> RuntimeSnapshot | None:
        latest = self.latest
        return latest if latest is not self._current else None

    @property
    def pending_candidate(self) -> RuntimeSnapshot | None:
        if self._pending is None:
            return None
        return self._pending.candidate

    @property
    def pending_transaction(self) -> SnapshotTransaction | None:
        """Expose the exact pending owner for its caller's failure cleanup."""

        return self._pending

    @property
    def retained_snapshot_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._snapshots))

    def generation_is_referenced_elsewhere(
        self,
        generation: PluginGeneration,
        *,
        excluding_snapshot_id: str,
    ) -> bool:
        return any(
            snapshot.snapshot_id != excluding_snapshot_id
            and (
                snapshot.state in {"validating", "committed"}
                or snapshot.lease_count > 0
            )
            and any(item is generation for item in snapshot.generations.values())
            for snapshot in self._snapshots.values()
        )

    def composition_is_referenced_elsewhere(
        self,
        root: CompositionSnapshotRoot,
        *,
        excluding_snapshot_id: str,
    ) -> bool:
        return any(
            snapshot.snapshot_id != excluding_snapshot_id
            and (
                snapshot.state in {"validating", "committed"}
                or snapshot.lease_count > 0
            )
            and snapshot.composition_root is root
            for snapshot in self._snapshots.values()
        )

    def install(self, snapshot: RuntimeSnapshot) -> None:
        if (
            self._current is not None
            or self._pending is not None
            or self._provisional is not None
        ):
            raise RuntimeError("RuntimeSnapshotStore 已安装初始快照")
        self._validate_composition(snapshot)
        self._adopt(snapshot)
        snapshot.state = "committed"
        self._current = snapshot
        self._latest = snapshot
        self._snapshots[snapshot.snapshot_id] = snapshot

    def begin_publish(
        self,
        candidate: RuntimeSnapshot,
        *,
        admission_gated: bool = False,
    ) -> SnapshotTransaction:
        if self._pending is not None or self._provisional is not None:
            raise RuntimeError("已有 RuntimeSnapshot 发布事务")
        if self.unpromoted_candidate is not None:
            raise RuntimeError("已有 RuntimeSnapshot 候选等待 promote/discard")
        if candidate.snapshot_id in self._snapshots:
            raise RuntimeError(f"RuntimeSnapshot 已存在: {candidate.snapshot_id}")
        self._validate_composition(candidate)
        self._adopt(candidate)
        transaction = SnapshotTransaction(previous=self._current, candidate=candidate)
        candidate.state = "validating"
        candidate.accepting_leases = False
        self._snapshots[candidate.snapshot_id] = candidate
        self._pending = transaction
        return transaction

    async def commit(
        self,
        transaction: SnapshotTransaction,
        *,
        before_open: Callable[[], None] | None = None,
        after_open: Callable[[], None] | None = None,
    ) -> None:
        self._require_pending(transaction)
        self._validate_composition(transaction.candidate)
        if before_open is not None:
            before_open()
        transaction.candidate.state = "committed"
        transaction.candidate.accepting_leases = True
        self._current = transaction.candidate
        self._latest = transaction.candidate
        self._pending = None
        previous = transaction.previous
        if previous is not None:
            previous.state = "retired"
            if after_open is not None:
                after_open()
            self._schedule_drain(previous)
        async with self._condition:
            self._condition.notify_all()

    async def commit_latest(
        self,
        transaction: SnapshotTransaction,
        *,
        before_open: Callable[[], None] | None = None,
    ) -> None:
        """Publish a validation candidate without changing the stable pointer."""

        # 1. Open only the explicitly selected candidate.
        self._require_pending(transaction)
        self._validate_composition(transaction.candidate)
        if before_open is not None:
            before_open()
        transaction.candidate.state = "committed"
        transaction.candidate.accepting_leases = True
        self._latest = transaction.candidate
        self._pending = None

        # 2. Wake latest waiters while stable readers stay on the previous snapshot.
        async with self._condition:
            self._condition.notify_all()

    async def commit_provisional(
        self,
        transaction: SnapshotTransaction,
    ) -> None:
        """Stage a closed candidate without exposing it as the published stable."""

        # 1. Validate and close both sides before the external publication step.
        self._require_pending(transaction)
        self._validate_composition(transaction.candidate)
        transaction.candidate.state = "committed"
        transaction.candidate.accepting_leases = False
        if transaction.previous is not None:
            transaction.previous.accepting_leases = False

        # 2. Keep discovery pinned to the old stable while retaining the target.
        self._latest = transaction.candidate
        self._pending = None
        self._provisional = transaction
        async with self._condition:
            self._condition.notify_all()

    async def promote_latest_provisional(self) -> SnapshotTransaction:
        """Stage a sealed latest candidate without exposing it as stable."""

        # 1. Validate the exact closed latest candidate before moving the pointer.
        if self._provisional is not None:
            raise RuntimeError("已有 RuntimeSnapshot provisional 发布事务")
        candidate = self.unpromoted_candidate
        if candidate is None:
            raise RuntimeError("没有等待 promote 的 RuntimeSnapshot 候选")
        if candidate.accepting_leases:
            raise RuntimeError("promote 前必须先暂停 candidate lease admission")
        self._validate_composition(candidate, require_validation=True)

        # 2. Keep the old stable visible but closed until the external step settles.
        previous = self._current
        if previous is not None:
            previous.accepting_leases = False
        transaction = SnapshotTransaction(previous=previous, candidate=candidate)
        self._latest = candidate
        candidate.accepting_leases = False
        self._provisional = transaction
        async with self._condition:
            self._condition.notify_all()
        return transaction

    async def finalize_provisional(
        self,
        transaction: SnapshotTransaction,
        *,
        before_open: Callable[[], None] | None = None,
        after_open: Callable[[], None] | None = None,
    ) -> None:
        """Open a provisional stable and retire its rollback snapshot."""

        # 1. Complete fallible projection work while the old stable stays visible.
        self._require_provisional(transaction)
        self._validate_composition(transaction.candidate)
        if before_open is not None:
            before_open()

        # 2. Switch the stable pointer synchronously around the owner callback.
        transaction.candidate.state = "committed"
        previous = transaction.previous
        self._current = transaction.candidate
        try:
            if after_open is not None:
                after_open()
        except BaseException:
            self._current = previous
            raise

        # 3. Open the new stable only after all publication work succeeded.
        transaction.candidate.accepting_leases = True
        if previous is not None:
            previous.state = "retired"
            previous.accepting_leases = False
        self._provisional = None
        if previous is not None:
            self._schedule_drain(previous)
        async with self._condition:
            self._condition.notify_all()

    async def rollback_provisional(
        self,
        transaction: SnapshotTransaction,
        *,
        keep_candidate_latest: bool,
        reopen_previous: bool = True,
    ) -> None:
        """Restore the previous stable before disposing or retrying the candidate."""

        # 1. Reopen the old pointer; the candidate was never publicly current.
        self._require_provisional(transaction)
        candidate = transaction.candidate
        previous = transaction.previous
        if self._current is not previous:
            raise RuntimeError("RuntimeSnapshot provisional stable 指针已漂移")
        if previous is not None:
            previous.state = "committed"
            previous.accepting_leases = reopen_previous
        self._provisional = None

        # 2. Either retain latest for normal discard or restore the pending transaction.
        candidate.accepting_leases = False
        if keep_candidate_latest:
            candidate.state = "committed"
            self._latest = candidate
        else:
            candidate.state = "validating"
            self._latest = previous
            self._pending = transaction
        async with self._condition:
            self._condition.notify_all()

    async def promote_latest(
        self,
        *,
        before_open: Callable[[], None] | None = None,
        after_open: Callable[[], None] | None = None,
    ) -> SnapshotTransaction:
        """Atomically make the ready latest snapshot stable and retire the old stable."""

        # 1. Switch the public pointer without rebuilding the validated snapshot.
        if self._provisional is not None:
            raise RuntimeError("RuntimeSnapshot provisional 发布事务尚未结束")
        candidate = self.unpromoted_candidate
        if candidate is None:
            raise RuntimeError("没有等待 promote 的 RuntimeSnapshot 候选")
        if candidate.accepting_leases:
            raise RuntimeError("promote 前必须先暂停 candidate lease admission")
        self._validate_composition(candidate, require_validation=True)
        if before_open is not None:
            before_open()
        previous = self._current
        self._current = candidate
        self._latest = candidate
        candidate.accepting_leases = True

        # 2. manager owner 切换完成后，旧 stable 才能开始 drain。
        if previous is not None:
            previous.state = "retired"
            previous.accepting_leases = False
        try:
            if after_open is not None:
                after_open()
        except BaseException:
            self._current = previous
            self._latest = candidate
            candidate.accepting_leases = True
            if previous is not None:
                previous.state = "committed"
                previous.accepting_leases = True
            async with self._condition:
                self._condition.notify_all()
            raise
        if previous is not None:
            self._schedule_drain(previous)
        async with self._condition:
            self._condition.notify_all()
        return SnapshotTransaction(previous=previous, candidate=candidate)

    async def discard_latest(
        self,
        expected: RuntimeSnapshot | None = None,
    ) -> RuntimeSnapshot:
        """Discard the ready latest snapshot without changing stable."""

        # 1. Remove candidate admission once; retries resume its failed drain.
        if self._provisional is not None:
            raise RuntimeError("RuntimeSnapshot provisional 发布事务尚未结束")
        candidate = self.unpromoted_candidate
        if candidate is None:
            if expected is None or expected.state != "aborted":
                raise RuntimeError("没有等待 discard 的 RuntimeSnapshot 候选")
            candidate = expected
            if candidate.snapshot_id not in self._snapshots:
                return candidate
        elif expected is not None and candidate is not expected:
            raise RuntimeError("等待 discard 的 RuntimeSnapshot 候选不一致")
        if candidate.state != "aborted":
            candidate.state = "aborted"
            candidate.accepting_leases = False
            self._latest = self._current
        await self.wait_for_no_leases(candidate)
        self._schedule_drain(candidate)

        # 2. Wait for validation leases and candidate-owned resources to drain.
        await self._await_drain_tasks((candidate.snapshot_id,))
        self._raise_drain_failures((candidate.snapshot_id,))
        async with self._condition:
            self._condition.notify_all()
        return candidate

    async def abort(
        self,
        transaction: SnapshotTransaction,
        *,
        reopen_previous: bool = True,
    ) -> None:
        self._require_pending(transaction)
        transaction.candidate.state = "aborted"
        transaction.candidate.accepting_leases = False
        if self._current is transaction.previous and transaction.previous is not None:
            transaction.previous.accepting_leases = reopen_previous
        self._pending = None
        self._schedule_drain(transaction.candidate)
        await self._await_drain_tasks((transaction.candidate.snapshot_id,))
        self._raise_drain_failures((transaction.candidate.snapshot_id,))
        async with self._condition:
            self._condition.notify_all()

    async def quiesce_current(self) -> RuntimeSnapshot | None:
        snapshot = self.pause_admission()
        if snapshot is None:
            return None
        try:
            await self.wait_for_no_leases(snapshot)
        except BaseException:
            await self.resume(snapshot)
            raise
        return snapshot

    def pause_admission(self) -> RuntimeSnapshot | None:
        snapshot = self._current
        if snapshot is not None:
            snapshot.accepting_leases = False
        return snapshot

    def pause_candidate_admission(
        self,
        expected: RuntimeSnapshot,
    ) -> RuntimeSnapshot:
        """Atomically seal the exact unpromoted candidate against new leases."""

        candidate = self.unpromoted_candidate
        if candidate is None or candidate is not expected:
            raise RuntimeError("等待 promote 的 RuntimeSnapshot 候选不一致")
        candidate.accepting_leases = False
        return candidate

    def seal_candidate_validation(self, expected: RuntimeSnapshot) -> None:
        """Seal the Core-observed receipt after validation leases have drained."""

        candidate = self.unpromoted_candidate
        if candidate is None or candidate is not expected:
            raise RuntimeError("等待封存验证回执的 RuntimeSnapshot 候选不一致")
        if candidate.accepting_leases or candidate.lease_count:
            raise RuntimeError("封存验证回执前必须暂停并排空 candidate lease")
        self._seal_composition_validation(candidate)

    def seal_pending_validation(self, expected: RuntimeSnapshot) -> None:
        """封存尚未公开的 direct candidate 组合验证事实。"""

        candidate = self.pending_candidate
        if candidate is None or candidate is not expected:
            raise RuntimeError("等待封存验证回执的 pending candidate 不一致")
        self._seal_composition_validation(candidate)

    def _seal_composition_validation(self, candidate: RuntimeSnapshot) -> None:
        """在无 lease 的隔离 Root 上保存不可变验证证明。"""

        if candidate.accepting_leases or candidate.lease_count:
            raise RuntimeError("封存验证回执前必须暂停并排空 candidate lease")
        self._validate_composition(candidate)
        root = candidate.composition_root
        candidate.composition_validation_identity = (
            None if root is None else root.validation_identity()
        )
        candidate.composition_validation_root_token = (
            None if root is None else root.instance_token
        )

    async def wait_for_no_leases(self, snapshot: RuntimeSnapshot) -> None:
        async with self._condition:
            while snapshot.lease_count:
                await self._condition.wait()

    async def resume(self, snapshot: RuntimeSnapshot | None) -> None:
        if snapshot is None:
            return
        if snapshot.state == "committed" and (
            self._current is snapshot or self.unpromoted_candidate is snapshot
        ):
            snapshot.accepting_leases = True
        async with self._condition:
            self._condition.notify_all()

    async def acquire(
        self,
        snapshot_id: str | None = None,
        *,
        selector: RuntimeSelector = "stable",
    ) -> RuntimeSnapshotLease:
        async with self._condition:
            while True:
                snapshot = (
                    self._selected(selector)
                    if snapshot_id is None
                    else self._snapshots.get(snapshot_id)
                )
                if snapshot is None:
                    raise RuntimeError("RuntimeSnapshot 不可用")
                if snapshot.state != "committed":
                    raise RuntimeError(f"RuntimeSnapshot 不可租用: {snapshot.state}")
                if snapshot.accepting_leases:
                    return self._claim_lease(snapshot)
                await self._condition.wait()

    async def acquire_composition_root(
        self,
        root: CompositionSnapshotRoot,
    ) -> RuntimeSnapshotLease:
        """Lease the committed snapshot that owns one exact composition Root."""

        async with self._condition:
            while True:
                snapshot = next(
                    (
                        item
                        for item in self._snapshots.values()
                        if item.composition_root is root
                        and item.state in {"validating", "committed"}
                    ),
                    None,
                )
                if snapshot is None:
                    raise TurnAdmissionRetiredError(
                        "composition Root 已退役，Turn 尚未进入 admission"
                    )
                if (
                    snapshot is self._current
                    and snapshot.state == "committed"
                    and snapshot.accepting_leases
                ):
                    return self._claim_lease(snapshot)
                await self._condition.wait()

    async def close(self) -> None:
        if self._pending is not None or self._provisional is not None:
            raise RuntimeError("RuntimeSnapshot 发布事务尚未结束")
        leased = [
            snapshot.snapshot_id
            for snapshot in self._snapshots.values()
            if snapshot.lease_count
        ]
        if leased:
            raise RuntimeError(
                f"RuntimeSnapshot 仍有 lease: {', '.join(sorted(leased))}"
            )
        await self.retry_drains()
        latest = self.unpromoted_candidate
        self._latest = self._current
        if latest is not None:
            latest.state = "aborted"
            latest.accepting_leases = False
            self._schedule_drain(latest)
        current = self._current
        self._current = None
        self._latest = None
        if current is not None:
            current.state = "retired"
            self._schedule_drain(current)
            await self.retry_drains()

    def lease(
        self,
        snapshot_id: str | None = None,
        *,
        selector: RuntimeSelector = "stable",
    ) -> RuntimeSnapshotLease:
        snapshot = (
            self._selected(selector)
            if snapshot_id is None
            else self._snapshots.get(snapshot_id)
        )
        if snapshot is None:
            raise RuntimeError("RuntimeSnapshot 不可用")
        if snapshot.state != "committed":
            raise RuntimeError(f"RuntimeSnapshot 不可租用: {snapshot.state}")
        if not snapshot.accepting_leases:
            raise RuntimeError("RuntimeSnapshot 暂停接收新 lease")
        return self._claim_lease(snapshot)

    def retain_publication_target(
        self,
        transaction: SnapshotTransaction,
    ) -> RuntimeSnapshotLease:
        """Retain the closed exact target for one Core publication participant."""

        if self._pending is not transaction and self._provisional is not transaction:
            raise RuntimeError("RuntimeSnapshot publication target 已失效")
        candidate = transaction.candidate
        if self._snapshots.get(candidate.snapshot_id) is not candidate:
            raise RuntimeError("RuntimeSnapshot publication target 未被 Store 持有")
        return self._claim_lease(candidate)

    def _claim_lease(self, snapshot: RuntimeSnapshot) -> RuntimeSnapshotLease:
        snapshot.lease_count += 1
        for generation in snapshot.generations.values():
            generation.lease_count += 1
        return RuntimeSnapshotLease(
            self,
            snapshot,
            self._validation_candidate_plugin_ids(snapshot),
        )

    def fork_lease(self, source: RuntimeSnapshotLease) -> RuntimeSnapshotLease:
        snapshot = source.snapshot
        if (
            not source.active
            or self._snapshots.get(snapshot.snapshot_id) is not snapshot
        ):
            raise RuntimeError("RuntimeSnapshot lease 不可复制")
        snapshot.lease_count += 1
        for generation in snapshot.generations.values():
            generation.lease_count += 1
        return RuntimeSnapshotLease(
            self,
            snapshot,
            source.validation_candidate_plugin_ids,
        )

    def _validation_candidate_plugin_ids(
        self,
        snapshot: RuntimeSnapshot,
    ) -> frozenset[str]:
        stable = self._current
        if stable is None or snapshot is not self.unpromoted_candidate:
            return frozenset()
        return frozenset(
            plugin_id
            for plugin_id, generation in snapshot.generations.items()
            if stable.generations.get(plugin_id) is not generation
        )

    async def release_lease(self, snapshot: RuntimeSnapshot) -> None:
        if snapshot.lease_count <= 0:
            raise RuntimeError(
                f"RuntimeSnapshot lease 计数失衡: {snapshot.snapshot_id}"
            )
        snapshot.lease_count -= 1
        for generation in snapshot.generations.values():
            generation.lease_count -= 1
        self._schedule_drain(snapshot)
        async with self._condition:
            self._condition.notify_all()

    async def wait_for_generation_drained(
        self,
        generation: PluginGeneration,
    ) -> None:
        async with self._condition:
            while generation.lease_count:
                await self._condition.wait()
        await self.retry_drains()

    def _schedule_drain(self, snapshot: RuntimeSnapshot) -> None:
        if (
            self._snapshots.get(snapshot.snapshot_id) is not snapshot
            or snapshot.state not in {"retired", "aborted"}
            or snapshot.lease_count
        ):
            return
        existing = self._drain_tasks.get(snapshot.snapshot_id)
        if existing is not None and not existing.done():
            return
        _ = self._drain_failures.pop(snapshot.snapshot_id, None)
        self._drain_tasks[snapshot.snapshot_id] = asyncio.create_task(
            self._run_drain(snapshot),
            name=f"runtime_snapshot_drain:{snapshot.snapshot_id}",
        )

    async def _run_drain(self, snapshot: RuntimeSnapshot) -> None:
        try:
            if self._on_drained is not None:
                await self._on_drained(snapshot)
        except (asyncio.CancelledError, Exception) as error:
            self._drain_failures[snapshot.snapshot_id] = error
        else:
            _ = self._snapshots.pop(snapshot.snapshot_id, None)
        finally:
            _ = self._drain_tasks.pop(snapshot.snapshot_id, None)
            async with self._condition:
                self._condition.notify_all()

    async def retry_drains(self) -> None:
        await self._await_drain_tasks(tuple(self._drain_tasks))
        for snapshot in tuple(self._snapshots.values()):
            self._schedule_drain(snapshot)
        attempted = tuple(self._drain_tasks)
        await self._await_drain_tasks(attempted)
        self._raise_drain_failures(attempted)

    async def _await_drain_tasks(self, snapshot_ids: tuple[str, ...]) -> None:
        tasks = [
            task
            for snapshot_id in snapshot_ids
            if (task := self._drain_tasks.get(snapshot_id)) is not None
        ]
        if tasks:
            await asyncio.gather(*tasks)

    def _raise_drain_failures(self, snapshot_ids: tuple[str, ...]) -> None:
        failures = [
            (snapshot_id, self._drain_failures[snapshot_id])
            for snapshot_id in snapshot_ids
            if snapshot_id in self._drain_failures
        ]
        if not failures:
            return
        snapshot_id, error = failures[0]
        raise RuntimeError(f"RuntimeSnapshot drain 失败: {snapshot_id}") from error

    def _require_pending(self, transaction: SnapshotTransaction) -> None:
        if self._pending is not transaction:
            raise RuntimeError("RuntimeSnapshot 发布事务已失效")

    def _require_provisional(self, transaction: SnapshotTransaction) -> None:
        if self._provisional is not transaction:
            raise RuntimeError("RuntimeSnapshot provisional 发布事务已失效")

    def _adopt(self, snapshot: RuntimeSnapshot) -> None:
        snapshot.claim(self._token)

    @staticmethod
    def _validate_composition(
        snapshot: RuntimeSnapshot,
        *,
        require_validation: bool = False,
    ) -> None:
        root = snapshot.composition_root
        if root is None:
            if (
                snapshot.composition_topology is not None
                or snapshot.mobile_ui_registry is not None
                or snapshot.mobile_ui_registry_identity is not None
                or snapshot.channel_registry is not None
                or snapshot.channel_registry_identity is not None
                or snapshot.channel_catalog is not None
                or snapshot.mcp_server_registry is not None
                or snapshot.mcp_server_registry_identity is not None
                or snapshot.managed_process_registry is not None
                or snapshot.managed_process_registry_identity is not None
                or snapshot.background_job_catalog is not None
                or snapshot.background_job_catalog_identity is not None
                or snapshot.plugin_tool_catalog is not None
                or snapshot.plugin_tool_catalog_identity is not None
            ):
                raise RuntimeError(
                    "RuntimeSnapshot composition identity 缺少 Root Context"
                )
            return
        if snapshot.mobile_ui_registry_identity != (
            None
            if snapshot.mobile_ui_registry is None
            else snapshot.mobile_ui_registry.identity
        ):
            raise RuntimeError("RuntimeSnapshot Mobile UI descriptor 在编译后发生变化")
        if snapshot.channel_registry_identity != (
            None
            if snapshot.channel_registry is None
            else snapshot.channel_registry.identity
        ):
            raise RuntimeError("RuntimeSnapshot channel descriptor 在编译后发生变化")
        if (
            snapshot.channel_registry is not None
            and snapshot.channel_registry.root_instance_token is not root.instance_token
        ):
            raise RuntimeError("RuntimeSnapshot channel registry 不属于 exact Root")
        if (
            snapshot.channel_catalog is not None
            and snapshot.channel_catalog.root_instance_token is not root.instance_token
        ):
            raise RuntimeError("RuntimeSnapshot channel catalog 不属于 exact Root")
        if snapshot.mcp_server_registry_identity != (
            None
            if snapshot.mcp_server_registry is None
            else snapshot.mcp_server_registry.identity
        ):
            raise RuntimeError("RuntimeSnapshot MCP descriptor 在编译后发生变化")
        if (
            snapshot.mcp_server_registry is not None
            and snapshot.mcp_server_registry.root_instance_token
            is not root.instance_token
        ):
            raise RuntimeError("RuntimeSnapshot MCP registry 不属于 exact Root")
        if snapshot.managed_process_registry_identity != (
            None
            if snapshot.managed_process_registry is None
            else snapshot.managed_process_registry.identity
        ):
            raise RuntimeError(
                "RuntimeSnapshot managed process descriptor 在编译后发生变化"
            )
        if (
            snapshot.managed_process_registry is not None
            and snapshot.managed_process_registry.root_instance_token
            is not root.instance_token
        ):
            raise RuntimeError(
                "RuntimeSnapshot managed process registry 不属于 exact Root"
            )
        if snapshot.background_job_catalog_identity != (
            None
            if snapshot.background_job_catalog is None
            else snapshot.background_job_catalog.identity
        ):
            raise RuntimeError(
                "RuntimeSnapshot background job catalog 在编译后发生变化"
            )
        if (
            snapshot.background_job_catalog is not None
            and snapshot.background_job_catalog.root_instance_token
            is not root.instance_token
        ):
            raise RuntimeError(
                "RuntimeSnapshot background job catalog 不属于 exact Root"
            )
        if snapshot.plugin_tool_catalog_identity != (
            None
            if snapshot.plugin_tool_catalog is None
            else snapshot.plugin_tool_catalog.identity
        ):
            raise RuntimeError("RuntimeSnapshot plugin Tool catalog 在编译后发生变化")
        if (
            snapshot.plugin_tool_catalog is not None
            and snapshot.plugin_tool_catalog.root_instance_token
            is not root.instance_token
        ):
            raise RuntimeError("RuntimeSnapshot plugin Tool catalog 不属于 exact Root")
        topology = snapshot.composition_topology
        if topology is None:
            raise RuntimeError("RuntimeSnapshot composition Root 缺少 TopologyView")
        receipt = root.receipt()
        if receipt.incident_overflowed or receipt.external_effects:
            raise RuntimeError(
                "RuntimeSnapshot 插件组合拓扑未就绪: "
                f"required_pending={receipt.required_pending}, "
                f"required_degraded={receipt.required_degraded}, "
                f"incident_overflowed={receipt.incident_overflowed}, "
                f"external_effects={receipt.external_effects}"
            )
        if not receipt.ready:
            raise RuntimeError(
                "RuntimeSnapshot 插件组合拓扑未就绪: "
                f"required_pending={receipt.required_pending}, "
                f"required_degraded={receipt.required_degraded}, "
                f"incident_overflowed={receipt.incident_overflowed}, "
                f"external_effects={receipt.external_effects}"
            )
        if root.topology_identity() != topology.identity:
            raise RuntimeError("RuntimeSnapshot 插件组合拓扑在编译后发生变化")
        if root.composition_revision != topology.composition_revision:
            raise RuntimeError("RuntimeSnapshot 插件组合拓扑在编译后发生过结构变化")
        if require_validation:
            if (
                snapshot.composition_validation_identity is None
                or snapshot.composition_validation_root_token is None
            ):
                raise RuntimeError("RuntimeSnapshot 插件组合候选缺少 Core 验证回执")
            if (
                snapshot.composition_validation_root_token is root.instance_token
                and root.validation_identity()
                != snapshot.composition_validation_identity
            ):
                raise RuntimeError("RuntimeSnapshot 插件组合验证回执在封存后发生变化")

    def _selected(self, selector: RuntimeSelector) -> RuntimeSnapshot | None:
        if selector == "stable":
            return self._current
        if selector == "latest":
            return self.latest
        raise ValueError(f"未知 RuntimeSnapshot selector: {selector}")
