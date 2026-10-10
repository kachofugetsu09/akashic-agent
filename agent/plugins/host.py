"""宿主能力装配；安装控制器只交入明确端口与实时只读事实。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from agent.plugins.execution import (
    CodeOwner,
    ExecutionAccess,
)
from agent.plugin_composition import (
    CompositionError,
    CompositionRoot,
    ServiceKey,
)
from agent.plugin_composition.context import Context
from agent.plugin_composition.credentials import CREDENTIALS, CredentialClients
from agent.plugin_composition.execution import EXECUTION
from agent.plugin_composition.host import HOST_INFO, HostInfo
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG, PluginConfig
from agent.plugin_composition.plugin_updates import (
    PLUGIN_UPDATES,
    PluginInstallPort,
    PluginUpdates,
)
from agent.plugin_composition.requests import RequestContext
from agent.plugin_composition.runtime_catalog import (
    RUNTIME_CATALOG,
    build_runtime_catalog,
)
from agent.plugin_composition.tasks import TASKS, PluginTasks
from agent.plugins.channel_credentials import CoreProviderClientFactory
from agent.plugins.composable import ComposablePlugin
from agent.plugins.generation import PluginGeneration
from agent.restart import RESTART_GATE, RestartGate


async def provide_host_services(
    root: CompositionRoot,
    mount_order: tuple[PluginGeneration, ...],
    *,
    boot_id: str,
    resolve_command: Callable[
        [PluginGeneration, tuple[str, ...], str], tuple[str, ...]
    ],
    runtime_generations: Callable[
        [], tuple[Mapping[str, PluginGeneration], Mapping[str, list[PluginGeneration]]]
    ],
    runtime_updating: Callable[[], bool],
    live_root: Callable[[], CompositionRoot | None],
    installer: PluginInstallPort,
    tasks: PluginTasks,
    restart_gate: RestartGate,
    host_ready: Callable[[], bool] | None,
) -> tuple[ExecutionAccess, CredentialClients]:
    """组装真实宿主端口与只读投影；不拥有安装选择或第二份运行状态。"""
    await root.context.provide(
        HOST_INFO,
        HostInfo(boot_id=boot_id, validation=False, ready=host_ready or (lambda: True)),
    )
    execution = ExecutionAccess(
        root.instance_token,
        {
            (item.plugin_id, item.generation_id): CodeOwner(
                item.generation_id,
                item.code_dir,
                lambda command, cwd, item=item: resolve_command(item, command, cwd),
            )
            for item in mount_order
        },
        candidate=False,
    )
    await root.context.provide(EXECUTION, execution)
    requested = {
        key
        for generation in mount_order
        for key in cast(ComposablePlugin, generation.instance).inject
    }
    # Host services remain available when a later local generation arrives.
    requested.update(
        {
            RUNTIME_CATALOG,
            PLUGIN_UPDATES,
            RESTART_GATE,
        }
    )
    if RUNTIME_CATALOG in requested:
        if root is not live_root():
            raise RuntimeError("runtime catalog 只在当前 live Root 提供")

        def read_runtime_catalog(
            context: Context | RequestContext,
        ) -> dict[str, object]:
            """Read live runtime facts only from the exact owner scope."""

            if isinstance(context, RequestContext):
                context = context._require_context(
                    RUNTIME_CATALOG, read_runtime_catalog
                )
            if (
                root is not live_root()
                or context.root_instance_token is not root.instance_token
            ):
                raise RuntimeError("runtime catalog 不属于当前 live Root")
            if RUNTIME_CATALOG not in context._declared_dependencies():
                raise CompositionError(
                    "UNDECLARED_SERVICE",
                    "当前 Fiber 未声明 runtime catalog 依赖",
                )
            context.require_runtime_identity(RUNTIME_CATALOG, read_runtime_catalog)
            catalog = build_runtime_catalog(root, *runtime_generations())
            catalog["updating"] = runtime_updating()
            return catalog

        _ = await root.context.provide(RUNTIME_CATALOG, read_runtime_catalog)
    clients = CredentialClients(
        {
            (generation.plugin_id, generation.generation_id): CoreProviderClientFactory(
                generation.data_dir,
                generation.config_projection,
                generation.config_revision,
            )
            for generation in mount_order
        }
    )
    _ = await root.context.provide(CREDENTIALS, clients)
    root._defer_internal_cleanup("credential_clients", clients.aclose)  # pyright: ignore[reportPrivateUsage]
    _ = await root.context.provide(PLUGIN_CONFIG, PluginConfig(installer))
    if PLUGIN_UPDATES in requested:
        _ = await root.context.provide(
            PLUGIN_UPDATES,
            PluginUpdates(installer),
        )
    _ = await root.context.provide(RESTART_GATE, restart_gate)
    _ = await root.context.provide(TASKS, tasks)

    return execution, clients


def check_host_dependencies(
    root: CompositionRoot, generations: tuple[PluginGeneration, ...]
) -> None:
    """只对实际请求且缺席的宿主能力失败，插件依赖由组合图负责。"""
    host_keys: set[ServiceKey[Any]] = {
        HOST_INFO, EXECUTION, RUNTIME_CATALOG, CREDENTIALS,
        PLUGIN_UPDATES, PLUGIN_CONFIG, RESTART_GATE, TASKS,
    }
    for generation in generations:
        plugin = cast(ComposablePlugin, generation.instance)
        for key in plugin.inject:
            # Unknown keys may be provided by another plugin Fiber; let the kernel
            # report PENDING. Only known host-owned capabilities are a migration gate.
            if key not in host_keys or root.context.get(key) is not None:
                continue
            raise RuntimeError(
                f"宿主能力尚未迁移，阻止启用 {generation.plugin_id}: {key.name}"
            )
