"""宿主能力装配；安装控制器只交入明确端口与实时只读事实。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from agent.control.frame_book import CONTROL_FRAMES, FrameBook
from agent.control.timer import AsyncioOneShotTimer
from agent.host_bridge.plugin_execution import (
    CodeOwner,
    ControllerAccess,
    ExecutionAccess,
)
from agent.plugin_composition import (
    INTERACTION_UNDO,
    TIMERS,
    CompositionError,
    CompositionRoot,
    InteractionUndoService,
    PluginTimers,
    ServiceKey,
)
from agent.plugin_composition.artifacts import (
    ARTIFACT_IMPORT,
    ARTIFACT_READ,
    ArtifactImport,
    ArtifactRead,
)
from agent.plugin_composition.bindings import BINDINGS, Bindings
from agent.plugin_composition.channel_io import (
    CHANNEL_ATTACHMENT_IMPORT,
    CHANNEL_ATTACHMENT_READ,
    CHANNEL_IDENTITY,
    INPUT_CUSTODY,
    ChannelAttachmentImport,
    ChannelAttachmentRead,
    ChannelIdentity,
    InputCustody,
    unavailable,
    unavailable_input_custody,
)
from agent.plugin_composition.context import Context
from agent.plugin_composition.credentials import CREDENTIALS, CredentialClients
from agent.plugin_composition.execution import EXECUTION, WORKLOAD_CONTROLLER
from agent.plugin_composition.host import HOST_INFO, HostInfo
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG,
    MESSAGE_EMBEDDINGS,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMISSION,
    MessageWriters,
    OwnerState,
    SessionAdmission,
)
from agent.plugin_composition.plugin_updates import (
    PLUGIN_UPDATES,
    PluginInstallPort,
    PluginUpdates,
)
from agent.plugin_composition.processes import PROCESSES, PluginProcesses
from agent.plugin_composition.requests import RequestContext
from agent.plugin_composition.runtime_catalog import (
    RUNTIME_CATALOG,
    RUNTIME_MCP_DETAIL,
    RuntimeCatalogUnavailable,
    build_runtime_catalog,
)
from agent.plugin_composition.tasks import TASKS, PluginTasks
from agent.plugin_composition.ui import DASHBOARD_ROUTES
from agent.plugin_contracts.ui import MESSAGE_DISPLAY, MOBILE_UI
from agent.plugins.archive import PluginArchive
from agent.plugins.channel_credentials import CoreProviderClientFactory
from agent.plugins.composable import ComposablePlugin
from agent.plugins.generation import PluginGeneration
from agent.plugins.interaction_undo import InteractionUndoCoordinator
from agent.restart import RESTART_GATE, RestartGate
from agent.workloads.client import WorkloadController
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.channels.attachment_import import ChannelOutboundAttachmentImporter
from session.embedding_store import MessageEmbeddings
from session.identities import ChannelIdentities, ChannelIdentityWriteReceipt
from session.log import MessageCatalog, MessageLog, MessagePage


async def provide_host_services(
    root: CompositionRoot,
    mount_order: tuple[PluginGeneration, ...],
    *,
    boot_id: str,
    dashboard_routes: tuple[object, ...] | None,
    input_custody: InputCustody | None,
    channel_identities: ChannelIdentities | None,
    attachments: ChannelAttachmentArtifactStore | None,
    resolve_command: Callable[
        [PluginGeneration, tuple[str, ...], str], tuple[str, ...]
    ],
    workload_controller: WorkloadController | None,
    workspace_id: str,
    message_log: MessageLog | None,
    archive: PluginArchive,
    generation_for_context: Callable[[Context], PluginGeneration],
    runtime_generations: Callable[
        [], tuple[Mapping[str, PluginGeneration], Mapping[str, list[PluginGeneration]]]
    ],
    live_root: Callable[[], CompositionRoot | None],
    installer: PluginInstallPort,
    tasks: PluginTasks,
    processes: PluginProcesses,
    restart_gate: RestartGate,
    control_frames: FrameBook,
    session_manager: Any,
) -> tuple[ExecutionAccess, CredentialClients]:
    """组装真实宿主端口与只读投影；不拥有安装选择或第二份运行状态。"""
    artifact_read = None if attachments is None else ArtifactRead(attachments.acquire)
    artifact_import = (
        None
        if attachments is None
        else ArtifactImport(
            ChannelOutboundAttachmentImporter(attachments).import_source
        )
    )
    interaction_undo = (
        None if session_manager is None else InteractionUndoCoordinator(session_manager)
    )

    def resolve_identity(channel: str, provider_identity: str) -> str | None:
        if channel_identities is None:
            raise RuntimeError("Channel identities 未绑定")
        return channel_identities.resolve(channel, provider_identity)

    async def remember_identity(
        channel: str, provider_identity: str, recipient: str
    ) -> ChannelIdentityWriteReceipt:
        if channel_identities is None:
            raise RuntimeError("Channel identities 未绑定")
        return channel_identities.remember(channel, provider_identity, recipient)

    async def rollback_identity(receipt: object) -> bool:
        if not isinstance(receipt, ChannelIdentityWriteReceipt):
            raise TypeError("channel identity rollback receipt 类型无效")
        if channel_identities is None:
            raise RuntimeError("Channel identities 未绑定")
        return channel_identities.rollback(receipt)

    await root.context.provide(
        HOST_INFO,
        HostInfo(boot_id=boot_id, validation=False),
    )
    await root.context.provide(
        DASHBOARD_ROUTES,
        () if dashboard_routes is None else dashboard_routes,
    )
    custody = input_custody
    await root.context.provide(
        INPUT_CUSTODY, unavailable_input_custody() if custody is None else custody
    )
    if channel_identities is None:
        identity = ChannelIdentity(unavailable, unavailable, unavailable)
    else:
        identity = ChannelIdentity(
            resolve_identity,
            remember_identity,
            rollback_identity,
        )
    await root.context.provide(CHANNEL_IDENTITY, identity)
    await root.context.provide(
        CHANNEL_ATTACHMENT_IMPORT,
        ChannelAttachmentImport(
            unavailable if attachments is None else attachments.import_bytes,
        ),
    )
    await root.context.provide(
        CHANNEL_ATTACHMENT_READ,
        ChannelAttachmentRead(
            unavailable if attachments is None else attachments.resolve_refs,
            unavailable if attachments is None else attachments.acquire,
        ),
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
    await root.context.provide(
        WORKLOAD_CONTROLLER,
        ControllerAccess(execution, workload_controller, workspace_id),
    )
    requested = {
        key
        for generation in mount_order
        for key in cast(ComposablePlugin, generation.instance).inject
    }
    # Host services remain available when a later local generation arrives.
    requested.update(
        {
            RUNTIME_CATALOG,
            RUNTIME_MCP_DETAIL,
            PLUGIN_UPDATES,
            RESTART_GATE,
            CONTROL_FRAMES,
            PROCESSES,
            TIMERS,
            MESSAGE_DISPLAY,
            MOBILE_UI,
        }
    )
    if artifact_import is not None:
        requested.add(ARTIFACT_IMPORT)
    if interaction_undo is not None:
        requested.add(INTERACTION_UNDO)
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
            context.require_runtime_owner(RUNTIME_CATALOG, read_runtime_catalog)
            return build_runtime_catalog(
                root,
                *runtime_generations(),
            )

        _ = await root.context.provide(RUNTIME_CATALOG, read_runtime_catalog)
    if RUNTIME_MCP_DETAIL in requested:
        if root is not live_root():
            raise RuntimeError("MCP detail 只在当前 live Root 提供")

        async def read_runtime_mcp_detail(
            context: Context | RequestContext,
            owner_id: str,
            name: str,
        ) -> list[dict[str, object]]:
            """Inspect one target under caller and contributor owner scopes."""
            from agent.plugin_composition.mcp_slots import MCP_SERVERS

            if isinstance(context, RequestContext):
                context = context._require_context(
                    RUNTIME_MCP_DETAIL, read_runtime_mcp_detail
                )
            if (
                root is not live_root()
                or context.root_instance_token is not root.instance_token
            ):
                raise RuntimeError("MCP detail 不属于当前 live Root")
            context.require_declared_runtime_owner(
                RUNTIME_MCP_DETAIL, read_runtime_mcp_detail
            )
            service = root.context.get(MCP_SERVERS)
            if service is None:
                raise RuntimeCatalogUnavailable(
                    "mcp_provider_unavailable", "MCP provider 尚未在当前 Root 提供"
                )
            if service.root_instance_token is not root.instance_token:
                raise RuntimeError("MCP provider 不属于当前 Root")
            return await service.inspect(
                context, read_runtime_mcp_detail, owner_id, name
            )

        _ = await root.context.provide(RUNTIME_MCP_DETAIL, read_runtime_mcp_detail)
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
    if PLUGIN_UPDATES in requested:
        _ = await root.context.provide(
            PLUGIN_UPDATES,
            PluginUpdates(installer),
        )
    message_services: set[ServiceKey[Any]] = {
        MESSAGE_CATALOG,
        MESSAGE_EMBEDDINGS,
        MESSAGE_WRITERS,
        OWNER_STATE,
        SESSION_ADMISSION,
        BINDINGS,
    }
    if RESTART_GATE in requested:
        _ = await root.context.provide(RESTART_GATE, restart_gate)
    if CONTROL_FRAMES in requested:
        _ = await root.context.provide(CONTROL_FRAMES, control_frames)
    # Host capabilities are owned by the live process, outside plugin dependencies.
    if requested & message_services and message_log is None:
        raise RuntimeError("消息能力需要 bootstrap 提供已迁移的 MessageLog")
    if message_log is not None:
        log = message_log
        _ = await root.context.provide(MESSAGE_CATALOG, MessageCatalog(log))
        _ = await root.context.provide(MESSAGE_EMBEDDINGS, MessageEmbeddings(log))
        _ = await root.context.provide(MESSAGE_WRITERS, MessageWriters(log))
        _ = await root.context.provide(OWNER_STATE, OwnerState(log))
        _ = await root.context.provide(SESSION_ADMISSION, SessionAdmission(log))
        _ = await root.context.provide(
            BINDINGS, Bindings(log, archive, root, generation_for_context)
        )
    if TASKS in requested or message_log is not None:
        _ = await root.context.provide(TASKS, tasks)
    if PROCESSES in requested:
        _ = await root.context.provide(PROCESSES, processes)
    if artifact_read is not None:
        _ = await root.context.provide(ARTIFACT_READ, artifact_read)
    if ARTIFACT_IMPORT in requested and artifact_import is not None:
        _ = await root.context.provide(ARTIFACT_IMPORT, artifact_import)
    if TIMERS in requested:
        _ = await root.context.provide(TIMERS, PluginTimers(AsyncioOneShotTimer()))

    # Client UI and message display are neutral projections.  The host
    # publishes stable names; each display request opens only its provider
    # Context scope while retaining the same live Root.
    host_ui_requested = {
        key.name
        for key in requested
        if key.name
        in {
            "core.message_display.v1",
            "core.mobile_ui.v1",
        }
    }
    if "core.message_display.v1" in host_ui_requested:
        from agent.plugin_composition.message_view import project_message_rows

        async def display_message_page(
            page: MessagePage,
            *,
            display_only: bool,
        ) -> list[dict[str, object]]:
            return await project_message_rows(
                root,
                page,
                display_only=display_only,
            )

        _ = await root.context.provide(
            MESSAGE_DISPLAY,
            display_message_page,
        )
    if "core.mobile_ui.v1" in host_ui_requested:
        from agent.plugins.mobile_ui import PluginMobileUiProvider

        mobile_ui = PluginMobileUiProvider(root)
        _ = await root.context.provide(
            MOBILE_UI,
            mobile_ui,
        )
        root._defer_internal_cleanup(  # pyright: ignore[reportPrivateUsage]
            "mobile_ui_provider.close",
            mobile_ui.aclose,
        )
    if any(
        INTERACTION_UNDO in cast(ComposablePlugin, item.instance).inject
        for item in mount_order
    ):
        if interaction_undo is None:
            raise RuntimeError("INTERACTION_UNDO 需要 Session owner")
        interaction_undo = InteractionUndoService(interaction_undo.undo_latest)
        _ = await root.context.provide(INTERACTION_UNDO, interaction_undo)
    return execution, clients


def check_host_dependencies(
    root: CompositionRoot, generations: tuple[PluginGeneration, ...]
) -> None:
    """只对实际请求且缺席的宿主能力失败，插件依赖由组合图负责。"""
    host_keys: set[ServiceKey[Any]] = {
        HOST_INFO,
        DASHBOARD_ROUTES,
        INPUT_CUSTODY,
        CHANNEL_IDENTITY,
        CHANNEL_ATTACHMENT_IMPORT,
        CHANNEL_ATTACHMENT_READ,
        EXECUTION,
        WORKLOAD_CONTROLLER,
        RUNTIME_CATALOG,
        RUNTIME_MCP_DETAIL,
        CREDENTIALS,
        PLUGIN_UPDATES,
        RESTART_GATE,
        CONTROL_FRAMES,
        MESSAGE_CATALOG,
        MESSAGE_EMBEDDINGS,
        MESSAGE_WRITERS,
        OWNER_STATE,
        SESSION_ADMISSION,
        BINDINGS,
        TASKS,
        PROCESSES,
        ARTIFACT_READ,
        ARTIFACT_IMPORT,
        TIMERS,
        INTERACTION_UNDO,
        MESSAGE_DISPLAY,
        MOBILE_UI,
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
