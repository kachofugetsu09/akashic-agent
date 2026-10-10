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
from agent.plugin_composition.artifacts import ARTIFACT_IMPORT, ARTIFACT_READ
from session.artifact_services import ArtifactImport, ArtifactRead
from agent.plugin_composition.bindings import BINDINGS
from session.bindings import Bindings
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
from agent.plugin_composition.execution import EXECUTION
from agent.plugin_composition.host import HOST_INFO, HostInfo
from session.services import MessageWriters, OwnerState, SessionAdmin, SessionAdmission
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG,
    MESSAGE_EMBEDDINGS,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMIN,
    SESSION_ADMISSION,
)
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
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.channels.attachment_import import ChannelOutboundAttachmentImporter
from session.embedding_store import MessageEmbeddings
from session.identities import ChannelIdentities, ChannelIdentityWriteReceipt
from session.log import MessageCatalog, MessageLog


async def provide_host_services(
    root: CompositionRoot,
    mount_order: tuple[PluginGeneration, ...],
    *,
    boot_id: str,
    input_custody: InputCustody | None,
    channel_identities: ChannelIdentities | None,
    attachments: ChannelAttachmentArtifactStore | None,
    resolve_command: Callable[
        [PluginGeneration, tuple[str, ...], str], tuple[str, ...]
    ],
    message_log: MessageLog | None,
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
    artifact_read = None if attachments is None else ArtifactRead(attachments.acquire)
    artifact_import = (
        None
        if attachments is None
        else ArtifactImport(
            ChannelOutboundAttachmentImporter(attachments).import_source
        )
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
        HostInfo(boot_id=boot_id, validation=False, ready=host_ready or (lambda: True)),
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
    if artifact_import is not None:
        requested.add(ARTIFACT_IMPORT)
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
    message_services: set[ServiceKey[Any]] = {
        MESSAGE_CATALOG,
        MESSAGE_EMBEDDINGS,
        MESSAGE_WRITERS,
        OWNER_STATE,
        SESSION_ADMIN,
        SESSION_ADMISSION,
        BINDINGS,
    }
    if RESTART_GATE in requested:
        _ = await root.context.provide(RESTART_GATE, restart_gate)
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
        _ = await root.context.provide(SESSION_ADMIN, SessionAdmin(log))
        _ = await root.context.provide(
            BINDINGS, Bindings(log, root.context)
        )
    if TASKS in requested or message_log is not None:
        _ = await root.context.provide(TASKS, tasks)
    if artifact_read is not None:
        _ = await root.context.provide(ARTIFACT_READ, artifact_read)
    if ARTIFACT_IMPORT in requested and artifact_import is not None:
        _ = await root.context.provide(ARTIFACT_IMPORT, artifact_import)

    return execution, clients


def check_host_dependencies(
    root: CompositionRoot, generations: tuple[PluginGeneration, ...]
) -> None:
    """只对实际请求且缺席的宿主能力失败，插件依赖由组合图负责。"""
    host_keys: set[ServiceKey[Any]] = {
        HOST_INFO,
        INPUT_CUSTODY,
        CHANNEL_IDENTITY,
        CHANNEL_ATTACHMENT_IMPORT,
        CHANNEL_ATTACHMENT_READ,
        EXECUTION,
        RUNTIME_CATALOG,
        CREDENTIALS,
        PLUGIN_UPDATES,
        PLUGIN_CONFIG,
        RESTART_GATE,
        MESSAGE_CATALOG,
        MESSAGE_EMBEDDINGS,
        MESSAGE_WRITERS,
        OWNER_STATE,
        SESSION_ADMIN,
        SESSION_ADMISSION,
        BINDINGS,
        TASKS,
        ARTIFACT_READ,
        ARTIFACT_IMPORT,
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
