from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

import uvicorn

from agent.plugin_composition import MODEL_CALL_STATS, MODEL_CATALOG
from agent.plugin_composition.commands import COMMANDS
from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentReadLease,
    AttachmentRef,
    ChannelFactoryContext,
    ChannelPresentationPorts,
    ChannelReady,
    ChannelRuntimePorts,
    DeliveryStatus,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    StopReceipt,
)
from agent.plugin_composition.messages import MESSAGE_CATALOG

from .capabilities import (
    MESSAGE_DISPLAY,
    MOBILE_UI,
    MODEL_SELECTION,
    REPLY_STATUS,
    WEB_UI,
)
from .config import AkashicClientsConfig
from .attachments import AttachmentStore
from .web_chat import WebChatChannel
from .mobile_realtime.channel import MobileRealtimeChannel
from .mobile_realtime.gateway import MobileGatewayRuntime
from .mobile_realtime.gateway import build_mobile_gateway_runtime, build_mobile_gateway_server
from .chat_api import build_chat_server
from .runtime_inspection import ScopedRpcRuntimeInspection
from .services import (
    MessageCatalogPort,
    ModelCatalogReader,
    ModelSelectionReader,
    ModelStatsReader,
    ReplyStatusPort,
)
from .services import ArtifactReadLeasePort, MobileUiProvider, WebUiProvider
from agent.plugin_composition.message_view import MessageDisplayReader
from .scoped_capabilities import (
    ScopedCommandCatalog,
    ScopedMessageDisplay,
    ScopedMobileUiProvider,
    ScopedWebUiProvider,
    open_request_scope,
)


_SERVER_START_TIMEOUT_SECONDS = 10.0


class _ChannelArtifactReadLease:
    """Adapt a host bounded read lease to the client artifact protocol."""

    def __init__(self, ref: AttachmentRef, lease: AttachmentReadLease) -> None:
        self.ref = ref
        self._lease = lease

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        return await self._lease.read_bytes(max_bytes=max_bytes)

    async def read_chunk(self, *, offset: int, max_bytes: int) -> bytes:
        if offset < 0 or max_bytes <= 0:
            raise ValueError("artifact chunk 范围无效")
        return await self._lease.read_chunk(offset=offset, max_bytes=max_bytes)

    async def aclose(self) -> None:
        await self._lease.aclose()


class _ChannelArtifactStore:
    """Compose the host's import/read/resolution atoms for one binding."""

    def __init__(self, context: ChannelFactoryContext) -> None:
        if context.attachment_import is None or context.attachment_read is None:
            raise RuntimeError("akashic channel 缺少 attachment import/read port")
        self._import = context.attachment_import
        self._read = context.attachment_read
        if not callable(getattr(self._read, "resolve_refs", None)):
            raise TypeError("akashic channel attachment_read 缺少 resolve_refs(ids)")

    async def import_bytes(
        self,
        data: bytes,
        *,
        kind: object,
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef:
        if not isinstance(kind, AttachmentKind):
            raise TypeError("artifact kind 必须是 AttachmentKind")
        return await self._import.import_bytes(
            data,
            kind=kind,
            filename=filename,
            media_type=media_type,
        )

    def resolve_refs(self, artifact_ids: tuple[str, ...]) -> tuple[AttachmentRef, ...]:
        refs = cast(Any, self._read).resolve_refs(artifact_ids)
        if not isinstance(refs, tuple) or any(not isinstance(ref, AttachmentRef) for ref in refs):
            raise TypeError("artifact resolver 必须返回 AttachmentRef tuple")
        if tuple(ref.artifact_id for ref in refs) != artifact_ids:
            raise RuntimeError("artifact resolver 未保留请求顺序")
        return refs

    async def acquire(self, ref: AttachmentRef) -> ArtifactReadLeasePort:
        lease = await self._read.acquire(ref)
        return _ChannelArtifactReadLease(ref, lease)


async def _stop_started_children(
    children: Sequence[Any],
    *,
    primary: BaseException,
    message: str,
) -> None:
    """Stop every started child and preserve all rollback failures."""

    results = await asyncio.gather(
        *(child.stop() for child in reversed(children)),
        return_exceptions=True,
    )
    errors = tuple(result for result in results if isinstance(result, BaseException))
    if errors:
        raise BaseExceptionGroup(message, (primary, *errors))
    raise primary


def _close_children(
    children: Sequence[Any],
    *,
    message: str,
    primary: BaseException | None = None,
) -> None:
    """Close every child admission and preserve all failures."""

    errors: list[BaseException] = []
    for child in reversed(children):
        try:
            child.close_admission()
        except BaseException as error:
            errors.append(error)
    if errors:
        causes = (primary, *errors) if primary is not None else tuple(errors)
        raise BaseExceptionGroup(message, causes)
    if primary is not None:
        raise primary


@dataclass(slots=True)
class _ClientGeneration:
    """Hold only immutable plugin config until the exact channel is started."""

    config: AkashicClientsConfig
    workspace: Any
    # A generation can be rebound to a new snapshot while its plugin fiber
    # remains alive.  The binding token, rather than generation_id, owns one
    # adapter lifecycle.
    adapters: dict[str, "_GenerationAkashicAdapter"]

    def __init__(self, config: AkashicClientsConfig, workspace: Any) -> None:
        self.config = config
        self.workspace = workspace
        self.adapters = {}


_GENERATIONS: dict[str, _ClientGeneration] = {}


def register_generation(
    generation_id: str,
    config: AkashicClientsConfig,
    workspace: Any,
) -> None:
    """Register config only; service values are resolved in an exact request scope."""

    if generation_id in _GENERATIONS:
        raise RuntimeError(f"akashic clients generation 已注册: {generation_id}")
    _GENERATIONS[generation_id] = _ClientGeneration(config, workspace)


def unregister_generation(generation_id: str) -> None:
    """Release factory inputs after the exact generation has stopped."""

    state = _GENERATIONS.get(generation_id)
    if state is not None:
        active = tuple(adapter for adapter in state.adapters.values() if adapter.started)
        if active:
            raise RuntimeError("akashic clients generation 在 adapter 停止前被释放")
        if state.adapters:
            raise RuntimeError("akashic clients generation 仍保留未完成的 channel binding")
    _GENERATIONS.pop(generation_id, None)


def build_akashic_channel(
    context: ChannelFactoryContext,
) -> "_GenerationAkashicAdapter":
    """Build one binding adapter over the generation's ordinary client owners."""

    state = _GENERATIONS.get(context.generation_id)
    if state is None:
        raise RuntimeError(
            f"akashic clients 缺少 generation service binding: {context.generation_id}"
        )
    if context.binding_token in state.adapters:
        raise RuntimeError("同一 akashic clients binding token 不允许重复创建 channel")
    adapter = _GenerationAkashicAdapter(state, context)
    state.adapters[context.binding_token] = adapter
    return adapter


class _GenerationAkashicAdapter:
    """Own Web/Mobile providers while exposing one Core channel binding."""

    def __init__(
        self,
        state: _ClientGeneration,
        context: ChannelFactoryContext,
    ) -> None:
        self._state = state
        self._context = context
        self._binding_token = context.binding_token
        self._config = state.config
        self._workspace = state.workspace
        self._reply_status: ReplyStatusPort | None = None
        self._model_catalog_reader: ModelCatalogReader | None = None
        self._model_selection_reader: ModelSelectionReader | None = None
        self._model_stats_reader: ModelStatsReader | None = None
        self._runtime_inspection: ScopedRpcRuntimeInspection | None = None
        self._message_display: MessageDisplayReader = ScopedMessageDisplay(
            self._open_request_scope
        )
        self._mobile_ui_provider: MobileUiProvider = ScopedMobileUiProvider(
            self._open_request_scope
        )
        self._web_ui_provider: WebUiProvider = ScopedWebUiProvider(
            self._open_request_scope
        )
        self._command_catalog = ScopedCommandCatalog()
        self._artifact_store: _ChannelArtifactStore | None = None
        self._web = WebChatChannel("akashic") if state.config.web.enabled else None
        self._mobile: MobileRealtimeChannel | None = None
        self._mobile_runtime: MobileGatewayRuntime | None = None
        self._upload_store: AttachmentStore | None = None
        self._mobile_presentation: ChannelPresentationPorts | None = None
        self._web_adapter = (
            None if self._web is None else self._web.build_v3_adapter(context)
        )
        self._mobile_adapter: Any | None = None
        self._runtime_ports: ChannelRuntimePorts | None = None
        self._servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []
        self._started_children: list[Any] = []
        self._started = False
        self._stopped = False
        self._stopping = False

        if self._web is None and not state.config.mobile_realtime.enabled:
            raise ValueError("akashic channel 至少需要启用 Web 或 Mobile")

    @property
    def started(self) -> bool:
        return self._started and not self._stopped

    async def _resolve_capabilities(self) -> None:
        """Validate declared capabilities without retaining provider objects."""

        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic channel 缺少 host request scope")
        self._reply_status = self._follow_reply_status
        self._model_catalog_reader = self._read_model_catalog
        self._model_selection_reader = self._read_model_selection
        self._model_stats_reader = self._read_model_stats
        self._runtime_inspection = ScopedRpcRuntimeInspection(open_scope)
        async with self._open_request_scope() as scope:
            for key in (
                MESSAGE_CATALOG,
                COMMANDS,
                MESSAGE_DISPLAY,
                MOBILE_UI,
                WEB_UI,
            ):
                _ = scope.require(key)
        self._artifact_store = _ChannelArtifactStore(self._context)
        if self._context.data_root is None:
            raise RuntimeError("akashic clients 缺少 plugin data root")
        self._upload_store = AttachmentStore(self._context.data_root / "uploads")
        if self._web is not None:
            self._web.bind_message_scope(
                self._message_scope,
                reply_status=self._reply_status,
            )
            self._web.bind_message_display(self._message_display)

    @asynccontextmanager
    async def _open_request_scope(self) -> AsyncIterator[Any]:
        """Open one exact binding scope for a short client operation."""

        opener = self._context.open_scope
        if opener is None:
            raise RuntimeError("akashic channel 缺少 host request scope")
        async with open_request_scope(opener) as scope:
            yield scope

    @asynccontextmanager
    async def _message_scope(self) -> AsyncIterator[MessageCatalogPort]:
        """Resolve the message catalog only for one HTTP or WebSocket operation."""

        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic message catalog 缺少 host request scope")
        async with self._open_request_scope() as scope:
            yield cast(MessageCatalogPort, scope.require(MESSAGE_CATALOG))

    @asynccontextmanager
    async def _mobile_ui_scope(self) -> AsyncIterator[MobileUiProvider]:
        """Expose one exact Mobile UI provider to an HTTP operation."""

        async with self._open_request_scope() as scope:
            yield cast(MobileUiProvider, scope.require(MOBILE_UI))

    def _read_command_catalog(self) -> tuple[tuple[str, str], ...]:
        """Build the command projection from the active request scope."""

        return self._command_catalog()

    async def _follow_reply_status(self, session_id: str):
        """Keep the reply status read inside this subscription's exact scope."""

        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic reply status 缺少 host request scope")
        async with open_scope() as scope:
            reader = cast(ReplyStatusPort, scope.require(REPLY_STATUS))
            async for frame in reader.follow(session_id):
                yield frame

    async def _read_model_catalog(self) -> Any:
        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic model catalog 缺少 host request scope")
        async with open_scope() as scope:
            return scope.require(MODEL_CATALOG).snapshot()

    async def _read_model_selection(self, metadata: dict[str, object]) -> Any:
        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic model selection 缺少 host request scope")
        async with open_scope() as scope:
            return scope.require(MODEL_SELECTION).read_saved(metadata)

    async def _read_model_stats(self, call_id: str) -> Any:
        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic model stats 缺少 host request scope")
        async with open_scope() as scope:
            return scope.require(MODEL_CALL_STATS)(call_id)

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None:
        if self._stopped:
            raise RuntimeError("akashic channel 已停止")
        if self._runtime_ports is not None:
            raise RuntimeError("akashic channel runtime 不允许替换")
        self._runtime_ports = ports
        if self._web_adapter is not None:
            self._web_adapter.attach_runtime(ports)
        if self._mobile_adapter is not None:
            self._mobile_adapter.attach_runtime(ports)

    def attach_presentation(self, ports: ChannelPresentationPorts) -> None:
        """Bind the exact turn stream to the enabled transport owner."""

        if ports.turn_stream is None:
            raise RuntimeError("akashic channel 缺少 turn stream port")
        if self._mobile_presentation is not None:
            raise RuntimeError("akashic Mobile presentation 不允许替换")
        if self._web is not None:
            self._web.attach_presentation(ports)
        self._mobile_presentation = ports
        if self._mobile is not None:
            self._mobile.attach_presentation(ports)

    async def _start_server(self, server: uvicorn.Server, *, name: str) -> None:
        spawn_owned = self._context.spawn_owned
        if spawn_owned is None:
            raise RuntimeError(f"akashic {name} 缺少 host-owned task scope")
        task = await spawn_owned(server.serve(), name=name)
        try:
            async with asyncio.timeout(_SERVER_START_TIMEOUT_SECONDS):
                while True:
                    if server.started:
                        break
                    if task.done():
                        task.result()
                        raise RuntimeError(f"akashic {name} 在监听就绪前退出")
                    await asyncio.sleep(0)
        except BaseException as error:
            server.should_exit = True
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        f"akashic {name} 启动失败且清理失败",
                        (error, cleanup_error),
                    ) from error
            raise
        self._servers.append((server, task))

    async def _start_web(self) -> None:
        if self._web is None or self._web_adapter is None:
            return
        await self._web.start()
        self._started_children.append(self._web)
        _ = await self._web_adapter.start()
        self._started_children.append(self._web_adapter)
        socket_path = self._config.web.socket_path or str(
            self._workspace / "runtime" / "chat.sock"
        )
        artifact_store = self._artifact_store
        if artifact_store is None:
            raise RuntimeError("akashic Web 缺少 artifact store")
        server = build_chat_server(
            workspace=self._workspace,
            channel=self._web,
            runtime_inspection=self._runtime_inspection,
            model_catalog_reader=self._model_catalog_reader,
            model_selection_reader=self._model_selection_reader,
            message_display=self._message_display,
            mobile_ui_scope=self._mobile_ui_scope,
            web_ui_provider=self._web_ui_provider,
            attachment_store=self._upload_store,
            artifact_store=artifact_store,
            mobile_pairing_admin=(
                None if self._mobile_runtime is None else self._mobile_runtime.admin
            ),
            reply_status=self._reply_status,
            message_scope=self._message_scope,
            uds=socket_path,
        )
        await self._start_server(server, name="akashic-web")

    async def _start_mobile(self) -> None:
        config = self._config.mobile_realtime
        if not config.enabled:
            return
        if self._runtime_ports is None or self._runtime_ports.durable_inbound is None:
            raise RuntimeError("akashic Mobile 缺少 durable inbound host port")
        if self._mobile_runtime is None or self._mobile is None:
            raise RuntimeError("akashic Mobile runtime 尚未准备")
        if self._mobile_adapter is None:
            raise RuntimeError("akashic Mobile binding adapter 尚未准备")
        upload_store = self._upload_store
        if upload_store is None:
            raise RuntimeError("akashic Mobile 缺少 upload store")
        _keyset = self._mobile_runtime.keyset
        _ = await self._mobile_adapter.start()
        self._started_children.append(self._mobile_adapter)
        await self._mobile.start(
            host_boot_id=self._context.boot_id,
            durable_inbound=self._runtime_ports.durable_inbound,
            attachment_store=upload_store,
        )
        server = build_mobile_gateway_server(self._mobile_runtime, _keyset)
        await self._start_server(server, name="akashic-mobile")

    def _prepare_mobile(self) -> None:
        """Construct Mobile storage before Web routes expose pairing operations."""

        if not self._config.mobile_realtime.enabled:
            return
        if self._context.data_root is None:
            raise RuntimeError("akashic Mobile 缺少 plugin data root")
        self._mobile_runtime, _keyset = build_mobile_gateway_runtime(
            self._config.mobile_realtime,
            self._workspace,
        )
        self._mobile = self._mobile_runtime.channel
        self._mobile_adapter = self._mobile.build_v3_adapter(self._context)
        if self._runtime_ports is not None:
            self._mobile_adapter.attach_runtime(self._runtime_ports)
        if self._runtime_inspection is None or self._model_catalog_reader is None:
            raise RuntimeError("akashic Mobile capability 尚未解析")
        if self._artifact_store is None:
            raise RuntimeError("akashic Mobile capability 尚未解析")
        self._mobile.bind_message_scope(self._message_scope, self._reply_status)
        self._mobile.bind_message_display(self._message_display)
        self._mobile.bind_runtime_inspection(self._runtime_inspection)
        self._mobile.bind_model_catalog(self._model_catalog_reader)
        model_selection_reader = self._model_selection_reader
        model_stats_reader = self._model_stats_reader
        if model_selection_reader is None or model_stats_reader is None:
            raise RuntimeError("akashic Mobile 缺少 model reader")
        self._mobile.bind_model_selection(model_selection_reader)
        self._mobile.bind_model_stats(model_stats_reader)
        self._mobile.bind_mobile_ui_provider(
            self._mobile_ui_provider,
            scope=self._mobile_ui_scope,
        )
        self._mobile.bind_channel_attachment_store(self._artifact_store)
        self._mobile.bind_command_catalog(self._read_command_catalog)
        if self._mobile_presentation is not None:
            self._mobile.attach_presentation(self._mobile_presentation)

    async def start(self) -> ChannelReady:
        if self._started:
            raise RuntimeError("akashic channel 重复 start")
        if self._stopped:
            raise RuntimeError("akashic channel 已停止")
        try:
            await self._resolve_capabilities()
            self._prepare_mobile()
            await self._start_web()
            await self._start_mobile()
        except BaseException as error:
            await self._rollback_start(error)
        self._started = True
        return ChannelReady(self._binding_token)

    async def _rollback_start(self, primary: BaseException) -> None:
        failures: list[BaseException] = []
        for server, task in reversed(self._servers):
            server.should_exit = True
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                failures.append(error)
        self._servers.clear()
        results = await asyncio.gather(
            *(child.stop() for child in reversed(self._started_children)),
            return_exceptions=True,
        )
        failures.extend(item for item in results if isinstance(item, BaseException))
        if self._web is not None and self._web not in self._started_children:
            try:
                await self._web.stop()
            except BaseException as error:
                failures.append(error)
        self._started_children.clear()
        if self._mobile_runtime is not None:
            try:
                await self._mobile_runtime.stop()
            except BaseException as error:
                failures.append(error)
        if failures:
            self._stopping = False
            raise BaseExceptionGroup("akashic channel start rollback 失败", (primary, *failures))
        self._stopped = True
        self._stopping = False
        self._release_generation_binding()
        raise primary

    def open_admission(self) -> None:
        children: list[Any] = [
            child
            for child in (self._web_adapter, self._mobile_adapter)
            if child is not None
        ]
        opened: list[Any] = []
        try:
            for child in children:
                child.open_admission()
                opened.append(child)
        except BaseException as error:
            _close_children(opened, primary=error, message="akashic channel admission rollback 失败")

    def close_admission(self) -> None:
        children = [child for child in (self._web_adapter, self._mobile_adapter) if child is not None]
        _close_children(children, message="akashic channel admission close 失败")

    async def deliver(self, request: ProviderDeliveryRequest) -> ProviderDeliveryReceipt:
        children = [child for child in (self._web_adapter, self._mobile_adapter) if child is not None]
        results = await asyncio.gather(
            *(child.deliver(request) for child in children),
            return_exceptions=True,
        )
        receipts = tuple(item for item in results if isinstance(item, ProviderDeliveryReceipt))
        errors = [str(item) for item in results if isinstance(item, BaseException)]
        errors.extend(receipt.error for receipt in receipts if receipt.error is not None)
        provider_ids = tuple(dict.fromkeys(
            provider_id for receipt in receipts for provider_id in receipt.provider_ids
        ))
        if any(isinstance(item, BaseException) or item.status is DeliveryStatus.FAILED for item in results):
            return ProviderDeliveryReceipt(
                request.delivery_id,
                DeliveryStatus.FAILED,
                provider_ids=provider_ids,
                error="; ".join(errors) or "akashic channel delivery failed",
            )
        status = (
            DeliveryStatus.DELIVERED
            if any(receipt.status is DeliveryStatus.DELIVERED for receipt in receipts)
            else DeliveryStatus.REJECTED
        )
        return ProviderDeliveryReceipt(
            request.delivery_id,
            status,
            provider_ids=provider_ids,
            error=None if status is DeliveryStatus.DELIVERED else "; ".join(errors),
        )

    async def stop(self) -> StopReceipt:
        if self._stopped:
            return StopReceipt(self._binding_token, resources_closed=True)
        self._stopping = True
        errors: list[BaseException] = []
        remaining_servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []
        for server, task in reversed(self._servers):
            server.should_exit = True
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                errors.append(error)
                remaining_servers.append((server, task))
        self._servers = list(reversed(remaining_servers))
        receipts: list[StopReceipt] = []
        started_children = tuple(self._started_children)
        remaining_children: list[Any] = []
        for child in reversed(self._started_children):
            try:
                result = await child.stop()
            except BaseException as error:
                errors.append(error)
                remaining_children.append(child)
            else:
                if isinstance(result, StopReceipt):
                    receipts.append(result)
                    if not result.resources_closed or result.failures:
                        remaining_children.append(child)
                        errors.append(
                            RuntimeError(
                                f"akashic child stop 未完成: {result.binding_token}"
                            )
                        )
        self._started_children = list(reversed(remaining_children))
        if self._web is not None and self._web not in started_children:
            try:
                await self._web.stop()
            except BaseException as error:
                errors.append(error)
        self._started_children.clear()
        if self._mobile_runtime is not None:
            try:
                await self._mobile_runtime.stop()
            except BaseException as error:
                errors.append(error)
        if errors:
            self._stopping = False
            raise BaseExceptionGroup("akashic channel stop 失败", tuple(errors))
        self._stopped = True
        self._stopping = False
        self._release_generation_binding()
        return StopReceipt(
            self._binding_token,
            resources_closed=all(receipt.resources_closed for receipt in receipts),
            failures=tuple(failure for receipt in receipts for failure in receipt.failures),
        )

    def _release_generation_binding(self) -> None:
        """Release this exact adapter only after every owned resource settled."""

        current = self._state.adapters.get(self._binding_token)
        if current is self:
            self._state.adapters.pop(self._binding_token, None)


__all__ = [
    "build_akashic_channel",
    "register_generation",
    "unregister_generation",
]
