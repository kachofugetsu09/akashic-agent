from __future__ import annotations

from agent.plugin_composition.models import MODEL_CALL_STATS, ModelCallStats
from agent.plugin_composition.model_settings_http import ModelControlUnavailable

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import uvicorn

from agent.plugin_composition import MODEL_CATALOG
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
    PLUGIN_UI,
    MODEL_SELECTION,
    REPLY_STATUS,
    WEB_UI,
)
from .config import AkashicClientsConfig
from .attachments import AttachmentStore
from .web_chat import WebChatChannel
from .chat_api import build_chat_server
from .runtime_inspection import ScopedRpcRuntimeInspection
from .services import (
    MessageCatalogPort,
    ModelCatalogReader,
    ModelSelectionReader,
    ReplyStatusPort,
)
from .services import ArtifactReadLeasePort, PluginUiProvider, WebUiProvider
from agent.plugin_composition.message_view import MessageDisplayReader
from .scoped_capabilities import (
    ScopedMessageDisplay,
    ScopedPluginUiProvider,
    ScopedWebUiProvider,
    open_request_scope,
)


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


async def _stop_owned_children(
    children: Sequence[Any],
) -> tuple[list[Any], list[StopReceipt], list[BaseException]]:
    """Stop children while retaining every incomplete owner for retry."""

    receipts: list[StopReceipt] = []
    remaining: list[Any] = []
    errors: list[BaseException] = []
    for child in reversed(children):
        try:
            result = await child.stop()
        except BaseException as error:
            errors.append(error)
            remaining.append(child)
            continue
        if result is None:
            continue
        if not isinstance(result, StopReceipt):
            errors.append(RuntimeError("akashic child stop 必须返回 StopReceipt 或 None"))
            remaining.append(child)
            continue
        receipts.append(result)
        if not result.resources_closed or result.failures:
            remaining.append(child)
            errors.append(
                RuntimeError(
                    f"akashic child stop 未完成: {result.binding_token}"
                )
            )
    return list(reversed(remaining)), receipts, errors


async def _stop_server(server: uvicorn.Server, task: asyncio.Task[Any]) -> None:
    """等待 Uvicorn 完成启动和关闭，不能把取消 Task 当成资源关闭。"""

    server.should_exit = True
    # serve 在 startup 中取消会跳过 shutdown；调用方取消只中断等待，owner 留待重试。
    await asyncio.shield(task)


@dataclass(slots=True)
class _ClientGeneration:
    """Hold only immutable plugin config until the exact channel is started."""

    config: AkashicClientsConfig
    workspace: Any
    # 固定输入与实际 adapter 只属于本次 apply。
    adapters: dict[str, "_GenerationAkashicAdapter"]

    def __init__(self, config: AkashicClientsConfig, workspace: Any) -> None:
        self.config = config
        self.workspace = workspace
        self.adapters = {}


def build_akashic_channel_factory(config: AkashicClientsConfig, workspace: Any):
    """为本次 apply 保留固定输入，实际 binding 由贡献 Scope 关闭。"""
    state = _ClientGeneration(config, workspace)

    def build(context: ChannelFactoryContext) -> _GenerationAkashicAdapter:
        if state.adapters:
            raise RuntimeError("本次 apply 的 Channel binding 尚未释放")
        adapter = _GenerationAkashicAdapter(state, context)
        state.adapters[context.binding_token] = adapter
        return adapter

    return build


class _GenerationAkashicAdapter:
    """Own the Web client while exposing one Core channel binding."""

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
        self._runtime_inspection: ScopedRpcRuntimeInspection | None = None
        self._message_display: MessageDisplayReader = ScopedMessageDisplay(
            self._open_request_scope
        )
        self._plugin_ui_provider: PluginUiProvider = ScopedPluginUiProvider(
            self._open_request_scope
        )
        self._web_ui_provider: WebUiProvider = ScopedWebUiProvider(
            self._open_request_scope
        )
        self._artifact_store: _ChannelArtifactStore | None = None
        self._web = WebChatChannel("akashic")
        self._upload_store: AttachmentStore | None = None
        self._web_adapter = self._web.build_v3_adapter(context)
        self._runtime_ports: ChannelRuntimePorts | None = None
        self._servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []
        self._started_children: list[Any] = []
        self._socket_nodes: list[tuple[Path, int, int]] = []
        self._started = False
        self._stopped = False
        self._stopping = False

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
        self._runtime_inspection = ScopedRpcRuntimeInspection(open_scope)
        async with self._open_request_scope() as scope:
            for key in (
                MESSAGE_CATALOG,
                MESSAGE_DISPLAY,
                PLUGIN_UI,
                WEB_UI,
            ):
                _ = scope.require(key)
        self._artifact_store = _ChannelArtifactStore(self._context)
        if self._context.data_root is None:
            raise RuntimeError("akashic clients 缺少 plugin data root")
        self._upload_store = AttachmentStore(self._context.data_root / "uploads")
        self._web.bind_message_scope(
            self._message_scope,
            reply_status=self._reply_status,
        )
        self._web.bind_message_display(self._message_display)

    @asynccontextmanager
    async def _open_request_scope(self) -> AsyncIterator[Any]:
        """Open one exact binding scope for a short client operation."""

        if self._stopping or self._stopped:
            raise RuntimeError("akashic channel 已关闭请求接纳")
        opener = self._context.open_scope
        if opener is None:
            raise RuntimeError("akashic channel 缺少 host request scope")
        async with open_request_scope(opener) as scope:
            yield scope

    @asynccontextmanager
    async def _message_scope(self) -> AsyncIterator[MessageCatalogPort]:
        """Resolve the message catalog only for one short reader acquisition."""

        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic message catalog 缺少 host request scope")
        async with self._open_request_scope() as scope:
            yield cast(MessageCatalogPort, scope.require(MESSAGE_CATALOG))

    @asynccontextmanager
    async def _plugin_ui_scope(self) -> AsyncIterator[PluginUiProvider]:
        """Expose one exact Plugin UI provider to an HTTP operation."""

        async with self._open_request_scope() as scope:
            yield cast(PluginUiProvider, scope.require(PLUGIN_UI))

    async def _follow_reply_status(self, session_id: str):
        """Acquire the reply reader briefly, then own its long follow locally."""

        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic reply status 缺少 host request scope")
        async with self._open_request_scope() as scope:
            reader = cast(ReplyStatusPort, scope.require(REPLY_STATUS))
        async with aclosing(reader.follow(session_id)) as frames:
            async for frame in frames:
                if isinstance(frame, Mapping):
                    yield dict(frame)
                    continue
                if not isinstance(frame, (tuple, list)):
                    raise TypeError("reply status provider 必须返回 mapping 或 item sequence")
                if any(not isinstance(item, Mapping) for item in frame):
                    raise TypeError("reply status item 必须是 mapping")
                yield {
                    "version": 2,
                    "session_id": session_id,
                    "snapshot_id": self._context.snapshot_id,
                    "available": True,
                    "items": [dict(item) for item in frame],
                }

    async def _read_model_call_stats(self, call_id: str) -> ModelCallStats:
        """借用实际统计 owner；缺席不关闭聊天，也不获取模型修改权限。"""
        async with self._open_request_scope() as scope:
            with scope.borrow(MODEL_CALL_STATS) as reader:
                if reader is None:
                    raise ModelControlUnavailable("模型调用统计不可用")
                return reader(call_id)

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

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None:
        if self._stopped:
            raise RuntimeError("akashic channel 已停止")
        if self._runtime_ports is not None:
            raise RuntimeError("akashic channel runtime 不允许替换")
        self._runtime_ports = ports
        self._web_adapter.attach_runtime(ports)

    def attach_presentation(self, ports: ChannelPresentationPorts) -> None:
        """Bind the exact turn stream to the enabled transport owner."""

        if ports.turn_stream is None:
            raise RuntimeError("akashic channel 缺少 turn stream port")
        self._web.attach_presentation(ports)

    async def _start_server(self, server: uvicorn.Server, *, name: str) -> None:
        """先登记监听 owner，再等待真实就绪或启动失败。"""
        spawn_owned = self._context.spawn_owned
        if spawn_owned is None:
            raise RuntimeError(f"akashic {name} 缺少 host-owned task scope")
        task = await spawn_owned(server.serve(), name=name)
        self._servers.append((server, task))
        # 本地空 lifespan 与 socket bind 不需要时钟期限；其他插件占用事件循环不是启动失败。
        while True:
            if task.done():
                task.result()
                raise RuntimeError(f"akashic {name} 在监听就绪前退出")
            if server.started:
                return
            await asyncio.sleep(0)

    async def _start_web(self) -> None:
        await self._web.start()
        self._started_children.append(self._web)
        _ = await self._web_adapter.start()
        self._started_children.append(self._web_adapter)
        public_path = (self._workspace / "runtime" / "web-chat.sock").absolute()
        socket_path = Path(self._config.web.socket_path).absolute() if self._config.web.socket_path else public_path
        public_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path != public_path and (public_path.exists() or public_path.is_symlink()):
            raise FileExistsError(f"聊天公共 socket 路径已被占用: {public_path}")
        artifact_store = self._artifact_store
        if artifact_store is None:
            raise RuntimeError("akashic Web 缺少 artifact store")
        server = build_chat_server(
            workspace=self._workspace,
            channel=self._web,
            runtime_inspection=self._runtime_inspection,
            model_catalog_reader=self._model_catalog_reader,
            model_call_stats_reader=self._read_model_call_stats,
            model_selection_reader=self._model_selection_reader,
            message_display=self._message_display,
            plugin_ui_scope=self._plugin_ui_scope,
            web_ui_provider=self._web_ui_provider,
            attachment_store=self._upload_store,
            artifact_store=artifact_store,
            reply_status=self._reply_status,
            message_scope=self._message_scope,
            uds=str(socket_path),
        )
        await self._start_server(server, name="akashic-web")
        node = socket_path.lstat()
        self._socket_nodes.append((socket_path, node.st_dev, node.st_ino))
        if socket_path != public_path:
            # listener 就绪后才发布；原子创建拒绝覆盖其他 owner 的节点。
            public_path.symlink_to(socket_path)
            node = public_path.lstat()
            self._socket_nodes.append((public_path, node.st_dev, node.st_ino))

    def _close_socket_nodes(self) -> None:
        """listener 排空后只移除本次创建且身份未变的临时节点。"""
        while self._socket_nodes:
            path, device, inode = self._socket_nodes[-1]
            try:
                node = path.lstat()
            except FileNotFoundError:
                pass
            else:
                if (node.st_dev, node.st_ino) == (device, inode):
                    path.unlink()
            self._socket_nodes.pop()

    async def start(self) -> ChannelReady:
        if self._started:
            raise RuntimeError("akashic channel 重复 start")
        if self._stopped:
            raise RuntimeError("akashic channel 已停止")
        try:
            await self._resolve_capabilities()
            await self._start_web()
        except BaseException as error:
            await self._rollback_start(error)
        self._started = True
        return ChannelReady(self._binding_token)

    async def _rollback_start(self, primary: BaseException) -> None:
        failures: list[BaseException] = []
        remaining_servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []
        for server, task in reversed(self._servers):
            try:
                await _stop_server(server, task)
            except BaseException as error:
                failures.append(error)
                remaining_servers.append((server, task))
        self._servers = list(reversed(remaining_servers))
        if not self._servers:
            try:
                self._close_socket_nodes()
            except OSError as error:
                failures.append(error)
            remaining_children, _receipts, child_failures = await _stop_owned_children(
                tuple(self._started_children)
            )
            failures.extend(child_failures)
            if self._web not in self._started_children:
                try:
                    await self._web.stop()
                except BaseException as error:
                    failures.append(error)
                    remaining_children.append(self._web)
            self._started_children = remaining_children
        if failures:
            self._stopping = False
            raise BaseExceptionGroup("akashic channel start rollback 失败", (primary, *failures))
        self._stopped = True
        self._stopping = False
        self._release_generation_binding()
        raise primary

    def open_admission(self) -> None:
        children: list[Any] = [self._web_adapter]
        opened: list[Any] = []
        try:
            for child in children:
                child.open_admission()
                opened.append(child)
        except BaseException as error:
            _close_children(opened, primary=error, message="akashic channel admission rollback 失败")

    def close_admission(self) -> None:
        children = [self._web_adapter]
        _close_children(children, message="akashic channel admission close 失败")

    async def deliver(self, request: ProviderDeliveryRequest) -> ProviderDeliveryReceipt:
        children = [self._web_adapter]
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
            try:
                await _stop_server(server, task)
            except BaseException as error:
                errors.append(error)
                remaining_servers.append((server, task))
        self._servers = list(reversed(remaining_servers))
        if self._servers:
            self._stopping = False
            raise BaseExceptionGroup("akashic channel stop 失败", tuple(errors))

        self._close_socket_nodes()
        started_children = tuple(self._started_children)
        remaining_children, receipts, child_failures = await _stop_owned_children(
            started_children
        )
        errors.extend(child_failures)
        if self._web not in started_children:
            extra_remaining, extra_receipts, extra_failures = await _stop_owned_children(
                (self._web,)
            )
            remaining_children.extend(extra_remaining)
            receipts.extend(extra_receipts)
            errors.extend(extra_failures)
        self._started_children = remaining_children
        if self._started_children:
            self._stopping = False
            raise BaseExceptionGroup("akashic channel stop 失败", tuple(errors))

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
]
