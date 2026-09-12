from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn

from agent.plugin_composition import MODEL_CALL_STATS, MODEL_CATALOG
from agent.plugin_composition.channels import (
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
    MODEL_SELECTION,
    REPLY_STATUS,
)
from .config import AkashicClientsConfig
from .web_chat import WebChatChannel
from .mobile_realtime.channel import MobileRealtimeChannel
from .mobile_realtime.gateway import MobileGatewayRuntime
from .chat_api import build_chat_server
from .runtime_inspection import ScopedRpcRuntimeInspection
from .model_control import ScopedModelRpcControl
from .services import (
    MessageCatalogPort,
    ModelCatalogReader,
    ModelSelectionReader,
    ModelStatsReader,
    ReplyStatusPort,
)


_SERVER_START_TIMEOUT_SECONDS = 10.0


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
    adapter: "_GenerationAkashicAdapter | None" = None


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
    if state is not None and state.adapter is not None and state.adapter.started:
        raise RuntimeError("akashic clients generation 在 adapter 停止前被释放")
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
    if state.adapter is not None:
        raise RuntimeError("同一 akashic clients generation 不允许重复创建 channel")
    adapter = _GenerationAkashicAdapter(state, context)
    state.adapter = adapter
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
        self._model_control: ScopedModelRpcControl | None = None
        self._web = WebChatChannel("akashic") if state.config.web.enabled else None
        self._mobile: MobileRealtimeChannel | None = None
        self._mobile_runtime: MobileGatewayRuntime | None = None
        self._web_adapter = (
            None if self._web is None else self._web.build_v3_adapter(context)
        )
        self._mobile_adapter: Any | None = None
        self._runtime_ports: ChannelRuntimePorts | None = None
        self._servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []
        self._started_children: list[Any] = []
        self._started = False
        self._stopped = False

        if self._web is None and not state.config.mobile_realtime.enabled:
            raise ValueError("akashic channel 至少需要启用 Web 或 Mobile")

    @property
    def started(self) -> bool:
        return self._started and not self._stopped

    async def _resolve_capabilities(self) -> None:
        """Prepare request-scoped capability readers without retaining providers."""

        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic channel 缺少 host request scope")
        self._reply_status = self._follow_reply_status
        self._model_catalog_reader = self._read_model_catalog
        self._model_selection_reader = self._read_model_selection
        self._model_stats_reader = self._read_model_stats
        self._runtime_inspection = ScopedRpcRuntimeInspection(open_scope)
        self._model_control = ScopedModelRpcControl(open_scope)
        if self._web is not None:
            self._web.bind_message_scope(
                self._message_scope,
                reply_status=self._reply_status,
            )

    @asynccontextmanager
    async def _message_scope(self) -> AsyncIterator[MessageCatalogPort]:
        """Resolve the message catalog only for one HTTP or WebSocket operation."""

        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic message catalog 缺少 host request scope")
        async with open_scope() as scope:
            yield scope.require(MESSAGE_CATALOG)

    async def _follow_reply_status(self, session_id: str):
        """Keep the reply status read inside this subscription's exact scope."""

        open_scope = self._context.open_scope
        if open_scope is None:
            raise RuntimeError("akashic reply status 缺少 host request scope")
        async with open_scope() as scope:
            reader = scope.require(REPLY_STATUS)
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

    def attach_presentation(self, ports: ChannelPresentationPorts) -> None:
        """Bind the exact turn stream to the enabled transport owner."""

        if ports.turn_stream is None:
            raise RuntimeError("akashic channel 缺少 turn stream port")
        if self._web is None:
            raise RuntimeError(
                "akashic Mobile presentation host port 尚未接入，不能启用无 Web channel"
            )
        self._web.attach_presentation(ports)

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
        socket_path = self._config.web.socket_path or str(
            self._workspace / "runtime" / "chat.sock"
        )
        server = build_chat_server(
            workspace=self._workspace,
            channel=self._web,
            runtime_inspection=self._runtime_inspection,
            model_catalog_reader=self._model_catalog_reader,
            model_selection_reader=self._model_selection_reader,
            model_control=self._model_control,
            reply_status=self._reply_status,
            message_scope=self._message_scope,
            uds=socket_path,
        )
        await self._start_server(server, name="akashic-web")

    async def _start_mobile(self) -> None:
        config = self._config.mobile_realtime
        if not config.enabled:
            return
        # The Mobile continuity owner still needs the host's boot identity and
        # durable inbound port.  The formal channel host will supply those in
        # the next host-port revision; do not initialize its database before
        # that contract is present.
        raise RuntimeError("akashic Mobile continuity host port 尚未接入")

    async def start(self) -> ChannelReady:
        if self._started:
            raise RuntimeError("akashic channel 重复 start")
        if self._stopped:
            raise RuntimeError("akashic channel 已停止")
        try:
            await self._resolve_capabilities()
            await self._start_web()
            await self._start_mobile()
        except BaseException as error:
            await self._rollback_start(error)
        self._started = True
        return ChannelReady(self._binding_token)

    async def _rollback_start(self, primary: BaseException) -> None:
        self._stopped = True
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
            raise BaseExceptionGroup("akashic channel start rollback 失败", (primary, *failures))
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
        self._stopped = True
        errors: list[BaseException] = []
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
        self._servers.clear()
        receipts: list[StopReceipt] = []
        for child in reversed(self._started_children):
            try:
                result = await child.stop()
            except BaseException as error:
                errors.append(error)
            else:
                if isinstance(result, StopReceipt):
                    receipts.append(result)
        if self._web is not None and self._web not in self._started_children:
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
            raise BaseExceptionGroup("akashic channel stop 失败", tuple(errors))
        return StopReceipt(
            self._binding_token,
            resources_closed=all(receipt.resources_closed for receipt in receipts),
            failures=tuple(failure for receipt in receipts for failure in receipt.failures),
        )


__all__ = [
    "build_akashic_channel",
    "register_generation",
    "unregister_generation",
]
