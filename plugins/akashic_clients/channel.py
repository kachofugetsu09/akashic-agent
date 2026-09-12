from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import uvicorn

from agent.plugin_composition.channels import (
    ChannelFactoryContext,
    ChannelReady,
    ChannelRuntimePorts,
    DeliveryStatus,
    InboundIdentity,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    StopReceipt,
)
from .services import ClientChannel, ClientChannelContext as ChannelContext
from .config import AkashicClientsConfig
from .web_chat import WebChatChannel
from .mobile_realtime.channel import MobileRealtimeChannel
from .mobile_realtime.gateway import (
    MobileGatewayRuntime,
    build_mobile_gateway_runtime,
    build_mobile_gateway_server,
)
from .chat_api import build_chat_server
from .services import AkashicClientServices


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


class AkashicNativeAdapter:
    """Expose Web and Mobile through one exact Core channel binding."""

    def __init__(
        self,
        children: Sequence[Any],
        context: ChannelFactoryContext,
    ) -> None:
        self._binding_token = context.binding_token
        self._children: tuple[Any, ...] = tuple(
            child.build_v3_adapter(context) for child in children
        )

    async def start(self) -> ChannelReady:
        started: list[Any] = []
        try:
            for child in self._children:
                _ = await child.start()
                started.append(child)
        except BaseException as error:
            await _stop_started_children(
                started,
                primary=error,
                message="Akashic adapter start rollback 失败",
            )
        return ChannelReady(self._binding_token)

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None:
        for child in self._children:
            child.attach_runtime(ports)

    def open_admission(self) -> None:
        opened: list[Any] = []
        try:
            for child in self._children:
                child.open_admission()
                opened.append(child)
        except BaseException as error:
            _close_children(
                opened,
                primary=error,
                message="Akashic adapter admission rollback 失败",
            )

    def close_admission(self) -> None:
        _close_children(
            self._children,
            message="Akashic adapter admission close 失败",
        )

    async def deliver(
        self,
        request: ProviderDeliveryRequest,
    ) -> ProviderDeliveryReceipt:
        """Project one logical delivery to both clients and settle it once."""

        results = await asyncio.gather(
            *(child.deliver(request) for child in self._children),
            return_exceptions=True,
        )
        receipts = tuple(
            result for result in results if isinstance(result, ProviderDeliveryReceipt)
        )
        provider_ids = tuple(
            dict.fromkeys(
                provider_id
                for receipt in receipts
                for provider_id in receipt.provider_ids
            )
        )
        errors = [
            str(result) if isinstance(result, BaseException) else result.error
            for result in results
            if isinstance(result, BaseException) or result.error is not None
        ]
        if any(
            isinstance(result, BaseException) or result.status is DeliveryStatus.FAILED
            for result in results
        ):
            return ProviderDeliveryReceipt(
                request.delivery_id,
                DeliveryStatus.FAILED,
                provider_ids=provider_ids,
                error="; ".join(errors) or "Akashic adapter 投递结果未知",
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
            error=(
                None
                if status is DeliveryStatus.DELIVERED
                else "; ".join(errors) or "Akashic 没有可用客户端"
            ),
        )

    async def stop(self) -> StopReceipt:
        results = await asyncio.gather(
            *(child.stop() for child in reversed(self._children)),
            return_exceptions=True,
        )
        errors = tuple(
            result for result in results if isinstance(result, BaseException)
        )
        if errors:
            raise BaseExceptionGroup("Akashic adapter stop 失败", errors)
        receipts = tuple(
            result for result in results if isinstance(result, StopReceipt)
        )
        return StopReceipt(
            self._binding_token,
            resources_closed=all(receipt.resources_closed for receipt in receipts),
            failures=tuple(
                failure for receipt in receipts for failure in receipt.failures
            ),
        )


class AkashicChannel:
    """Own one Core channel while Web and Mobile keep their transport state."""

    name = "akashic"
    v3_inbound_identity = InboundIdentity.PROVIDER_MESSAGE_ID

    def __init__(
        self,
        web: ClientChannel | None = None,
        mobile: ClientChannel | None = None,
    ) -> None:
        self.web = web
        self.mobile = mobile
        self._children = tuple(child for child in (web, mobile) if child is not None)
        if not self._children:
            raise ValueError("Akashic channel 至少需要一个 client adapter")

    async def start(self, ctx: ChannelContext) -> None:
        started: list[ClientChannel] = []
        try:
            for child in self._children:
                await child.start(ctx)
                started.append(child)
        except BaseException as error:
            await _stop_started_children(
                started,
                primary=error,
                message="Akashic channel start rollback 失败",
            )

    async def stop(self) -> None:
        results = await asyncio.gather(
            *(child.stop() for child in reversed(self._children)),
            return_exceptions=True,
        )
        errors = tuple(
            result for result in results if isinstance(result, BaseException)
        )
        if errors:
            raise BaseExceptionGroup("Akashic channel stop 失败", errors)

    def build_v3_adapter(self, context: ChannelFactoryContext) -> AkashicNativeAdapter:
        return AkashicNativeAdapter(self._children, context)


@dataclass(slots=True)
class _ClientGeneration:
    """Hold one generation's client owners until its channel factory is called."""

    config: AkashicClientsConfig
    services: AkashicClientServices
    adapter: "_GenerationAkashicAdapter | None" = None


_GENERATIONS: dict[str, _ClientGeneration] = {}


def register_generation(
    generation_id: str,
    config: AkashicClientsConfig,
    services: AkashicClientServices,
) -> None:
    """Register immutable generation inputs for the synchronous channel factory."""

    if generation_id in _GENERATIONS:
        raise RuntimeError(f"akashic clients generation 已注册: {generation_id}")
    _GENERATIONS[generation_id] = _ClientGeneration(config, services)


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
        self._services = state.services
        self._config = state.config
        self._web = WebChatChannel("akashic") if state.config.web.enabled else None
        self._mobile: MobileRealtimeChannel | None = None
        self._mobile_runtime: MobileGatewayRuntime | None = None
        self._web_adapter = (
            None if self._web is None else self._web.build_v3_adapter(context)
        )
        self._mobile_adapter: Any | None = None
        self._runtime_ports: ChannelRuntimePorts | None = None
        self._servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []
        self._started_children: list[ClientChannel] = []
        self._started = False
        self._stopped = False

        if self._web is None and not state.config.mobile_realtime.enabled:
            raise ValueError("akashic channel 至少需要启用 Web 或 Mobile")
        self._bind_web_services()

    @property
    def started(self) -> bool:
        return self._started and not self._stopped

    def _bind_web_services(self) -> None:
        web = self._web
        if web is None:
            return
        services = self._services
        if services.message_catalog is None:
            raise RuntimeError("akashic Web 缺少 MessageCatalog service")
        web.bind_message_readers(services.message_catalog, services.reply_status)
        if services.message_display is not None:
            web.bind_message_display(services.message_display)
        web.bind_attachment_store(services.attachment_store)
        if services.artifact_store is not None:
            web.bind_artifact_store(services.artifact_store)

    def _bind_mobile_services(self, mobile: MobileRealtimeChannel) -> None:
        services = self._services
        if services.message_catalog is None:
            raise RuntimeError("akashic Mobile 缺少 MessageCatalog service")
        if services.artifact_store is None:
            raise RuntimeError("akashic Mobile 缺少 ArtifactStore service")
        mobile.bind_messages(services.message_catalog, services.reply_status)
        mobile.bind_channel_attachment_store(services.artifact_store)
        if services.message_display is not None:
            mobile.bind_message_display(services.message_display)
        if services.mobile_ui_provider is not None:
            mobile.bind_mobile_ui_provider(services.mobile_ui_provider)
        if services.runtime_inspection is not None:
            mobile.bind_runtime_inspection(services.runtime_inspection)
        if services.model_catalog_reader is not None:
            mobile.bind_model_catalog(services.model_catalog_reader)
        if services.model_selection_reader is not None:
            mobile.bind_model_selection(services.model_selection_reader)
        if services.model_stats_reader is not None:
            mobile.bind_model_stats(services.model_stats_reader)

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None:
        if self._stopped:
            raise RuntimeError("akashic channel 已停止")
        if self._runtime_ports is not None:
            raise RuntimeError("akashic channel runtime 不允许替换")
        self._runtime_ports = ports
        if self._web_adapter is not None:
            self._web_adapter.attach_runtime(ports)

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
        await self._web.start(self._services.channel_context)
        self._started_children.append(self._web)
        _ = await self._web_adapter.start()
        socket_path = self._config.web.socket_path or self._services.chat_socket_path
        if not socket_path:
            raise RuntimeError("akashic Web 缺少 Unix socket path")
        server = build_chat_server(
            workspace=self._services.workspace,
            channel=self._web,
            mobile_pairing_admin=self._services.mobile_pairing_admin,
            runtime_inspection=self._services.runtime_inspection,
            message_display=self._services.message_display,
            plugin_ui_provider=self._services.mobile_ui_provider,
            web_ui_provider=self._services.web_ui_provider,
            model_catalog_reader=self._services.model_catalog_reader,
            model_selection_reader=self._services.model_selection_reader,
            model_control=self._services.model_control,
            messages=self._services.message_catalog,
            reply_status=self._services.reply_status,
            attachment_store=self._services.attachment_store,
            artifact_store=self._services.artifact_store,
            uds=socket_path,
        )
        await self._start_server(server, name="akashic-web")

    async def _start_mobile(self) -> None:
        config = self._config.mobile_realtime
        if not config.enabled:
            return
        runtime, keyset = build_mobile_gateway_runtime(
            config,
            self._services.workspace,
            webui_source_repository=self._services.webui_source_repository,
        )
        mobile = runtime.channel
        self._bind_mobile_services(mobile)
        self._mobile_runtime = runtime
        self._mobile = mobile
        self._mobile_adapter = mobile.build_v3_adapter(self._context)
        if self._runtime_ports is None:
            raise RuntimeError("akashic Mobile channel 缺少 Core runtime ports")
        self._mobile_adapter.attach_runtime(self._runtime_ports)
        await mobile.start(self._services.channel_context)
        self._started_children.append(mobile)
        _ = await self._mobile_adapter.start()
        server = build_mobile_gateway_server(runtime, keyset)
        await self._start_server(server, name="akashic-mobile")

    async def start(self) -> ChannelReady:
        if self._started:
            raise RuntimeError("akashic channel 重复 start")
        if self._stopped:
            raise RuntimeError("akashic channel 已停止")
        try:
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
    "AkashicChannel",
    "AkashicNativeAdapter",
    "build_akashic_channel",
    "register_generation",
    "unregister_generation",
]
