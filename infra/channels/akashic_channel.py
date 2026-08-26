from __future__ import annotations

import asyncio
from collections.abc import Sequence

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
from infra.channels.contract import Channel, ChannelContext


async def _stop_started_children(
    children: Sequence[object],
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
    children: Sequence[object],
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
        children: Sequence[object],
        context: ChannelFactoryContext,
    ) -> None:
        self._binding_token = context.binding_token
        self._children = tuple(child.build_v3_adapter(context) for child in children)

    async def start(self) -> ChannelReady:
        started: list[object] = []
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
        opened: list[object] = []
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
            isinstance(result, BaseException) or result.status is DeliveryStatus.UNKNOWN
            for result in results
        ):
            return ProviderDeliveryReceipt(
                request.delivery_id,
                DeliveryStatus.UNKNOWN,
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
        web: Channel | None = None,
        mobile: Channel | None = None,
    ) -> None:
        self.web = web
        self.mobile = mobile
        self._children = tuple(child for child in (web, mobile) if child is not None)
        if not self._children:
            raise ValueError("Akashic channel 至少需要一个 client adapter")

    async def start(self, ctx: ChannelContext) -> None:
        started: list[Channel] = []
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


__all__ = ["AkashicChannel", "AkashicNativeAdapter"]
