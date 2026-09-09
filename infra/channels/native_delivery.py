"""Shared delivery semantics for Core-owned native channel adapters."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent.plugin_composition.channels import (
    AttachmentRef,
    ChannelAdapter,
    ChannelFactoryContext,
    ChannelReady,
    DeliveryStatus,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    StopReceipt,
)

logger = logging.getLogger(__name__)

SendText = Callable[[ProviderDeliveryRequest], Awaitable[None]]
SendAttachment = Callable[
    [ProviderDeliveryRequest, AttachmentRef, bytes], Awaitable[None]
]
ValidateRecipient = Callable[[str], Any]


class NativeChannelDeliveryAdapter(ChannelAdapter):
    """Adapt one already-started provider owner to the Core delivery ABI."""

    def __init__(
        self,
        context: ChannelFactoryContext,
        *,
        channel_name: str,
        validate_recipient: ValidateRecipient,
        send_text: SendText,
        send_attachment: SendAttachment,
    ) -> None:
        if context.attachment_read is None:
            raise RuntimeError(f"{channel_name} native adapter 缺少 attachment_read")
        self._channel_name = channel_name
        self._binding_token = context.binding_token
        self._attachment_read = context.attachment_read
        self._validate_recipient = validate_recipient
        self._send_text = send_text
        self._send_attachment = send_attachment

    async def start(self) -> ChannelReady:
        """Acknowledge an existing provider owner without starting it twice."""

        return ChannelReady(self._binding_token)

    async def deliver(
        self,
        request: ProviderDeliveryRequest,
    ) -> ProviderDeliveryReceipt:
        """Deliver text and attachments in order with non-retryable status mapping."""

        if request.binding_token != self._binding_token:
            raise RuntimeError(
                f"{self._channel_name} native adapter binding token 不匹配"
            )

        # 1. Reject routing and attachment integrity errors before provider effect.
        provider_called = False
        try:
            self._validate_recipient(request.recipient)
            if request.body:
                provider_called = True
                await self._send_text(request)
            for attachment in request.attachments:
                payload = await self._read_attachment(attachment)
                provider_called = True
                await self._send_attachment(request, attachment, payload)
        except Exception as error:
            status = (
                DeliveryStatus.FAILED if provider_called else DeliveryStatus.REJECTED
            )
            logger.warning(
                "[%s] native delivery failed delivery_id=%s provider_called=%s error=%s",
                self._channel_name,
                request.delivery_id,
                provider_called,
                error,
            )
            return ProviderDeliveryReceipt(
                request.delivery_id,
                status,
                error=str(error),
            )
        return ProviderDeliveryReceipt(request.delivery_id, DeliveryStatus.DELIVERED)

    async def stop(self) -> StopReceipt:
        """Release only the Core adapter; the existing provider owner stops elsewhere."""

        return StopReceipt(self._binding_token, resources_closed=True)

    async def _read_attachment(self, ref: AttachmentRef) -> bytes:
        """Read and verify one exact Core-owned attachment lease."""

        lease = await self._attachment_read.acquire(ref)
        try:
            if lease.ref != ref:
                raise RuntimeError("attachment read lease ref 不匹配")
            payload = await lease.read_bytes(max_bytes=max(ref.size_bytes, 1))
            if len(payload) != ref.size_bytes:
                raise ValueError(
                    f"attachment size 不匹配: expected={ref.size_bytes} actual={len(payload)}"
                )
            if hashlib.sha256(payload).hexdigest() != ref.sha256:
                raise ValueError("attachment sha256 不匹配")
            return payload
        finally:
            await lease.aclose()


__all__ = ["NativeChannelDeliveryAdapter"]
