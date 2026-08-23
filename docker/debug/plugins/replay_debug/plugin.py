from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent.plugin_composition import (
    CHANNELS,
    ChannelCapability,
    ChannelDefinition,
    ChannelFactoryContext,
    ChannelReady,
    Context,
    CredentialRef,
    DeliveryStatus,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    ServiceView,
    StopReceipt,
)
from core.clock import clock_from_env

api_version = 3
name = "replay_debug"
version = "3.0.0"
desc = "Debug-only outbound capture channel"
author = "Akashic Core"
inject = (CHANNELS,)


class Config(BaseModel):
    """Carry the opaque fixture credential required by the v3 channel seam."""

    replay_token: CredentialRef | None = None


def is_active(_services: ServiceView) -> bool:
    """Enable replay declarations only when the debug replay profile is mounted."""

    return _replay_source_enabled()


async def apply(ctx: Context, config: object) -> None:
    """Register typed replay source, MCP, and optional outbound capture effects."""

    # 1. The replay profile owns all declarations; a normal debug runtime stays inert.
    if not _replay_source_enabled():
        return

    # 2. The capture channel is present only with an explicit isolated fixture token.
    if _capture_channel_enabled(config):
        await ctx.require(CHANNELS).register(
            ctx,
            ChannelDefinition(
                name="replay",
                capabilities=frozenset({ChannelCapability.OUTBOUND}),
                factory_export="build_channel",
                inbound_identity=None,
                credential_paths=("replay_token",),
            ),
        )


class ReplayCaptureAdapter:
    """Capture text deliveries into the replay outbox with replay-clock timestamps."""

    def __init__(self, context: ChannelFactoryContext) -> None:
        self._binding_token = context.binding_token
        self._outbox_path = _required_path("AKASHIC_REPLAY_OUTBOX_FILE")

    async def start(self) -> ChannelReady:
        """Start without opening external resources or admitting delivery yet."""

        return ChannelReady(self._binding_token)

    async def deliver(
        self,
        request: ProviderDeliveryRequest,
    ) -> ProviderDeliveryReceipt:
        """Capture one text delivery and reject unsupported attachment payloads."""

        if request.attachments:
            return ProviderDeliveryReceipt(
                request.delivery_id,
                DeliveryStatus.REJECTED,
                error="replay capture 仅支持文本消息",
            )
        self._append(
            {
                "type": "text",
                "chat_id": request.recipient,
                "message": request.body,
            }
        )
        return ProviderDeliveryReceipt(
            request.delivery_id,
            DeliveryStatus.DELIVERED,
        )

    async def stop(self) -> StopReceipt:
        """Close the capture adapter; its append-only outbox needs no extra teardown."""

        return StopReceipt(self._binding_token, resources_closed=True)

    def _append(self, payload: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "captured_at": clock_from_env().now().isoformat(),
            **payload,
        }
        with self._outbox_path.open("a", encoding="utf-8") as handle:
            _ = handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_channel(context: ChannelFactoryContext) -> ReplayCaptureAdapter:
    """Build one exact-generation replay capture adapter for Core's channel host."""

    return ReplayCaptureAdapter(context)


def _replay_source_enabled() -> bool:
    return bool(
        os.environ.get("AKASHIC_REPLAY_CLOCK_FILE", "").strip()
        and os.environ.get("AKASHIC_REPLAY_EVENTS_FILE", "").strip()
    )


def _capture_channel_enabled(config: object) -> bool:
    token = getattr(config, "replay_token", None)
    return bool(
        os.environ.get("AKASHIC_REPLAY_OUTBOX_FILE", "").strip()
        and isinstance(token, CredentialRef)
    )


def _required_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return Path(value)
