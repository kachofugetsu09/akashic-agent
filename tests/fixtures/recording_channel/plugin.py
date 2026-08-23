from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

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
    StopReceipt,
)

api_version = 3
name = "recording_channel"
version = "3.0.0"
desc = "SQLite-backed ordinary outbound Channel for isolated E2E evidence"
author = "Akashic Core"
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()
inject = (CHANNELS,)


class Config(BaseModel):
    """Carry the isolated receipt database path into the formal Channel factory."""

    receipt_db: str
    token: CredentialRef


class RecordingChannel:
    """Persist one typed receipt for every accepted ordinary Channel delivery."""

    def __init__(self, context: ChannelFactoryContext) -> None:
        self._binding_token = context.binding_token
        raw_path = context.config.get("receipt_db")
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError("recording Channel 缺少 receipt_db")
        self._path = Path(raw_path)
        self._initialize()

    async def start(self) -> ChannelReady:
        return ChannelReady(self._binding_token)

    async def deliver(
        self, request: ProviderDeliveryRequest
    ) -> ProviderDeliveryReceipt:
        """Commit the exact delivery identity before returning its provider receipt."""

        with self._transaction() as connection:
            prior = connection.execute(
                "SELECT recipient, control_turn_id FROM deliveries "
                "WHERE delivery_id = ?",
                (request.delivery_id,),
            ).fetchone()
            identity = (request.recipient, request.control_turn_id)
            if prior is None:
                connection.execute(
                    "INSERT INTO deliveries(delivery_id, recipient, control_turn_id) "
                    "VALUES (?, ?, ?)",
                    (request.delivery_id, *identity),
                )
            elif tuple(prior) != identity:
                raise RuntimeError("recording Channel delivery identity conflict")
        return ProviderDeliveryReceipt(
            request.delivery_id,
            DeliveryStatus.DELIVERED,
            (f"recording:{request.delivery_id}",),
        )

    async def stop(self) -> StopReceipt:
        return StopReceipt(self._binding_token, resources_closed=True)

    def _initialize(self) -> None:
        with self._transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS deliveries("
                "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                "delivery_id TEXT NOT NULL UNIQUE, "
                "recipient TEXT NOT NULL, "
                "control_turn_id TEXT)"
            )

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


def build_channel(context: ChannelFactoryContext) -> RecordingChannel:
    """Build the exact-generation ordinary recording Channel."""

    return RecordingChannel(context)


async def apply(ctx: Context, config: object) -> None:
    """Register one ordinary outbound Channel through the public v3 slot."""

    if not isinstance(config, Config):
        raise TypeError("recording Channel config 必须通过 Config 校验")
    await ctx.require(CHANNELS).register(
        ctx,
        ChannelDefinition(
            name="recording",
            capabilities=frozenset({ChannelCapability.OUTBOUND}),
            factory_export="build_channel",
            inbound_identity=None,
            credential_paths=("token",),
        ),
    )
