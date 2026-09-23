"""验证普通 Root/Fiber 的输入、身份与 Channel validation 边界。"""
from __future__ import annotations

import asyncio
from contextlib import closing
from datetime import UTC, datetime
import sqlite3

import pytest

from agent.plugin_composition import CompositionRoot, FiberState, PluginRuntime, ServiceKey
from agent.plugin_composition.channel_io import (
    ChannelAttachmentImport, ChannelAttachmentRead, ChannelIdentity, InputCustody,
    CHANNEL_ATTACHMENT_IMPORT, CHANNEL_ATTACHMENT_READ, CHANNEL_IDENTITY, INPUT_CUSTODY,
)
from agent.plugin_composition.channels import (
    CHANNEL_INPUT, CHANNELS, ChannelCapability, ChannelDefinition,
    ChannelInboundMessage, InboundIdentity, RawInbound,
)
from agent.plugin_composition.host import HOST_INFO, HostInfo
from agent.plugin_composition.messages import (
    MESSAGE_WRITERS, SESSION_ADMISSION, MessageWriters, SessionAdmission,
)
from bus.queue import MessageBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.channels import plugin as channels_plugin
from session.admissions import SessionAdmissions
from session.artifact_store import ArtifactStore
from session.identities import ChannelIdentities, ChannelIdentityWriteReceipt
from session.inbound_store import InboundHandoffStore
from session.log import MessageLog, SessionAttributes
from session.message import ContentPart, ContentReferences, Input


VALIDATION_SERVICE = ServiceKey[object]("test.validation")


class ValidationBus(MessageBus):
    """Inject one close failure before delegating to the real Bus close chain."""

    def __init__(self, fail_close: bool) -> None:
        super().__init__()
        self.fail_close = fail_close
        self.close_attempts = 0

    async def aclose(self) -> None:
        self.close_attempts += 1
        if self.fail_close:
            self.fail_close = False
            raise OSError("validation message bus close failed")
        await super().aclose()


def inbound(number: int) -> RawInbound:
    """Build one durable handoff for the isolated validation store."""

    return RawInbound(
        f"message-{number}",
        ChannelInboundMessage(
            "probe", "user", "chat", "validation input", datetime.now(UTC),
            {
                "durable_inbound": True,
                "durable_handoff_id": f"handoff-{number}",
                "provider_message_id": f"message-{number}",
                "session_key_override": "validation",
                "require_existing_session": False,
            },
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_validation_custody_owns_inputs_and_retains_failed_close(tmp_path, fail_close):
    """真实验证 Fiber 不写正式库，失败 Effect 保留原 owner 供 retry。"""

    formal_path = tmp_path / "formal.db"
    formal_log = MessageLog(formal_path)
    with closing(sqlite3.connect(formal_path)) as db:
        formal_before = tuple(db.iterdump())

    isolated_path = tmp_path / "validation.db"
    messages = MessageLog(isolated_path)
    admissions = SessionAdmissions(isolated_path)
    identities = ChannelIdentities(isolated_path)
    inbound_store = InboundHandoffStore(isolated_path)
    message_bus = ValidationBus(fail_close)
    message_bus.bind_durable_inbound_store(inbound_store)
    message_bus.bind_session_admission_owner(admissions)
    custody = InputCustody(
        message_bus.prepare_channel_input,
        message_bus.complete_channel_input,
        message_bus.retain_channel_input,
        message_bus.reserve_durable_inbound,
        message_bus.defer_durable_inbound,
        message_bus.settle_rejected_inbound,
        message_bus.has_pending_durable_inbound,
        message_bus.pending_durable_attachment_refs,
        message_bus.recover_durable_inbounds,
    )

    async def remember(channel: str, identity: str, recipient: str) -> object:
        return identities.remember(channel, identity, recipient)

    async def rollback(receipt: object) -> bool:
        if not isinstance(receipt, ChannelIdentityWriteReceipt):
            raise TypeError("validation identity rollback receipt 类型无效")
        return identities.rollback(receipt)

    channel_identity = ChannelIdentity(identities.resolve, remember, rollback)
    artifact_metadata = ArtifactStore(tmp_path / "artifact-metadata.db")
    artifact_root = tmp_path / "attachments"
    artifact_root.mkdir()
    attachments = ChannelAttachmentArtifactStore(
        workspace=artifact_root, metadata_store=artifact_metadata,
    )

    root = CompositionRoot("validation-root")
    try:
        await root.context.provide(HOST_INFO, HostInfo("validation-boot", True))
        await root.context.provide(INPUT_CUSTODY, custody)
        await root.context.provide(CHANNEL_IDENTITY, channel_identity)
        await root.context.provide(
            CHANNEL_ATTACHMENT_IMPORT,
            ChannelAttachmentImport(attachments.import_bytes),
        )
        await root.context.provide(
            CHANNEL_ATTACHMENT_READ,
            ChannelAttachmentRead(attachments.resolve_refs, attachments.acquire),
        )

        async def reject_input(*args: object, **kwargs: object) -> None:
            raise AssertionError("validation probe 不得执行正式 Channel input")

        await root.context.provide(CHANNEL_INPUT, reject_input)
        await root.mount(
            channels_plugin.apply,
            name="channels",
            inject=channels_plugin.inject,
            runtime=PluginRuntime(
                "channels", "validation-channels", tmp_path, tmp_path / "channels-data",
                tmp_path / "workspace", {},
            ),
        )

        factory_calls = 0
        bus_effect = None

        async def probe(ctx):
            def factory(_context):
                nonlocal factory_calls
                factory_calls += 1
                raise AssertionError("validation factory must not start")

            await ctx.require(CHANNELS).register(
                ctx,
                ChannelDefinition(
                    "validation-probe",
                    frozenset({ChannelCapability.INBOUND}),
                    factory,
                    InboundIdentity.PROVIDER_MESSAGE_ID,
                ),
            )

        rejected = await root.mount(
            probe,
            name="validation-probe",
            inject=(CHANNELS, CHANNEL_INPUT),
            runtime=PluginRuntime(
                "validation-probe", "validation-probe", tmp_path, tmp_path / "probe-data",
                tmp_path / "workspace", {},
            ),
        )
        assert rejected.state == FiberState.FAILED
        assert factory_calls == 0

        async def program(ctx):
            nonlocal bus_effect
            writers = MessageWriters(messages)
            admissions_service = SessionAdmission(messages)
            await ctx.provide(MESSAGE_WRITERS, writers)
            await ctx.provide(SESSION_ADMISSION, admissions_service)

            async def run(entered):
                entered.set()
                admissions_service.ensure(ctx, "validation", SessionAttributes())
                writer = writers.bind(
                    ctx,
                    author="user",
                    source="conversation",
                    body_types=(Input,),
                    content={"text": lambda part: ContentReferences()},
                )
                message = writer("validation").append(
                    "validation-input",
                    Input((ContentPart("text", "validation input"),)),
                )
                await channel_identity.remember("probe", "user", "chat")
                assert await custody.reserve_durable_inbound(inbound(2))
                return message

            await ctx.provide(VALIDATION_SERVICE, run)
            bus_effect = await ctx.effect(lambda: message_bus.aclose, label="validation-message-bus")

        program_fiber = await root.mount(
            program,
            name="validation-program",
            inject=(CHANNEL_IDENTITY, INPUT_CUSTODY),
            runtime=PluginRuntime(
                "validation-program", "validation-program", tmp_path, tmp_path / "program-data",
                tmp_path / "workspace", {},
            ),
        )
        entered = asyncio.Event()
        async with program_fiber.context.runtime_scope():
            result = await program_fiber.context.require(VALIDATION_SERVICE)(entered)
            assert result.message_id == "validation-input"
            assert identities.resolve("probe", "user") == "chat"
            assert messages.reader("validation").get("validation-input") == result
            assert await custody.reserve_durable_inbound(inbound(1))
            await custody.settle_rejected_inbound(
                channel="probe", session_key="validation", provider_message_id="message-1",
            )
            assert not custody.has_pending_durable_inbound(
                channel="probe", session_key="validation", provider_message_id="message-1",
            )

        if fail_close:
            with pytest.raises(BaseExceptionGroup) as error:
                await root.dispose()
            assert len(error.value.exceptions) == 1
            assert isinstance(error.value.exceptions[0], OSError)
            assert str(error.value.exceptions[0]) == "validation message bus close failed"
            assert program_fiber.state == FiberState.UNLOADING
            assert bus_effect is not None
            assert bus_effect.label == "validation-message-bus"
            assert bus_effect in program_fiber.effects
            assert message_bus.close_attempts == 1
            assert not message_bus._closed
            assert not bus_effect._closed
            await root.dispose()
            assert message_bus.close_attempts == 2
            assert message_bus._closed
            assert bus_effect._closed
            assert program_fiber.effects == []
        else:
            await root.dispose()
        assert message_bus.close_attempts == (2 if fail_close else 1)
        assert [row["handoff_id"] for row in inbound_store.list_inbound_handoffs()] == ["handoff-2"]
    finally:
        await root.dispose()
        if not message_bus._closed:
            await message_bus.aclose()
        inbound_store.close()
        identities.close()
        admissions.close()
        messages.close()
        artifact_metadata.close()
        formal_log.close()

    with closing(sqlite3.connect(formal_path)) as db:
        assert tuple(db.iterdump()) == formal_before
