"""Ledger 拥有唯一消息库、附件和入站交接的完整生命周期。"""
from __future__ import annotations

from contextlib import ExitStack

from agent.plugin_composition import Context
from plugins.ledger.contract import (
    ARTIFACT_IMPORT, ARTIFACT_READ, BINDINGS, CHANNEL_ATTACHMENT_IMPORT,
    CHANNEL_ATTACHMENT_READ, CHANNEL_IDENTITY, INPUT_CUSTODY, MESSAGE_CATALOG,
    MESSAGE_EMBEDDINGS, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMIN, SESSION_ADMISSION,
    ChannelAttachmentImport, ChannelAttachmentRead, ChannelIdentity, InputCustody,
)
from .admissions import SessionAdmissions
from .artifact_services import ArtifactImport, ArtifactRead
from .artifact_store import ArtifactStore
from .attachment_import import ChannelOutboundAttachmentImporter
from .attachments import ChannelAttachmentArtifactStore
from .bindings import Bindings
from .embedding_store import MessageEmbeddings
from .identities import ChannelIdentities, ChannelIdentityWriteReceipt
from .inbound_store import InboundHandoffStore
from .log import MessageCatalog, MessageLog
from .custody import InboundCustody
from .services import MessageWriters, OwnerState, SessionAdmin, SessionAdmission

api_version = 3
name = "ledger"
version = "1.0.0"
desc = "消息、同库事务、附件与入站交接的持久 owner"
inject = ()
workspace_files = ("sessions.db", "sessions-derived.db")
workspace_roots = ("uploads",)


async def apply(ctx: Context) -> None:
    """先登记连接与队列的关闭责任，再发布窄读写端口。"""
    path = ctx.runtime.workspace / "sessions.db"
    # 1. 部分构造失败时关闭已经取得的连接；旧库先通过 schema 检查。
    with ExitStack() as cleanup:
        log = MessageLog(path)
        cleanup.callback(log.close)
        embeddings = MessageEmbeddings(log, ctx.runtime.workspace / "sessions-derived.db")
        cleanup.callback(embeddings.close)
        metadata = ArtifactStore(path)
        cleanup.callback(metadata.close)
        admissions = SessionAdmissions(path)
        cleanup.callback(admissions.close)
        identities = ChannelIdentities(path)
        cleanup.callback(identities.close)
        inbounds = InboundHandoffStore(path)
        cleanup.callback(inbounds.close)
        admissions.clear_stale()
        custody = InboundCustody(inbounds, admissions)
        attachments = ChannelAttachmentArtifactStore(
            workspace=ctx.runtime.workspace, metadata_store=metadata,
        )
        stores = cleanup.pop_all()

    async def close() -> None:
        # 队列持有的租约先收束；失败时保留连接以供原 Effect 重试。
        await custody.aclose()
        stores.close()

    await ctx.effect(lambda: close, label="ledger.storage")

    # 2. 身份回执只由创建它的本 generation 解释。
    async def remember(channel: str, provider: str, recipient: str) -> object:
        return identities.remember(channel, provider, recipient)

    async def rollback(receipt: object) -> bool:
        if not isinstance(receipt, ChannelIdentityWriteReceipt):
            raise TypeError("Channel identity receipt 不属于当前 Ledger")
        return identities.rollback(receipt)

    await ctx.provide(CHANNEL_IDENTITY, ChannelIdentity(identities.resolve, remember, rollback))
    await ctx.provide(CHANNEL_ATTACHMENT_IMPORT, ChannelAttachmentImport(attachments.import_bytes))
    await ctx.provide(CHANNEL_ATTACHMENT_READ, ChannelAttachmentRead(attachments.resolve_refs, attachments.acquire))
    await ctx.provide(INPUT_CUSTODY, InputCustody(
        custody.prepare_channel_input, custody.complete_channel_input, custody.retain_channel_input,
        custody.reserve_durable_inbound, custody.defer_durable_inbound, custody.settle_rejected_inbound,
        custody.has_pending_durable_inbound, custody.pending_durable_attachment_refs,
        custody.recover_durable_inbounds,
    ))
    await ctx.provide(MESSAGE_CATALOG, MessageCatalog(log))
    await ctx.provide(MESSAGE_EMBEDDINGS, embeddings)
    await ctx.provide(MESSAGE_WRITERS, MessageWriters(log))
    await ctx.provide(OWNER_STATE, OwnerState(log))
    await ctx.provide(SESSION_ADMIN, SessionAdmin(log))
    await ctx.provide(SESSION_ADMISSION, SessionAdmission(log))
    await ctx.provide(BINDINGS, Bindings(log, ctx))
    await ctx.provide(ARTIFACT_READ, ArtifactRead(attachments.acquire))
    await ctx.provide(ARTIFACT_IMPORT, ArtifactImport(ChannelOutboundAttachmentImporter(attachments).import_source))
