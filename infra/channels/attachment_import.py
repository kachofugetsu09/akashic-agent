from __future__ import annotations

import mimetypes
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
)
from bus.events import (
    AttachmentKind as LegacyAttachmentKind,
    ChannelAttachment,
)
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.channels.base import AttachmentStore
from infra.mobile_realtime.remote_media import snapshot_remote_media


_MAX_IMPORT_BYTES = 50 * 1024 * 1024


class ChannelOutboundAttachmentImporter:
    """Import model-produced media through the same Core artifact boundary."""

    def __init__(self, store: ChannelAttachmentArtifactStore) -> None:
        self._store = store

    async def import_source(self, source: str, kind: AttachmentKind) -> AttachmentRef:
        """将已授权路径或公网 URL 固定为一个已发布附件引用。"""
        refs = await import_channel_attachments(self._store, (
            ChannelAttachment(kind=LegacyAttachmentKind(kind.value), source=source),
        ))
        return refs[0]

    async def import_media(
        self,
        media: tuple[str, ...],
    ) -> tuple[AttachmentRef, ...]:
        return await import_channel_attachments(
            self._store,
            tuple(
                ChannelAttachment(
                    kind=LegacyAttachmentKind.IMAGE,
                    source=source,
                )
                for source in media
            ),
        )


async def import_channel_attachments(
    store: ChannelAttachmentArtifactStore,
    attachments: tuple[ChannelAttachment, ...],
) -> tuple[AttachmentRef, ...]:
    """Import authorized outbound sources before one exact provider attempt."""

    # 1. 每个来源先成为 Core-owned immutable artifact。
    refs: list[AttachmentRef] = []
    for attachment in attachments:
        kind = AttachmentKind(attachment.kind.value)
        if attachment.source.startswith(("http://", "https://")):
            refs.append(await _import_remote(store, attachment, kind))
            continue
        source = Path(attachment.source).expanduser().absolute()
        refs.append(
            await store.adopt_file(
                source,
                allowed_root=source.parent,
                kind=kind,
                filename=attachment.filename or source.name,
                media_type=mimetypes.guess_type(
                    attachment.filename or source.name
                )[0],
            )
        )
    return tuple(refs)


async def _import_remote(
    store: ChannelAttachmentArtifactStore,
    attachment: ChannelAttachment,
    kind: AttachmentKind,
) -> AttachmentRef:
    """Fetch one public URL into a bounded temporary owner, then adopt it."""

    # 1. 下载器逐跳固定公网地址，并在退出时删除非权威临时文件。
    with TemporaryDirectory(prefix="akashic-channel-fetch-") as temp_dir:
        root = Path(temp_dir)
        snapshot = await snapshot_remote_media(
            attachment.source,
            AttachmentStore(root),
            max_bytes=_MAX_IMPORT_BYTES,
        )

        # 2. 只有 artifact transaction 成功后才向调用方返回 opaque ref。
        return await store.adopt_file(
            snapshot.path,
            allowed_root=root,
            kind=kind,
            filename=attachment.filename or snapshot.filename,
            media_type=snapshot.content_type,
        )


__all__ = [
    "ChannelOutboundAttachmentImporter",
    "import_channel_attachments",
]
