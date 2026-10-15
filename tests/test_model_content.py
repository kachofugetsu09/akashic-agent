"""附件能力和有界读取由模型内容 owner 解释。"""

import json
import pytest
from plugins.ledger.artifact_services import ArtifactRead

from plugins.models.content import load_artifacts
from plugins.ledger.contract import AttachmentKind, AttachmentRef, AttachmentReadLease


async def reject_read(ref: AttachmentRef) -> AttachmentReadLease:
    raise AssertionError("此展示不应读取附件")


@pytest.mark.asyncio
async def test_text_model_preserves_identity_without_reading_image_data():
    """不支持图片的模型不会读取原图，也不会误称已经提供图片内容。"""
    ref = AttachmentRef("image", AttachmentKind.IMAGE, "photo.png", "image/png",
                        100 * 1024 * 1024, "0" * 64)
    blocks = (await load_artifacts(ArtifactRead(reject_read), (ref,), accepts_images=False, current_artifact_ids=frozenset()))[ref.artifact_id]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    label = json.loads(blocks[0]["text"])
    assert label["artifact"]["artifact_id"] == ref.artifact_id
    assert label["artifact"]["filename"] == ref.filename
    assert label["artifact"]["size_bytes"] == ref.size_bytes
    assert "当前模型不接收图片" in label["image_status"]


class _MemoryLease:
    """内存里的附件字节，记录是否被关闭。"""

    def __init__(self, ref: AttachmentRef, data: bytes):
        self.ref = ref
        self.data = data
        self.closed = False

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        return self.data[:max_bytes]

    async def read_chunk(self, *, offset: int, max_bytes: int) -> bytes:
        return self.data[offset:offset + max_bytes]

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_file_attachment_keeps_its_identity():
    """文本附件保留身份标签并把内容交给模型；二进制文件只给身份、不读取字节。"""
    data = "第一行\nsecond line\n".encode()
    text_ref = AttachmentRef("notes", AttachmentKind.FILE, "notes.txt", "text/plain", len(data), "1" * 64)
    leases: list[_MemoryLease] = []

    async def read(ref: AttachmentRef) -> AttachmentReadLease:
        leases.append(_MemoryLease(ref, data))
        return leases[-1]

    blocks = (await load_artifacts(ArtifactRead(read), (text_ref,), accepts_images=True, current_artifact_ids=frozenset()))[text_ref.artifact_id]
    label = json.loads(blocks[0]["text"])
    assert label["artifact"]["artifact_id"] == text_ref.artifact_id
    assert label["artifact"]["filename"] == "notes.txt"
    assert "image_status" not in label and "text_status" not in label
    assert blocks[1]["text"] == f"<attachment_text artifact_id=\"{text_ref.artifact_id}\">\n第一行\nsecond line\n\n</attachment_text>"
    assert [lease.closed for lease in leases] == [True]

    binary_ref = AttachmentRef("archive", AttachmentKind.FILE, "archive.zip", "application/zip", 10, "2" * 64)
    blocks = (await load_artifacts(ArtifactRead(reject_read), (binary_ref,), accepts_images=True, current_artifact_ids=frozenset()))[binary_ref.artifact_id]
    assert len(blocks) == 1
    assert json.loads(blocks[0]["text"])["artifact"]["filename"] == "archive.zip"
