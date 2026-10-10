"""附件能力和有界读取由模型内容 owner 解释。"""

import json
import pytest
from session.artifact_services import ArtifactRead

from plugins.models.content import load_artifacts
from session.artifacts import AttachmentKind, AttachmentRef, AttachmentReadLease


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


@pytest.mark.asyncio
async def test_file_attachment_keeps_its_identity():
    ref = AttachmentRef("file", AttachmentKind.FILE, "notes.txt", "text/plain", 10, "1" * 64)
    blocks = (await load_artifacts(ArtifactRead(reject_read), (ref,), accepts_images=True, current_artifact_ids=frozenset()))[ref.artifact_id]
    label = json.loads(blocks[0]["text"])
    assert label["artifact"]["artifact_id"] == ref.artifact_id
    assert "image_status" not in label
