"""Model attachment projection is text-only and does not open files."""

import json

from plugins.models.content import describe_artifacts
from session.artifacts import AttachmentKind, AttachmentRef


def test_image_placeholder_preserves_identity_without_image_data():
    """Even a large unavailable image is safe to describe without acquiring it."""
    ref = AttachmentRef("image", AttachmentKind.IMAGE, "photo.png", "image/png",
                        100 * 1024 * 1024, "0" * 64)
    blocks = describe_artifacts((ref,))[ref.artifact_id]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    label = json.loads(blocks[0]["text"])
    assert label["artifact"]["artifact_id"] == ref.artifact_id
    assert label["artifact"]["filename"] == ref.filename
    assert label["artifact"]["size_bytes"] == ref.size_bytes
    assert "未提供图片内容" in label["image_status"]


def test_file_attachment_keeps_its_identity():
    ref = AttachmentRef("file", AttachmentKind.FILE, "notes.txt", "text/plain", 10, "1" * 64)
    blocks = describe_artifacts((ref,))[ref.artifact_id]
    label = json.loads(blocks[0]["text"])
    assert label["artifact"]["artifact_id"] == ref.artifact_id
    assert "image_status" not in label
