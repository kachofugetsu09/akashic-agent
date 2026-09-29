"""Image request limits apply to bytes, not the number of history messages."""

import io

import pytest
from PIL import Image

from agent.media import MAX_IMAGE_FILE_BYTES, MAX_IMAGE_TOTAL_BYTES, encode_image_bytes
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.models import content
from session.artifact_store import ArtifactStore
from session.artifacts import AttachmentKind, AttachmentReadLease, AttachmentRef


class UnusedReader:
    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease:
        raise AssertionError("Over-budget or text-only requests must not read images")


@pytest.mark.asyncio
@pytest.mark.parametrize("sizes,reason", [
    ([MAX_IMAGE_FILE_BYTES + 1], "单张图片"),
    ([MAX_IMAGE_FILE_BYTES, MAX_IMAGE_FILE_BYTES, 1], "含历史消息"),
])
async def test_raw_image_limits_fail_before_acquiring_files(sizes, reason):
    """Reject excessive raw bytes before reading any artifact."""
    refs = tuple(AttachmentRef(
        f"image-{index}", AttachmentKind.IMAGE, None, "image/png", size, "0" * 64,
    ) for index, size in enumerate(sizes))
    with pytest.raises(ValueError, match=reason):
        await content.load_artifacts(UnusedReader(), refs, accepts_images=True)


@pytest.mark.asyncio
async def test_text_only_model_keeps_image_identity_without_reading():
    """A model without vision receives an explicit attachment label."""
    ref = AttachmentRef("image", AttachmentKind.IMAGE, None, "image/png",
                        MAX_IMAGE_TOTAL_BYTES + 1, "0" * 64)
    result = await content.load_artifacts(UnusedReader(), (ref,), accepts_images=False)
    assert len(result[ref.artifact_id]) == 1
    label = result[ref.artifact_id][0]
    assert label["type"] == "text"
    assert "当前模型不接收图片" in label["text"]
    assert ref.artifact_id in label["text"]


@pytest.mark.asyncio
async def test_encoded_image_total_remains_bounded(tmp_path, monkeypatch):
    """Check the encoded total after real artifact reads and image conversion."""
    data = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(data, format="PNG")
    raw = data.getvalue()
    monkeypatch.setattr(content, "MAX_IMAGE_DATA_URI_TOTAL_BYTES", 2 * len(encode_image_bytes(raw)) - 1)
    metadata = ArtifactStore(tmp_path / "artifacts.db")
    try:
        store = ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=metadata)
        ref = await store.import_bytes(raw, kind=AttachmentKind.IMAGE,
                                       filename="image.png", media_type="image/png")
        with pytest.raises(ValueError, match="编码字节上限"):
            await content.load_artifacts(store, (ref, ref), accepts_images=True)
    finally:
        metadata.close()
