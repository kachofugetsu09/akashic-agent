from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from infra.channels.base import AttachmentStore
from infra.mobile_realtime.attachments import (
    AttachmentChunk,
    AttachmentTransferService,
    decode_attachment_chunk,
    encode_attachment_chunk,
)
from infra.mobile_realtime.storage import (
    AttachmentStateError,
    DeviceRecord,
    MobileRealtimeStorage,
)


ATTACHMENT_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


@pytest.fixture
def storage(tmp_path: Path) -> MobileRealtimeStorage:
    value = MobileRealtimeStorage(tmp_path / "mobile.db")
    value.register_device(
        DeviceRecord(
            device_id="device-1",
            public_key="public-key",
            display_name="Pixel",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("attachments-v1",),
        )
    )
    try:
        yield value
    finally:
        value.close()


def test_binary_chunk_round_trip_and_rejects_oversized_header() -> None:
    chunk = AttachmentChunk(ATTACHMENT_ID, 4096, b"payload")

    assert decode_attachment_chunk(encode_attachment_chunk(chunk)) == chunk

    with pytest.raises(ValueError, match="header 长度无效"):
        decode_attachment_chunk((2048).to_bytes(4, "big") + b"{}data")


def test_upload_resumes_from_persisted_offset_and_verifies_digest(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    content = b"mobile attachment payload"
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "uploads"),
        max_attachment_bytes=1024,
    )
    record = service.begin_upload(
        device_id="device-1",
        attachment_id=ATTACHMENT_ID,
        session_id="mobile:00000000-0000-0000-0000-000000000001",
        filename="image.png",
        content_type="image/png",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )

    first, _ = service.append_chunk(
        device_id="device-1",
        chunk=AttachmentChunk(ATTACHMENT_ID, 0, content[:8]),
    )
    resumed = service.begin_upload(
        device_id="device-1",
        attachment_id=ATTACHMENT_ID,
        session_id=record.session_id,
        filename=record.filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
    )
    assert resumed.transferred_bytes == first.transferred_bytes == 8

    service.append_chunk(
        device_id="device-1",
        chunk=AttachmentChunk(ATTACHMENT_ID, 8, content[8:]),
    )
    with pytest.raises(AttachmentStateError, match="当前上传会话"):
        service.finish_upload(
            device_id="device-1",
            session_id="mobile:00000000-0000-0000-0000-000000000099",
            attachment_id=ATTACHMENT_ID,
        )
    unfinished = storage.read_attachment(ATTACHMENT_ID)
    assert unfinished is not None
    assert unfinished.state == "transferring"

    ready = service.finish_upload(
        device_id="device-1",
        session_id=record.session_id,
        attachment_id=ATTACHMENT_ID,
    )

    assert ready.state == "ready"
    assert Path(ready.local_path).read_bytes() == content
    assert service.resolve_uploads(
        device_id="device-1",
        session_id=ready.session_id,
        attachment_ids=[ATTACHMENT_ID],
    ) == [ready.local_path]
    with pytest.raises(AttachmentStateError, match="未就绪或不属于"):
        service.resolve_uploads(
            device_id="device-2",
            session_id=ready.session_id,
            attachment_ids=[ATTACHMENT_ID],
        )
    with pytest.raises(AttachmentStateError, match="未就绪或不属于"):
        service.resolve_uploads(
            device_id="device-1",
            session_id="mobile:00000000-0000-0000-0000-000000000099",
            attachment_ids=[ATTACHMENT_ID],
        )
    assert (
        service.finish_upload(
            device_id="device-1",
            session_id=record.session_id,
            attachment_id=ATTACHMENT_ID,
        )
        == ready
    )


def test_upload_rejects_wrong_offset_and_digest(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "uploads"),
        max_attachment_bytes=1024,
    )
    service.begin_upload(
        device_id="device-1",
        attachment_id=ATTACHMENT_ID,
        session_id="mobile:00000000-0000-0000-0000-000000000001",
        filename="bad.bin",
        content_type="application/octet-stream",
        size_bytes=3,
        sha256=hashlib.sha256(b"good").hexdigest(),
    )

    with pytest.raises(AttachmentStateError, match="offset 不连续"):
        service.append_chunk(
            device_id="device-1",
            chunk=AttachmentChunk(ATTACHMENT_ID, 1, b"bad"),
        )

    service.append_chunk(
        device_id="device-1",
        chunk=AttachmentChunk(ATTACHMENT_ID, 0, b"bad"),
    )
    with pytest.raises(AttachmentStateError, match="SHA-256"):
        service.finish_upload(
            device_id="device-1",
            session_id="mobile:00000000-0000-0000-0000-000000000001",
            attachment_id=ATTACHMENT_ID,
        )

    restarted = service.begin_upload(
        device_id="device-1",
        attachment_id=ATTACHMENT_ID,
        session_id="mobile:00000000-0000-0000-0000-000000000001",
        filename="bad.bin",
        content_type="application/octet-stream",
        size_bytes=3,
        sha256=hashlib.sha256(b"good").hexdigest(),
    )
    assert restarted.state == "transferring"
    assert restarted.transferred_bytes == 0
    assert Path(restarted.local_path).stat().st_size == 0


def test_begin_recovers_when_file_is_shorter_than_committed_offset(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    content = b"resume"
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "uploads"),
        max_attachment_bytes=1024,
    )
    record = service.begin_upload(
        device_id="device-1",
        attachment_id=ATTACHMENT_ID,
        session_id="mobile:00000000-0000-0000-0000-000000000001",
        filename="resume.bin",
        content_type="application/octet-stream",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    _ = service.append_chunk(
        device_id="device-1",
        chunk=AttachmentChunk(ATTACHMENT_ID, 0, content[:4]),
    )
    Path(record.local_path).write_bytes(b"r")

    recovered = service.begin_upload(
        device_id="device-1",
        attachment_id=ATTACHMENT_ID,
        session_id=record.session_id,
        filename=record.filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
    )

    assert recovered.transferred_bytes == 0
    assert Path(recovered.local_path).stat().st_size == 0


@pytest.mark.parametrize(
    "filename",
    ["../secret", "folder\\secret", "bad\x00name", "bad\nname", "bad\x7fname"],
)
def test_begin_rejects_filename_that_is_not_plain(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
    filename: str,
) -> None:
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "uploads"),
        max_attachment_bytes=1024,
    )

    with pytest.raises(ValueError, match="纯文件名"):
        service.begin_upload(
            device_id="device-1",
            attachment_id=ATTACHMENT_ID,
            session_id="mobile:00000000-0000-0000-0000-000000000001",
            filename=filename,
            content_type="application/octet-stream",
            size_bytes=1,
            sha256=hashlib.sha256(b"x").hexdigest(),
        )


def test_concurrent_begin_reuses_one_attachment_record(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "uploads"),
        max_attachment_bytes=1024,
    )

    def begin() -> object:
        return service.begin_upload(
            device_id="device-1",
            attachment_id=ATTACHMENT_ID,
            session_id="mobile:00000000-0000-0000-0000-000000000001",
            filename="same.bin",
            content_type="application/octet-stream",
            size_bytes=1,
            sha256=hashlib.sha256(b"x").hexdigest(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(lambda _: begin(), range(2))

    assert first == second
    assert len(tuple((tmp_path / "uploads").iterdir())) == 1
