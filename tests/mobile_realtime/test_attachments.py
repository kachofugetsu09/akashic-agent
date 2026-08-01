from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from infra.channels.base import AttachmentStore
from infra.mobile_realtime.attachments import (
    AttachmentChunk,
    AttachmentRequestError,
    AttachmentTransferService,
    MAX_ATTACHMENT_CHUNK_BYTES,
    decode_attachment_chunk,
    encode_attachment_chunk,
)
from infra.mobile_realtime.storage import (
    AttachmentRecord,
    AttachmentStateError,
    DeviceRecord,
    MobileRealtimeStorage,
    UnknownDeviceError,
)


ATTACHMENT_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


def test_outbound_attachment_and_all_device_inbox_rows_roll_back_together(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "attachments"),
        max_attachment_bytes=1024,
    )
    source = tmp_path / "report.pdf"
    source.write_bytes(b"report")
    candidates = service.snapshot_outbound_batch(
        session_id="mobile:session-1",
        local_media_paths=(source,),
    )

    with pytest.raises(UnknownDeviceError, match="missing-device"):
        storage.commit_outbound_event(
            candidates,
            device_ids=("device-1", "missing-device"),
            event_id="event-atomic",
            envelope_builder=lambda _records: '{"kind":"event"}',
            created_at=datetime.now(timezone.utc),
        )

    assert storage.read_attachment(candidates[0].attachment_id) is None
    assert storage.count_durable_events("device-1") == 0
    assert Path(candidates[0].local_path).exists()
    service.cleanup_outbound_candidates(candidates)
    assert not Path(candidates[0].local_path).exists()


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[MobileRealtimeStorage]:
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


def test_outbound_registration_copies_and_stably_reuses_content(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    content = b"outbound image"
    source = tmp_path / "answer.png"
    source.write_bytes(content)
    root = tmp_path / "outbound"
    service = AttachmentTransferService(
        storage,
        AttachmentStore(root),
        max_attachment_bytes=1024,
    )
    concurrent_storage = MobileRealtimeStorage(storage.db_path)
    concurrent_service = AttachmentTransferService(
        concurrent_storage,
        AttachmentStore(root),
        max_attachment_bytes=1024,
    )
    session_id = "mobile:00000000-0000-0000-0000-000000000001"

    def register(value: AttachmentTransferService) -> AttachmentRecord:
        return value.register_outbound(
            session_id=session_id,
            local_media_path=source,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = executor.map(
                register,
                (service, concurrent_service),
            )
    finally:
        concurrent_storage.close()

    assert second == first
    assert first.direction == "outbound"
    assert first.device_id is None
    assert first.state == "ready"
    assert first.transferred_bytes == len(content)
    assert first.sha256 == hashlib.sha256(content).hexdigest()
    assert first.content_type == "image/png"
    assert Path(first.local_path) != source
    assert Path(first.local_path).read_bytes() == content
    assert Path(first.local_path).stat().st_mode & 0o777 == 0o444
    assert len(tuple(root.iterdir())) == 1

    other_session = service.register_outbound(
        session_id="mobile:00000000-0000-0000-0000-000000000002",
        local_media_path=source,
    )
    assert other_session.attachment_id != first.attachment_id


def test_outbound_chunk_is_bounded_and_hides_local_path(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    content = b"a" * (MAX_ATTACHMENT_CHUNK_BYTES + 7)
    source = tmp_path / "archive.bin"
    source.write_bytes(content)
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "outbound"),
        max_attachment_bytes=len(content),
    )
    session_id = "mobile:00000000-0000-0000-0000-000000000001"
    record = service.register_outbound(
        session_id=session_id,
        local_media_path=source,
    )

    first = service.read_outbound_chunk(
        session_id=session_id,
        attachment_id=record.attachment_id,
        offset=0,
    )
    final = service.read_outbound_chunk(
        session_id=session_id,
        attachment_id=record.attachment_id,
        offset=len(first.data),
    )

    assert len(first.data) == MAX_ATTACHMENT_CHUNK_BYTES
    assert first.eof is False
    assert final.data == b"a" * 7
    assert final.eof is True
    assert "local_path" not in first.descriptor
    assert first.descriptor == {
        "attachment_id": record.attachment_id,
        "filename": "archive.bin",
        "content_type": "application/octet-stream",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_outbound_chunk_enforces_session_state_and_offset(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    source = tmp_path / "answer.txt"
    source.write_text("answer", encoding="utf-8")
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "outbound"),
        max_attachment_bytes=1024,
    )
    session_id = "mobile:00000000-0000-0000-0000-000000000001"
    record = service.register_outbound(
        session_id=session_id,
        local_media_path=source,
    )

    with pytest.raises(AttachmentStateError, match="下载会话"):
        service.read_outbound_chunk(
            session_id="mobile:00000000-0000-0000-0000-000000000099",
            attachment_id=record.attachment_id,
            offset=0,
        )
    with pytest.raises(AttachmentRequestError, match="不能为负数"):
        service.read_outbound_chunk(
            session_id=session_id,
            attachment_id=record.attachment_id,
            offset=-1,
        )
    with pytest.raises(AttachmentRequestError, match="必须小于文件大小"):
        service.read_outbound_chunk(
            session_id=session_id,
            attachment_id=record.attachment_id,
            offset=record.size_bytes,
        )


def test_outbound_registration_rejects_boundary_and_cleans_failed_copy(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    root = tmp_path / "outbound"
    service = AttachmentTransferService(
        storage,
        AttachmentStore(root),
        max_attachment_bytes=3,
    )
    oversized = tmp_path / "large.bin"
    oversized.write_bytes(b"four")

    with pytest.raises(AttachmentRequestError, match="1..3"):
        service.register_outbound(
            session_id="mobile:00000000-0000-0000-0000-000000000001",
            local_media_path=oversized,
        )
    assert not root.exists() or tuple(root.iterdir()) == ()

    invalid_name = tmp_path / "bad\nname"
    invalid_name.write_bytes(b"x")
    with pytest.raises(AttachmentRequestError, match="纯文件名"):
        service.register_outbound(
            session_id="mobile:00000000-0000-0000-0000-000000000001",
            local_media_path=invalid_name,
        )

    with pytest.raises(AttachmentRequestError, match="普通文件"):
        service.register_outbound(
            session_id="mobile:00000000-0000-0000-0000-000000000001",
            local_media_path=tmp_path,
        )


def test_outbound_registration_cleans_snapshot_when_database_fails(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    root = tmp_path / "outbound"
    source = tmp_path / "answer.bin"
    source.write_bytes(b"answer")
    service = AttachmentTransferService(
        storage,
        AttachmentStore(root),
        max_attachment_bytes=1024,
    )
    storage.close()

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        service.register_outbound(
            session_id="mobile:00000000-0000-0000-0000-000000000001",
            local_media_path=source,
        )
    assert not root.exists() or tuple(root.iterdir()) == ()


def test_outbound_batch_is_atomic_and_enforces_message_limits(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    root = tmp_path / "outbound"
    first_source = tmp_path / "first.bin"
    second_source = tmp_path / "second.bin"
    first_source.write_bytes(b"one")
    second_source.write_bytes(b"two")
    service = AttachmentTransferService(
        storage,
        AttachmentStore(root),
        max_attachment_bytes=5,
    )
    session_id = "mobile:00000000-0000-0000-0000-000000000001"

    with pytest.raises(AttachmentRequestError, match="总量"):
        service.register_outbound_batch(
            session_id=session_id,
            local_media_paths=(first_source, second_source),
        )
    assert tuple(root.iterdir()) == ()
    database = sqlite3.connect(storage.db_path)
    try:
        outbound_count = database.execute(
            "SELECT COUNT(*) FROM mobile_attachments WHERE direction = 'outbound'"
        ).fetchone()
    finally:
        database.close()
    assert outbound_count == (0,)

    with pytest.raises(AttachmentRequestError, match="1..10"):
        service.register_outbound_batch(
            session_id=session_id,
            local_media_paths=(first_source,) * 11,
        )


def test_outbound_batch_reuses_duplicate_content_in_one_transaction(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    source = tmp_path / "same.txt"
    source.write_text("same", encoding="utf-8")
    root = tmp_path / "outbound"
    service = AttachmentTransferService(
        storage,
        AttachmentStore(root),
        max_attachment_bytes=1024,
    )

    first, second = service.register_outbound_batch(
        session_id="mobile:00000000-0000-0000-0000-000000000001",
        local_media_paths=(source, source),
    )

    assert first == second
    assert len(tuple(root.iterdir())) == 1


def test_outbound_inherits_ready_upload_metadata_by_local_path(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    content = b"camera bytes"
    root = tmp_path / "attachments"
    service = AttachmentTransferService(
        storage,
        AttachmentStore(root),
        max_attachment_bytes=1024,
    )
    session_id = "mobile:00000000-0000-0000-0000-000000000001"
    upload = service.begin_upload(
        device_id="device-1",
        attachment_id=ATTACHMENT_ID,
        session_id=session_id,
        filename="camera-export",
        content_type="image/jpeg",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    service.append_chunk(
        device_id="device-1",
        chunk=AttachmentChunk(ATTACHMENT_ID, 0, content),
    )
    ready = service.finish_upload(
        device_id="device-1",
        session_id=session_id,
        attachment_id=ATTACHMENT_ID,
    )

    outbound = service.register_outbound(
        session_id=session_id,
        local_media_path=ready.local_path,
    )

    assert outbound.filename == upload.filename == "camera-export"
    assert outbound.content_type == upload.content_type == "image/jpeg"
    assert Path(outbound.local_path).suffix == ".bin"


def test_message_binding_survives_source_deletion_and_preserves_mime_identity(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    source = tmp_path / "reaction.bin"
    source.write_bytes(b"same-content")
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "outbound"),
        max_attachment_bytes=1024,
    )
    session_id = "mobile:00000000-0000-0000-0000-000000000001"
    first = service.register_outbound_batch(
        session_id=session_id,
        local_media_paths=(source,),
        metadata_overrides=(("reaction", "image/gif"),),
        message_id=f"{session_id}:1",
    )[0]
    different_mime = service.register_outbound_batch(
        session_id=session_id,
        local_media_paths=(source,),
        metadata_overrides=(("reaction", "application/octet-stream"),),
        message_id=f"{session_id}:2",
    )[0]

    source.unlink()
    restored = service.read_message_outbound(
        session_id=session_id,
        message_id=f"{session_id}:1",
    )

    assert restored == (first,)
    assert Path(first.local_path).read_bytes() == b"same-content"
    assert different_mime.attachment_id != first.attachment_id
    assert different_mime.content_type == "application/octet-stream"


def test_outbound_rejects_symlink_and_persistent_root_failure(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    symlink = tmp_path / "link.bin"
    symlink.symlink_to(source)
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "outbound"),
        max_attachment_bytes=1024,
    )

    with pytest.raises(AttachmentRequestError, match="符号链接"):
        service.register_outbound(
            session_id="mobile:00000000-0000-0000-0000-000000000001",
            local_media_path=symlink,
        )

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block", encoding="utf-8")
    strict_service = AttachmentTransferService(
        storage,
        AttachmentStore(blocker / "outbound"),
        max_attachment_bytes=1024,
    )
    with pytest.raises(NotADirectoryError):
        strict_service.register_outbound(
            session_id="mobile:00000000-0000-0000-0000-000000000001",
            local_media_path=source,
        )


def test_outbound_storage_batch_rolls_back_all_rows_on_primary_key_conflict(
    storage: MobileRealtimeStorage,
    tmp_path: Path,
) -> None:
    service = AttachmentTransferService(
        storage,
        AttachmentStore(tmp_path / "attachments"),
        max_attachment_bytes=1024,
    )
    service.begin_upload(
        device_id="device-1",
        attachment_id=ATTACHMENT_ID,
        session_id="mobile:00000000-0000-0000-0000-000000000001",
        filename="occupied.bin",
        content_type="application/octet-stream",
        size_bytes=1,
        sha256=hashlib.sha256(b"x").hexdigest(),
    )
    now = datetime.now(timezone.utc)

    def candidate(attachment_id: str, filename: str) -> AttachmentRecord:
        path = tmp_path / f"{filename}.canonical"
        path.write_bytes(b"x")
        return AttachmentRecord(
            attachment_id=attachment_id,
            device_id=None,
            session_id="mobile:00000000-0000-0000-0000-000000000001",
            direction="outbound",
            filename=filename,
            content_type="application/octet-stream",
            size_bytes=1,
            sha256=hashlib.sha256(b"x").hexdigest(),
            local_path=str(path),
            transferred_bytes=1,
            state="ready",
            created_at=now,
            updated_at=now,
        )

    first = candidate("01ARZ3NDEKTSV4RRFFQ69G5FAW", "first.bin")
    conflicting = candidate(ATTACHMENT_ID, "second.bin")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint"):
        storage.create_or_read_outbound_attachments((first, conflicting))

    assert storage.read_attachment(first.attachment_id) is None
    occupied = storage.read_attachment(ATTACHMENT_ID)
    assert occupied is not None
    assert occupied.direction == "upload"
