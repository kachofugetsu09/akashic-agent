from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from pathlib import Path

import pytest
from PIL import Image

from agent.plugin_composition.channels import AttachmentKind, AttachmentRef
from bootstrap.channel_attachment_import import import_channel_attachments
from bus.events import (
    AttachmentKind as LegacyAttachmentKind,
    ChannelAttachment,
)
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from session.store import SessionStore


@pytest.fixture
def stores(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_store = SessionStore(workspace / "sessions.db")
    artifact_store = ChannelAttachmentArtifactStore(
        workspace=workspace,
        session_store=session_store,
    )
    try:
        yield session_store, artifact_store
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_import_publishes_ready_metadata_and_verified_read_lease(stores) -> None:
    session_store, artifact_store = stores
    payload = b"immutable attachment"

    ref = await artifact_store.import_bytes(
        payload,
        kind=AttachmentKind.FILE,
        filename="evidence.txt",
        media_type="text/plain",
    )
    record = session_store.get_attachment(ref.artifact_id)
    assert record is not None
    assert record.storage_key == f"uploads/artifacts/{ref.artifact_id}.bin"
    assert record.state == "ready"
    assert not Path(record.storage_key).is_absolute()

    lease = await artifact_store.acquire(ref)
    assert lease.ref == ref
    assert await lease.read_bytes(max_bytes=len(payload)) == payload
    await lease.aclose()
    with pytest.raises(RuntimeError, match="已关闭"):
        await lease.read_bytes(max_bytes=len(payload))
    report = await artifact_store.validate_filesystem_integrity()
    assert report.ready_count == 1
    assert report.verified_bytes == len(payload)
    assert report.orphan_artifact_ids == ()
    assert report.incomplete_import_ids == ()


@pytest.mark.asyncio
async def test_import_owns_image_kind_from_file_signature(stores) -> None:
    _session_store, artifact_store = stores
    image_path = Path(_session_store.db_path).parent / "extensionless"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image_path, format="PNG")

    image_ref = await artifact_store.import_bytes(
        image_path.read_bytes(),
        kind=AttachmentKind.FILE,
        filename="extensionless",
        media_type=None,
    )
    assert image_ref.kind is AttachmentKind.IMAGE
    assert image_ref.media_type == "image/png"

    fake_ref = await artifact_store.import_bytes(
        b"not an image",
        kind=AttachmentKind.IMAGE,
        filename="forged.png",
        media_type="image/png",
    )
    assert fake_ref.kind is AttachmentKind.FILE


@pytest.mark.asyncio
async def test_ready_row_failure_retains_only_auditable_physical_orphan(
    stores,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_store, artifact_store = stores

    def fail_publication(**_kwargs):
        raise OSError("injected sqlite publication failure")

    monkeypatch.setattr(session_store, "register_ready_attachment", fail_publication)
    with pytest.raises(OSError, match="injected"):
        await artifact_store.import_bytes(
            b"orphan bytes",
            kind=AttachmentKind.FILE,
            filename="orphan.bin",
            media_type="application/octet-stream",
        )

    orphan_ids = artifact_store.audit_orphan_artifact_ids()
    assert len(orphan_ids) == 1
    assert session_store.get_attachment(orphan_ids[0]) is None
    intent = session_store.attachment_import(orphan_ids[0])
    assert intent is not None
    assert intent.phase == "file_published"
    assert intent.error == "OSError: injected sqlite publication failure"
    assert session_store.incomplete_attachment_imports() == (intent,)
    root = Path(session_store.db_path).parent / "uploads" / "artifacts"
    assert [path.name for path in root.glob("*.part")] == []
    assert (root / f"{orphan_ids[0]}.bin").read_bytes() == b"orphan bytes"
    report = await artifact_store.validate_filesystem_integrity()
    assert report.ready_count == 0
    assert report.orphan_artifact_ids == orphan_ids
    assert report.incomplete_import_ids == orphan_ids


@pytest.mark.asyncio
async def test_acquire_rejects_forged_metadata_and_file_drift(stores) -> None:
    session_store, artifact_store = stores
    ref = await artifact_store.import_bytes(
        b"original",
        kind=AttachmentKind.IMAGE,
        filename="photo.png",
        media_type="image/png",
    )
    forged = AttachmentRef(
        artifact_id=ref.artifact_id,
        kind=ref.kind,
        filename="other.png",
        media_type=ref.media_type,
        size_bytes=ref.size_bytes,
        sha256=ref.sha256,
    )
    with pytest.raises(ValueError, match="权威 metadata"):
        await artifact_store.acquire(forged)

    record = session_store.get_attachment(ref.artifact_id)
    assert record is not None
    path = Path(session_store.db_path).parent / record.storage_key
    path.write_bytes(b"drifted!")
    with pytest.raises(ValueError, match="size 已漂移|hash 已漂移"):
        await artifact_store.acquire(ref)


@pytest.mark.asyncio
async def test_import_rejects_symlinked_artifact_parent_before_write(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    os.symlink(outside, workspace / "uploads")
    session_store = SessionStore(workspace / "sessions.db")
    artifact_store = ChannelAttachmentArtifactStore(
        workspace=workspace,
        session_store=session_store,
    )
    try:
        with pytest.raises(ValueError, match="符号链接"):
            await artifact_store.import_bytes(
                b"secret",
                kind=AttachmentKind.FILE,
                filename=None,
                media_type=None,
            )

        assert list(outside.iterdir()) == []
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_import_finishes_durable_publication_before_restoring_cancellation(
    stores,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_store, artifact_store = stores
    entered = threading.Event()
    release = threading.Event()
    original = session_store.mark_attachment_import_file_published

    def blocking_mark(artifact_id: str, *, updated_at: str) -> None:
        entered.set()
        assert release.wait(timeout=5)
        original(artifact_id, updated_at=updated_at)

    monkeypatch.setattr(
        session_store,
        "mark_attachment_import_file_published",
        blocking_mark,
    )
    task = asyncio.create_task(
        artifact_store.import_bytes(
            b"cancel after admission",
            kind=AttachmentKind.FILE,
            filename="cancel.txt",
            media_type="text/plain",
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    report = session_store.validate_attachment_metadata_integrity()
    assert report.artifact_count == 1
    assert report.incomplete_import_ids == ()
    assert artifact_store.audit_orphan_artifact_ids() == ()
    root = Path(session_store.db_path).parent / "uploads" / "artifacts"
    assert list(root.glob("*.part")) == []


@pytest.mark.asyncio
async def test_adopted_file_remains_readable_after_source_is_deleted(stores) -> None:
    session_store, artifact_store = stores
    workspace = Path(session_store.db_path).parent
    source_root = workspace / "uploads" / "legacy"
    source_root.mkdir(parents=True)
    source = source_root / "photo.jpg"
    source.write_bytes(b"legacy provider bytes")

    ref = await artifact_store.adopt_file(
        source,
        allowed_root=workspace / "uploads",
        kind=AttachmentKind.IMAGE,
        filename="photo.jpg",
        media_type="image/jpeg",
    )
    source.unlink()

    lease = await artifact_store.acquire(ref)
    assert await lease.read_bytes(max_bytes=1024) == b"legacy provider bytes"
    await lease.aclose()


@pytest.mark.asyncio
async def test_outbound_import_returns_opaque_ref_without_exposing_source_path(
    stores,
    tmp_path: Path,
) -> None:
    _, artifact_store = stores
    source = tmp_path / "authorized-source.txt"
    source.write_bytes(b"authorized outbound bytes")

    refs = await import_channel_attachments(
        artifact_store,
        (
            ChannelAttachment(
                LegacyAttachmentKind.FILE,
                str(source),
                "evidence.txt",
            ),
        ),
    )
    source.unlink()

    assert len(refs) == 1
    assert refs[0].filename == "evidence.txt"
    assert str(source) not in repr(refs[0])
    lease = await artifact_store.acquire(refs[0])
    assert await lease.read_bytes(max_bytes=1024) == b"authorized outbound bytes"
    await lease.aclose()


@pytest.mark.asyncio
async def test_fixed_artifact_identity_is_idempotent_and_rejects_source_drift(
    stores,
) -> None:
    session_store, artifact_store = stores
    workspace = Path(session_store.db_path).parent
    source_root = workspace / "uploads" / "mobile"
    source_root.mkdir(parents=True)
    source = source_root / "upload.bin"
    source.write_bytes(b"mobile finalized bytes")

    expected = await artifact_store.inspect_file_with_artifact_id(
        source,
        allowed_root=source_root,
        artifact_id="mobile-fixed-artifact",
        kind=AttachmentKind.FILE,
        filename="upload.bin",
        media_type="application/octet-stream",
    )

    first = await artifact_store.adopt_file_with_artifact_id(
        source,
        allowed_root=source_root,
        expected_ref=expected,
    )
    second = await artifact_store.adopt_file_with_artifact_id(
        source,
        allowed_root=source_root,
        expected_ref=expected,
    )
    assert second == first
    assert len(session_store.list_attachments()) == 1

    source.write_bytes(b"different finalized bytes")
    with pytest.raises(ValueError, match="durable ref 不一致"):
        await artifact_store.adopt_file_with_artifact_id(
            source,
            allowed_root=source_root,
            expected_ref=expected,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prepared", "file_published"])
async def test_fixed_artifact_identity_recovers_published_bytes_before_ready_row(
    stores,
    phase: str,
) -> None:
    session_store, artifact_store = stores
    workspace = Path(session_store.db_path).parent
    source_root = workspace / "uploads" / "mobile"
    source_root.mkdir(parents=True)
    source = source_root / "recover.bin"
    payload = b"published before database ready"
    source.write_bytes(payload)
    artifact_id = f"recover-{phase}"
    expected = await artifact_store.inspect_file_with_artifact_id(
        source,
        allowed_root=source_root,
        artifact_id=artifact_id,
        kind=AttachmentKind.FILE,
        filename="recover.bin",
        media_type="application/octet-stream",
    )
    storage_key = f"uploads/artifacts/{artifact_id}.bin"
    created_at = "2026-08-17T00:00:00+00:00"
    _ = session_store.begin_attachment_import(
        artifact_id=artifact_id,
        storage_key=storage_key,
        expected_size_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        created_at=created_at,
    )
    artifact_root = workspace / "uploads" / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / f"{artifact_id}.bin").write_bytes(payload)
    if phase == "file_published":
        session_store.mark_attachment_import_file_published(
            artifact_id,
            updated_at=created_at,
        )

    ref = await artifact_store.adopt_file_with_artifact_id(
        source,
        allowed_root=source_root,
        expected_ref=expected,
    )

    assert ref.artifact_id == artifact_id
    intent = session_store.attachment_import(artifact_id)
    assert intent is not None and intent.phase == "artifact_committed"
    assert session_store.get_attachment(artifact_id) is not None


@pytest.mark.asyncio
async def test_adopt_rejects_symlink_and_source_mutation_without_ready_row(
    stores,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_store, artifact_store = stores
    workspace = Path(session_store.db_path).parent
    source_root = workspace / "uploads" / "legacy"
    source_root.mkdir(parents=True)
    outside = workspace / "outside.bin"
    outside.write_bytes(b"outside")
    symlink = source_root / "link.bin"
    symlink.symlink_to(outside)
    with pytest.raises(ValueError, match="符号链接"):
        await artifact_store.adopt_file(
            symlink,
            allowed_root=workspace / "uploads",
            kind=AttachmentKind.FILE,
            filename="link.bin",
            media_type="application/octet-stream",
        )
    assert session_store.validate_attachment_metadata_integrity().artifact_count == 0

    source = source_root / "mutable.bin"
    source.write_bytes(b"before")
    original_publish = artifact_store._publish_content

    def mutate_then_publish(ref, writer):
        source.write_bytes(b"after mutation")
        return original_publish(ref, writer)

    monkeypatch.setattr(artifact_store, "_publish_content", mutate_then_publish)
    with pytest.raises(ValueError, match="identity 已漂移"):
        await artifact_store.adopt_file(
            source,
            allowed_root=workspace / "uploads",
            kind=AttachmentKind.FILE,
            filename="mutable.bin",
            media_type="application/octet-stream",
        )

    report = session_store.validate_attachment_metadata_integrity()
    assert report.artifact_count == 0
    assert len(report.incomplete_import_ids) == 1
    root = workspace / "uploads" / "artifacts"
    assert list(root.glob("*.part")) == []
    assert list(root.glob("*.bin")) == []


@pytest.mark.asyncio
async def test_concurrent_first_imports_share_only_the_directory_owner(stores) -> None:
    session_store, artifact_store = stores

    refs = await asyncio.gather(
        *(
            artifact_store.import_bytes(
                f"payload-{index}".encode(),
                kind=AttachmentKind.FILE,
                filename=f"{index}.txt",
                media_type="text/plain",
            )
            for index in range(8)
        )
    )

    assert len({ref.artifact_id for ref in refs}) == 8
    report = await artifact_store.validate_filesystem_integrity()
    assert report.ready_count == 8
    assert report.incomplete_import_ids == ()
    assert session_store.validate_attachment_metadata_integrity().artifact_count == 8


@pytest.mark.asyncio
async def test_cancelled_acquire_closes_worker_owned_file_descriptor(
    stores,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_store, artifact_store = stores
    ref = await artifact_store.import_bytes(
        b"fd owner",
        kind=AttachmentKind.FILE,
        filename="fd.txt",
        media_type="text/plain",
    )
    entered = threading.Event()
    release = threading.Event()
    opened_fds: list[int] = []
    original = artifact_store._open_verified

    def blocking_open(record) -> int:
        fd = original(record)
        opened_fds.append(fd)
        entered.set()
        assert release.wait(timeout=5)
        return fd

    monkeypatch.setattr(artifact_store, "_open_verified", blocking_open)
    task = asyncio.create_task(artifact_store.acquire(ref))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(opened_fds) == 1
    with pytest.raises(OSError):
        os.fstat(opened_fds[0])


@pytest.mark.asyncio
async def test_cancelled_integrity_check_closes_worker_owned_file_descriptor(
    stores,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session_store, artifact_store = stores
    _ = await artifact_store.import_bytes(
        b"integrity fd owner",
        kind=AttachmentKind.FILE,
        filename="integrity.txt",
        media_type="text/plain",
    )
    entered = threading.Event()
    release = threading.Event()
    opened_fds: list[int] = []
    original = artifact_store._open_verified

    def blocking_open(record) -> int:
        fd = original(record)
        opened_fds.append(fd)
        entered.set()
        assert release.wait(timeout=5)
        return fd

    monkeypatch.setattr(artifact_store, "_open_verified", blocking_open)
    task = asyncio.create_task(artifact_store.validate_filesystem_integrity())
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(opened_fds) == 1
    with pytest.raises(OSError):
        os.fstat(opened_fds[0])


@pytest.mark.asyncio
async def test_adopt_rejects_source_path_replacement_after_fd_open(
    stores,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_store, artifact_store = stores
    workspace = Path(session_store.db_path).parent
    source_root = workspace / "uploads" / "legacy"
    source_root.mkdir(parents=True)
    source = source_root / "replace.bin"
    source.write_bytes(b"old inode")
    replacement = source_root / "replacement.bin"
    replacement.write_bytes(b"new inode")
    original_open = os.open
    source_open_count = 0

    def replace_after_copy_fd_open(path, flags, mode=0o777):
        nonlocal source_open_count
        fd = original_open(path, flags, mode)
        if Path(path) == source:
            source_open_count += 1
            if source_open_count == 2:
                replacement.replace(source)
        return fd

    monkeypatch.setattr(os, "open", replace_after_copy_fd_open)
    with pytest.raises(ValueError, match="identity 已漂移"):
        await artifact_store.adopt_file(
            source,
            allowed_root=workspace / "uploads",
            kind=AttachmentKind.FILE,
            filename="replace.bin",
            media_type="application/octet-stream",
        )

    report = session_store.validate_attachment_metadata_integrity()
    assert report.artifact_count == 0
    assert len(report.incomplete_import_ids) == 1
