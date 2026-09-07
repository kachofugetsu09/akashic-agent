from contextlib import closing
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from bootstrap.chat_api import create_chat_app
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.channels.web_chat_channel import WebChatChannel
from session.artifact_store import ARTIFACT_SCHEMA, ArtifactStore
from session.artifacts import AttachmentKind
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input


def publish(log, ref):
    writer = log.writer(
        "room", author="user", source="channel", body_types=(Input,),
        content={"file": lambda part: ContentReferences(artifact_ids=(part.value,))},
    )
    writer.append("message", Input((ContentPart("file", ref.artifact_id),)))


@pytest.mark.asyncio
async def test_independent_owners_reopen_fresh_db_and_keep_exact_message_and_bytes(tmp_path):
    path = tmp_path / "sessions.db"
    metadata = ArtifactStore(path)
    physical = ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=metadata)
    ref = await physical.import_bytes(b"original bytes", kind=AttachmentKind.FILE,
                                      filename="evidence.txt", media_type="text/plain")
    log = MessageLog(path)
    publish(log, ref)
    before = log.reader("room").snapshot()
    log.close()
    metadata.close()

    with closing(ArtifactStore(path)) as reopened, closing(MessageLog(path)) as log:
        physical = ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=reopened)
        assert log.reader("room").snapshot() == before
        assert log.reader("room").attachments("message") == (ref,)
        lease = await physical.acquire(ref)
        try:
            assert await lease.read_bytes(max_bytes=100) == b"original bytes"
        finally:
            await lease.aclose()
        log.validate_attachment_bindings()
        assert (await physical.validate_filesystem_integrity()).ready_count == 1
        with closing(sqlite3.connect(path)) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "turns" not in tables and "messages_fts" not in tables


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", ["file_published", "ready_insert"])
async def test_interrupted_artifact_publication_reopens_and_resumes_same_identity(tmp_path, monkeypatch, cut):
    path = tmp_path / "sessions.db"
    metadata = ArtifactStore(path)
    physical = ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=metadata)
    source = tmp_path / "source.txt"
    source.write_bytes(b"recoverable bytes")
    ref = await physical.inspect_file_with_artifact_id(
        source, allowed_root=tmp_path, artifact_id="fixed", kind=AttachmentKind.FILE,
        filename="source.txt", media_type="text/plain",
    )
    if cut == "ready_insert":
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("""CREATE TRIGGER cut_before_commit BEFORE UPDATE OF phase ON attachment_imports
                WHEN NEW.phase='artifact_committed' BEGIN SELECT RAISE(ABORT, 'injected cut'); END""")
    with monkeypatch.context() as patch:
        if cut == "file_published":
            patch.setattr(metadata, "mark_attachment_import_file_published", Mock(side_effect=OSError("injected cut")))
        with pytest.raises((OSError, sqlite3.IntegrityError), match="injected cut"):
            await physical.adopt_file_with_artifact_id(source, allowed_root=tmp_path, expected_ref=ref)
    assert metadata.get_attachment("fixed") is None
    metadata.close()
    if cut == "ready_insert":
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("DROP TRIGGER cut_before_commit")
    with closing(ArtifactStore(path)) as reopened:
        intent = reopened.attachment_import("fixed")
        assert intent is not None
        assert intent.phase == ("prepared" if cut == "file_published" else "file_published")
        physical = ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=reopened)
        assert await physical.adopt_file_with_artifact_id(source, allowed_root=tmp_path, expected_ref=ref) == ref
        assert (await physical.validate_filesystem_integrity()).incomplete_import_ids == ()
        assert len(reopened.list_attachments()) == 1


def test_concurrent_import_intent_reuses_first_committed_request(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(ArtifactStore(path)) as first, closing(ArtifactStore(path)) as second:
        barrier = Barrier(2)
        def accept(store):
            barrier.wait(timeout=5)
            return store.begin_attachment_import(artifact_id="shared", storage_key="uploads/artifacts/shared.bin",
                expected_size_bytes=3, expected_sha256="a" * 64, created_at="2026-09-06")
        with ThreadPoolExecutor(2) as pool:
            records = list(pool.map(accept, (first, second)))
        assert records[0] == records[1]
        assert first.incomplete_attachment_imports() == (records[0],)


@pytest.mark.parametrize("damage", ["missing_imports", "unknown_schema"])
def test_existing_artifact_state_is_not_repaired_or_reinterpreted(tmp_path, damage):
    path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        statement = ARTIFACT_SCHEMA["attachments"]
        if damage == "unknown_schema":
            statement = statement.replace("CHECK (state = 'ready')", "CHECK (state IN ('ready', 'lost'))")
        connection.execute(statement)
        connection.execute("INSERT INTO attachments VALUES (?,?,?,?,?,?,?,?,?)",
            ("existing", "uploads/artifacts/existing.bin", "file", None, None, 1, "a" * 64, "ready", "2026-09-06"))
        before = connection.execute("SELECT * FROM attachments").fetchall()
    with pytest.raises(RuntimeError, match="缺少 attachment_imports|schema 不匹配"):
        ArtifactStore(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT * FROM attachments").fetchall() == before
        assert connection.execute("SELECT 1 FROM sqlite_master WHERE name='attachment_imports'").fetchone() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["ordinal", "artifact_fk", "message_fk"])
async def test_message_binding_audit_detects_structural_damage_without_changing_bodies(tmp_path, damage):
    path = tmp_path / "sessions.db"
    with closing(ArtifactStore(path)) as metadata, closing(MessageLog(path)) as log:
        physical = ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=metadata)
        ref = await physical.import_bytes(b"file", kind=AttachmentKind.FILE, filename=None, media_type=None)
        publish(log, ref)
        with closing(sqlite3.connect(path)) as connection, connection:
            before = connection.execute("SELECT * FROM messages").fetchall()
            if damage == "ordinal":
                connection.execute("UPDATE message_attachments SET ordinal=1")
            elif damage == "artifact_fk":
                connection.execute("UPDATE message_attachments SET artifact_id='missing'")
            else:
                connection.execute("UPDATE message_attachments SET message_id='missing'")
        # 文件诊断不被授予消息库；各自检查自己的不变量。
        assert (await physical.validate_filesystem_integrity()).ready_count == 1
        with pytest.raises(ValueError, match="ordinal|foreign key"):
            log.validate_attachment_bindings()
        with closing(sqlite3.connect(path)) as connection:
            assert connection.execute("SELECT * FROM messages").fetchall() == before


def test_web_fallback_lifespan_owns_only_its_new_metadata_connection(tmp_path):
    channel = WebChatChannel()
    channel._ctx = SimpleNamespace()  # 只表示已绑定的 Channel，不提供 legacy SessionStore。
    app = create_chat_app(workspace=tmp_path, channel=channel)
    physical = channel.artifact_store
    assert physical is not None
    metadata = physical._metadata_store
    with TestClient(app):
        assert metadata.list_attachments() == ()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        metadata.list_attachments()

    # 注入的 Core 连接不归 Web app 关闭。
    with closing(ArtifactStore(tmp_path / "sessions.db")) as shared:
        channel = WebChatChannel()
        channel.bind_artifact_store(ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=shared))
        with TestClient(create_chat_app(workspace=tmp_path, channel=channel)):
            assert shared.list_attachments() == ()
        assert shared.list_attachments() == ()
