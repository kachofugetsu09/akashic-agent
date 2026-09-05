from session.message import ContentReferences
from contextlib import closing
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest
from yoyo import get_backend, read_migrations

from agent.migrations.context import bind_migration_context
from session.artifacts import AttachmentKind, AttachmentRef
from session.log import MessageLog, MessageConflict
from session.message import CallRef, ContentPart, Input, Output, ToolCall, ToolResult
from session.store import SessionStore


@pytest.fixture
def storage(tmp_path):
    path = tmp_path / "sessions.db"
    log = MessageLog(path)
    ref = AttachmentRef("image-1", AttachmentKind.IMAGE, "one.png", "image/png", 3, "a" * 64)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("INSERT INTO attachments VALUES (?,?,?,?,?,?,?,?,?)",
                           (ref.artifact_id, "artifact.bin", ref.kind, ref.filename, ref.media_type,
                            ref.size_bytes, ref.sha256, "ready", "2026-09-05T00:00:00+00:00"))
    try:
        yield path, log, ref
    finally:
        log.close()


def check_image(part):
    return ContentReferences(artifact_ids=(part.value,))


def writer(log, body, *, ref=None, checks=None):
    return log.writer("s", author="actor", source="conversation", body_types=(body,),
                      content={"image": check_image, **(checks or {})}, call_ref=ref,
                      check_call=lambda call: None)


@pytest.mark.parametrize("body", [Input, Output, ToolResult])
def test_any_content_body_pins_ordered_exact_artifacts_and_replays_once(storage, body):
    path, log, ref = storage
    parts = (ContentPart("image", ref.artifact_id), ContentPart("image", ref.artifact_id))
    call_ref = None
    if body is ToolResult:
        log.save_binding("tool", {})
        writer(log, Output).append("call", Output((ToolCall("tool", {}),), "continue"))
        call_ref = CallRef("call", 0)
        payload = ToolResult(call_ref, "success", parts)
    else:
        payload = Input(parts) if body is Input else Output(parts, "complete")
    target = writer(log, body, ref=call_ref)
    receipt = target.append("message", payload)
    assert target.append("message", payload) == receipt
    assert log.reader("s").attachments("message") == (ref, ref)
    with pytest.raises(LookupError):
        log.reader("other").attachments("message")
    with pytest.raises(MessageConflict):
        target.append("message", replace(payload, parts=parts[:1]))
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT ordinal,artifact_id FROM message_attachments ORDER BY ordinal").fetchall() == [(0, ref.artifact_id), (1, ref.artifact_id)]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_bad_ref_or_late_binding_failure_rolls_back_message_and_pins_but_keeps_artifact(storage):
    path, log, ref = storage
    target = writer(log, Input, checks={"bound": lambda part: ContentReferences(binding_ids=("missing-binding",))})
    with pytest.raises(ValueError, match="已发布"):
        target.append("forged", Input((ContentPart("image", "missing-artifact"),)))
    with pytest.raises(sqlite3.IntegrityError):
        target.append("failed", Input((ContentPart("image", ref.artifact_id), ContentPart("bound", "value"))))
    assert log.reader("s").snapshot() == ()
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM message_attachments").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM attachments").fetchone() == (1,)


def migration(tmp_path):
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "20260905_05_message_embeddings.py").write_text('from yoyo import step\nsteps = [step("SELECT 1")]\n')
    source = Path(__file__).parents[1] / "migrations/yoyo/20260905_06_message_artifacts.py"
    (directory / source.name).write_bytes(source.read_bytes())
    return read_migrations(str(directory))


@pytest.mark.parametrize("bad_direction", [False, True])
def test_yoyo_removes_only_proven_redundant_direction_and_keeps_every_reference(storage, tmp_path, bad_direction):
    path, log, ref = storage
    writer(log, Input, checks={"history.provenance": lambda part: ContentReferences()}).append(
        "old", Input((ContentPart("history.provenance", {"schema": "sessions.messages.v0", "role": "user"}),)))
    # 真实旧 owner 提供 schema；消息已处于前置 01 的不可变表示。
    old_path = tmp_path / "legacy-schema.db"
    old = SessionStore(old_path)
    old.close()
    with closing(sqlite3.connect(old_path)) as old_db:
        old_sql = old_db.execute("SELECT sql FROM sqlite_master WHERE name='message_attachments'").fetchone()[0]
        index_sql = old_db.execute("SELECT sql FROM sqlite_master WHERE name='idx_message_attachments_artifact'").fetchone()[0]
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP TABLE message_attachments")
        connection.execute(old_sql)
        connection.execute(index_sql)
        connection.execute("INSERT INTO message_attachments VALUES (?,?,?,?)", ("old", 0, ref.artifact_id, "outbound" if bad_direction else "inbound"))
        before = connection.execute("SELECT * FROM messages").fetchall()
        artifacts = connection.execute("SELECT * FROM attachments").fetchall()
    steps = migration(tmp_path)
    backend = get_backend(f'sqlite:///{tmp_path / "ledger.db"}')
    with backend, bind_migration_context(config_path=tmp_path / "config.toml", workspace=tmp_path):
        if bad_direction:
            with pytest.raises(RuntimeError, match="角色不一致"):
                backend.apply_migrations(backend.to_apply(steps))
            assert not (tmp_path / "backups/message-artifacts-v1").exists()
            return
        backend.apply_migrations(backend.to_apply(steps))
        steps[-1].module.migrate_message_artifacts(None)
    assert log.reader("s").attachments("old") == (ref,)
    writer(log, Output).append("new", Output((ContentPart("image", ref.artifact_id),), "complete"))
    backups = list((tmp_path / "backups/message-artifacts-v1").glob("*/sessions.db"))
    assert len(backups) == 1
    with closing(sqlite3.connect(backups[0])) as backup:
        assert backup.execute("SELECT * FROM message_attachments").fetchall() == [("old", 0, ref.artifact_id, "inbound")]
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT * FROM messages WHERE id='old'").fetchall() == before
        assert connection.execute("SELECT * FROM attachments").fetchall() == artifacts
        assert [row[1] for row in connection.execute("PRAGMA table_info(message_attachments)")] == ["message_id", "ordinal", "artifact_id"]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.asyncio
async def test_host_exposes_only_bounded_artifact_read_and_candidate_cannot_open(tmp_path):
    from agent.plugin_composition.artifacts import ARTIFACT_READ, ArtifactRead
    from agent.plugins.manager import PluginManager
    from agent.plugins.snapshot import lease_runtime_snapshot
    from bus.event_bus import EventBus
    from infra.channels.artifacts import ChannelAttachmentArtifactStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(workspace / "sessions.db")
    artifacts = ChannelAttachmentArtifactStore(workspace=workspace, session_store=store)
    ref = await artifacts.import_bytes(b"fixed bytes", kind=AttachmentKind.FILE,
                                       filename="evidence.txt", media_type="text/plain")
    sources = tmp_path / "plugins"
    (sources / "probe").mkdir(parents=True)
    (sources / "probe/plugin.py").write_text('''
from agent.plugin_composition import ServiceKey
api_version = 3
name = "probe"
version = "1.0.0"
inject = ()
async def apply(ctx, config):
    await ctx.provide(ServiceKey("probe"), ctx)
''')
    host = PluginManager([sources], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=tmp_path / "home", channel_attachment_store=artifacts)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            reader = snapshot.composition_root.context.require(ARTIFACT_READ)
            assert not hasattr(reader, "import_bytes") and not hasattr(reader, "resolve_refs")
            lease = await reader.acquire(ref)
            try:
                assert not hasattr(lease, "model_path")
                assert await lease.read_bytes(max_bytes=100) == b"fixed bytes"
                with pytest.raises(ValueError, match="上限"):
                    await lease.read_bytes(max_bytes=1)
            finally:
                await lease.aclose()
            with pytest.raises(RuntimeError, match="关闭"):
                await lease.read_bytes(max_bytes=100)
            with pytest.raises(RuntimeError, match="candidate"):
                await ArtifactRead(None).acquire(ref)
    finally:
        await host.terminate_all()
        store.close()
