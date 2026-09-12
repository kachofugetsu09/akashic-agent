from session.message import ContentReferences
from contextlib import closing
from dataclasses import replace
import sqlite3

import pytest

from session.artifacts import AttachmentKind, AttachmentRef
from session.log import MessageLog, MessageConflict
from session.message import CallRef, ContentPart, Input, Output, ToolCall, ToolResult
from session.artifact_store import ArtifactStore


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


@pytest.mark.asyncio
async def test_host_exposes_only_bounded_artifact_read_and_candidate_cannot_open(tmp_path):
    from agent.plugin_composition.artifacts import ARTIFACT_READ, ArtifactRead
    from agent.plugins.manager import PluginManager
    from agent.plugins.snapshot import lease_runtime_snapshot
    from bus.event_bus import EventBus
    from infra.channels.artifacts import ChannelAttachmentArtifactStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ArtifactStore(workspace / "sessions.db")
    artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=store)
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


def test_batch_artifacts_keep_order_and_scope_without_decoding_bodies(storage, monkeypatch):
    _path, log, ref = storage
    target = writer(log, Input)
    target.append("image", Input((ContentPart("image", ref.artifact_id),)))
    target.append("empty", Input(()))
    reader = log.reader("s")
    import session.log as messages
    def no_decode(_row):
        raise AssertionError("附件查询不能解码消息正文")
    monkeypatch.setattr(messages, "_message", no_decode)
    assert reader.attachments_for(("empty", "image", "image")) == (ref, ref)
    assert reader.attachments_for(()) == ()
    with pytest.raises(LookupError):
        reader.attachments_for(("image", "missing"))
    with pytest.raises(LookupError):
        log.reader("other").attachments_for(("image",))
