from contextlib import closing

import pytest

from plugins.compaction.records import SummaryRecord, SummaryRecords
from plugins.content.plugin import check_text
from session.log import MessageConflict, MessageLog, OwnerTransaction
from session.message import ContentPart, Input


def record(reference, *, parent=None, ids=("u1",)):
    return SummaryRecord(
        reference=reference, session_id="s", generation=1 if parent is None else parent.generation + 1,
        parent=None if parent is None else parent.reference, source_message_ids=ids,
        content="summary " + reference, model_call_ids=("call:" + reference,), trigger="soft_limit",
        context_window=32000, max_output_tokens=4096, keep_recent_tokens=20000,
        tokens_before=27000, tokens_after=18000,
    )


def inputs(log):
    writer = log.writer("s", author="user", source="conversation", body_types=(Input,), content={"text": check_text})
    writer.append("u1", Input((ContentPart("text", "first original input"),)))
    writer.append("u2", Input((ContentPart("text", "second original input"),)))


def test_summary_publication_is_create_only_and_stale_parent_cannot_win_after_restart(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        inputs(log)
        records = SummaryRecords(log.owner("compaction"))
        original = log.reader("s").snapshot()
        first = records.publish(record("first"), log.reader("s"), parent=None)
        second = records.publish(record("second", parent=first, ids=("u1", "u2")), log.reader("s"), parent=first)
        assert records.publish(first, log.reader("s"), parent=None) == first
        assert records.head("s") == second
        with pytest.raises(ValueError, match="漂移"):
            records.publish(first.model_copy(update={"content": "overwritten"}), log.reader("s"), parent=None)
        assert log.reader("s").snapshot() == original
    with closing(MessageLog(path)) as log:
        records = SummaryRecords(log.owner("compaction"))
        with pytest.raises(MessageConflict, match="parent"):
            records.publish(record("loser", parent=first, ids=("u1", "u2")), log.reader("s"), parent=first)
        assert records.read("loser") is None
        assert records.head("s") == second and records.read("first") == first
        assert log.reader("s").snapshot() == original


def test_failed_head_update_rolls_back_new_summary_and_rejects_wrong_source(tmp_path, monkeypatch):
    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        inputs(log)
        records = SummaryRecords(log.owner("compaction"))
        first = records.publish(record("first"), log.reader("s"), parent=None)
        with pytest.raises(ValueError, match="范围"):
            records.publish(record("wrong", parent=first, ids=("u1", "missing")), log.reader("s"), parent=first)
        assert records.read("wrong") is None
        save = OwnerTransaction.save
        def fail_head(self, key, value, *, expected_version):
            if key == "head:s":
                raise OSError("injected write failure")
            return save(self, key, value, expected_version=expected_version)
        monkeypatch.setattr(OwnerTransaction, "save", fail_head)
        with pytest.raises(OSError, match="write failure"):
            records.publish(record("next", parent=first, ids=("u1", "u2")), log.reader("s"), parent=first)
        assert records.head("s") == first
        assert records.read("next") is None


def test_summary_keeps_recent_start_and_only_extends_its_actual_range(tmp_path):
    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        inputs(log)
        writer = log.writer("s", author="user", source="conversation", body_types=(Input,), content={"text": check_text})
        writer.append("u3", Input((ContentPart("text", "third"),)))
        records = SummaryRecords(log.owner("compaction"))
        first = records.publish(record("recent", ids=("u2",)), log.reader("s"), parent=None)
        for ids in (("u1", "u2", "u3"), ("u2",)):
            with pytest.raises(ValueError, match="来源"):
                records.publish(record("bad", parent=first, ids=ids), log.reader("s"), parent=first)
        second = records.publish(record("extended", parent=first, ids=("u2", "u3")), log.reader("s"), parent=first)
        assert records.head("s") == second
        assert log.reader("s").get("u1").body.parts[0].value == "first original input"


@pytest.mark.asyncio
async def test_summary_use_reopens_original_archive_after_head_advance_and_source_removal(tmp_path):
    from pathlib import Path
    import shutil
    from agent.plugin_composition.bindings import BINDINGS, Bindings
    from agent.plugins.manager import PluginManager
    from agent.plugins.snapshot import lease_runtime_snapshot
    from bus.event_bus import EventBus
    from plugins.compaction.records import COMPACTION_SUMMARIES
    from plugins.context.api import check_summary
    from plugins.context.materials import MATERIALS
    from session.message import Output

    sources = tmp_path / "plugins"
    for name in ("context", "compaction", "turn_projection"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (sources / "compaction/akashic.plugin.toml").write_text(
        'schema_version = 1\nname = "compaction"\nversion = "4.0.0"\napi_version = 3\nentrypoint = "message_plugin.py"\n')
    context = sources / "context/plugin.py"
    context.write_text(context.read_text().replace(
        'summary_source: tuple[str, str] | None = None',
        'summary_source: tuple[str, str] | None = ("compaction", "compaction")'))
    provider = sources / "fixture_models"
    provider.mkdir()
    (provider / "plugin.py").write_text('''
from agent.plugin_composition import CHAT_MODELS
api_version = 3
name = "fixture_models"
version = "1.0.0"
inject = ()
async def apply(ctx, config):
    class Models:
        def execution(self):
            raise AssertionError("archive lookup must not open a model")
    await ctx.provide(CHAT_MODELS, Models())
''')
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        inputs(log)
        original = log.reader("s").snapshot()
        records = SummaryRecords(log.owner("plugin:compaction"))
        first = records.publish(record("first"), log.reader("s"), parent=None)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            async with ctx.require(MATERIALS).bind() as view:
                prepared = await view.prepare(log.reader("s").snapshot(), "conversation")
            reference = prepared.summary.reference
            metadata = ctx.require(BINDINGS).describe(reference, COMPACTION_SUMMARIES)
            assert metadata == {"record_ref": "first", "session_id": "s"}
        writer = log.writer("s", author="assistant", source="conversation", body_types=(Output,),
                            content={"text": check_text, "context.summary": check_summary})
        used = ContentPart("context.summary", {"reference": reference})
        writer.append("used", Output((ContentPart("text", "real successful reply"), used), "complete"))
        with pytest.raises(PermissionError):
            log.writer("s", author="user", source="conversation", body_types=(Input,),
                       content={"text": check_text}).append("forged", Input((used,)))
        records.publish(record("second", parent=first, ids=("u1", "u2")), log.reader("s"), parent=first)
    finally:
        await host.terminate_all()
        log.close()

    shutil.rmtree(sources)
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    bindings = Bindings(log, host._archive, host.open_binding)
    try:
        assert log.reader("s").snapshot()[:2] == original
        assert log.reader("s").get("used").body.parts[-1] == used
        async with bindings.open(reference, COMPACTION_SUMMARIES) as (lookup, metadata):
            restored = lookup.resolve(metadata, session_id="s")
            assert restored.model_dump() == first.model_dump()
            with pytest.raises(ValueError, match="Session"):
                lookup.resolve(metadata, session_id="elsewhere")
        assert SummaryRecords(log.owner("plugin:compaction")).head("s").reference == "second"
        assert not (tmp_path / "workspace/memory").exists()
    finally:
        await host.terminate_all()
        log.close()
