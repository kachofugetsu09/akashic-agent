import asyncio
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

import pytest

from agent.plugin_composition import CHAT_MODELS, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.compaction.records import COMPACTION_SUMMARIES, SummaryRecord, SummaryRecords
from plugins.content.plugin import check_text
from plugins.context.api import check_summary
from plugins.context.materials import MATERIALS
from plugins.turn_projection.plugin import TURN_PROJECTION
from plugins.markdown_memory.store import MarkdownProfileStore
from session.log import MessageLog, SessionAttributes
from session.message import ContentPart, Input, Output


@asynccontextmanager
async def application(tmp_path, *, start=False):
    sources = tmp_path / "plugins"
    if not sources.exists():
        for name in ("context", "compaction", "markdown_memory", "turn_projection"):
            shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("compaction", "markdown_memory"):
            (sources / name / "akashic.plugin.toml").write_text(
                f'schema_version = 1\nname = "{name}"\nversion = "4.0.0"\napi_version = 3\nentrypoint = "message_plugin.py"\n')
        settings = tmp_path / "workspace/plugin-data/context-builtin/config.local.toml"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text('summary_source = ["compaction", "compaction"]\nprompt_sources = {markdown_memory = "markdown_memory"}\n')
        provider = sources / "fixture_models"
        provider.mkdir()
        (provider / "plugin.py").write_text('''
import asyncio
import json
from pathlib import Path
from contextlib import asynccontextmanager
from types import SimpleNamespace
from agent.plugin_composition import CHAT_MODELS, ServiceKey
from agent.plugin_composition.models import BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities, ModelRole
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
api_version = 3
name = "fixture_models"
version = "1.0.0"
inject = ()
async def apply(ctx, config):
    completed = asyncio.Event()
    store = ModelsStore(ctx.data_root / "models.db", ctx.data_root / "backups")
    store.initialize()
    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools=()):
            return len(str(messages)) // 4
        async def complete(self, request):
            root = Path(TEST_ROOT)
            with (root / "requests.jsonl").open("a") as handle:
                handle.write(json.dumps(request.messages[0]["content"]) + "\\n")
            memory = (root / "workspace/memory/MEMORY.md").read_text()
            if not memory:
                memory = "# 用户长期记忆\\n\\n## 用户事实\\n\\n## 用户偏好\\n\\n## 用户明确要求长期记住的关键内容\\n"
            for fact in ("fact-one", "fact-two", "fact-three"):
                if fact in request.messages[0]["content"] and fact not in memory:
                    memory += "- " + fact + "\\n"
            completed.set()
            return LLMResponse(json.dumps({"memory": memory, "self": (root / "workspace/memory/SELF.md").read_text()}))
    descriptor = BoundModelDescriptor(
        binding_id="fixture", plugin_snapshot_id="fixture", model_revision=0,
        model_id="fixture", connection_id="fixture", driver_id="fixture", driver_contract_version="1",
        auth_identity="fixture", model="fixture", role=ModelRole.DEFAULT, reasoning_effort=None,
        capabilities=ModelCapabilities(context_window=32000), capability_sources=CapabilitySources(), capability_digest="fixture")
    model = _BoundChat(descriptor, Driver(), store)
    class Models:
        @asynccontextmanager
        async def independent_execution(self):
            yield SimpleNamespace(chat=lambda role: model)
        @asynccontextmanager
        async def execution(self):
            yield SimpleNamespace(chat=lambda role: model)
    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(ServiceKey("fixture.profile_response"), completed)
'''.replace("TEST_ROOT", repr(str(tmp_path))))
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        if start:
            await host.start_runtime()
        yield log, host
    finally:
        await host.terminate_all()
        log.close()


def profile_store(tmp_path):
    root = tmp_path / "workspace/memory"
    return MarkdownProfileStore(root / "MEMORY.md", root / "SELF.md", root / "markdown-profile-writes.db")


def legacy_part(raw, *, digest=None):
    import hashlib
    return ContentPart("history.provenance", {
        "schema": "sessions.messages.v0", "role": "user", "content_was_null": False,
        "extra": raw, "extra_sha256": digest or hashlib.sha256(raw.encode()).hexdigest(),
    })


@pytest.mark.asyncio
async def test_excluded_session_never_reaches_markdown_even_when_source_is_allowed(tmp_path):
    from plugins.markdown_memory.message_plugin import project
    async with application(tmp_path) as (log, host):
        log.ensure_session("s", SessionAttributes("internal", "excluded"))
        writer = log.writer("s", author="app", source="conversation", body_types=(Input, Output),
                            content={"text": check_text})
        writer.append("input", Input((ContentPart("text", "fact-one"),)))
        writer.append("answer", Output((ContentPart("text", "fact-two"),), "complete"))
        summary = publish(log, "internal-summary")
        used = await record_use(log, host, summary, "used-summary")
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            store = profile_store(tmp_path)
            await project(used, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock",
                sources=("conversation",), projection=ctx.require(TURN_PROJECTION))
            assert not (tmp_path / "requests.jsonl").exists()
            assert not store.is_applied(summary.reference)
            assert log.reader("s").get("input").body.parts[0].value == "fact-one"


@pytest.mark.asyncio
async def test_legacy_suppress_excludes_whole_turn_but_keeps_later_allowed_facts(tmp_path):
    from plugins.markdown_memory.message_plugin import project
    from session.message import ContentReferences
    async with application(tmp_path) as (log, host):
        writer = log.writer("s", author="migration", source="legacy-unattributed", body_types=(Input, Output),
            content={"text": check_text, "history.provenance": lambda part: ContentReferences()})
        writer.append("excluded-input", Input((ContentPart("text", "fact-one"),
            legacy_part('{"effects":{"post_commit":"suppress"}}'))))
        writer.append("excluded-answer", Output((ContentPart("text", "fact-two"),), "complete"))
        suppressed = publish(log, "suppressed-range")
        use = await record_use(log, host, suppressed, "suppressed-use")
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            store = profile_store(tmp_path)
            async def consume(message):
                await project(message, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                    models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock",
                    sources=("conversation", "legacy-unattributed"), projection=ctx.require(TURN_PROJECTION))
            await consume(use)
            assert not (tmp_path / "requests.jsonl").exists()
            assert not store.is_applied(suppressed.reference)
            writer.append("allowed-input", Input((ContentPart("text", "fact-three"), legacy_part('{}'))))
            writer.append("allowed-answer", Output((ContentPart("text", "allowed answer"),), "complete"))
            allowed = publish(log, "allowed-range", suppressed)
            used = await record_use(log, host, allowed, "allowed-use")
            await consume(used)
            payload = (tmp_path / "requests.jsonl").read_text()
            assert "fact-one" not in payload and "fact-two" not in payload
            assert "fact-three" in payload and "fact-three" in store.read_memory()
            assert store.is_applied(allowed.reference)
            assert log.reader("s").get("excluded-answer").body.parts[0].value == "fact-two"


@pytest.mark.asyncio
async def test_markdown_does_not_reintroduce_abandoned_late_result_from_raw_range(tmp_path):
    from plugins.markdown_memory.message_plugin import project
    from session.message import CallRef, Control, ToolCall, ToolResult
    async with application(tmp_path) as (log, host):
        # 这里只验证已有消息的读取；fixture 调用从未执行，也不测试工具授权。
        log.save_binding("fixture:unexecuted", {"fixture": "raw-message-read"})
        writer = log.writer("s", author="test", source="conversation", body_types=(Input, Output, Control),
                            content={"text": check_text}, check_call=lambda call: None)
        writer.append("input", Input((ContentPart("text", "fact-one"),)))
        called = writer.append("call", Output((ToolCall("fixture:unexecuted", {}),), "continue"))
        writer.append("abandon", Control("abandon", called.seq))
        parent = publish(log, "before-late-result")
        writer.append("new-input", Input((ContentPart("text", "fact-three"),)))
        ref = CallRef("call", 0)
        log.writer("s", author="fixture", source="conversation", body_types=(ToolResult,), call_ref=ref,
                   content={"text": check_text}).append("late", ToolResult(ref, "success", (ContentPart("text", "fact-two"),)))
        writer.append("answer", Output((ContentPart("text", "new answer"),), "complete"))
        child = publish(log, "after-late-result", parent)
        used = await record_use(log, host, child, "used")
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            store = profile_store(tmp_path)
            await project(used, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock",
                sources=("conversation",), projection=ctx.require(TURN_PROJECTION))
            assert "fact-two" not in (tmp_path / "requests.jsonl").read_text()
            assert "fact-three" in store.read_memory()
            assert store.is_applied(child.reference)
            assert log.reader("s").get("late").body.parts[0].value == "fact-two"


@pytest.mark.parametrize("raw,digest,error", [
    ('{"effects":{"post_commit":"suppress"}}', "wrong", "digest"),
    ('{"effects":{"post_commit":"suppress","post_commit":"allow"}}', None, "重复"),
    ('{"effects":{"post_commit":"unknown"}}', None, "unknown"),
    ('{"skip_post_memory":true}', None, "迁移"),
])
def test_legacy_effect_reader_rejects_unproven_or_ambiguous_metadata(raw, digest, error):
    from datetime import UTC, datetime
    from plugins.content.api import legacy_post_commit_effect
    from session.message import Message
    row = Message("old", "s", 0, datetime.now(UTC), "migration", "legacy-unattributed",
                  Input((legacy_part(raw, digest=digest),)))
    with pytest.raises(ValueError, match=error):
        legacy_post_commit_effect(row)


def publish(log, reference, parent=None):
    record = SummaryRecord(reference=reference, session_id="s", generation=1 if parent is None else parent.generation + 1,
        parent=None if parent is None else parent.reference,
        source_message_ids=tuple(message.message_id for message in log.reader("s").snapshot()),
        content="actual summary", model_call_ids=("summary-model:" + reference,), trigger="soft_limit",
        context_window=32000, max_output_tokens=4096, keep_recent_tokens=20000, tokens_before=27000, tokens_after=18000)
    return SummaryRecords(log.owner("plugin:compaction")).publish(record, log.reader("s"), parent=parent)


async def record_use(log, host, record, identity, finish="continue", source="conversation"):
    async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
        binding = snapshot.composition_root.context.require(BINDINGS).bind(COMPACTION_SUMMARIES,
            {"record_ref": record.reference, "session_id": record.session_id})
    return log.writer("s", author="assistant", source=source, body_types=(Output,),
        content={"text": check_text, "context.summary": check_summary}).append(identity,
            Output((ContentPart("text", "successful response"), ContentPart("context.summary", {"reference": binding})), finish))


async def wait_applied(tmp_path, host, reference):
    store = profile_store(tmp_path)
    if not store.is_applied(reference):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            done = snapshot.composition_root.context.require(ServiceKey("fixture.profile_response"))
            _ = await asyncio.wait_for(done.wait(), 5)
        # fixture 从返回响应到写完两个文件无挂起点，等待者在提交之后恢复。
    assert store.is_applied(reference)


@pytest.mark.asyncio
async def test_markdown_replays_output_after_restart_and_does_not_skip_unused_parent_facts(tmp_path):
    async with application(tmp_path) as (log, host):
        assert not (tmp_path / "workspace/memory").exists()
        inputs = log.writer("s", author="user", source="conversation", body_types=(Input,), content={"text": check_text})
        inputs.append("u1", Input((ContentPart("text", "fact-one"),)))
        unused = publish(log, "unused-first")
        inputs.append("u2", Input((ContentPart("text", "fact-two"),)))
        used = publish(log, "used-second", unused)
        await record_use(log, host, used, "before-crash")
        original = log.reader("s").snapshot()
        assert not (tmp_path / "requests.jsonl").exists()
    async with application(tmp_path, start=True) as (log, host):
        await wait_applied(tmp_path, host, used.reference)
        store = profile_store(tmp_path)
        assert not store.is_applied(unused.reference)
        memory = store.read_memory()
        assert "fact-one" in memory and "fact-two" in memory
        assert log.reader("s").snapshot() == original
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            async with snapshot.composition_root.context.require(MATERIALS).bind() as materials:
                prepared = await materials.prepare(log.reader("s").snapshot(), "conversation")
            assert "fact-two" in prepared.system_prompt
        await record_use(log, host, used, "duplicate-use", "complete")
    async with application(tmp_path, start=True) as (log, host):
        await wait_applied(tmp_path, host, used.reference)
        # 直接消费重复 Output 确认幂等边界，避免只靠 watcher 时间猜测。
        from plugins.markdown_memory.message_plugin import project
        from agent.plugin_composition import CHAT_MODELS
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            await project(log.reader("s").get("duplicate-use"), reader=log.reader("s"), bindings=ctx.require(BINDINGS),
                          store=profile_store(tmp_path), models=ctx.require(CHAT_MODELS),
                          lock_path=tmp_path / "workspace/memory/markdown-profile.lock", sources=("conversation",), projection=ctx.require(TURN_PROJECTION))
        assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1


@pytest.mark.asyncio
async def test_restart_finishes_saved_draft_after_only_memory_file_was_applied(tmp_path, monkeypatch):
    from plugins.markdown_memory.message_plugin import project

    async with application(tmp_path) as (log, host):
        log.writer("s", author="user", source="conversation", body_types=(Input,), content={"text": check_text}).append(
            "u", Input((ContentPart("text", "fact-one"),)))
        record = publish(log, "partial-files")
        used = await record_use(log, host, record, "used")
        store = profile_store(tmp_path)
        before_self = store.read_self()
        apply_document = store._apply_document
        def fail_self(source_ref, document, path):
            if document == "self":
                raise OSError("injected second document failure")
            apply_document(source_ref, document, path)
        monkeypatch.setattr(store, "_apply_document", fail_self)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            with pytest.raises(OSError, match="second document failure"):
                await project(used, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                    models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock",
                    sources=("conversation",), projection=ctx.require(TURN_PROJECTION))
        assert not store.is_applied(record.reference)
        assert "fact-one" in store.read_memory()
        assert store.read_self() == before_self
        assert store.read_draft(record.reference) is not None
        assert store.read_backup(record.reference, "memory") == ""
        original = log.reader("s").snapshot()
    async with application(tmp_path, start=True) as (log, host):
        store = profile_store(tmp_path)
        assert store.is_applied(record.reference)
        assert store.read_backup(record.reference, "self") == before_self
        assert log.reader("s").snapshot() == original
        assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1


@pytest.mark.asyncio
async def test_delayed_parent_output_cannot_reapply_older_facts_after_child(tmp_path):
    from plugins.markdown_memory.message_plugin import project
    async with application(tmp_path) as (log, host):
        writer = log.writer("s", author="user", source="conversation", body_types=(Input,), content={"text": check_text})
        writer.append("u1", Input((ContentPart("text", "fact-one"),)))
        parent = publish(log, "old")
        writer.append("u2", Input((ContentPart("text", "fact-two"),)))
        child = publish(log, "new", parent)
        first = await record_use(log, host, child, "child-first")
        late = await record_use(log, host, parent, "parent-late")
        store = profile_store(tmp_path)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            for message in (first, late):
                await project(message, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                              models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock",
                              sources=("conversation",), projection=ctx.require(TURN_PROJECTION))
        assert store.latest_applied("s") == (child.reference, child.generation)
        assert store.is_applied(child.reference) and not store.is_applied(parent.reference)
        assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["markdown_projection_order_v1", "markdown_self_draft_v1"])
async def test_restart_repairs_partial_sqlite_preparation_without_recomputing_model(tmp_path, monkeypatch, kind):
    from plugins.markdown_memory.message_plugin import project
    async with application(tmp_path) as (log, host):
        log.writer("s", author="user", source="conversation", body_types=(Input,), content={"text": check_text}).append(
            "u", Input((ContentPart("text", "fact-one"),)))
        record = publish(log, "partial-sql")
        used = await record_use(log, host, record, "used")
        store = profile_store(tmp_path)
        write_once = store._write_once
        def fail_prepare(source_ref, row_kind, payload):
            if row_kind == kind:
                raise OSError("injected partial preparation")
            return write_once(source_ref, row_kind, payload)
        monkeypatch.setattr(store, "_write_once", fail_prepare)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            with pytest.raises(OSError, match="partial preparation"):
                await project(used, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                    models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock", sources=("conversation",), projection=ctx.require(TURN_PROJECTION))
        assert store.read_draft(record.reference) is not None
        assert not store.is_applied(record.reference)
    async with application(tmp_path, start=True) as (log, host):
        store = profile_store(tmp_path)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            await project(log.reader("s").get("used"), reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                          models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock", sources=("conversation",), projection=ctx.require(TURN_PROJECTION))
        assert store.is_applied(record.reference)
        assert store.latest_applied("s") == (record.reference, record.generation)
        assert "fact-one" in store.read_memory()
        assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["waiting", "held"])
async def test_profile_lock_cancellation_closes_its_handle_and_allows_next_writer(tmp_path, monkeypatch, phase):
    import fcntl
    from plugins.markdown_memory.plugin import profile_lock
    path = tmp_path / "profile.lock"
    blocker = path.open("a+b")
    if phase == "waiting":
        fcntl.flock(blocker.fileno(), fcntl.LOCK_EX)
    opened, entered = asyncio.Event(), asyncio.Event()
    handles = []
    original_open = Path.open
    def tracked_open(self, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        if self == path:
            handles.append(handle)
            opened.set()
        return handle
    monkeypatch.setattr(Path, "open", tracked_open)
    async def write():
        async with profile_lock(path):
            entered.set()
            await asyncio.Event().wait()
    task = asyncio.create_task(write())
    try:
        await asyncio.wait_for((opened if phase == "waiting" else entered).wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(handles) == 1 and handles[0].closed
        fcntl.flock(blocker.fileno(), fcntl.LOCK_UN)
        async with profile_lock(path):
            assert not handles[-1].closed
        assert all(handle.closed for handle in handles)
    finally:
        blocker.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", ["MEMORY.md", "markdown-profile-writes.db", "PENDING.md"])
async def test_unstarted_markdown_does_not_treat_partial_state_as_initial(tmp_path, existing):
    async with application(tmp_path) as (log, host):
        memory = tmp_path / "workspace/memory"
        memory.mkdir()
        path = memory / existing
        path.write_bytes(b"preserved state")
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            async with snapshot.composition_root.context.require(MATERIALS).bind() as view:
                with pytest.raises(FileNotFoundError):
                    await asyncio.wait_for(view.prepare((), "conversation"), 5)
        assert tuple(memory.iterdir()) == (path,)
        assert path.read_bytes() == b"preserved state"


@pytest.mark.asyncio
@pytest.mark.parametrize("learning", ["excluded", "eligible"])
async def test_default_markdown_uses_programmatic_admission_for_real_summary_projection(tmp_path, learning):
    from plugins.markdown_memory.message_plugin import Config, project

    async with application(tmp_path) as (log, host):
        log.ensure_session("s", SessionAttributes("internal", learning))
        writer = log.writer("s", author="fixture", source="programmatic", body_types=(Input, Output),
                            content={"text": check_text})
        writer.append("input", Input((ContentPart("text", "fact-one"),)))
        writer.append("answer", Output((ContentPart("text", "fact-two"),), "complete"))
        summary = publish(log, "programmatic-summary")
        used = await record_use(log, host, summary, "use", source="programmatic")
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            store = profile_store(tmp_path)
            await project(used, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock",
                sources=Config().sources, projection=ctx.require(TURN_PROJECTION))
            assert store.is_applied(summary.reference) is (learning == "eligible")
            assert (tmp_path / "requests.jsonl").exists() is (learning == "eligible")
        assert log.reader("s").get("input").body.parts[0].value == "fact-one"
