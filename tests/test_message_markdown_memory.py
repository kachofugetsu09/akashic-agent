import asyncio
from typing import Literal, cast
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
from agent.plugin_contracts.content import check_text
from agent.plugin_contracts.context import check_summary
from plugins.context.materials import MATERIALS
from agent.plugin_contracts.turn_projection import TURN_PROJECTION
from plugins.markdown_memory.store import MarkdownProfileStore
from session.log import MessageLog, SessionAttributes
from session.message import ContentPart, Input, Output


@asynccontextmanager
async def application(tmp_path, *, start=False, transient_failure=False, draft_failures=0):
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
            source = json.loads(request.messages[0]["content"].split("本次精确来源：\\n", 1)[1])
            def user_input(row):
                return row["body"]["kind"] == "input" and (
                    row["author"] == "user" or row["source"] == "legacy-unattributed" and any(
                        part.get("kind") == "history.provenance" and part["value"]["role"] == "user"
                        for part in row["body"]["parts"]))
            for fact in ("fact-one", "fact-two", "fact-three"):
                if any(user_input(row)
                       and fact in json.dumps(row) for row in source) and fact not in memory:
                    memory += "- " + fact + "\\n"
            previous = (root / "workspace/memory/MEMORY.md").read_text()
            evidence = {line: [row["message_id"] for row in source if user_input(row) and line[2:] in json.dumps(row)]
                        for line in memory.splitlines() if line.startswith("- ") and line not in previous.splitlines()}
            completed.set()
            return LLMResponse(json.dumps({"additions": [
                {"document": "memory", "section": "## 用户明确要求长期记住的关键内容", "line": line, "message_ids": ids}
                for line, ids in evidence.items()]}))
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
    if draft_failures:
        fixture = sources / "fixture_models/plugin.py"
        fixture.write_text(fixture.read_text().replace("    completed = asyncio.Event()",
            "    completed = asyncio.Event()\n    attempts = 0").replace(
            "        async def complete(self, request):",
            "        async def complete(self, request):\n            nonlocal attempts\n            attempts += 1").replace(
            "            completed.set()",
            f"            if attempts <= {draft_failures}:\n                evidence = {{line: [identity[:12] for identity in ids] for line, ids in evidence.items()}}\n            completed.set()"))
    if transient_failure or draft_failures > 1:
        # 仅测试副本用事件控制重试等待，不修改全局 asyncio 时序。
        module = sources / "markdown_memory/message_plugin.py"
        module.write_text(module.read_text().replace("await asyncio.sleep(30)",
            'await ctx.require(ServiceKey("fixture.retry_release")).wait()').replace(
                "    CHAT_MODELS,", "    CHAT_MODELS, ServiceKey,"))
        fixture = sources / "fixture_models/plugin.py"
        fixture.write_text(fixture.read_text().replace("    completed = asyncio.Event()",
            "    completed = asyncio.Event()\n    failed = asyncio.Event()\n    release = asyncio.Event()").replace(
            '            memory = (root / "workspace/memory/MEMORY.md").read_text()',
            '            if not failed.is_set():\n                failed.set()\n                from agent.plugin_composition.models import TransportError\n                raise TransportError("temporary fixture failure")\n            memory = (root / "workspace/memory/MEMORY.md").read_text()').replace(
            '    await ctx.provide(CHAT_MODELS, Models())',
            '    await ctx.provide(ServiceKey("fixture.retry_failed"), failed)\n    await ctx.provide(ServiceKey("fixture.retry_release"), release)\n    await ctx.provide(CHAT_MODELS, Models())'))
    if draft_failures > 1:
        fixture = sources / "fixture_models/plugin.py"
        text = fixture.read_text()
        start = text.index("            if not failed.is_set():")
        end = text.index('            memory = (root / "workspace/memory/MEMORY.md").read_text()', start)
        text = text[:start] + text[end:]
        text = text.replace("            completed.set()",
            f"            if attempts == {draft_failures}:\n                failed.set()\n            completed.set()")
        fixture.write_text(text)
    if draft_failures:
        fixture = sources / "fixture_models/plugin.py"
        fixture.write_text(fixture.read_text().replace("            completed.set()",
            f"            if attempts > {draft_failures}:\n                completed.set()"))
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
    from plugins.markdown_memory.message_plugin import Config, project
    from session.message import ContentReferences
    async with application(tmp_path) as (log, host):
        writer = log.writer("s", author="legacy-attribution-unknown", source="legacy-unattributed", body_types=(Input, Output),
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
                    sources=Config().sources, projection=ctx.require(TURN_PROJECTION))
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
        writer = log.writer("s", author="user", source="conversation", body_types=(Input, Output, Control),
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
    from agent.plugin_contracts.content import legacy_post_commit_effect
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


async def record_use(
    log, host, record, identity,
    finish: Literal["continue", "complete", "quiet"] = "continue",
    source="conversation",
):
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
            message = log.reader("s").get("duplicate-use")
            assert message is not None
            await project(message, reader=log.reader("s"), bindings=ctx.require(BINDINGS),
                          store=profile_store(tmp_path), models=ctx.require(CHAT_MODELS),
                          lock_path=tmp_path / "workspace/memory/markdown-profile.lock", sources=("conversation",), projection=ctx.require(TURN_PROJECTION))
        assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1


@pytest.mark.asyncio
async def test_restart_finishes_saved_draft_after_only_memory_file_was_applied(tmp_path, monkeypatch):
    import fcntl
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
                # 第一份文件已安装时，其他读者仍不能取得整对档案的锁。
                with (tmp_path / "workspace/memory/markdown-profile.lock").open("rb") as lock:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        partial = store.read_writes(None, 100)
        assert "markdown_memory_applied_v1" in {row["kind"] for row in partial}
        assert "markdown_self_applied_v1" not in {row["kind"] for row in partial}
        original = log.reader("s").snapshot()
    async with application(tmp_path, start=True) as (log, host):
        store = profile_store(tmp_path)
        assert store.is_applied(record.reference)
        assert store.read_backup(record.reference, "self") == before_self
        from plugins.markdown_memory.store import MEMORY_WRITES
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            read = snapshot.composition_root.context.require(MEMORY_WRITES)
            pages = []
            after = None
            while page := read(after, 2):
                pages.extend(page)
                after = (cast(str, page[-1]["source_ref"]), cast(str, page[-1]["kind"]))
        assert len(pages) == len(store.read_writes(None, 100))
        assert "markdown_self_applied_v1" in {row["kind"] for row in pages}
        assert all(row in pages for row in partial)
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
            message = log.reader("s").get("used")
            assert message is not None
            await project(message, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                          models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock", sources=("conversation",), projection=ctx.require(TURN_PROJECTION))
        assert store.is_applied(record.reference)
        assert store.latest_applied("s") == (record.reference, record.generation)
        assert "fact-one" in store.read_memory()
        assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["waiting", "held"])
async def test_profile_lock_cancellation_closes_its_handle_and_allows_next_writer(tmp_path, monkeypatch, phase):
    import fcntl
    from plugins.markdown_memory.message_plugin import profile_lock
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
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_profile_model_work_keeps_materials_readable_and_updates_serial(tmp_path, monkeypatch, cancel_first):
    """模型停在确定性屏障时仍可读旧档案；另一次更新必须等待且可接替取消者。"""
    from plugins.markdown_memory import message_plugin as plugin

    async with application(tmp_path) as (log, host):
        log.writer("s", author="user", source="conversation", body_types=(Input,), content={"text": check_text}).append(
            "u", Input((ContentPart("text", "fact-one"),)))
        record = publish(log, "read-during-model")
        used = await record_use(log, host, record, "used")
        store = profile_store(tmp_path)
        before = (store.read_memory(), store.read_self(), store.read_writes(None, 100))
        entered, release, second_waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
        prepare_calls = []
        original_prepare, original_flock = plugin.prepare_profile_draft, plugin.fcntl.flock

        async def paused_prepare(*args, **kwargs):
            prepare_calls.append(asyncio.current_task())
            entered.set()
            await release.wait()
            return await original_prepare(*args, **kwargs)

        async def update():
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                ctx = snapshot.composition_root.context
                await plugin.project(used, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                    models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock",
                    sources=("conversation",), projection=ctx.require(TURN_PROJECTION))

        monkeypatch.setattr(plugin, "prepare_profile_draft", paused_prepare)
        first = asyncio.create_task(update())
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            second = asyncio.create_task(update())

            def tracked_flock(fd, operation):
                try:
                    return original_flock(fd, operation)
                except BlockingIOError:
                    if asyncio.current_task() is second:
                        second_waiting.set()
                    raise

            monkeypatch.setattr(plugin.fcntl, "flock", tracked_flock)
            await asyncio.wait_for(second_waiting.wait(), 5)
            assert prepare_calls == [first]
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                async with snapshot.composition_root.context.require(MATERIALS).bind() as materials:
                    prepared = await asyncio.wait_for(materials.prepare((), "conversation"), 1)
            assert before[1].strip() in prepared.system_prompt
            assert "fact-one" not in prepared.system_prompt
            assert (store.read_memory(), store.read_self(), store.read_writes(None, 100)) == before
            if cancel_first:
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert (store.read_memory(), store.read_self(), store.read_writes(None, 100)) == before
                assert store.read_draft(record.reference) is None
            release.set()
            await asyncio.gather(*([second] if cancel_first else [first, second]))
            assert len(prepare_calls) == (2 if cancel_first else 1)
            assert store.is_applied(record.reference)
            assert store.read_memory().count("fact-one") == 1
            assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1
        finally:
            tasks = [first, *([second] if second is not None else [])]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


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
async def test_an_update_lock_alone_does_not_create_a_partial_profile_state(tmp_path):
    async with application(tmp_path) as (_log, host):
        path = tmp_path / "workspace/memory/markdown-profile-update.lock"
        path.parent.mkdir()
        path.touch()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            async with snapshot.composition_root.context.require(MATERIALS).bind() as materials:
                prepared = await materials.prepare((), "conversation")
        assert "# Akashic 的自我认知" in prepared.system_prompt
        assert tuple(path.parent.iterdir()) == (path,)


@pytest.mark.asyncio
@pytest.mark.parametrize("learning", ["excluded", "eligible"])
async def test_default_markdown_uses_programmatic_admission_for_real_summary_projection(tmp_path, learning):
    from plugins.markdown_memory.message_plugin import Config, project

    async with application(tmp_path) as (log, host):
        log.ensure_session("s", SessionAttributes("internal", learning))
        writer = log.writer("s", author="user", source="programmatic", body_types=(Input, Output),
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


@pytest.mark.parametrize("evidence_id", ["assistant", "background", "missing"])
def test_new_user_fact_cannot_use_an_assistant_or_background_claim(tmp_path, evidence_id):
    """即使普通助手复述了后台结论，也不能独自把它升级成用户事实。"""
    from plugins.markdown_memory.message_plugin import check_evidence

    log = MessageLog(tmp_path / "facts.db")
    try:
        for identity, author, source, body in (
            ("user", "user", "conversation", Input((ContentPart("text", "查一下结果"),))),
            ("assistant", "assistant", "conversation", Output((ContentPart("text", "用户喜欢红色"),), "complete")),
            ("background", "assistant", "subagent:job", Output((ContentPart("text", "用户喜欢红色"),), "complete")),
        ):
            log.writer("s", author=author, source=source, body_types=(type(body),), content={"text": check_text}).append(identity, body)
        before = "# 用户长期记忆\n## 用户事实\n## 用户偏好\n## 用户明确要求长期记住的关键内容\n"
        draft = {"memory_before": before, "memory": before + "- 用户喜欢红色\n", "self_before": "", "self": "",
                 "evidence": {"memory": {"- 用户喜欢红色": [evidence_id]}, "self": {}}}
        with pytest.raises(ValueError, match="用户事实|实际消息"):
            check_evidence(draft, log.reader("s").snapshot())
        assert len(log.reader("s").snapshot()) == 3
    finally:
        log.close()


def test_moving_an_operation_note_into_user_facts_requires_new_evidence():
    """相同文字换成用户资料也改变含义，不能沿用旧操作记录的资格。"""
    from plugins.markdown_memory.message_plugin import check_evidence

    headings = "# 用户长期记忆\n## 用户事实\n## 用户偏好\n## 用户明确要求长期记住的关键内容\n"
    draft = {"memory_before": headings + "## 助手操作上下文\n- 红色主题\n",
             "memory": headings + "- 红色主题\n## 助手操作上下文\n",
             "self_before": "", "self": "", "evidence": {"memory": {}, "self": {}}}
    with pytest.raises(ValueError, match="实际消息"):
        check_evidence(draft, ())


@pytest.mark.asyncio
async def test_background_discovery_reads_only_new_eligible_messages(tmp_path, monkeypatch):
    from session.log import MessageCatalog, MessageReader

    first_done, next_head, finished, hold = (asyncio.Event() for _ in range(4))
    calls = []
    original = MessageReader.snapshot

    def snapshot(reader, **kwargs):
        assert reader.session_id != "excluded", "excluded history must not be decoded"
        calls.append((reader.session_id, kwargs))
        return original(reader, **kwargs)

    async def heads(_catalog):
        yield {"excluded": 0, "eligible": 0}
        first_done.set()
        await next_head.wait()
        yield {"excluded": 0, "eligible": 1}
        finished.set()
        await hold.wait()

    async with application(tmp_path) as (log, host):
        for name, learning in (("excluded", "excluded"), ("eligible", "eligible")):
            log.ensure_session(name, SessionAttributes("internal", cast(Literal["excluded", "eligible"], learning)))
            log.writer(name, author="user", source="conversation", body_types=(Input,),
                       content={"text": check_text}).append(name, Input((ContentPart("text", name),)))
        monkeypatch.setattr(MessageReader, "snapshot", snapshot)
        monkeypatch.setattr(MessageCatalog, "follow", heads)
        await host.start_runtime()
        await asyncio.wait_for(first_done.wait(), 2)
        log.writer("eligible", author="user", source="conversation", body_types=(Input,),
                   content={"text": check_text}).append("next", Input((ContentPart("text", "next"),)))
        next_head.set()
        await asyncio.wait_for(finished.wait(), 2)
        assert calls == [
            ("eligible", {"after_seq": -1, "through_seq": 0}),
            ("eligible", {"after_seq": 0, "through_seq": 1}),
        ]


@pytest.mark.asyncio
async def test_markdown_retries_transient_failure_without_new_messages(tmp_path):
    async with application(tmp_path, transient_failure=True) as (log, host):
        writer = log.writer("s", author="user", source="conversation", body_types=(Input,),
                            content={"text": check_text})
        writer.append("u1", Input((ContentPart("text", "fact-one"),)))
        summary = publish(log, "retry-summary")
        await record_use(log, host, summary, "used-summary")
        original = log.reader("s").snapshot()
        await host.start_runtime()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            await asyncio.wait_for(ctx.require(ServiceKey("fixture.retry_failed")).wait(), 5)
            assert not profile_store(tmp_path).is_applied(summary.reference)
            ctx.require(ServiceKey("fixture.retry_release")).set()
        await wait_applied(tmp_path, host, summary.reference)
        assert log.reader("s").snapshot() == original
        assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 2
        assert profile_store(tmp_path).read_memory().count("fact-one") == 1


@pytest.mark.asyncio
async def test_profile_input_omits_large_legacy_replay_but_preserves_user_evidence(tmp_path):
    """历史回放大小不再放大档案请求；学习资格和完整原文仍由原 Message 证明。"""
    import json
    from session.message import CallRef, ContentReferences, ToolCall, ToolResult
    from plugins.markdown_memory.message_plugin import project

    async with application(tmp_path) as (log, host):
        writer = log.writer("s", author="legacy-attribution-unknown", source="legacy-unattributed",
            body_types=(Input, Output), content={"text": check_text,
                "history.provenance": lambda part: ContentReferences(),
                "history.transcript": lambda part: ContentReferences(),
                "history.record": lambda part: ContentReferences()})
        writer.append("old-user", Input((ContentPart("text", "fact-one"),
            legacy_part(json.dumps({"provider_replay": "opaque-extra" * 100_000})),
            ContentPart("history.record", {"row": "opaque-record" * 100_000})) ))
        writer.append("old-answer", Output((ContentPart("text", "acknowledged"),
            ContentPart("history.transcript", {"raw": "opaque-tools" * 100_000})), "complete"))
        log.save_binding("fixture:replay", {"fixture": "read-only-history"})
        caller = log.writer("s", author="assistant", source="legacy-unattributed", body_types=(Output,),
                            content={"text": check_text}, check_call=lambda call: None)
        caller.append("replay-call", Output((ToolCall("fixture:replay", {}),), "continue"))
        call_ref = CallRef("replay-call", 0)
        log.writer("s", author="fixture", source="legacy-unattributed", body_types=(ToolResult,),
            call_ref=call_ref, content={"text": check_text, "history.transcript": lambda part: ContentReferences()}).append(
                "replay-result", ToolResult(call_ref, "success", (ContentPart("text", "result text"),
                    ContentPart("history.transcript", {"raw": "opaque-result" * 100_000}))))
        caller.append("replay-done", Output((ContentPart("text", "done"),), "complete"))
        original = log.reader("s").snapshot()
        summary = publish(log, "large-legacy")
        used = await record_use(log, host, summary, "use", source="legacy-unattributed")
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            store = profile_store(tmp_path)
            await project(used, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock",
                sources=("legacy-unattributed",), projection=ctx.require(TURN_PROJECTION))
        prompts = (tmp_path / "requests.jsonl").read_text()
        request = json.loads(prompts.splitlines()[0])
        rows = json.loads(request.split("本次精确来源：\n", 1)[1])
        assert rows[0]["body"]["parts"][1]["value"] == {"schema": "sessions.messages.v0", "role": "user"}
        assert len(prompts) < 20_000
        assert all(value not in prompts for value in ("opaque-extra", "opaque-tools", "opaque-result", "opaque-record"))
        assert "result text" in prompts
        assert "fact-one" in store.read_memory() and store.is_applied(summary.reference)
        assert log.reader("s").snapshot()[:len(original)] == original


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_second", [False, True])
@pytest.mark.parametrize("window,text_size,large_profile", [(32_000, 55_000, False), (200_000, 140_000, False), (200_000, 140_000, True)])
async def test_profile_batches_keep_whole_turns_and_write_only_after_all_succeed(tmp_path, fail_second, window, text_size, large_profile):
    """两批真实完整 Turn 的档案和证据合并；后批失败不写文件或收据。"""
    import json
    from types import SimpleNamespace
    from plugins.markdown_memory.message_plugin import prepare_profile_draft
    from plugins.compaction.message_summary import closed_groups
    from plugins.turn_projection.plugin import TurnProjection
    from agent.plugin_composition.models import ChatModels, LLMResponse, TransportError

    log = MessageLog(tmp_path / "messages.db")
    try:
        writer = log.writer("s", author="user", source="conversation", body_types=(Input, Output), content={"text": check_text})
        for index in range(2):
            writer.append(f"input-{index}", Input((ContentPart("text", f"fact-{index} " + "x" * text_size),)))
            writer.append(f"step-{index}", Output((ContentPart("text", "continue"),), "continue"))
            writer.append(f"answer-{index}", Output((ContentPart("text", "done"),), "complete"))
        original = log.reader("s").snapshot()
        groups = closed_groups(original, TurnProjection())
        store = profile_store(tmp_path)
        if large_profile:
            (tmp_path / "workspace/memory/MEMORY.md").write_text(
                "# 用户长期记忆\n## 用户事实\n- " + "existing-fact " * 4_000
                + "\n## 用户偏好\n## 用户明确要求长期记住的关键内容\n")
        before = (store.read_memory(), store.read_self(), store.read_writes(None, 20))
        requests = []

        async def complete(request):
            prompt = request.messages[0]["content"]
            rows = json.loads(prompt.split("本次精确来源：\n", 1)[1])
            requests.append(rows)
            assert (store.read_memory(), store.read_self(), store.read_writes(None, 20)) == before
            assert len(rows) == 3, "a complete Turn must stay in one request"
            if fail_second and len(requests) == 2:
                raise TransportError("injected second batch failure")
            fact = rows[0]["body"]["parts"][0]["value"].split()[0]
            line = "- " + fact
            response = json.dumps({"additions": [{"document": "memory", "section": "## 用户事实",
                "line": line, "message_ids": [rows[0]["message_id"]]}]})
            return LLMResponse(response)

        model = SimpleNamespace(descriptor=SimpleNamespace(capabilities=SimpleNamespace(context_window=window, max_output_tokens=32_768)),
                                estimate_context_tokens=lambda messages: len(str(messages)) // 4, complete=complete)

        @asynccontextmanager
        async def execution():
            yield SimpleNamespace(chat=lambda role: model)

        models = cast(ChatModels, SimpleNamespace(independent_execution=execution))
        if fail_second:
            with pytest.raises(TransportError, match="second batch"):
                await prepare_profile_draft(groups, before[0], before[1], models)
        else:
            draft = await prepare_profile_draft(groups, before[0], before[1], models)
            assert draft["memory_before"] == before[0]
            assert isinstance(draft["memory"], str)
            assert "- fact-0" in draft["memory"] and "- fact-1" in draft["memory"]
            if large_profile:
                assert "- " + "existing-fact " * 4_000 + "\n" in draft["memory"]
            assert draft["evidence"] == {"memory": {"- fact-0": ["input-0"], "- fact-1": ["input-1"]}, "self": {}}
        assert len(requests) == 2
        assert (store.read_memory(), store.read_self(), store.read_writes(None, 20)) == before
        assert log.reader("s").snapshot() == original
    finally:
        log.close()


@pytest.mark.asyncio
async def test_profile_keeps_allowed_turn_inside_mixed_source_group(tmp_path):
    """交错来源共用批次切点，不把被抑制来源的资格传播给用户 Turn。"""
    from plugins.markdown_memory.message_plugin import Config, project
    from session.message import ContentReferences

    async with application(tmp_path) as (log, host):
        legacy = log.writer("s", author="legacy-attribution-unknown", source="legacy-unattributed",
            body_types=(Input, Output), content={"text": check_text,
                "history.provenance": lambda part: ContentReferences()})
        user = log.writer("s", author="user", source="conversation", body_types=(Input, Output),
            content={"text": check_text})
        legacy.append("excluded-input", Input((ContentPart("text", "fact-one"),
            legacy_part('{"effects":{"post_commit":"suppress"}}'))))
        user.append("allowed-input", Input((ContentPart("text", "fact-three"),)))
        legacy.append("excluded-answer", Output((ContentPart("text", "fact-two"),), "complete"))
        user.append("allowed-answer", Output((ContentPart("text", "done"),), "complete"))
        original = log.reader("s").snapshot()
        summary = publish(log, "mixed-source")
        used = await record_use(log, host, summary, "used")
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            store = profile_store(tmp_path)
            await project(used, reader=log.reader("s"), bindings=ctx.require(BINDINGS), store=store,
                models=ctx.require(CHAT_MODELS), lock_path=tmp_path / "workspace/memory/markdown-profile.lock",
                sources=Config().sources, projection=ctx.require(TURN_PROJECTION))
        prompt = (tmp_path / "requests.jsonl").read_text()
        assert "fact-one" not in prompt and "fact-two" not in prompt
        assert "fact-three" in store.read_memory() and store.is_applied(summary.reference)
        assert log.reader("s").snapshot()[:4] == original


@pytest.mark.asyncio
@pytest.mark.parametrize("draft_failures", [1, 2])
async def test_invalid_model_draft_repairs_or_retries_without_partial_writes(tmp_path, draft_failures):
    """缩短的证据 ID 不得提交；一轮修正失败后 follower 仍能从原回执重试。"""
    async with application(tmp_path, draft_failures=draft_failures) as (log, host):
        writer = log.writer("s", author="user", source="conversation", body_types=(Input,),
                            content={"text": check_text})
        writer.append("legacy-message:" + "a" * 64, Input((ContentPart("text", "fact-one"),)))
        summary = publish(log, "draft-repair")
        await record_use(log, host, summary, "used-summary")
        original = log.reader("s").snapshot()
        store = profile_store(tmp_path)
        before = (store.read_memory(), store.read_self(), store.read_writes(None, 20))
        await host.start_runtime()
        if draft_failures == 2:
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                ctx = snapshot.composition_root.context
                await asyncio.wait_for(ctx.require(ServiceKey("fixture.retry_failed")).wait(), 5)
                assert (store.read_memory(), store.read_self(), store.read_writes(None, 20)) == before
                assert not store.is_applied(summary.reference)
                ctx.require(ServiceKey("fixture.retry_release")).set()
        await wait_applied(tmp_path, host, summary.reference)
        assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == draft_failures + 1
        assert store.read_memory().count("fact-one") == 1
        assert log.reader("s").snapshot() == original


@pytest.mark.parametrize("case", ["no_change", "new_fact", "empty_memory", "self", "multiline", "move_existing", "assistant_claim", "truncated"])
def test_profile_additions_preserve_owner_files_and_enforce_evidence(tmp_path, case):
    """模型无法用增量绕过旧事实保留、单行出处与真实用户证据边界。"""
    import json
    from agent.plugin_composition.models import LLMResponse
    from plugins.markdown_memory.message_plugin import _check_profile_response

    log = MessageLog(tmp_path / "messages.db")
    try:
        user = log.writer("s", author="user", source="conversation", body_types=(Input,), content={"text": check_text})
        assistant = log.writer("s", author="assistant", source="conversation", body_types=(Output,), content={"text": check_text})
        user.append("user-fact", Input((ContentPart("text", "请记住我喜欢红色"),)))
        assistant.append("assistant-claim", Output((ContentPart("text", "用户喜欢红色"),), "complete"))
        store = profile_store(tmp_path)
        memory = "# 用户长期记忆\n## 用户事实\n## 用户偏好\n## 用户明确要求长期记住的关键内容\n## 助手操作上下文\n- 已有操作记录\n"
        if case == "empty_memory":
            memory = ""
        before_self = store.read_self()
        line = "- 用户喜欢红色"
        if case == "multiline":
            line += "\n- 没有证据的事实"
        elif case == "move_existing":
            line = "- 已有操作记录"
        additions = [] if case == "no_change" else [{"document": "memory", "section": "## 用户偏好", "line": line,
            "message_ids": ["assistant-claim" if case == "assistant_claim" else "user-fact"]}]
        if case == "self":
            additions[0].update(document="self", section="## 我对当前用户的理解")
        response = LLMResponse(json.dumps({"additions": additions}), finish_reason="length" if case == "truncated" else "stop")
        if case not in {"no_change", "new_fact", "empty_memory", "self"}:
            with pytest.raises(ValueError):
                _check_profile_response(response, log.reader("s").snapshot(), memory, before_self)
        else:
            draft = _check_profile_response(response, log.reader("s").snapshot(), memory, before_self)
            assert draft["self_before"] == before_self
            assert draft["memory_before"] == memory
            if case == "no_change":
                assert draft["memory"] == memory
                assert draft["self"] == before_self
                assert draft["evidence"] == {"memory": {}, "self": {}}
            elif case == "self":
                assert draft["memory"] == memory
                assert isinstance(draft["self"], str)
                assert before_self.split("## 我们关系的定义")[0] in draft["self"]
                assert "- 用户喜欢红色\n## 我们关系的定义" in draft["self"]
                assert draft["evidence"] == {"memory": {}, "self": {"- 用户喜欢红色": ["user-fact"]}}
            else:
                expected = memory or "# 用户长期记忆\n\n## 用户事实\n\n## 用户偏好\n\n## 用户明确要求长期记住的关键内容\n"
                expected = expected.replace("## 用户明确要求长期记住的关键内容", "- 用户喜欢红色\n## 用户明确要求长期记住的关键内容")
                assert draft["memory"] == expected
                assert draft["self"] == before_self
                assert draft["evidence"] == {"memory": {"- 用户喜欢红色": ["user-fact"]}, "self": {}}
    finally:
        log.close()


@pytest.mark.parametrize("case", ["envelope", "entry", "section", "ids", "duplicate"])
def test_profile_rejects_invalid_model_additions_before_building_a_durable_draft(tmp_path, case):
    """畸形外部增量只形成可重试错误，不能产生部分持久草稿。"""
    import json
    from agent.plugin_composition.models import LLMResponse
    from plugins.markdown_memory.message_plugin import _check_profile_response, _InvalidDraft

    store = profile_store(tmp_path)
    before = (store.read_memory(), store.read_self(), store.read_writes(None, 10))
    entry: dict[str, object] = {"document": "memory", "section": "## 用户事实", "line": "- 新事实", "message_ids": ["missing"]}
    additions = [entry]
    payload: dict[str, object] = {"additions": additions}
    if case == "envelope":
        payload = {"memory": "overwrite", "additions": []}
    elif case == "entry":
        entry["replace"] = "old fact"
    elif case == "section":
        entry["section"] = "# 用户长期记忆"
    elif case == "ids":
        entry["message_ids"] = [None]
    else:
        additions.append(entry)
    with pytest.raises(_InvalidDraft):
        _check_profile_response(LLMResponse(json.dumps(payload)), (), before[0], before[1])
    assert (store.read_memory(), store.read_self(), store.read_writes(None, 10)) == before
