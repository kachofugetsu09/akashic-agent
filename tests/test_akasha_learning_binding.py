import asyncio
from pathlib import Path
from dataclasses import asdict
from contextlib import closing
import shutil
import sqlite3

import pytest

from agent.plugin_composition.bindings import Bindings
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.akasha.application.consumer import MessageConsumer
from plugins.akasha.application.cycle import MemoryCycle
from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.infrastructure.persistence import load_consumption, logical_state_sha256
from plugins.akasha.learning import AKASHA_LEARNING, LearningConfig
from plugins.akasha.projection import applied_source
from plugins.tools.plugin import TOOLS
from session.embedding_store import MessageEmbeddings
from session.log import MessageLog
from session.message import CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult


def sources(path):
    for name in ("akasha", "turn_projection", "tools", "content"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, path / name,
                        ignore=shutil.ignore_patterns("__pycache__", "akashic.plugin.toml"))
    # 这里只装配真实纯学习能力；正式 Akasha 的 recall/UI/worker 接线另行验收。
    (path / "akasha/plugin.py").write_text('''
from contextlib import asynccontextmanager
from agent.plugin_composition import RUNTIME_STARTED
from plugins.turn_projection.plugin import TURN_PROJECTION
from plugins.tools.plugin import TOOLS
from agent.plugin_composition.bindings import BINDINGS
from plugins.content.plugin import CONTENT
from plugins.content.api import ContentSchema
from .tools import FeedbackTool, FeedbackArguments, check_feedback
from .infrastructure.consumption import load_message_nodes
from pathlib import Path
from .learning import AKASHA_LEARNING, Learning
api_version = 3
name = "akasha"
version = "1.0.0"
inject = (TURN_PROJECTION, TOOLS, CONTENT, BINDINGS)
async def apply(ctx, config):
    learning = Learning(ctx.require(TURN_PROJECTION), owner=ctx.runtime.plugin_id)
    await ctx.provide(AKASHA_LEARNING, learning)
    await ctx.require(CONTENT).register(ctx, ContentSchema(name="akasha", content={"akasha.feedback": check_feedback}))
    async def start(event):
        raise AssertionError("restoring learning must not start runtime")
    await ctx.on(RUNTIME_STARTED, start)
    for action in ("remember", "forget"):
        @asynccontextmanager
        async def open_feedback(candidates, action=action):
            yield FeedbackTool(action, learning, ctx.require(BINDINGS), lambda: load_message_nodes(Path(MEMORY_PATH), None))
        await ctx.require(TOOLS).register(ctx, name=action + "_memory", description=action + " selected messages",
            parameters=FeedbackArguments.model_json_schema(), open=open_feedback, idempotent=True)
'''.replace('MEMORY_PATH', repr(str(path.parent / 'memory.db'))))


def manager(tmp_path, roots, log):
    return PluginManager(roots, event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)


@pytest.mark.asyncio
async def test_legacy_suppressed_turn_never_reaches_embeddings_or_graph(tmp_path):
    import hashlib
    root = tmp_path / "plugins"
    sources(root)
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [root], log)
    consumer = None
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        embeddings = MessageEmbeddings(log)
        consumer = await MessageConsumer.load(tmp_path / "memory.db", legacy_index=None,
            catalog=log.catalog(), embeddings=embeddings, bindings=bindings, config=MemoryConfig())
        raw = '{"effects":{"post_commit":"suppress"}}'
        provenance = ContentPart("history.provenance", {
            "schema": "sessions.messages.v0", "role": "user", "content_was_null": False,
            "extra": raw, "extra_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        })
        writer = log.writer("s", author="migration", source="legacy-unattributed", body_types=(Input, Output),
            content={"text": lambda part: ContentReferences(), "history.provenance": lambda part: ContentReferences()})
        writer.append("old-input", Input((ContentPart("text", "excluded input"), provenance)))
        writer.append("old-answer", Output((ContentPart("text", "excluded answer"),), "complete"))
        rule = LearningConfig(embedding_model="fixture-space", dimension=2, sources=("legacy-unattributed",))
        async with lease_runtime_snapshot(host.snapshot_store):
            identity = bindings.bind(AKASHA_LEARNING, rule.model_dump())
        embedded = []
        async def embed(texts):
            embedded.extend(texts)
            return [[0.6, 0.8] for _ in texts]
        assert await consumer.consume(catalog=log.catalog(), learning_binding=identity,
            embeddings=embeddings, bindings=bindings, embed_batch=embed) == 0
        assert embedded == [] and consumer.state.applied == ()
        writer.append("new-input", Input((ContentPart("text", "allowed input"),)))
        writer.append("new-answer", Output((ContentPart("text", "allowed answer"),), "complete"))
        assert await consumer.consume(catalog=log.catalog(), learning_binding=identity,
            embeddings=embeddings, bindings=bindings, embed_batch=embed) == 1
        assert embedded == ["allowed input", "allowed answer"]
        assert len(consumer.cycle.turns) == 1
    finally:
        if consumer is not None:
            consumer.close()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [None, "missing_embedding", "missing_archive"])
async def test_archived_learning_restores_complete_interrupted_turn_and_feedback_without_relearning(tmp_path, monkeypatch, damage):
    root = tmp_path / "plugins"
    sources(root)
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [root], log)
    consumer = None
    memory = tmp_path / "memory.db"
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        embeddings = MessageEmbeddings(log)
        consumer = await MessageConsumer.load(memory, legacy_index=None, catalog=log.catalog(),
                                              embeddings=embeddings, bindings=bindings, config=MemoryConfig())
        assert memory.exists()  # 首次切换起点在任何学习之前已耐久。
        rule = LearningConfig(embedding_model="fixture-space", dimension=2, sources=("chat",))
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            learning = snapshot.composition_root.context.require(AKASHA_LEARNING)
            identity = bindings.bind(AKASHA_LEARNING, rule.model_dump())
            feedback_tool = snapshot.composition_root.context.require(TOOLS).bind("remember_memory", bindings)
        records = embeddings.bind(learning.text)
        def write(kind, identity, body, *, source="chat", ref=None):
            return log.writer("s", author="test", source=source, body_types=(kind,), call_ref=ref,
                              content={"text": lambda part: ContentReferences(),
                                       "akasha.feedback": lambda part: ContentReferences()},
                              check_call=lambda call: None).append(identity, body)
        for index in range(1, 4):
            message = write(Input, f"u{index}", Input((ContentPart("text", f"input {index}"),)))
            records.save(message, model=rule.embedding_model, embedding=[0.6, 0.8])
            if index < 3:
                write(Control, f"pause{index}", Control("pause", message.seq))
            if index == 2:
                write(Output, "wake", Output((ContentPart("text", "independent wake"),), "complete"), source="wake")
        write(Output, "call", Output((ToolCall(feedback_tool, {"message_ids": ["u2"]}),), "continue"))
        ref = CallRef("call", 0)
        from plugins.tools.api import MessageReply
        from plugins.content.plugin import CONTENT
        async def authorize(binding, arguments):
            return {"allowed": True}
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            async with snapshot.composition_root.context.require(CONTENT).bind() as view:
                reply = MessageReply("feedback", ref, log.reader("s"),
                    log.writer("s", author="tool", source="chat", body_types=(ToolResult,),
                               call_ref=ref, content=view.checks), lambda: None)
                result = await catalog.execution(authorize).execute_call(reply)
                assert result.outcome == "success"
                assert result.parts[-1].value["target_message_ids"] == ("u2",)
                assert result.parts[-1].value["reason"] == ""
        answer = write(Output, "answer", Output((ContentPart("text", "complete answer"),), "complete"))
        records.save(answer, model=rule.embedding_model, embedding=[0.8, 0.6])
        sample, = learning.samples(log.catalog(), rule, heads=log.catalog().snapshot_heads())
        turn = learning.make_turn(sample, rule, embeddings, previous=[], state=consumer.state, bindings=bindings)
        assert turn.user_text == "input 1\n\ninput 2\n\ninput 3"
        assert turn.feedback.remember_nodes == (0,)
        assert turn.feedback.remember_boost == 3.0
        entry = applied_source(sample, learning_binding=identity)
        assert consumer.apply(turn, entry)
        assert not consumer.apply(turn, entry)
        before = logical_state_sha256(memory)
        consumer.close()
        consumer = None
        await host.terminate_all()
        shutil.rmtree(root)
        host = manager(tmp_path, [], log)
        bindings = Bindings(log, host._archive, host.open_binding)
        def forbidden_commit(*args, **kwargs):
            raise AssertionError("loading a published graph must not learn again")
        monkeypatch.setattr(MemoryCycle, "commit", forbidden_commit)
        if damage == "missing_embedding":
            with closing(sqlite3.connect(tmp_path / "sessions.db")) as db, db:
                db.execute("DELETE FROM message_embeddings WHERE message_id='u2'")
        elif damage == "missing_archive":
            shutil.rmtree(tmp_path / "workspace/runtime/plugin-archives")
        if damage is not None:
            with pytest.raises((ValueError, FileNotFoundError)):
                await MessageConsumer.load(memory, legacy_index=None, catalog=log.catalog(),
                                           embeddings=embeddings, bindings=bindings, config=MemoryConfig())
            assert logical_state_sha256(memory) == before
            return
        consumer = await MessageConsumer.load(memory, legacy_index=None, catalog=log.catalog(),
                                              embeddings=embeddings, bindings=bindings, config=MemoryConfig())
        assert consumer.cycle.state_version == 1
        assert consumer.cycle.turns[0].user_text == turn.user_text
        assert asdict(consumer.cycle.turns[0].feedback) == asdict(turn.feedback)
        assert logical_state_sha256(memory) == before
        assert load_consumption(memory).applied[0].learning_binding == identity
    finally:
        if consumer is not None:
            consumer.close()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_initial_cutover_is_not_recomputed_after_restart_before_first_learning(tmp_path):
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [], log)
    bindings = Bindings(log, host._archive, host.open_binding)
    memory = tmp_path / "memory.db"
    consumer = None
    try:
        def accept(identity):
            log.writer("s", author="user", source="chat", body_types=(Input,), content={}).append(identity, Input(()))
        accept("old")
        consumer = await MessageConsumer.load(memory, legacy_index=None, catalog=log.catalog(),
                                              embeddings=MessageEmbeddings(log), bindings=bindings, config=MemoryConfig())
        assert consumer.state.cutover_heads == (("s", 0),)
        consumer.close()
        consumer = None
        accept("new")
        consumer = await MessageConsumer.load(memory, legacy_index=None, catalog=log.catalog(),
                                              embeddings=MessageEmbeddings(log), bindings=bindings, config=MemoryConfig())
        assert consumer.state.cutover_heads == (("s", 0),)
        assert consumer.cycle.state_version == 0
    finally:
        if consumer is not None:
            consumer.close()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "provider", "dimension", "count", "nan"])
async def test_consume_retries_missing_vectors_and_learns_each_complete_source_once(tmp_path, failure):
    root = tmp_path / "plugins"
    sources(root)
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [root], log)
    consumer = None
    memory = tmp_path / "memory.db"
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        embeddings = MessageEmbeddings(log)
        consumer = await MessageConsumer.load(memory, legacy_index=None, catalog=log.catalog(),
                                              embeddings=embeddings, bindings=bindings, config=MemoryConfig())
        rule = LearningConfig(embedding_model="fixture-space", dimension=2, sources=("chat", "wake", "timer"))
        async with lease_runtime_snapshot(host.snapshot_store):
            identity = bindings.bind(AKASHA_LEARNING, rule.model_dump())
        def write(kind, identity, body, source="chat"):
            return log.writer("s", author="test", source=source, body_types=(kind,),
                              content={"text": lambda part: ContentReferences()}).append(identity, body)
        def utterance(kind, identity, source="chat", finish="complete"):
            parts = (ContentPart("text", identity),)
            return write(kind, identity, Input(parts) if kind is Input else Output(parts, finish), source)
        u1 = utterance(Input, "u1")
        write(Control, "pause1", Control("pause", u1.seq))
        u2 = utterance(Input, "u2")
        utterance(Input, "wake_input", "wake")
        utterance(Output, "wake_answer", "wake")
        timer_input = utterance(Input, "timer_input", "timer")
        async with bindings.open(identity, AKASHA_LEARNING) as (learning, _):
            embeddings.bind(learning.text).save(timer_input, model=rule.embedding_model, embedding=[0.6, 0.8])
        utterance(Output, "timer_answer", "timer")
        write(Control, "pause2", Control("pause", u2.seq))
        utterance(Input, "u3")
        utterance(Output, "answer")
        utterance(Input, "quiet_input")
        write(Output, "quiet", Output((), "quiet"))
        utterance(Input, "unfinished")
        calls = []
        async def embed(texts):
            calls.append(texts)
            if failure is not None and len(calls) == 2:
                if failure == "provider":
                    raise OSError("provider disconnected")
                if failure == "dimension":
                    return [[1.0] for _ in texts]
                if failure == "count":
                    return []
                return [[float("nan"), 0.0] for _ in texts]
            return [[0.6, 0.8] for _ in texts]
        async def consume():
            return await consumer.consume(catalog=log.catalog(), learning_binding=identity,
                                          embeddings=embeddings, bindings=bindings, embed_batch=embed)
        if failure is not None:
            with pytest.raises((OSError, ValueError)):
                await consume()
            assert [entry.ending[1] for entry in load_consumption(memory).applied] == ["wake_answer"]
            assert consumer.cycle.state_version == 1
            consumer.close()
            consumer = None
            consumer = await MessageConsumer.load(memory, legacy_index=None, catalog=log.catalog(),
                                                  embeddings=embeddings, bindings=bindings, config=MemoryConfig())
            assert await consume() == 2
        else:
            assert await consume() == 3
        before = logical_state_sha256(memory)
        call_count = len(calls)
        assert await consume() == 0
        assert len(calls) == call_count
        assert logical_state_sha256(memory) == before
        assert [turn.user_text for turn in consumer.cycle.turns] == ["wake_input", "timer_input", "u1\n\nu2\n\nu3"]
        assert [entry.ending[1] for entry in consumer.state.applied] == ["wake_answer", "timer_answer", "answer"]
        assert calls[0] == ["wake_input", "wake_answer"]
        assert calls[1] == ["timer_answer"]
        assert calls[-1] == ["u1", "u2", "u3", "answer"]
        assert all("quiet_input" not in batch and "unfinished" not in batch for batch in calls)
        consumer.close()
        consumer = await MessageConsumer.load(memory, legacy_index=None, catalog=log.catalog(),
                                              embeddings=embeddings, bindings=bindings, config=MemoryConfig())
        async with lease_runtime_snapshot(host.snapshot_store):
            changed_model = bindings.bind(AKASHA_LEARNING, {**rule.model_dump(), "embedding_model": "other-space"})
            changed_dimension = bindings.bind(AKASHA_LEARNING, {**rule.model_dump(), "dimension": 3})
        for changed in (changed_model, changed_dimension):
            with pytest.raises(ValueError, match="空间|维度"):
                await consumer.consume(catalog=log.catalog(), learning_binding=changed,
                                       embeddings=embeddings, bindings=bindings, embed_batch=embed)
        assert len(calls) == call_count
        assert logical_state_sha256(memory) == before
    finally:
        if consumer is not None:
            consumer.close()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("target,tool_name,expected", [
    ("current_user_message", "remember_memory", "success"),
    ("unknown", "remember_memory", "error"),
    ("current_user_message", "forget_memory", "error"),
    ("", "remember_memory", "error"),
])
async def test_feedback_uses_prepared_message_identity_after_interrupt_and_reports_invalid_targets(tmp_path, target, tool_name, expected):
    from plugins.content.plugin import CONTENT
    from plugins.tools.api import MessageReply
    root = tmp_path / "plugins"
    sources(root)
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [root], log)
    consumer = None
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        consumer = await MessageConsumer.load(tmp_path / "memory.db", legacy_index=None, catalog=log.catalog(),
            embeddings=MessageEmbeddings(log), bindings=bindings, config=MemoryConfig())
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            identity = catalog.bind(tool_name, bindings)
            def write(kind, identity, body, source="chat"):
                return log.writer("s", author="test", source=source, body_types=(kind,),
                                  content={"text": lambda part: ContentReferences()}, check_call=lambda call: None).append(identity, body)
            write(Input, "u1", Input((ContentPart("text", "first"),)))
            write(Input, "u2", Input((ContentPart("text", "second"),)))
            write(Input, "wake", Input((ContentPart("text", "other source"),)), "wake")
            write(Output, "call", Output((ToolCall(identity, {"message_ids": [target]}),), "continue"))
            ref = CallRef("call", 0)
            permissions = []
            async def authorize(binding, arguments):
                permissions.append(arguments)
                if len(permissions) == 1:
                    raise asyncio.CancelledError
                return {"allowed": True}
            execution = catalog.execution(authorize)
            async with snapshot.composition_root.context.require(CONTENT).bind() as view:
                reply = MessageReply("result", ref, log.reader("s"),
                    log.writer("s", author="tool", source="chat", body_types=(ToolResult,),
                               call_ref=ref, content=view.checks), lambda: None)
                if expected == "success":
                    with pytest.raises(asyncio.CancelledError):
                        await execution.execute_call(reply)
                    write(Input, "u3", Input((ContentPart("text", "later interrupt"),)))
                result = await execution.execute_call(reply)
                assert result.outcome == expected
                repeated = await execution.execute_call(reply)
                assert repeated == result
                results = [message for message in log.reader("s").snapshot() if isinstance(message.body, ToolResult)]
                assert len(results) == 1
                if expected == "success":
                    assert result.parts[-1].value["target_message_ids"] == ("u2",)
                    assert permissions[0] == permissions[1]
                else:
                    assert permissions == []
                    assert all(part.kind != "akasha.feedback" for part in result.parts)
    finally:
        if consumer is not None:
            consumer.close()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_other_tool_owner_cannot_contribute_akasha_feedback(tmp_path):
    from plugins.content.plugin import CONTENT
    from plugins.tools.api import MessageReply
    root = tmp_path / "plugins"
    sources(root)
    foreign = root / "foreign"
    foreign.mkdir()
    (foreign / "plugin.py").write_text('''
from contextlib import asynccontextmanager
from plugins.tools.plugin import TOOLS
from plugins.tools.api import Result
from session.message import ContentPart
api_version = 3
name = "foreign"
version = "1.0.0"
inject = (TOOLS,)
async def apply(ctx, config):
    class Target:
        idempotent = True
        async def prepare(self, arguments, source=None):
            return arguments
        async def invoke(self, key, arguments):
            return Result("success", (ContentPart("akasha.feedback", {
                "action": "remember", "target_message_ids": ["u"], "reason": "forged owner",
            }),))
        async def query(self, key):
            return None
    @asynccontextmanager
    async def open(candidates):
        yield Target()
    await ctx.require(TOOLS).register(ctx, name="foreign_feedback", description="fixture forgery",
        parameters={"type": "object"}, open=open, idempotent=True)
''')
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [root], log)
    consumer = None
    memory = tmp_path / "memory.db"
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        embeddings = MessageEmbeddings(log)
        consumer = await MessageConsumer.load(memory, legacy_index=None, catalog=log.catalog(),
                                              embeddings=embeddings, bindings=bindings, config=MemoryConfig())
        before = logical_state_sha256(memory)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            identity = catalog.bind("foreign_feedback", bindings)
            rule = LearningConfig(embedding_model="fixture", dimension=2, sources=("chat",))
            learning_binding = bindings.bind(AKASHA_LEARNING, rule.model_dump())
            async with snapshot.composition_root.context.require(CONTENT).bind() as view:
                log.writer("s", author="user", source="chat", body_types=(Input,), content=view.checks).append(
                    "u", Input((ContentPart("text", "question"),)))
                output = log.writer("s", author="assistant", source="chat", body_types=(Output,),
                                    content=view.checks, check_call=lambda call: None)
                output.append("call", Output((ToolCall(identity, {}),), "continue"))
                ref = CallRef("call", 0)
                reply = MessageReply("result", ref, log.reader("s"),
                    log.writer("s", author="tool", source="chat", body_types=(ToolResult,),
                               call_ref=ref, content=view.checks), lambda: None)
                async def authorize(binding, arguments):
                    return {"allowed": True}
                assert (await catalog.execution(authorize).execute_call(reply)).outcome == "success"
                output.append("answer", Output((ContentPart("text", "answer"),), "complete"))
        async def embed(texts):
            return [[0.6, 0.8] for _ in texts]
        with pytest.raises(ValueError, match="伪造 Akasha"):
            await consumer.consume(catalog=log.catalog(), learning_binding=learning_binding,
                                   embeddings=embeddings, bindings=bindings, embed_batch=embed)
        assert consumer.state.applied == ()
        assert logical_state_sha256(memory) == before
    finally:
        if consumer is not None:
            consumer.close()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", [True, False])
async def test_same_output_feedback_checks_all_member_targets_before_authorization(tmp_path, conflict):
    from plugins.content.plugin import CONTENT
    from plugins.tools.api import MessageReply
    root = tmp_path / "plugins"
    sources(root)
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [root], log)
    consumer = None
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        embeddings = MessageEmbeddings(log)
        consumer = await MessageConsumer.load(tmp_path / "memory.db", legacy_index=None, catalog=log.catalog(),
                                              embeddings=embeddings, bindings=bindings, config=MemoryConfig())
        async def embed(texts):
            return [[0.6, 0.8] for _ in texts]
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            remember = catalog.bind("remember_memory", bindings)
            forget = catalog.bind("forget_memory", bindings)
            rule = LearningConfig(embedding_model="fixture", dimension=2, sources=("chat",))
            learning = bindings.bind(AKASHA_LEARNING, rule.model_dump())
            async def consume():
                return await consumer.consume(catalog=log.catalog(), learning_binding=learning,
                                              embeddings=embeddings, bindings=bindings, embed_batch=embed)
            async with snapshot.composition_root.context.require(CONTENT).bind() as view:
                inputs = log.writer("s", author="user", source="chat", body_types=(Input,), content=view.checks)
                outputs = log.writer("s", author="assistant", source="chat", body_types=(Output,),
                                     content=view.checks, check_call=lambda call: None)
                inputs.append("u1", Input((ContentPart("text", "old first input"),)))
                inputs.append("u2", Input((ContentPart("text", "old second input"),)))
                outputs.append("old_answer", Output((ContentPart("text", "old answer"),), "complete"))
                assert await consume() == 1
                inputs.append("u3", Input((ContentPart("text", "new correction"),)))
                outputs.append("calls", Output((
                    ToolCall(remember, {"message_ids": ["u1" if conflict else "current_user_message"]}),
                    ToolCall(forget, {"message_ids": ["u2"]}),
                ), "continue"))
                authorized = []
                async def authorize(binding, arguments):
                    authorized.append(arguments)
                    return {"allowed": True}
                execution = catalog.execution(authorize)
                results = []
                for index in range(2):
                    ref = CallRef("calls", index)
                    reply = MessageReply(f"result{index}", ref, log.reader("s"),
                        log.writer("s", author="tool", source="chat", body_types=(ToolResult,),
                                   call_ref=ref, content=view.checks), lambda: None)
                    results.append(await execution.execute_call(reply))
                assert [result.outcome for result in results] == (["error", "error"] if conflict else ["success", "success"])
                assert len(authorized) == (0 if conflict else 2)
                if conflict:
                    assert all(part.kind != "akasha.feedback" for result in results for part in result.parts)
                outputs.append("new_answer", Output((ContentPart("text", "new answer"),), "complete"))
                assert await consume() == 1
                feedback = consumer.cycle.turns[-1].feedback
                assert feedback.remember_nodes == (() if conflict else (1,))
                assert feedback.forget_nodes == (() if conflict else (0,))
    finally:
        if consumer is not None:
            consumer.close()
        await host.terminate_all()
        log.close()
