"""真实候选的独立消息/Task 与清理；不把程序验证冒充正式来源启动。"""
import asyncio
from contextlib import closing

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input
from tests.test_plugin_install import _commit, _write_v3_plugin

REPLY_MODULE = '''
from agent.plugin_composition import CHAT_MODELS, ServiceKey
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS
from plugins.content.plugin import CONTENT
from plugins.context.plugin import CONTEXT
from plugins.context.materials import MATERIALS
from plugins.conversation.program import run_reply
from plugins.models.projection import MODEL_CALLS
from plugins.react.plugin import REACT
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.log import SessionAttributes
from session.message import ContentPart, ContentReferences, Input
api_version = 3
name = "probe"
version = "1.0.0"
inject = (CHAT_MODELS, MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION, TASKS,
          CONTENT, CONTEXT, MATERIALS, MODEL_CALLS, REACT, TOOLS, TURN_PROJECTION)
async def apply(ctx, config):
    async def validate():
        ctx.require(SESSION_ADMISSION).ensure(ctx, "validation", SessionAttributes("internal", "excluded"))
        reader = ctx.require(MESSAGE_CATALOG).reader("validation")
        writer = ctx.require(MESSAGE_WRITERS).bind(
            ctx, author="probe", source="validation", body_types=(Input,),
            content={"text": lambda part: ContentReferences()},
        )("validation")
        writer.append("input", Input((ContentPart("text", "verify updated candidate"),)))
        async def authorize(binding, arguments):
            return {"decision": "allowed"}
        async def program(task):
            task.on_close(writer.expire)
            return await run_reply(
                ctx, task, reader, "validation", models=ctx.require(CHAT_MODELS),
                content=ctx.require(CONTENT), context=ctx.require(CONTEXT), tools=ctx.require(TOOLS),
                react=ctx.require(REACT), materials=ctx.require(MATERIALS),
                turn_projection=ctx.require(TURN_PROJECTION), read_call=ctx.require(MODEL_CALLS),
                authorize=authorize, tool_names=("write_evidence",), max_output_tokens=100, max_steps=4,
            )
        task = await ctx.require(TASKS).open(ctx).admit("validation", lambda slot: slot.start(program))
        return await task.join()
    await ctx.provide(ServiceKey("test.validation"), validate)
'''


@pytest.mark.asyncio
async def test_validation_runs_real_reply_model_projection_and_tool_records(tmp_path):
    """只替换外部模型 Driver；程序、投影和工具结算使用实际插件。"""
    from tests.test_default_reply import application
    from session.message import Output, ToolResult

    async with application(tmp_path, replying=False, start=False, provider_effect_data=True) as (log, host):
        source = tmp_path / "source"
        _write_v3_plugin(source, name="probe", module_source=REPLY_MODULE)
        _commit(source)
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        before = log.catalog().snapshot_heads()
        async with host.open_validation(result.update_id) as scope:
            validation = next(iter(host._validation_hosts.values()))
            output = await scope.require(ServiceKey("test.validation"))()
            assert output.body.finish == "complete"
            assert any(isinstance(part, ContentPart) and part.value == "finished" for part in output.body.parts)
            rows = validation.messages.reader("validation").snapshot()
            assert tuple(type(row.body) for row in rows) == (Input, Output, ToolResult, Output)
            tool_result = next(row.body for row in rows if isinstance(row.body, ToolResult))
            assert tool_result.outcome == "success"
            effects = list(validation.workspace.rglob("effect.txt"))
            assert len(effects) == 1 and effects[0].read_text() == "once\n"
        assert log.catalog().snapshot_heads() == before
        for generation in host.current_snapshot.generations.values():
            assert not (generation.data_dir / "effect.txt").exists()
        with closing(MessageLog(validation.workspace / "sessions.db")) as recovered:
            assert recovered.reader("validation").snapshot() == rows

MODULE = '''
from agent.plugin_composition import ServiceKey, RUNTIME_STARTED
from agent.plugin_composition.messages import MESSAGE_WRITERS, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS
from session.log import SessionAttributes
from session.message import ContentPart, ContentReferences, Input, Output
api_version = 3
name = "probe"
version = "1.0.0"
inject = (MESSAGE_WRITERS, SESSION_ADMISSION, TASKS)
async def apply(ctx, config):
    async def forbidden(event):
        raise AssertionError("program validation started an automatic source")
    await ctx.on(RUNTIME_STARTED, forbidden)
    async def validate(entered, release):
        ctx.require(SESSION_ADMISSION).ensure(ctx, "validation", SessionAttributes("internal", "excluded"))
        writer = ctx.require(MESSAGE_WRITERS).bind(
            ctx, author="probe", source="validation", body_types=(Input, Output),
            content={"text": lambda part: ContentReferences()},
        )("validation")
        writer.append("input", Input((ContentPart("text", "check the candidate"),)))
        async def program(task):
            task.on_close(writer.expire)
            entered.set()
            await release.wait()
            path = ctx.runtime.data_dir / "history.txt"
            assert path.read_text() == "formal history"
            path.write_text("isolated effect")
            return writer.append("output", Output((ContentPart("text", "old"),), "complete"))
        task = await ctx.require(TASKS).open(ctx).admit("validation", lambda slot: slot.start(program))
        return await task.join()
    await ctx.provide(ServiceKey("test.validation"), validate)
'''


def prepare(tmp_path):
    source, workspace, home = (tmp_path / name for name in ("source", "workspace", "home"))
    _write_v3_plugin(source, name="probe", module_source=MODULE)
    _commit(source)
    old = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    (old.data_path / "history.txt").write_text("formal history")
    log = MessageLog(workspace / "sessions.db")
    log.writer("formal", author="user", source="conversation", body_types=(Input,),
               content={"text": lambda part: ContentReferences()}).append(
        "formal-input", Input((ContentPart("text", "existing formal message"),)))
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, message_log=log,
                         installed_cache_root=home / "cache")
    return source, workspace, old, log, host


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["complete", "cancel", "shutdown"])
async def test_validation_owns_separate_messages_tasks_and_keeps_evidence(tmp_path, finish):
    source, workspace, old, log, host = prepare(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    task = None
    try:
        await host.load_all()
        (source / "plugin.py").write_text(MODULE.replace('ContentPart("text", "old")', 'ContentPart("text", "new")'))
        _commit(source)
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        (source / "plugin.py").unlink()
        formal_before = log.reader("formal").snapshot()
        scope = None
        async def validate():
            nonlocal scope
            async with host.open_validation(result.update_id) as scope:
                return await scope.require(ServiceKey("test.validation"))(entered, release)
        task = asyncio.create_task(validate())
        await asyncio.wait_for(entered.wait(), 10)
        assert len(host._validation_hosts) == 1
        validation = next(iter(host._validation_hosts.values()))
        assert validation.workspace != workspace
        with pytest.raises(RuntimeError, match="验证尚未退出"):
            host.start_update_publication(result.update_id)
        assert not host.update_is_publishing(result.update_id)
        if finish == "complete":
            release.set()
            output = await task
            assert output.body.parts[0].value == "new"
        elif finish == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await host.terminate_all()
            assert task.cancelled()
        assert host._validation_hosts == {}
        assert log.reader("formal").snapshot() == formal_before
        assert log.reader("validation").snapshot() == ()
        assert (old.data_path / "history.txt").read_text() == "formal history"
        assert validation.workspace.exists()
        with closing(MessageLog(validation.workspace / "sessions.db")) as recovered:
            rows = recovered.reader("validation").snapshot()
            assert tuple(row.message_id for row in rows) == (("input", "output") if finish == "complete" else ("input",))
        with pytest.raises(RuntimeError, match="关闭"):
            scope.require(ServiceKey("test.validation"))
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_shutdown_waits_for_validation_cleanup_already_in_progress(tmp_path, monkeypatch):
    """调用者已经退出程序时，关闭宿主不能并行关闭同一份连接。"""
    source, _, _, log, host = prepare(tmp_path)
    cleanup_entered, release_cleanup = asyncio.Event(), asyncio.Event()
    task = shutdown = None
    try:
        await host.load_all()
        (source / "plugin.py").write_text(MODULE + "\nmarker = 'new'\n")
        _commit(source)
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        calls = 0
        async def validate():
            async with host.open_validation(result.update_id):
                validation = next(iter(host._validation_hosts.values()))
                original = validation.manager.stop_validation_resources
                async def close_resources():
                    nonlocal calls
                    calls += 1
                    cleanup_entered.set()
                    await release_cleanup.wait()
                    await original()
                monkeypatch.setattr(validation.manager, "stop_validation_resources", close_resources)
        task = asyncio.create_task(validate())
        await asyncio.wait_for(cleanup_entered.wait(), 10)
        validation = next(iter(host._validation_hosts.values()))
        assert not validation.active and not validation.closed
        retry_entered = asyncio.Event()
        original_retry = host.retry_validation_cleanup
        async def retry(identity):
            retry_entered.set()
            await original_retry(identity)
        monkeypatch.setattr(host, "retry_validation_cleanup", retry)
        shutdown = asyncio.create_task(host.terminate_all())
        await asyncio.wait_for(retry_entered.wait(), 10)
        assert not shutdown.done() and calls == 1
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(task, shutdown), 10)
        assert calls == 1 and validation.closed
        assert host._validation_hosts == {}
    finally:
        release_cleanup.set()
        await asyncio.gather(*(item for item in (task, shutdown) if item is not None), return_exceptions=True)
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_validation_copy_failure_closes_connections_and_releases_candidate(tmp_path):
    source, workspace, old, log, host = prepare(tmp_path)
    try:
        await host.load_all()
        (source / "plugin.py").write_text(MODULE + "\nmarker = 'new'\n")
        _commit(source)
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        (host.ready_candidate.data_dir / "invalid-link").symlink_to(old.data_path / "history.txt")
        with pytest.raises(RuntimeError, match="符号链接"):
            async with host.open_validation(result.update_id):
                pytest.fail("invalid data was copied")
        assert host._validation_hosts == {}
        assert host.latest_snapshot.lease_count == 0
        for database in (workspace / "runtime/plugin-update-validation").glob("*/workspace/sessions.db"):
            with closing(MessageLog(database)) as recovered:
                assert recovered.reader("validation").snapshot() == ()
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_validation_mcp_failure_keeps_real_owner_and_candidate_pin_for_retry(tmp_path, monkeypatch):
    from tests.test_mcp_binding_scope import SERVICE, write_plugin

    source, workspace, home = (tmp_path / name for name in ("source", "workspace", "home"))
    write_plugin(source)
    _commit(source)
    install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    log = MessageLog(workspace / "sessions.db")
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, message_log=log,
                         installed_cache_root=home / "cache")
    try:
        await host.load_all()
        for name in ("first", "second"):
            path = source / name / "server.py"
            path.write_text(path.read_text().replace("fixed A", "fixed B"))
        _commit(source)
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        failed = False
        process = None
        validation = None
        with pytest.raises(RuntimeError, match="cleanup|清理"):
            async with host.open_validation(result.update_id) as scope:
                validation = next(iter(host._validation_hosts.values()))
                child = validation.manager
                runtime = child._composition_generation_host
                async with scope.require(SERVICE)() as server:
                    async with server.route() as route:
                        assert (await route.call("ping", {})).output == "fixed B"
                original = runtime._mcp_host._cleanup_entry
                async def fail_once(entry):
                    nonlocal failed, process
                    if not failed:
                        failed = True
                        process = entry.client._process
                        raise OSError("injected live MCP cleanup failure")
                    await original(entry)
                monkeypatch.setattr(runtime._mcp_host, "_cleanup_entry", fail_once)
        assert failed and process is not None and process.returncode is None
        assert host.latest_snapshot.lease_count == 1
        assert tuple(host._validation_hosts) == (validation.identity,)
        with pytest.raises(RuntimeError, match="资源尚未清理"):
            host.start_update_publication(result.update_id)
        await host.retry_validation_cleanup(validation.identity)
        assert process.returncode is not None
        assert host._validation_hosts == {}
        assert host.latest_snapshot.lease_count == 0
        formal = host._composition_generation_host.get(host.current_snapshot.generations["probe@lab"].generation_id)
        async with formal.mcp.server("first").route() as route:
            assert (await route.call("ping", {})).output == "fixed A"
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_validation_host_construction_failure_releases_candidate_scope(tmp_path, monkeypatch):
    """宿主尚未登记时的失败也必须归还候选，不能留下永久发布等待。"""
    source, _, _, log, host = prepare(tmp_path)
    try:
        await host.load_all()
        (source / "plugin.py").write_text(MODULE + "\nmarker = 'new'\n")
        _commit(source)
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        def fail_build(lease):
            raise OSError("injected validation construction failure")
        monkeypatch.setattr(host, "_build_validation_host", fail_build)
        with pytest.raises(OSError, match="construction failure"):
            async with host.open_validation(result.update_id):
                pytest.fail("validation entered after construction failed")
        assert host._validation_hosts == {}
        assert host.latest_snapshot.lease_count == 0
    finally:
        await host.terminate_all()
        log.close()


def memory_sources(root):
    """使用两个实际记忆入口，只替换外部 embedding provider。"""
    from pathlib import Path
    import shutil
    for name in ("akasha", "markdown_memory"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, root / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
        (root / name / "akashic.plugin.toml").write_text(
            f'schema_version = 1\nname = "{name}"\nversion = "4.0.0"\napi_version = 3\nentrypoint = "message_plugin.py"\n')
    settings = root.parent / "workspace/plugin-data/context-builtin/config.local.toml"
    settings.parent.mkdir(parents=True, exist_ok=True)
    with settings.open("a") as handle:
        handle.write('prompt_sources = {markdown_memory = "markdown_memory"}\n')
    provider = root / "fixture_embeddings"
    provider.mkdir()
    (provider / "plugin.py").write_text('''
from contextlib import asynccontextmanager
from agent.plugin_composition import EMBEDDINGS
from agent.plugin_composition.models import EmbeddingSpaceDescriptor, EmbeddingResult
api_version = 3
name = "fixture_embeddings"
version = "1.0.0"
inject = ()
async def apply(ctx, config):
    descriptor = EmbeddingSpaceDescriptor(
        plugin_snapshot_id="fixture", model_revision=0, model_id="fixture", connection_id="fixture",
        driver_id="fixture", driver_contract_version="1", auth_identity="fixture",
        connection_fingerprint="fixture", model="fixture", dimensions=2,
        normalization="unit", capability_digest="fixture")
    class Model:
        async def embed(self, texts):
            with (ctx.runtime.data_dir / "embeddings.txt").open("a") as output:
                output.write(repr(list(texts)) + "\\n")
            return EmbeddingResult(tuple((0.6, 0.8) for text in texts))
    model = Model()
    model.descriptor = descriptor
    class Embeddings:
        def describe(self, *, model_id=None):
            return descriptor
        @asynccontextmanager
        async def bind(self, *, model_id=None):
            yield model
    await ctx.provide(EMBEDDINGS, Embeddings())
''')


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [False, True])
async def test_validation_reply_reads_real_memory_without_starting_learning(tmp_path, history):
    """初始态与已发布图均走真实回复，验证查询不写正式图、向量或档案。"""
    from agent.plugin_composition import EMBEDDINGS
    from agent.plugin_composition.bindings import BINDINGS
    from agent.plugin_composition.messages import MESSAGE_EMBEDDINGS
    from agent.plugins.snapshot import lease_runtime_snapshot
    from plugins.akasha.application.consumer import MessageConsumer
    from plugins.akasha.domain.model import MemoryConfig
    from plugins.akasha.infrastructure.persistence import logical_state_sha256
    from plugins.akasha.learning import AKASHA_LEARNING, LearningConfig
    from plugins.akasha.recalls import RecallRecords
    from plugins.content.plugin import CONTENT
    from plugins.markdown_memory.plugin import start_store
    from plugins.markdown_memory.store import MarkdownProfileStore
    from session.message import Output
    from tests.test_default_reply import application
    import shutil

    async with application(tmp_path, replying=False, start=False, provider_effect_data=True,
                           extra_sources=memory_sources) as (log, host):
        memory = tmp_path / "workspace/memory"
        graph = memory / "akasha.db"
        if history:
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                ctx = snapshot.composition_root.context
                api = ctx.require(EMBEDDINGS)
                bindings = ctx.require(BINDINGS)
                vectors = ctx.require(MESSAGE_EMBEDDINGS)
                consumer = await MessageConsumer.load(graph, legacy_index=None, catalog=log.catalog(),
                    embeddings=vectors, bindings=bindings, config=MemoryConfig())
                try:
                    async with ctx.require(CONTENT).bind() as content:
                        writer = log.writer("past", author="user", source="conversation",
                            body_types=(Input, Output), content=content.checks)
                        writer.append("historical-input", Input((ContentPart("text", "remember my hiking route"),)))
                        writer.append("historical-answer", Output((ContentPart("text", "we walked to the blue lake"),), "complete"))
                    descriptor = api.describe()
                    binding = bindings.bind(AKASHA_LEARNING, LearningConfig(
                        embedding_model=descriptor.identity, dimension=2, sources=("conversation",)).model_dump())
                    async def embed(texts):
                        async with api.bind() as model:
                            return [list(vector) for vector in (await model.embed(texts)).vectors]
                    assert await consumer.consume(catalog=log.catalog(), learning_binding=binding,
                        embeddings=vectors, bindings=bindings, embed_batch=embed) == 1
                finally:
                    consumer.close()
            profile = MarkdownProfileStore(memory / "MEMORY.md", memory / "SELF.md",
                                           memory / "markdown-profile-writes.db")
            await start_store(profile, memory / "markdown-profile.lock", memory / "PENDING.md",
                              memory / "PENDING.snapshot.md", memory / "PENDING.retired.md")
            (memory / "MEMORY.md").write_text("known hiking preference")
        source = tmp_path / "source"
        _write_v3_plugin(source, name="probe", module_source=REPLY_MODULE)
        _commit(source)
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        shutil.rmtree(tmp_path / "plugins")
        (source / "plugin.py").unlink()
        before = tuple(log._connection.iterdump())
        files_before = {path.name: path.read_bytes() for path in memory.iterdir() if path.is_file()} if memory.exists() else {}
        graph_before = logical_state_sha256(graph) if history else None
        async with host.open_validation(result.update_id) as scope:
            validation = next(iter(host._validation_hosts.values()))
            output = await asyncio.wait_for(scope.require(ServiceKey("test.validation"))(), 20)
            assert output.body.finish == "complete"
            records = RecallRecords(validation.messages.owner("plugin:akasha")).list()
            assert records
            assert all(record.graph_version == (1 if history else 0) for _, record in records)
            if history:
                assert all(record.presented_message_ids == ("historical-input", "historical-answer")
                           for _, record in records)
                assert validation.messages.reader("past").snapshot() == log.reader("past").snapshot()
                assert logical_state_sha256(validation.workspace / "memory/akasha.db") == graph_before
            else:
                assert all(record.presented_message_ids == () for _, record in records)
                assert not (validation.workspace / "memory/akasha.db").exists()
                assert not (validation.workspace / "memory/MEMORY.md").exists()
            calls = scope.require(ServiceKey("fixture.calls"))
            prompt = str(calls[0].messages)
            assert ("known hiking preference" if history else "Akashic 自我认知") in prompt
        assert tuple(log._connection.iterdump()) == before
        assert ({path.name: path.read_bytes() for path in memory.iterdir() if path.is_file()}
                if memory.exists() else {}) == files_before
        assert (logical_state_sha256(graph) if history else None) == graph_before


@pytest.mark.asyncio
async def test_validation_copies_wal_history_archived_workspace_and_artifact_bytes(tmp_path):
    """当前组件不再声明旧数据时，实际旧 binding 和附件仍从独立副本打开。"""
    from agent.plugin_composition.artifacts import ARTIFACT_READ
    from agent.plugin_composition.bindings import BINDINGS
    from agent.plugins.snapshot import lease_runtime_snapshot
    from infra.channels.artifacts import ChannelAttachmentArtifactStore
    from session.artifact_store import ArtifactStore
    from session.artifacts import AttachmentKind

    source, workspace, home = (tmp_path / name for name in ("source", "workspace", "home"))
    original = MODULE.replace('api_version = 3',
        'workspace_roots = ("legacy",)\nworkspace_files = ("old-setting.txt",)\napi_version = 3') + '''
    async def history():
        return (ctx.workspace_root("legacy") / "old.txt").read_text() + ctx.workspace_file("old-setting.txt").read_text()
    await ctx.provide(ServiceKey("test.history"), history)
'''
    _write_v3_plugin(source, name="probe", module_source=original)
    _commit(source)
    _ = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    (workspace / "legacy").mkdir()
    (workspace / "legacy/old.txt").write_text("old workspace")
    (workspace / "old-setting.txt").write_text(" and file")
    log = MessageLog(workspace / "sessions.db")
    _ = log._connection.execute("PRAGMA journal_mode=WAL").fetchall()
    artifacts = ArtifactStore(workspace / "sessions.db")
    physical = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=artifacts)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, message_log=log,
                         installed_cache_root=home / "cache", channel_attachment_store=physical)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            reference = snapshot.composition_root.context.require(BINDINGS).bind(ServiceKey("test.history"), {})
        attachment = await physical.import_bytes(b"historical attachment bytes", kind=AttachmentKind.FILE,
                                                filename="history.txt", media_type="text/plain")
        log.writer("past", author="user", source="conversation", body_types=(Input,),
            content={"file": lambda part: ContentReferences(artifact_ids=(part.value,))}).append(
            "past-input", Input((ContentPart("file", attachment.artifact_id),)))
        assert (workspace / "sessions.db-wal").stat().st_size > 0
        (source / "plugin.py").write_text(MODULE)
        _commit(source)
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        (source / "plugin.py").unlink()
        before = tuple(log._connection.iterdump())
        async with host.open_validation(result.update_id) as scope:
            validation = next(iter(host._validation_hosts.values()))
            assert validation.messages.reader("past").snapshot() == log.reader("past").snapshot()
            async with scope.require(BINDINGS).open(reference, ServiceKey("test.history")) as (read, _):
                assert await read() == "old workspace and file"
            lease = await scope.require(ARTIFACT_READ).acquire(attachment)
            try:
                assert await lease.read_bytes(max_bytes=attachment.size_bytes) == b"historical attachment bytes"
            finally:
                await lease.aclose()
            (validation.workspace / "legacy/old.txt").write_text("only in the copy")
        assert (workspace / "legacy/old.txt").read_text() == "old workspace"
        assert tuple(log._connection.iterdump()) == before
    finally:
        await host.terminate_all()
        artifacts.close()
        log.close()
