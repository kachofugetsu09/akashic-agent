"""真实候选的独立消息/Task 与清理；不把程序验证冒充正式来源启动。"""
import asyncio
from contextlib import closing

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace


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
from plugins.reply_program.program import run_reply
from plugins.models.projection import MODEL_CALLS
from plugins.react.plugin import REACT
from plugins.tools.plugin import ALL_TOOLS, TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.log import SessionAttributes
from session.message import ContentPart, ContentReferences, Input
api_version = 3
name = "probe"
version = "1.0.0"
inject = (CHAT_MODELS, MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION, TASKS,
          CONTENT, CONTEXT, MATERIALS, MODEL_CALLS, REACT, TOOLS, ALL_TOOLS, TURN_PROJECTION, ServiceKey("tools.cleanup.v1"))
async def apply(ctx):
    async def validate():
        ctx.require(SESSION_ADMISSION).ensure(ctx, "validation", SessionAttributes("internal", "excluded"))
        reader = ctx.require(MESSAGE_CATALOG).reader("validation")
        writer = ctx.require(MESSAGE_WRITERS).bind(
            ctx, author="user", source="validation", body_types=(Input,),
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
                cleanup=ctx.require(ServiceKey("tools.cleanup.v1")),
                turn_projection=ctx.require(TURN_PROJECTION), read_call=ctx.require(MODEL_CALLS),
                authorize=authorize, tool_view=ctx.require(ALL_TOOLS)(),
                max_output_tokens=100, max_steps=4,
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
            snapshot = validation.snapshot_store.current
            assert snapshot is not None
            for plugin_id, generation in snapshot.generations.items():
                assert generation in validation.generations
                assert generation is not host.latest_snapshot.generations[plugin_id]
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
async def apply(ctx):
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
            assert not path.exists()
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
    initialize_plugin_workspace(workspace)
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
                original = validation.stop_resources
                async def close_resources():
                    nonlocal calls
                    calls += 1
                    cleanup_entered.set()
                    await release_cleanup.wait()
                    await original()
                monkeypatch.setattr(validation, "stop_resources", close_resources)
        task = asyncio.create_task(validate())
        await asyncio.wait_for(cleanup_entered.wait(), 10)
        validation = next(iter(host._validation_hosts.values()))
        assert not validation.active and not validation.closed
        joined = asyncio.Event()
        original_finish = host._finish_termination
        async def finish(previous):
            joined.set()
            await original_finish(previous)
        monkeypatch.setattr(host, "_finish_termination", finish)
        shutdown = asyncio.create_task(host.terminate_all())
        await joined.wait()
        assert not shutdown.done() and calls == 1
        release_cleanup.set()
        results = await asyncio.gather(task, shutdown, return_exceptions=True)
        assert results[1] is None
        assert results[0] is None or isinstance(results[0], asyncio.CancelledError)
        assert calls == 1 and validation.closed
        assert host._validation_hosts == {}
    finally:
        release_cleanup.set()
        await asyncio.gather(*(item for item in (task, shutdown) if item is not None), return_exceptions=True)
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_validation_mcp_failure_keeps_real_owner_and_candidate_pin_for_retry(tmp_path, monkeypatch):
    from tests.test_mcp_binding_scope import SERVICE, write_plugin, select_mcp_provider

    source, workspace, home = (tmp_path / name for name in ("source", "workspace", "home"))
    write_plugin(source)
    providers = tmp_path / "providers"
    select_mcp_provider(providers)
    _commit(source)
    install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    log = MessageLog(workspace / "sessions.db")
    initialize_plugin_workspace(workspace)
    host = PluginManager([providers], event_bus=EventBus(), workspace=workspace, message_log=log,
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
                # 已安装 provider 使用自己的模块 namespace，按实际会话 host 注入故障。
                async with scope.require(SERVICE)() as server:
                    from agent.plugin_composition.mcp_slots import MCP_SERVERS
                    service = validation.root.context.require(MCP_SERVERS)
                    session = service._sessions[server.generation_id]
                    actual = session._host._cleanup_entry
                    async def fail_actual(entry):
                        nonlocal failed, process
                        if not failed:
                            failed = True
                            process = entry.client._process
                            raise OSError("injected live MCP cleanup failure")
                        await actual(entry)
                    monkeypatch.setattr(session._host, "_cleanup_entry", fail_actual)
                    async with server.route() as route:
                        assert (await route.call("ping", {})).output == "fixed B"
        assert failed and process is not None and process.returncode is None
        assert host.read_update(result.update_id).error
        assert host.latest_snapshot.lease_count == 1
        assert tuple(host._validation_hosts) == (validation.identity,)
        with pytest.raises(RuntimeError, match="资源尚未清理|调用失败"):
            host.start_update_publication(result.update_id)
        await host.retry_validation_cleanup(validation.identity)
        assert process.returncode is not None
        assert host._validation_hosts == {}
        assert host.latest_snapshot.lease_count == 0
        async with host.current_snapshot.composition_root.service_value(SERVICE)() as formal:
            async with formal.route() as route:
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


@pytest.mark.asyncio
async def test_candidate_has_only_its_own_messages_and_plugin_data(tmp_path):
    """插件自行准备隔离数据；Core 不复制正式消息、文件或历史 binding。"""
    source, workspace, old, log, host = prepare(tmp_path)
    try:
        await host.load_all()
        (old.data_path / "private-link").symlink_to(old.data_path / "history.txt")
        (source / "plugin.py").write_text(MODULE + "\nmarker = 'new'\n")
        _commit(source)
        result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        before = tuple(log._connection.iterdump())
        async with host.open_validation(result.update_id):
            validation = next(iter(host._validation_hosts.values()))
            assert validation.messages.reader("formal").snapshot() == ()
            assert validation.messages.read_bindings() == ()
            assert not (validation.workspace / "plugin-data/probe-lab/history.txt").exists()
            assert not (validation.workspace / "plugin-data/probe-lab/private-link").exists()
            actual = validation.snapshot_store.current
            assert actual is not None
            assert {key: value.archive_ref for key, value in actual.generations.items()} == {
                key: value.archive_ref for key, value in host.latest_snapshot.generations.items()
            }
        assert tuple(log._connection.iterdump()) == before
        assert (old.data_path / "history.txt").read_text() == "formal history"
        assert validation.workspace.exists()
    finally:
        await host.terminate_all()
        log.close()
