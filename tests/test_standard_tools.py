import asyncio
import base64
from collections.abc import Mapping
from contextlib import asynccontextmanager
from functools import partial
import inspect
import json
from pathlib import Path
import shutil
from typing import cast

import httpx
from PIL import Image
import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition.config_input import save_config

from agent.media import encode_image_data_uri
from agent.plugin_composition import CompositionError, PROCESSES, PluginProcesses, ServiceKey
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.context import CompositionRoot
from agent.plugin_composition.messages import BINDINGS, OWNER_STATE, OwnerState
from agent.plugin_composition.model import FiberState, PluginRuntime
from agent.process_runtime import (
    ExecutionCleanupFailure,
    ExecutionCleanupReport,
    ShellProcessManager,
)
from agent.plugin_composition.runtime_lifecycle import (
    RUNTIME_STARTED,
    RUNTIME_STARTING,
)
from agent.plugin_composition.tasks import PluginTasks, TASKS
from agent.plugin_composition.tasks import Tasks
from agent.plugins.manager import PluginManager
from agent.plugins.archive import PluginArchive
from agent.restart import RestartGate
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.standard_tools.plugin import register_shell
from plugins.standard_tools.shell import SHELL_OWNERS, TOOL_CLEANUP
from plugins.content.plugin import CONTENT, check_text
from plugins.context.materials import MATERIALS
from plugins.context.plugin import CONTEXT
from plugins.reply_program.program import run_reply
from plugins.turn_projection.plugin import TURN_PROJECTION
from plugins.tools.api import MessageReply
from session.message import Control, CallRef, ContentPart, Input, Output, ToolCall, ToolResult
from plugins.standard_web.web import WebTool
from plugins.tools.execution import ToolExecution
from plugins.tools.plugin import ALL_TOOLS, TOOLS, ToolCatalog, open_tool
from plugins.standard_web.search import WebSearchTool


from tests.test_message_push_plugin import storage
from tests.model_plugin_fakes import build_test_chat_models


class _UnusedModelProvider:
    model = "unused-test-model"
    context_window = 1024
    max_tool_schemas = None

    def estimate_context_tokens(self, messages, tools=()):
        raise AssertionError("controlled reply must not estimate model context")

    def estimate_appended_message_tokens(self, messages):
        raise AssertionError("controlled reply must not estimate appended messages")

    async def chat(self, **kwargs):
        raise AssertionError("controlled reply must not call the model")


def _unexpected_call_read(identity: str) -> Mapping[str, object]:
    raise AssertionError(f"controlled reply unexpectedly read model call {identity}")


def environment(tmp_path, *, reply=False, models=True):
    source = tmp_path / "plugins"
    for name in (
        "tools",
        "content",
        "context",
        "assets",
        "standard_tools",
        *(("turn_projection", "sources") if reply else ()),
        *(("ui", "models") if reply and models else ()),
    ):
        shutil.copytree(
            Path(__file__).parents[1] / "plugins" / name,
            source / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    probe = source / "probe"
    probe.mkdir()
    (probe / "plugin.py").write_text('''from agent.plugin_composition import ServiceKey
api_version = 3
name = "probe"
version = "1.0.0"
inject = (ServiceKey("core.bindings"), *REPLY_INJECT)
async def apply(ctx):
    await ctx.provide(ServiceKey("standard-tools-probe"), ctx)
'''.replace('REPLY_INJECT', '(ServiceKey("source.check.v1"), ServiceKey("models.selection.v1"))' if reply else '()'))
    workspace = tmp_path / "workspace"
    store, log = storage(workspace)
    context_config = workspace / "plugin-data/context-builtin"
    context_config.parent.mkdir(parents=True, exist_ok=True)
    save_config(context_config, {"prompt_sources": {"skills": "standard_tools"}})
    artifacts = ChannelAttachmentArtifactStore(
        workspace=workspace, metadata_store=store
    )
    initialize_plugin_workspace(workspace)
    host = PluginManager(
        [source],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
        message_log=log,
        channel_attachment_store=artifacts,
    )
    return host, store, log, artifacts, source


def _local_runtime(tmp_path, plugin_id):
    return PluginRuntime(
        plugin_id,
        plugin_id + ":local",
        tmp_path / "plugins",
        tmp_path / plugin_id / "data",
        tmp_path / "workspace",
        {},
    )


async def _mount_local_shell_root(tmp_path):
    """Build one real local Root for Shell cleanup lifecycle tests."""
    workspace = tmp_path / "local-workspace"
    store, log = storage(workspace)
    archive = PluginArchive(tmp_path / "local-archives")
    root = CompositionRoot("local-shell-root")
    tasks = PluginTasks()
    processes = PluginProcesses(
        factory=lambda: ShellProcessManager(output_dir=tmp_path / "shell-output")
    )
    await root.context.provide(TASKS, tasks)
    await root.context.provide(PROCESSES, processes)
    await root.context.provide(OWNER_STATE, OwnerState(log))
    generation_refs = {}

    def generation_lookup(ctx):
        runtime = ctx.runtime
        if runtime is None:
            raise AssertionError("local binding must point at a plugin runtime")
        ref = generation_refs.setdefault(
            runtime.plugin_id,
            archive.save_descriptor(
                {"plugin_id": runtime.plugin_id, "generation_id": runtime.generation_id}
            ),
        )
        return type("LocalGeneration", (), {
            "plugin_id": runtime.plugin_id,
            "archive_ref": ref,
        })()

    bindings = Bindings(log, archive, root, generation_lookup)
    await root.context.provide(BINDINGS, bindings)
    contexts = {}
    hard_consumer_unloading = asyncio.Event()
    effect_closes = {"shell": 0, "hard_consumer": 0, "unrelated": 0}

    async def mount_tools(ctx):
        contexts["tools"] = ctx
        admission = ctx.require(TASKS).open(ctx)
        contexts["tools_admission"] = admission
        catalog = ToolCatalog(ctx, admission)
        contexts["catalog"] = catalog
        await ctx.provide(TOOLS, catalog)

    tools_fiber = await root.mount(
        mount_tools,
        name="local-tools",
        inject=(TASKS, BINDINGS, OWNER_STATE),
        runtime=_local_runtime(tmp_path, "local-tools"),
    )

    async def mount_shell(ctx):
        contexts["shell"] = ctx

        async def close_effect():
            effect_closes["shell"] += 1

        await ctx.effect(lambda: close_effect, label="local-shell-effect")
        contexts["shell_refs"] = await register_shell(ctx)

    shell_fiber = await root.mount(
        mount_shell,
        name="local-shell",
        inject=(TOOLS, TASKS, PROCESSES, BINDINGS),
        runtime=_local_runtime(tmp_path, "local-shell"),
    )

    async def mount_hard_consumer(ctx):
        contexts["hard_consumer"] = ctx

        async def close_effect():
            effect_closes["hard_consumer"] += 1
            hard_consumer_unloading.set()

        await ctx.effect(lambda: close_effect, label="local-shell-hard-consumer")

    hard_consumer_fiber = await root.mount(
        mount_hard_consumer,
        name="local-shell-hard-consumer",
        inject=(TOOL_CLEANUP,),
        runtime=_local_runtime(tmp_path, "local-shell-hard-consumer"),
    )

    async def mount_caller(ctx):
        contexts["caller"] = ctx

    caller_fiber = await root.mount(
        mount_caller,
        name="local-caller",
        inject=(TOOL_CLEANUP,),
        runtime=_local_runtime(tmp_path, "local-caller"),
    )

    async def mount_unrelated(ctx):
        contexts["unrelated"] = ctx

        async def close_effect():
            effect_closes["unrelated"] += 1

        await ctx.effect(lambda: close_effect, label="local-unrelated-effect")
        await ctx.on(RUNTIME_STARTING, lambda _event: None)
        await ctx.on(RUNTIME_STARTED, lambda _event: None)

    unrelated_fiber = await root.mount(
        mount_unrelated,
        name="local-unrelated",
        runtime=_local_runtime(tmp_path, "local-unrelated"),
    )
    return {
        "root": root,
        "store": store,
        "log": log,
        "archive": archive,
        "bindings": bindings,
        "tasks": tasks,
        "processes": processes,
        "tools_fiber": tools_fiber,
        "shell_fiber": shell_fiber,
        "hard_consumer_fiber": hard_consumer_fiber,
        "caller_fiber": caller_fiber,
        "unrelated_fiber": unrelated_fiber,
        "tools_ctx": contexts["tools"],
        "tools_admission": contexts["tools_admission"],
        "catalog": contexts["catalog"],
        "shell_ctx": contexts["shell"],
        "shell_refs": contexts["shell_refs"],
        "hard_consumer_ctx": contexts["hard_consumer"],
        "caller_ctx": contexts["caller"],
        "unrelated_ctx": contexts["unrelated"],
        "hard_consumer_unloading": hard_consumer_unloading,
        "effect_closes": effect_closes,
    }


@pytest.mark.asyncio
async def test_standard_file_tools_keep_typed_errors_and_model_safe_image_artifact(tmp_path):
    host, store, log, artifacts, source = environment(tmp_path)
    tasks = Tasks()
    allowed = []

    async def authorize(identity, final):
        allowed.append(final)
        return {"allowed": True}

    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root is not None
            bindings = Bindings(log, host._archive, snapshot.composition_root)
            ctx = snapshot.composition_root.context
            tools = ctx.require(TOOLS)
            view = ctx.require(ALL_TOOLS)()
            read = tools.bind(view.select("read_file"), bindings)
            write = tools.bind(
                view.select("write_file"),
                bindings,
                configuration={"allowed_dir": str(tmp_path / "job")},
            )
            edit = tools.bind(
                view.select("edit_file"),
                bindings,
                configuration={"allowed_dir": str(tmp_path / "job")},
            )
        shutil.rmtree(source)
        execution = ToolExecution(
            log.owner("plugin:tools"), tasks, partial(open_tool, bindings), authorize,
            task_key="effects",
        )
        missing = await execution.execute("missing", read, {"path": str(tmp_path / "missing")})
        assert missing.outcome == "error" and "不存在" in cast(str, missing.parts[0].value)
        replayed = await execution.execute("missing", read, {"path": str(tmp_path / "missing")})
        assert (replayed.outcome, replayed.parts) == (missing.outcome, missing.parts)
        escaped = await execution.execute("escape", write, {"path": "../outside", "content": "bad"})
        assert escaped.outcome == "error" and not (tmp_path / "outside").exists()
        written = await execution.execute("write", write, {"path": "record.txt", "content": "alpha\nalpha\n"})
        assert written.outcome == "success"
        edited = await execution.execute("ambiguous", edit, {"path": "record.txt", "old_text": "alpha", "new_text": "beta"})
        assert edited.outcome == "error"
        assert (tmp_path / "job/record.txt").read_text() == "alpha\nalpha\n"
        picture = tmp_path / "source.png"
        Image.new("RGB", (20, 20), (50, 100, 150)).save(picture)
        expected = base64.b64decode(encode_image_data_uri(picture).partition(",")[2])
        result = await execution.execute("image", read, {"path": str(picture)})
        assert result.outcome == "success"
        reference = cast(str, next(part.value for part in result.parts if part.kind == "artifact_ref"))
        assert store.get_attachment(reference) is not None
        ref = artifacts.resolve_refs((reference,))[0]
        lease = await artifacts.acquire(ref)
        try:
            assert await lease.read_bytes(max_bytes=10000) == expected
        finally:
            await lease.aclose()
        assert len(allowed) == 5
    finally:
        await tasks.close()
        await host.terminate_all()
        log.close()
        store.close()


@pytest.mark.asyncio
async def test_standard_shell_config_and_cleanup_use_same_archived_job_owner(tmp_path):
    host, store, log, _artifacts, source = environment(tmp_path)
    tasks = Tasks()
    permissions = []

    async def authorize(identity, final):
        permissions.append(final)
        return {"allowed": True}

    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root is not None
            bindings = Bindings(log, host._archive, snapshot.composition_root)
            ctx = snapshot.composition_root.context
            catalog = ctx.require(TOOLS)
            view = ctx.require(ALL_TOOLS)()
            configuration = {
                "owner_key": "job-a",
                "working_dir": str(tmp_path),
                "allow_network": False,
            }
            command = catalog.bind(
                view.select("shell"), bindings, configuration=configuration
            )
            stdin = catalog.bind(
                view.select("write_stdin"), bindings, configuration=configuration
            )
            foreign = catalog.bind(
                view.select("write_stdin"),
                bindings,
                configuration={**configuration, "owner_key": "job-b"},
            )
            cleanup = bindings.bind(SHELL_OWNERS, {})
        shutil.rmtree(source)
        execution = ToolExecution(
            log.owner("plugin:tools"), tasks, partial(open_tool, bindings), authorize,
            task_key="effects",
        )
        blocked = await execution.execute("network", command, {"command": "curl https://example.com", "description": "network"})
        assert blocked.outcome == "error" and permissions == []
        started = await execution.execute("start", command, {
            "command": "printf READY; read line; printf 'GOT:%s' \"$line\"", "description": "controlled PTY",
            "shell": "/usr/bin/bash", "login": False, "tty": True, "yield_time_ms": 250,
        })
        assert started.outcome == "success"
        identity = cast(str, json.loads(cast(str, started.parts[0].value))["execution_id"])
        wrong = await execution.execute("wrong-owner", foreign, {"execution_id": identity, "chars": "PING\n", "yield_time_ms": 1000})
        assert wrong.outcome == "error"
        completed = await execution.execute("stdin", stdin, {"execution_id": identity, "chars": "PING\n", "yield_time_ms": 1000})
        assert completed.outcome == "success" and "GOT:PING" in json.loads(cast(str, completed.parts[0].value))["output"]
        waiting = await execution.execute("wait", command, {"command": "sleep 30", "description": "wait", "yield_time_ms": 250})
        identity = cast(str, json.loads(cast(str, waiting.parts[0].value))["execution_id"])
        async with bindings.open(cleanup, SHELL_OWNERS) as (owners, _):
            report = await owners.release("job-a")
        assert report.cleaned_execution_ids == (identity,) and not report.failures
    finally:
        await tasks.close()
        await host.terminate_all()
        log.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply,media,error", [
    ('{"jsonrpc":"2.0","id":1,"result":{"content":[]}}', "application/json", False),
    ('data: {"jsonrpc":"2.0","id":1,"result":{"content":[]}}\n\n', "text/event-stream", False),
    ('data: invalid\n\n', "text/event-stream", True),
    (': keepalive\n\n', "text/event-stream", True),
    ('event: error\ndata: {"jsonrpc":"2.0","id":1,"result":{"content":[]}}\n\n', "text/event-stream", True),
    (': keepalive\n\nevent: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\ndata: {"jsonrpc":"2.0","id":1,\ndata: "result":{"content":[]}}\n\n', "text/event-stream", False),
    ('{"jsonrpc":"2.0","id":1,"result":{"content":[],"isError":true}}', "application/json", True),
    ('{"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"unavailable"}}', "application/json", True),
    ('{"jsonrpc":"2.0","id":2,"result":{"content":[]}}', "application/json", True),
])
async def test_web_search_only_reports_empty_success_from_confirmed_response(monkeypatch, reply, media, error):
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=reply, headers={"content-type": media}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=transport, **kwargs))
    tool = WebTool(WebSearchTool())
    prepared = await tool.prepare({"query": "test"})
    assert isinstance(prepared, Mapping)
    result = await tool.invoke("request", prepared)
    assert result.outcome == ("error" if error else "success")
    if not error:
        assert json.loads(cast(str, result.parts[0].value))["result"] == ""


async def start_shell_call(log, bindings, tasks, binding, source, identity):
    """保存真实调用和回执，启动一个等待显式清理的进程。"""
    reader = log.reader("shared")
    text = {"text": check_text}
    def check_call(call):
        if call.binding_id != binding:
            raise PermissionError("fixture only grants its original Shell binding")
    output = log.writer("shared", author="assistant", source=source, body_types=(Output,), content=text, check_call=check_call)
    message = output.append(identity, Output((ToolCall(binding, {
        "command": "sleep 30", "description": "cleanup fixture", "yield_time_ms": 250,
    }),), "continue"))
    result_writer = log.writer("shared", author="tool", source=source, body_types=(ToolResult,), content=text,
                               call_ref=CallRef(message.message_id, 0))
    reply = MessageReply(identity + "-result", CallRef(message.message_id, 0), reader, result_writer, lambda: None)

    async def allow(identity, arguments):
        return {"allowed": True}

    execution = ToolExecution(
        log.owner("plugin:tools"), tasks, partial(open_tool, bindings), allow,
        task_key="effects",
    )
    result = await execution.execute_call(reply)
    assert result.outcome == "success"
    return cast(str, json.loads(cast(str, result.parts[0].value))["execution_id"])


@pytest.mark.asyncio
async def test_shell_cleanup_uses_original_binding_and_keeps_other_source_running(tmp_path):
    host, store, log, _artifacts, source = environment(tmp_path)
    tasks = Tasks()
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root is not None
            bindings = Bindings(log, host._archive, snapshot.composition_root)
            root = snapshot.composition_root.context
            tool = root.require(TOOLS).bind(
                root.require(ALL_TOOLS)().select("shell"), bindings
            )
        first = await start_shell_call(log, bindings, tasks, tool, "conversation", "first")
        second = await start_shell_call(log, bindings, tasks, tool, "wake", "second")
        binding_ids = tuple(row[0] for row in log._connection.execute(
            "SELECT binding_id FROM bindings ORDER BY binding_id"
        ))
        shutil.rmtree(source)
        # 清理使用稳定 owner key；不因源码目录变化跳过同一进程集合的终止。
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert isinstance(snapshot.composition_root, CompositionRoot)
            ctx = snapshot.composition_root.context
            assert ctx.get(SHELL_OWNERS) is not None
            cleanup = ctx.require(TOOL_CLEANUP)
            async with cleanup(log.reader("shared"), "conversation", 0):
                pass
            backend = host._plugin_processes._manager
            assert first not in await backend.active_execution_ids()
            assert second in await backend.active_execution_ids()
            async with cleanup(log.reader("shared"), "wake", 0):
                pass
            assert await backend.active_execution_ids() == []
        assert tuple(row[0] for row in log._connection.execute(
            "SELECT binding_id FROM bindings ORDER BY binding_id"
        )) == binding_ids
    finally:
        await tasks.close()
        await host.terminate_all()
        log.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("unload_owner", ["tools", "shell"])
async def test_local_shell_cleanup_survives_hard_owner_unload(tmp_path, monkeypatch, unload_owner):
    """C1: public cleanup keeps Shell-owned drain and process cleanup alive."""
    graph = await _mount_local_shell_root(tmp_path)
    old_call = None
    cleanup_call = None
    unload_call = None
    probe_call = None
    release_old = asyncio.Event()
    old_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    try:
        async with graph["tools_ctx"].runtime_scope():
            binding = graph["catalog"].bind(
                graph["shell_refs"][0], graph["bindings"],
                configuration={},
            )
        descriptor = graph["bindings"].describe(binding, TOOLS)
        backend = graph["processes"]._backend()
        original_exec = backend.exec_command
        original_terminate = backend.terminate_owner

        async def delayed_exec(**kwargs):
            old_started.set()
            await release_old.wait()
            return await original_exec(**kwargs)

        async def delayed_terminate(owner):
            cleanup_started.set()
            await release_cleanup.wait()
            return await original_terminate(owner)

        monkeypatch.setattr(backend, "exec_command", delayed_exec)
        monkeypatch.setattr(backend, "terminate_owner", delayed_terminate)
        old_call = asyncio.create_task(
            start_shell_call(
                graph["log"], graph["bindings"], graph["tools_admission"],
                binding, "conversation", "local-c1-old",
            )
        )
        await old_started.wait()

        async def caller_cleanup():
            async with graph["caller_ctx"].runtime_scope():
                cleanup = graph["caller_ctx"].require(TOOL_CLEANUP)
                async with cleanup(
                    graph["log"].reader("shared"), "conversation", 0,
                    drain=graph["catalog"].drain_calls,
                ):
                    unload = (
                        graph["tools_fiber"]
                        if unload_owner == "tools" else graph["shell_fiber"]
                    )
                    nonlocal unload_call
                    unload_call = asyncio.create_task(unload.dispose())
                    # A real hard consumer enters UNLOADING before Shell can
                    # dispatch STOPPING; the old call still holds its permit.
                    await graph["hard_consumer_unloading"].wait()
                    assert graph["shell_fiber"].state is FiberState.UNLOADING
                    assert graph["effect_closes"]["hard_consumer"] == 1

                    probe_body = asyncio.Event()

                    async def probe_public_cleanup():
                        async with cleanup(
                            graph["log"].reader("shared"), "conversation", 0,
                            drain=graph["catalog"].drain_calls,
                        ):
                            probe_body.set()

                    nonlocal probe_call
                    probe_call = asyncio.create_task(probe_public_cleanup())
                    with pytest.raises(CompositionError) as error:
                        await probe_call
                    assert error.value.code == "OWNER_UNAVAILABLE"
                    assert not probe_body.is_set()
                    unrelated_token = graph["unrelated_ctx"].fiber.activation_token
                    unrelated_state = graph["unrelated_fiber"].state
                    unrelated_context = graph["unrelated_ctx"]
                    unrelated_effects = graph["effect_closes"]["unrelated"]
                    async with unrelated_context.runtime_scope():
                        assert graph["unrelated_fiber"].state is unrelated_state
                        assert graph["unrelated_ctx"].fiber.activation_token is unrelated_token
                        assert unrelated_context is graph["unrelated_ctx"]
                    assert graph["unrelated_fiber"].state is FiberState.ACTIVE
                    assert graph["unrelated_ctx"].fiber.activation_token is unrelated_token
                    assert graph["effect_closes"]["unrelated"] == unrelated_effects

        cleanup_call = asyncio.create_task(caller_cleanup())
        await graph["hard_consumer_unloading"].wait()
        release_old.set()
        await old_call
        reader = graph["log"].reader("shared")
        original_messages = (
            reader.get("local-c1-old"), reader.get("local-c1-old-result")
        )
        await cleanup_started.wait()
        assert not cleanup_call.done()
        assert graph["bindings"].describe(binding, TOOLS) == descriptor
        release_cleanup.set()
        await cleanup_call
        await unload_call
        assert graph["shell_fiber"]._in_flight_calls == {}
        assert graph["effect_closes"]["shell"] == 1
        assert original_messages[0] is not None and original_messages[1] is not None
        assert (
            reader.get("local-c1-old"), reader.get("local-c1-old-result")
        ) == original_messages
    finally:
        release_old.set()
        release_cleanup.set()
        pending = tuple(
            task for task in (old_call, cleanup_call, unload_call, probe_call)
            if task is not None and not task.done()
        )
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await graph["root"].dispose()
        await graph["tasks"].close()
        await graph["processes"].close()
        graph["log"].close()
        graph["store"].close()


@pytest.mark.asyncio
async def test_local_shell_cleanup_failure_keeps_owner_for_explicit_retry(tmp_path, monkeypatch):
    """C3: a real cleanup report retains the old owner until retry."""
    graph = await _mount_local_shell_root(tmp_path)
    try:
        async with graph["tools_ctx"].runtime_scope():
            binding = graph["catalog"].bind(
                graph["shell_refs"][0], graph["bindings"],
                configuration={},
            )
        descriptor = graph["bindings"].describe(binding, TOOLS)
        old_id = await start_shell_call(
            graph["log"], graph["bindings"], graph["tools_admission"],
            binding, "conversation", "local-c3-old",
        )
        new_id = await start_shell_call(
            graph["log"], graph["bindings"], graph["tools_admission"],
            binding, "wake", "local-c3-new",
        )
        reader = graph["log"].reader("shared")
        original_messages = tuple(
            reader.get(message_id)
            for message_id in (
                "local-c3-old", "local-c3-old-result",
                "local-c3-new", "local-c3-new-result",
            )
        )
        backend = graph["processes"]._backend()
        original_terminate = backend.terminate_owner
        failed = True

        async def fail_once(owner):
            nonlocal failed
            if failed:
                failed = False
                return ExecutionCleanupReport(
                    (old_id,), (),
                    (ExecutionCleanupFailure(old_id, "OSError", "controlled failure"),),
                )
            return await original_terminate(owner)

        monkeypatch.setattr(backend, "terminate_owner", fail_once)

        async def cleanup_old():
            async with graph["caller_ctx"].runtime_scope():
                cleanup = graph["caller_ctx"].require(TOOL_CLEANUP)
                async with cleanup(
                    graph["log"].reader("shared"), "conversation", 0,
                    drain=graph["catalog"].drain_calls,
                ):
                    pass

        await cleanup_old()
        backend_ids = await backend.active_execution_ids()
        assert old_id in backend_ids and new_id in backend_ids
        assert any(
            incident.kind == "shell_cleanup_failed"
            for incident in graph["root"].recent_incidents()
        )
        assert graph["bindings"].describe(binding, TOOLS) == descriptor
        assert reader.get("local-c3-old") is not None
        assert reader.get("local-c3-old-result") is not None
        assert tuple(
            reader.get(message_id)
            for message_id in (
                "local-c3-old", "local-c3-old-result",
                "local-c3-new", "local-c3-new-result",
            )
        ) == original_messages

        await cleanup_old()
        backend_ids = await backend.active_execution_ids()
        assert old_id not in backend_ids and new_id in backend_ids
        assert tuple(
            reader.get(message_id)
            for message_id in (
                "local-c3-old", "local-c3-old-result",
                "local-c3-new", "local-c3-new-result",
            )
        ) == original_messages
    finally:
        await graph["root"].dispose()
        await graph["tasks"].close()
        await graph["processes"].close()
        graph["log"].close()
        graph["store"].close()


@pytest.mark.asyncio
async def test_local_shell_cleanup_closes_scope_when_raw_child_creation_fails(
    tmp_path, monkeypatch,
):
    """C4: synchronous raw-child failure closes the captured Shell scope."""
    graph = await _mount_local_shell_root(tmp_path)
    try:
        async with graph["tools_ctx"].runtime_scope():
            binding = graph["catalog"].bind(
                graph["shell_refs"][0], graph["bindings"],
                configuration={},
            )
        await start_shell_call(
            graph["log"], graph["bindings"], graph["tools_admission"],
            binding, "conversation", "local-c4-call",
        )
        captured = []
        drain_started = asyncio.Event()

        def fail_create_task(operation, **_kwargs):
            captured.append(operation)
            raise RuntimeError("controlled create_task failure")

        async def record_drain(_calls):
            drain_started.set()

        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "create_task", fail_create_task)
            with pytest.raises(RuntimeError, match="controlled create_task failure"):
                async with graph["caller_ctx"].runtime_scope():
                    cleanup = graph["caller_ctx"].require(TOOL_CLEANUP)
                    async with cleanup(
                        graph["log"].reader("shared"), "conversation", 0,
                        drain=record_drain,
                    ):
                        pass
        assert len(captured) == 1
        assert inspect.getcoroutinestate(captured[0]) == inspect.CORO_CLOSED
        assert not drain_started.is_set()
        assert graph["shell_fiber"]._in_flight_calls == {}
        async with graph["shell_ctx"].runtime_scope():
            pass
        assert not any(
            incident.kind == "shell_cleanup_failed"
            for incident in graph["root"].recent_incidents()
        )
    finally:
        monkeypatch.undo()
        await graph["root"].dispose()
        await graph["tasks"].close()
        await graph["processes"].close()
        graph["log"].close()
        graph["store"].close()


@pytest.mark.asyncio
async def test_local_shell_task_canceled_before_start_releases_scope_and_root_permit(tmp_path, monkeypatch):
    """C4: a real Shell cleanup Task canceled before user code starts leaks nothing."""
    graph = await _mount_local_shell_root(tmp_path)
    gate = RestartGate(boot_id="local-c4", supervised=False)
    permit = gate.acquire()
    caller_task = None
    cleanup_task = None
    caller_settled = asyncio.Event()
    cleanup_settled = asyncio.Event()
    cleanup_body_started = asyncio.Event()
    started = asyncio.Event()
    captured = []
    before_cancel = {}
    try:
        async with graph["tools_ctx"].runtime_scope():
            binding = graph["catalog"].bind(
                graph["shell_refs"][0], graph["bindings"], configuration={}
            )
        reader = graph["log"].reader("shared")
        controls = graph["log"].writer(
            "shared", author="user", source="conversation",
            body_types=(Control,), content={},
        )
        old_id = None

        async def record_cleanup(_calls):
            cleanup_body_started.set()

        async def caller_program(task):
            nonlocal old_id
            cleanup = graph["caller_ctx"].require(TOOL_CLEANUP)
            async with cleanup(
                reader, "conversation", 0, task=task,
                drain=record_cleanup,
            ):
                old_id = await start_shell_call(
                    graph["log"], graph["bindings"], graph["tools_admission"],
                    binding, "conversation", "local-c4-cancel",
                )
                started.set()
                await asyncio.Event().wait()

        async with graph["shell_ctx"].runtime_scope():
            shell_admission = graph["tasks"].open(graph["shell_ctx"])
        original_admit = shell_admission.admit

        async def cancel_shell_cleanup_before_run(key, callback):
            def start_cleanup(slot):
                nonlocal cleanup_task
                owned = callback(slot)
                if (
                    isinstance(key, tuple)
                    and key[:4] == ("shell-cleanup", "shared", "conversation", 0)
                ):
                    cleanup_task = owned
                    captured.append(owned)
                    before_cancel["permits"] = gate.permit_count
                    before_cancel["shell_calls"] = bool(
                        graph["shell_fiber"]._in_flight_calls
                    )
                    owned.on_done(cleanup_settled.set)
                    owned.cancel()
                return owned

            return await original_admit(key, start_cleanup)

        with monkeypatch.context() as patch:
            patch.setattr(shell_admission, "admit", cancel_shell_cleanup_before_run)
            async with graph["caller_ctx"].runtime_scope():
                admission = graph["tasks"].open(graph["caller_ctx"])
                caller_task = await admission.admit(
                    ("caller", "local-c4-cancel"),
                    lambda slot: slot.start(
                        caller_program, child_permit=permit.child
                    ),
                )
            caller_task.on_done(permit.release)
            caller_task.on_done(caller_settled.set)
            await started.wait()
            old_message = reader.get("local-c4-cancel")
            assert old_message is not None
            controls.append(
                "local-c4-abandon", Control("abandon", old_message.seq)
            )
            caller_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller_task.join()
            await caller_settled.wait()
            assert cleanup_task is not None
            with pytest.raises(asyncio.CancelledError):
                await cleanup_task.join()
            await cleanup_settled.wait()

        assert captured == [cleanup_task]
        assert before_cancel == {"permits": 2, "shell_calls": True}
        assert not cleanup_body_started.is_set()
        assert graph["shell_fiber"]._in_flight_calls == {}
        assert gate.permit_count == 0
        assert old_id is not None
    finally:
        if caller_task is not None and not caller_task.done:
            caller_task.cancel()
            await asyncio.gather(caller_task.join(), return_exceptions=True)
        if cleanup_task is not None and not cleanup_task.done:
            await asyncio.gather(cleanup_task.join(), return_exceptions=True)
        permit.release()
        await graph["root"].dispose()
        await graph["tasks"].close()
        await graph["processes"].close()
        graph["log"].close()
        graph["store"].close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["complete", "cleanup_failure", "cancel", "recover"])
async def test_reply_closes_real_shell_after_settlement_without_changing_output(tmp_path, monkeypatch, case):
    host, store, log, _artifacts, _source = environment(tmp_path, reply=True)
    tasks = Tasks()
    entered, release = asyncio.Event(), asyncio.Event()
    cleaning, clean_release = asyncio.Event(), asyncio.Event()
    execution_id = None
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root is not None
            bindings = Bindings(log, host._archive, snapshot.composition_root)
            root = snapshot.composition_root.context
            ctx = root.require(ServiceKey("standard-tools-probe"))
            catalog = root.require(TOOLS)
            binding = catalog.bind(
                root.require(ALL_TOOLS)().select("shell"),
                bindings,
                configuration=(
                    {"owner_key": "explicit-job"} if case == "complete" else {}
                ),
            )
            reader = log.reader("shared")
            log.writer("shared", author="user", source="conversation", body_types=(Input,), content={"text": check_text}).append(
                "input", Input((ContentPart("text", "work"),)))
            if case == "recover":
                execution_id = await start_shell_call(log, bindings, tasks, binding, "conversation", "old-call")

            # 此测试控制 ReAct 阶段，只验证 run_reply 的真实资源与消息边界。
            models = build_test_chat_models(_UnusedModelProvider())

            async def controlled_react(reader, output, *, tools, **kwargs):
                nonlocal execution_id
                if case != "recover":
                    assert tools.schemas
                    message = output.append(
                        "call",
                        Output(
                            (
                                ToolCall(
                                    binding,
                                    {
                                        "command": "sleep 30",
                                        "description": "reply lifecycle",
                                        "yield_time_ms": 250,
                                    },
                                ),
                            ),
                            "continue",
                        ),
                    )
                    result = await tools.execute(CallRef(message.message_id, 0))
                    execution_id = json.loads(result.parts[0].value)["execution_id"]
                backend = host._plugin_processes._manager
                if case == "cleanup_failure":
                    async def denied(execution):
                        raise PermissionError("controlled cleanup denial")
                    monkeypatch.setattr(backend, "_terminate_confirmed", denied)
                if case == "cancel":
                    original = backend.terminate_owner
                    async def delayed(owner):
                        cleaning.set()
                        await clean_release.wait()
                        return await original(owner)
                    monkeypatch.setattr(backend, "terminate_owner", delayed)
                    entered.set()
                    await release.wait()
                return output.append("done", Output((ContentPart("text", "completed reply"),), "complete"))

            async def allow(identity, arguments):
                return {"allowed": True}

            async def program(task):
                return await run_reply(
                    ctx,
                    task,
                    reader,
                    "conversation",
                    models=models,
                    content=root.require(CONTENT),
                    context=root.require(CONTEXT),
                    tools=catalog,
                    cleanup=root.require(TOOL_CLEANUP),
                    react=controlled_react,
                    materials=root.require(MATERIALS),
                    turn_projection=root.require(TURN_PROJECTION),
                    read_call=_unexpected_call_read,
                    authorize=allow,
                    tool_view=None,
                    fixed_bindings={"shell": binding},
                    max_output_tokens=100,
                    max_steps=4,
                )

            task = await root.require(TASKS).open(ctx).admit("reply", lambda slot: slot.start(program))
            if case == "cancel":
                await asyncio.wait_for(entered.wait(), 10)
                task.cancel()
                await asyncio.wait_for(cleaning.wait(), 10)
                assert not task.done
                clean_release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task.join()
                assert reader.get("done") is None
            else:
                result = await task.join()
                assert result == reader.get("done")
                assert result.body.finish == "complete"
            backend = host._plugin_processes._manager
            remaining = await backend.active_execution_ids()
            if case == "cleanup_failure":
                assert remaining == [execution_id]
                assert any(item.kind == "shell_cleanup_failed" for item in snapshot.composition_root.recent_incidents())
                with pytest.raises(RuntimeError, match="shell cleanup 未确认"):
                    await start_shell_call(log, bindings, tasks, binding, "conversation", "blocked")
                other = await start_shell_call(log, bindings, tasks, binding, "wake", "other")
                assert set(await backend.active_execution_ids()) == {execution_id, other}
                assert reader.get("done") == result
                monkeypatch.undo()
            else:
                assert remaining == []
    finally:
        release.set()
        clean_release.set()
        monkeypatch.undo()
        await tasks.close()
        await host.terminate_all()
        log.close()
        store.close()
