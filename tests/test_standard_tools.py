import asyncio
import base64
from collections.abc import Mapping
from contextlib import asynccontextmanager
from functools import partial
import json
from pathlib import Path
import shutil
from typing import cast

import httpx
from PIL import Image
import pytest

from agent.media import encode_image_data_uri
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.tasks import TASKS
from agent.plugin_composition.tasks import Tasks
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.standard_tools.shell import SHELL_OWNERS, shell_cleanup
from agent.plugin_contracts.content import CONTENT, check_text
from agent.plugin_contracts.context import MATERIALS
from agent.plugin_contracts.context import CONTEXT
from plugins.conversation.program import run_reply
from agent.plugin_contracts.turn_projection import TURN_PROJECTION
from plugins.tools.api import MessageReply
from session.message import CallRef, ContentPart, Input, Output, ToolCall, ToolResult
from plugins.standard_web.web import WebTool
from plugins.tools.execution import ToolExecution
from agent.plugin_contracts.tools import ALL_TOOLS, TOOLS
from plugins.tools.plugin import open_tool
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


def environment(tmp_path, *, reply=False):
    source = tmp_path / "plugins"
    for name in (
        "tools",
        "content",
        "context",
        "standard_tools",
        *(("turn_projection",) if reply else ()),
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
inject = (ServiceKey("core.bindings"),)
async def apply(ctx, config):
    await ctx.provide(ServiceKey("standard-tools-probe"), ctx)
''')
    workspace = tmp_path / "workspace"
    store, log = storage(workspace)
    context_config = workspace / "plugin-data/context-builtin/config.local.toml"
    context_config.parent.mkdir(parents=True, exist_ok=True)
    context_config.write_text('prompt_sources = {skills = "standard_tools"}\n')
    artifacts = ChannelAttachmentArtifactStore(
        workspace=workspace, metadata_store=store
    )
    host = PluginManager(
        [source],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
        message_log=log,
        channel_attachment_store=artifacts,
    )
    return host, store, log, artifacts, source


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
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
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
        execution = ToolExecution(log.owner("plugin:tools"), tasks, partial(open_tool, bindings), authorize, task_key="effects")
        missing = await execution.execute("missing", read, {"path": str(tmp_path / "missing")})
        assert missing.outcome == "error" and "不存在" in cast(str, missing.parts[0].value)
        assert await execution.execute("missing", read, {"path": str(tmp_path / "missing")}) == missing
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
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
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
        execution = ToolExecution(log.owner("plugin:tools"), tasks, partial(open_tool, bindings), authorize, task_key="effects")
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
    result = await tool.invoke("request", await tool.prepare({"query": "test"}))
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

    execution = ToolExecution(log.owner("plugin:tools"), tasks, partial(open_tool, bindings), allow, task_key="effects")
    result = await execution.execute_call(reply)
    assert result.outcome == "success"
    return cast(str, json.loads(cast(str, result.parts[0].value))["execution_id"])


@pytest.mark.asyncio
async def test_shell_cleanup_uses_original_binding_and_keeps_other_source_running(tmp_path):
    host, store, log, _artifacts, source = environment(tmp_path)
    tasks = Tasks()
    probe = ServiceKey("standard-tools-probe")
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            tool = root.require(TOOLS).bind(
                root.require(ALL_TOOLS)().select("shell"), bindings
            )
            probe_binding = bindings.bind(probe, {})
        first = await start_shell_call(log, bindings, tasks, tool, "conversation", "first")
        second = await start_shell_call(log, bindings, tasks, tool, "wake", "second")
        shutil.rmtree(source)
        # 当前 Root 只有 probe；清理必须从实际 ToolCall 归档找回 Shell owner。
        async with bindings.open(probe_binding, probe) as (ctx, _):
            assert ctx.get(SHELL_OWNERS) is None
            async with shell_cleanup(ctx, log.reader("shared"), "conversation", 0):
                pass
            backend = host._plugin_processes._manager
            assert first not in await backend.active_execution_ids()
            assert second in await backend.active_execution_ids()
            async with shell_cleanup(ctx, log.reader("shared"), "wake", 0):
                pass
            assert await backend.active_execution_ids() == []
    finally:
        await tasks.close()
        await host.terminate_all()
        log.close()
        store.close()


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
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
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
