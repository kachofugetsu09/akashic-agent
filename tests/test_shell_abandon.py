import asyncio
import json

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.tasks import TASKS, Tasks
from agent.plugins.snapshot import lease_runtime_snapshot
from agent.plugins.manager import PluginManager
from agent.restart import RestartGate
from bus.event_bus import EventBus
from plugins.content.plugin import CONTENT, check_text
from plugins.context.materials import MATERIALS
from plugins.context.plugin import CONTEXT
from plugins.conversation.program import run_reply
from plugins.tools.plugin import ALL_TOOLS, TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.message import CallRef, ContentPart, Control, Input, Output, ToolCall, ToolResult
from tests.model_plugin_fakes import build_test_chat_models
from tests.test_standard_tools import environment, start_shell_call, _UnusedModelProvider, _unexpected_call_read


@pytest.mark.asyncio
async def test_real_tools_watcher_restarts_and_settles_offline_abandon_once(tmp_path):
    host, store, log, artifacts, source = environment(tmp_path)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            binding = ctx.require(TOOLS).bind(
                ctx.require(ALL_TOOLS)().select("shell"),
                Bindings(log, host._archive, host.open_binding),
            )
        inputs = log.writer(
            "s", author="user", source="conversation", body_types=(Input,), content={}
        )
        outputs = log.writer(
            "s",
            author="agent",
            source="conversation",
            body_types=(Output,),
            content={},
            check_call=lambda call: None,
        )
        controls = log.writer(
            "s", author="user", source="conversation", body_types=(Control,), content={}
        )

        def append(index):
            inputs.append(f"input-{index}", Input(()))
            call = outputs.append(f"call-{index}", Output((ToolCall(binding, {
                "command": "printf 'must not run'", "description": "offline abandon",
            }),), "continue"))
            controls.append(f"abandon-{index}", Control("abandon", call.seq))

        async def result(index):
            async for message in log.reader("s").follow():
                if message.message_id == f"tool-result:call-{index}:0":
                    return message
            raise AssertionError("工具结果订阅提前结束")

        append(1)
        await host.start_runtime()
        first = await asyncio.wait_for(result(1), 2)
        assert first.body.outcome == "denied"
        await host.terminate_all()
        append(2)
        host = PluginManager([source], event_bus=EventBus(), workspace=tmp_path / "workspace",
                             installed_cache_root=tmp_path / "cache", message_log=log,
                             channel_attachment_store=artifacts)
        await host.load_all()
        await host.start_runtime()
        second = await asyncio.wait_for(result(2), 2)
        assert second.body.outcome == "denied"
        assert log.reader("s").get(first.message_id) == first
        assert [m for m in log.reader("s").snapshot() if isinstance(m.body, ToolResult)] == [first, second]
        assert host._plugin_processes._manager is None
    finally:
        await host.terminate_all()
        log.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_first", [False, True])
async def test_abandon_keeps_old_cleanup_permit_and_does_not_kill_new_process(tmp_path, monkeypatch, pause_first):
    host, store, log, _artifacts, _source = environment(tmp_path, reply=True)
    tasks = Tasks()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    old_process = None
    gate = RestartGate(boot_id="test", supervised=True, commit=lambda _: None)
    try:
        await host.load_all()
        await host.start_runtime()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            ctx = root.require(ServiceKey("standard-tools-probe"))
            catalog = root.require(TOOLS)
            binding = catalog.bind(root.require(ALL_TOOLS)().select("shell"), bindings)
            reader = log.reader("shared")
            inputs = log.writer("shared", author="user", source="conversation", body_types=(Input,), content={"text": check_text})
            inputs.append("input", Input((ContentPart("text", "old work"),)))
            controls = log.writer("shared", author="user", source="conversation", body_types=(Control,), content={})
            models = build_test_chat_models(_UnusedModelProvider())

            async def controlled_react(reader, output, *, tools, **kwargs):
                nonlocal old_process
                assert tools.schemas
                message = output.append("old-call", Output((ToolCall(tools.bind("shell"), {
                    "command": "sleep 30", "description": "abandon lifecycle", "yield_time_ms": 250,
                }),), "continue"))
                result = await tools.execute(CallRef(message.message_id, 0))
                old_process = json.loads(result.parts[0].value)["execution_id"]
                backend = host._plugin_processes._manager
                original = backend.terminate_owner
                async def delayed(owner):
                    cleaning.set()
                    await release.wait()
                    return await original(owner)
                monkeypatch.setattr(backend, "terminate_owner", delayed)
                entered.set()
                await asyncio.Event().wait()
                raise AssertionError("受控回复应由测试取消")

            async def allow(identity, arguments):
                return {"allowed": True}

            async def program(task):
                return await run_reply(
                    ctx, task, reader, "conversation", models=models, content=root.require(CONTENT),
                    context=root.require(CONTEXT), tools=catalog, react=controlled_react,
                    materials=root.require(MATERIALS), turn_projection=root.require(TURN_PROJECTION),
                    read_call=_unexpected_call_read, authorize=allow, tool_names=("shell",),
                    fixed_bindings={"shell": binding}, max_output_tokens=100, max_steps=4,
                )

            permit = gate.acquire()
            owner = root.require(TASKS).open(ctx)
            task = await owner.admit("reply", lambda slot: slot.start(program, child_permit=permit.child))
            task.on_done(permit.release)
            await asyncio.wait_for(entered.wait(), 10)
            stop = controls.append("first-stop", Control("pause" if pause_first else "abandon", reader.head()))
            task.cancel()
            await asyncio.wait_for(cleaning.wait(), 10)
            end = stop.seq if pause_first else stop.body.through_seq
            cleanup = await owner.admit(("shell-cleanup", "shared", "conversation", 0, end), lambda slot: slot.current)
            assert cleanup is not None
            if pause_first:
                assert not task.done
                controls.append("abandon", Control("abandon", reader.head()))
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task.join(), 1)
            assert gate.permit_count == 1 and not release.is_set()
            inputs.append("new-input", Input((ContentPart("text", "new work"),)))
            new_process = await start_shell_call(log, bindings, tasks, binding, "conversation", "new-call")
            backend = host._plugin_processes._manager
            assert old_process in await backend.active_execution_ids()
            release.set()
            await asyncio.wait_for(cleanup.join(), 10)
            assert gate.permit_count == 0
            assert old_process not in await backend.active_execution_ids()
            assert new_process in await backend.active_execution_ids()
            assert reader.get("tool-result:old-call:0").body.outcome == "success"
    finally:
        release.set()
        await tasks.close()
        await host.terminate_all()
        log.close()
        store.close()
