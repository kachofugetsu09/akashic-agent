import asyncio
import json
from pathlib import Path
import shutil

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition import ServiceKey
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.delivery.records import DeliveryRecords
from plugins.message_push.tool import message_id
from plugins.turn_projection.plugin import TurnProjection
from plugins.sources.session import SourceSession
from plugins.content.plugin import check_text
from plugins.tools.execution import ToolExecution
from plugins.tools.plugin import ALL_TOOLS, TOOLS, open_tool
from agent.plugin_composition.tasks import Tasks
from session.log import MessageLog, OwnerTransaction, SessionAttributes
from session.artifact_store import ArtifactStore
from session.message import ContentPart, Control, Input, Output
from tests.test_delivery_bindings import sources


def storage(workspace):
    """用当前 Message owner 初始化消息与附件测试库。"""
    workspace.mkdir()
    log = MessageLog(workspace / "sessions.db")
    store = ArtifactStore(workspace / "sessions.db")
    return store, log


@pytest.mark.asyncio
async def test_push_completes_while_target_turn_is_active_and_appends_one_output(tmp_path):
    """A caller push finishes while a real source Task holds its original prompt."""
    source = tmp_path / "plugins"
    sources(source)
    for name in ("content", "tools", "message_push"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    workspace = tmp_path / "workspace"
    store, log = storage(workspace)
    initialize_plugin_workspace(workspace)
    artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=store)
    bus = EventBus()
    host = PluginManager([source], event_bus=bus, workspace=workspace,
                         installed_cache_root=tmp_path / "home", message_log=log,
                         channel_attachment_store=artifacts)
    target_tasks, caller_tasks = Tasks(), Tasks()
    release = asyncio.Event()
    target_task = None
    try:
        await host.load_all()
        await host.start_runtime()
        root = host.live_root
        sender = host.generation("test_sender")
        assert root is not None and sender is not None and sender.fiber is not None
        bindings = root.service_value(BINDINGS)
        tools = root.service_value(TOOLS)
        all_tools = root.service_value(ALL_TOOLS)
        assert bindings is not None and tools is not None and all_tools is not None
        binding = await tools.bind_scoped(all_tools().select("message_push"), bindings)

        # 1. Hold a real source Task after it reads its fixed Input.
        log.ensure_session("test:room", SessionAttributes())
        reader = log.reader("test:room")
        target = SourceSession(reader=reader,
            inputs=log.writer("test:room", author="user", source="conversation", body_types=(Input,),
                              content={"text": check_text}),
            controls=log.writer("test:room", author="app", source="conversation", body_types=(Control,), content={}),
            tasks=target_tasks)
        accepted = await target.accept("target-input", Input((ContentPart("text", "original target prompt"),)))
        prompt_read = asyncio.Event()
        seen_prompt = []

        async def target_program(_task, target_reader, _source):
            seen_prompt.append(target_reader.snapshot())
            prompt_read.set()
            await release.wait()
            log.writer("test:room", author="assistant", source="conversation", body_types=(Output,),
                       content={"text": check_text}).append(
                           "target-terminal", Output((ContentPart("text", "target finished"),), "complete"))

        target_task = await target.start(target_program)
        assert target_task is not None
        await asyncio.wait_for(prompt_read.wait(), 5)
        assert seen_prompt == [(accepted,)]
        projection = TurnProjection()
        active = projection.project(reader.snapshot(), "conversation")[-1]
        assert active.status == "open" and target_task.active

        async def authorize(_binding, _arguments):
            return {"approved": True}

        execution = ToolExecution(log.owner("plugin:tools"), caller_tasks,
                                  lambda key: open_tool(bindings, key), authorize, task_key="effects")
        # 2. The caller's tool and Delivery finish before the target is released.
        async with sender.fiber.context.runtime_scope():
            activity = sender.fiber.context.require(ServiceKey("fixture.delivery"))().activity("test", "room")
        with activity:
            answer = await asyncio.wait_for(execution.execute("while-target-open", binding, {
                "target_channel": "test", "target_chat_id": "room", "message": "independent push",
            }), 5)
        assert answer.outcome == "success"
        caller_receipt = log.owner("plugin:tools").read("program:while-target-open")
        assert caller_receipt is not None and caller_receipt.value["phase"] == "done"
        identity = message_id("program:while-target-open")
        receipt = DeliveryRecords(log.owner("plugin:delivery"), "message_push").read(identity, "test")[1]
        assert receipt.phase == "delivered"
        sent = [json.loads(line) for line in next(workspace.rglob("sent.jsonl")).read_text().splitlines()]
        assert len(sent) == 1 and sent[0][1:3] == ["room", identity]
        during = reader.snapshot()
        assert len(during) == 2 and during[0] == accepted
        assert during[1].message_id == identity and during[1].source == "message_push"
        assert during[1].body.parts[0].value == "independent push"
        assert projection.project(during, "conversation")[-1] == active
        assert seen_prompt == [(accepted,)] and target_task.active and not release.is_set()
        # 3. Release and join the exact target owner, then read its terminal log.
        release.set()
        await asyncio.wait_for(target_task.join(), 5)
        settled = reader.snapshot()
        assert [row.source for row in settled] == ["conversation", "message_push", "conversation"]
        assert settled[-1].message_id == "target-terminal"
        assert projection.project(settled, "conversation")[-1].status == "complete"
    finally:
        release.set()
        await caller_tasks.close()
        await target_tasks.close()
        await host.terminate_all()
        log.close()
        store.close()
        await bus.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [False, True])
async def test_push_keeps_artifacts_and_original_sender_after_crash_without_resending(tmp_path, monkeypatch, confirmed):
    source = tmp_path / "plugins"
    sources(source)
    for name in ("content", "tools", "message_push"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    sender = source / "test_sender/plugin.py"
    code = sender.read_text().replace("idempotent = True", "idempotent = False").replace("idempotent=True", "idempotent=False")
    if not confirmed:
        code = code.replace('async def query(self, key, address):', 'async def query(self, key, address):\n            return None')
    sender.write_text(code)
    workspace = tmp_path / "workspace"
    store, log = storage(workspace)
    initialize_plugin_workspace(workspace)
    artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=store)
    def manager(paths):
        return PluginManager(paths, event_bus=EventBus(), workspace=workspace,
            installed_cache_root=tmp_path / "home", message_log=log, channel_attachment_store=artifacts)
    host = manager([source])
    tasks = Tasks()
    restored = None
    file = tmp_path / "evidence.txt"
    file.write_bytes(b"original artifact bytes")
    parameters = {"target_channel": "test", "target_chat_id": "room", "message": "literal [MEDIA:x]", "file": str(file)}
    permission = []
    async def authorize(binding, final):
        permission.append(final)
        file.unlink()
        return {"approved": True}
    try:
        await host.load_all()
        await host.start_runtime()
        root = host.live_root
        sender = host.generation("test_sender")
        assert root is not None and sender is not None and sender.fiber is not None
        bindings = root.service_value(BINDINGS)
        tools = root.service_value(TOOLS)
        all_tools = root.service_value(ALL_TOOLS)
        assert bindings is not None and tools is not None and all_tools is not None
        binding = await tools.bind_scoped(all_tools().select("message_push"), bindings)
        async with sender.fiber.context.runtime_scope():
            activity = sender.fiber.context.require(ServiceKey("fixture.delivery"))().activity("test", "room")
        execution = ToolExecution(log.owner("plugin:tools"), tasks, lambda key: open_tool(bindings, key),
                                  authorize, task_key="effects")
        invalid = await execution.execute("bad-route", binding, {**parameters, "target_channel": "missing"})
        assert invalid.outcome == "error" and permission == []
        assert store.list_attachments() == () and log.reader("test:room").snapshot() == ()
        save = OwnerTransaction.save
        def crash(self, key, value, **kwargs):
            # 模拟发送回执和工具结算均未落盘的进程中断切点。
            if (key.startswith("delivery:") and value.get("phase") == "delivered") or value.get("phase") == "done":
                raise OSError("receipt disk unavailable")
            return save(self, key, value, **kwargs)
        with activity, monkeypatch.context() as patch:
            patch.setattr(OwnerTransaction, "save", crash)
            with pytest.raises(OSError, match="receipt disk unavailable"):
                async with asyncio.timeout(10):
                    await execution.execute("push-once", binding, parameters)
        identity = message_id("program:push-once")
        messages = log.reader("test:room").snapshot()
        assert len(messages) == 1 and messages[0].message_id == identity
        assert messages[0].source == "message_push"
        assert messages[0].body.parts[0].value == "literal [MEDIA:x]"
        refs = log.reader("test:room").attachments(identity)
        assert len(refs) == 1
        read = await artifacts.acquire(refs[0])
        assert await read.read_bytes(max_bytes=100) == b"original artifact bytes"
        await read.aclose()
        assert len(permission) == 1 and not file.exists()
        await host.terminate_all()
        log.close()
        store.close()
        store = ArtifactStore(workspace / "sessions.db")
        artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=store)
        log = MessageLog(workspace / "sessions.db")
        restored = manager([source])
        await restored.load_all()
        await restored.start_runtime()
        root = restored.live_root
        assert root is not None
        recovered = root.service_value(BINDINGS)
        assert recovered is not None
        async def no_new_authorization(*_):
            pytest.fail("query original send must not reauthorize or reprepare")
        execution = ToolExecution(log.owner("plugin:tools"), tasks, lambda key: open_tool(recovered, key),
                                  no_new_authorization, task_key="effects")
        answer = await execution.execute("push-once", binding, parameters)
        assert answer.outcome == ("success" if confirmed else "error")
        repeated = await execution.execute("push-once", binding, parameters)
        assert (repeated.outcome, repeated.parts) == (answer.outcome, answer.parts)
        assert len(log.reader("test:room").snapshot()) == 1
        record = DeliveryRecords(log.owner("plugin:delivery"), "message_push").read(identity, "test")[1]
        assert record.phase == ("delivered" if confirmed else "failed")
        sent = [json.loads(line) for line in next(workspace.rglob("sent.jsonl")).read_text().splitlines()]
        assert len(sent) == 1 and sent[0][1:] == ["room", identity, "original-A"]
        assert next(workspace.rglob("receiver-starts")).read_text().splitlines() == ["started", "started"]
    finally:
        await tasks.close()
        if restored is not None:
            await restored.terminate_all()
        await host.terminate_all()
        log.close()
        store.close()
