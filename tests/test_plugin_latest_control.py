"""真实普通工具可顺序启动、查询和撤销；Event 固定模型完成边界。"""
import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from agent.plugin_composition.bindings import BINDINGS
from agent.plugins.snapshot import lease_runtime_snapshot
from plugins.content.plugin import check_text
from plugins.models.state import _BoundChat
from plugins.tools.api import MessageReply
from plugins.tools.plugin import ALL_TOOLS, TOOLS
from session.message import CallRef, ContentPart, Input, Output, ToolCall, ToolResult
from tests.test_default_reply import application
from tests.test_plugin_install import _commit, _write_v3_plugin


@asynccontextmanager
async def update_tools(tmp_path):
    """工具从实际 Message 读取发起 session，不直接伪造 prepared arguments。"""
    async with application(tmp_path, replying=False, updates=True, provider_effect_data=True) as (log, host):
        source = tmp_path / "new-plugin"
        _write_v3_plugin(source, name="probe", module_source='''
api_version = 3
name = "probe"
version = "1.0.0"
async def apply(ctx):
    pass
''')
        _commit(source)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            tools = root.require(TOOLS)
            sequence = 0

            async def call(name, arguments, session_id="test:room"):
                nonlocal sequence
                sequence += 1
                identity = f"control:{sequence}"
                reader = log.reader(session_id)
                writer = log.writer(session_id, author="user", source="conversation", body_types=(Input,),
                                    content={"text": check_text})
                writer.append(identity + ":input", Input((ContentPart("text", "manage my plugin update"),)))
                binding = tools.bind(root.require(ALL_TOOLS)().select(name), root.require(BINDINGS))
                output = log.writer(session_id, author="assistant", source="conversation", body_types=(Output,),
                                    content={}, check_call=lambda call: None)
                output.append(identity, Output((ToolCall(binding, arguments),), "continue"))
                results = log.writer(session_id, author="tool", source="conversation", body_types=(ToolResult,),
                                     content={"text": check_text}, call_ref=CallRef(identity, 0))
                reply = MessageReply(identity + ":result", CallRef(identity, 0), reader, results, lambda: None)

                async def authorize(binding, arguments):
                    return {"approved": True}

                return await tools.execution(authorize).execute_call(reply)

            installed = await call("plugin_install", {"source": str(source), "marketplace": "lab",
                "validation_prompt": "Call write_evidence and report its result.", "validation_tools": ["write_evidence"]})
            assert installed.outcome == "success"
            identity = json.loads(installed.parts[0].value)["update_id"]
            yield host, snapshot, identity, call


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["finish", "revert", "fail"])
async def test_sequential_run_status_and_revert_keep_real_results(tmp_path, monkeypatch, action):
    entered, release, publication = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = _BoundChat.complete
    calls = 0

    async def complete(self, request):
        nonlocal calls
        calls += 1
        response = await original(self, request)
        if response.content == "finished":
            entered.set()
            await release.wait()
            if action == "fail":
                raise RuntimeError("controlled call failure")
        return response

    monkeypatch.setattr(_BoundChat, "complete", complete)
    async with asyncio.timeout(30), update_tools(tmp_path) as (host, snapshot, identity, call):
        before = (tmp_path / "workspace/runtime/plugin-stable.json").read_bytes()
        original_publish = host.start_update_publication

        def publish(update_id):
            # 这里只剩测试调用者租约；程序的 Task/Scope 和隔离 host 已真实释放。
            assert snapshot.lease_count == 1
            assert not host._validation_hosts
            original_publish(update_id)
            publication.set()

        monkeypatch.setattr(host, "start_update_publication", publish)
        try:
            started = await call("plugin_latest", {"update_id": identity, "action": "run"})
            assert started.outcome == "success"
            accepted = json.loads(started.parts[0].value)
            assert accepted["update_id"] == identity and accepted["handle"]
            await entered.wait()
            assert not release.is_set() and not publication.is_set()

            status = await call("plugin_latest", {"update_id": identity, "action": "status"})
            process = json.loads(status.parts[0].value)
            assert process["call"] == accepted
            assert process["task"]["active"] and not process["task"]["done"]
            assert process["messages"]
            observed = host.read_validation_messages(identity, "plugin-validation:" + identity)
            assert any(isinstance(message.body, ToolResult) for message in observed)
            for forbidden in ("status", "revert", "run"):
                denied = await call("plugin_latest", {"update_id": identity, "action": forbidden}, "test:other")
                assert denied.outcome != "success"
            assert host.read_update(identity).phase == "armed"

            active_host = next(iter(host._validation_hosts.values()))
            if action == "revert":
                reverted = await call("plugin_latest", {"update_id": identity, "action": "revert"})
                assert reverted.outcome == "success"
                assert host.read_update(identity).phase == "rolled_back"
                assert active_host.task.cancelled()
            else:
                release.set()
                if action == "finish":
                    await publication.wait()
                else:
                    with pytest.raises(RuntimeError, match="controlled call failure"):
                        await active_host.task
                    assert host.read_update(identity).error

            assert not host._validation_hosts
            finished = await call("plugin_latest", {"update_id": identity, "action": "status"})
            result = json.loads(finished.parts[0].value)
            assert result["call"] == accepted
            assert result["task"] is None
            assert result["messages"][:len(process["messages"])] == process["messages"]
            rows = host.read_validation_messages(identity, "plugin-validation:" + identity)
            if action == "finish":
                assert isinstance(rows[-1].body, Output) and rows[-1].body.finish == "complete"
                assert rows[-1].body.parts[0].value == "finished"
            else:
                assert not publication.is_set()
            with pytest.raises(RuntimeError, match="不得重跑"):
                await call("plugin_latest", {"update_id": identity, "action": "run"})
            assert calls == 2
            assert (tmp_path / "workspace/runtime/plugin-stable.json").read_bytes() == before
            if action != "revert":
                reverted = await call("plugin_latest", {"update_id": identity, "action": "revert"})
                assert reverted.outcome == "success"
        finally:
            release.set()
