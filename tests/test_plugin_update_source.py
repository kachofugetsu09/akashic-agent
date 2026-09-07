"""真实安装工具、候选回复和独立完成通知，不写父 Turn terminal。"""
import asyncio
from contextlib import AsyncExitStack, closing
import json

import pytest

from agent.plugin_composition.bindings import BINDINGS
from agent.plugins.snapshot import lease_runtime_snapshot
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from plugins.content.plugin import check_text
from plugins.conversation.plugin import check_origin
from plugins.tools.api import MessageReply
from plugins.tools.plugin import TOOLS
from session.log import MessageLog, OwnerTransaction
from session.message import CallRef, ContentPart, Input, Output, ToolCall, ToolResult
from tests.test_default_reply import application
from tests.test_plugin_install import _commit, _write_v3_plugin


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [True, False, None])
async def test_update_source_validates_and_reports_without_parent_terminal(tmp_path, passed, monkeypatch):
    delivered = asyncio.Event()
    original_save = OwnerTransaction.save
    def save(self, key, value, **kwargs):
        result = original_save(self, key, value, **kwargs)
        if key.startswith("delivery:") and value.get("phase") == "delivered":
            delivered.set()
        return result
    monkeypatch.setattr(OwnerTransaction, "save", save)
    async with AsyncExitStack() as cleanup, application(
        tmp_path, replying=False, updates=True, validation_passed=passed is not False,
        provider_effect_data=True, start=passed is not None,
    ) as (log, host):
        source = tmp_path / "new-plugin"
        _write_v3_plugin(source, name="probe", module_source='''
from agent.plugin_composition import ServiceKey
api_version = 3
name = "probe"
version = "1.0.0"
inject = ()
async def apply(ctx, config):
    await ctx.provide(ServiceKey("test.updated"), "candidate")
''')
        _commit(source)
        parameters = {"source": str(source), "marketplace": "lab",
                      "validation_prompt": "Call write_evidence and check its result.",
                      "validation_tools": ["write_evidence"]}
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            tools = root.require(TOOLS)
            binding = tools.bind("plugin_install", root.require(BINDINGS))
            reader = log.reader("test:room")
            inputs = log.writer("test:room", author="user", source="conversation", body_types=(Input,),
                content={"text": check_text, "channel.origin": check_origin})
            inputs.append("user-input", Input((ContentPart("text", "update the plugin"),
                ContentPart("channel.origin", {"channel": "test", "chat_id": "room", "sender": "user"}))))
            output = log.writer("test:room", author="assistant", source="conversation", body_types=(Output,),
                                content={}, check_call=lambda call: None)
            output.append("install-call", Output((ToolCall(binding, parameters),), "continue"))
            result_writer = log.writer("test:room", author="tool", source="conversation", body_types=(ToolResult,),
                                       content={"text": check_text}, call_ref=CallRef("install-call", 0))
            reply = MessageReply("install-result", CallRef("install-call", 0), reader, result_writer, lambda: None)
            async def authorize(binding, arguments):
                return {"approved": True}
            result = await tools.execution(authorize).execute_call(reply)
            assert result.outcome == "success"
            receipt = json.loads(result.parts[0].value)
            identity = receipt["update_id"]
            assert receipt["phase"] == "armed"
        async def restart():
            nonlocal log, host, reader
            await host.terminate_all()
            log.close()
            log = MessageLog(tmp_path / "sessions.db")
            cleanup.callback(log.close)
            host = PluginManager([tmp_path / "plugins"], event_bus=EventBus(), workspace=tmp_path / "workspace",
                                 installed_cache_root=tmp_path / "home/cache", message_log=log)
            cleanup.push_async_callback(host.terminate_all)
            reader = log.reader("test:room")
            await host.load_all()
            await host.start_runtime()
        if passed is None:
            (source / "plugin.py").unlink()
            await restart()
        # 只等待真实追加通知；原 conversation 保持 open，没有 terminal 来驱动发布。
        async with asyncio.timeout(20):
            async for _ in log.catalog().follow():
                rows = reader.snapshot()
                reports = [row for row in rows if row.message_id == identity + ":complete"]
                if reports:
                    break
        assert len(reports) == 1 and reports[0].source == "plugin_update"
        update = host.reload_journal.update(identity)
        assert update.phase == ("committed" if passed else "rolled_back")
        assert not any(isinstance(row.body, Output) and row.body.finish == "complete"
                       for row in rows if row.source == "conversation")
        databases = list((tmp_path / "workspace/runtime/plugin-update-validation").glob("*/workspace/sessions.db"))
        assert len(databases) == (0 if passed is None else 1)
        if databases:
            with closing(MessageLog(databases[0])) as validation:
                validation_rows = validation.reader("plugin-validation:" + identity).snapshot()
                assert tuple(type(row.body) for row in validation_rows) == (Input, Output, ToolResult, Output)
                assert validation.reader("plugin-validation:" + identity).attributes.learning == "excluded"
            assert (next(databases[0].parent.rglob("effect.txt"))).read_text() == "once\n"
        for generation in host.current_snapshot.generations.values():
            assert not (generation.data_dir / "effect.txt").exists()
        # 完成 Message 先提交，发送随后由实际 Delivery owner 结算。
        from plugins.delivery.records import DeliveryRecords
        delivery = DeliveryRecords(log.owner("plugin:delivery"), "plugin_update")
        await asyncio.wait_for(delivered.wait(), 10)
        await host.terminate_all()
        assert delivery.read(identity + ":complete", "test")[1].phase == "delivered"
        sent = [json.loads(line) for line in next((tmp_path / "workspace/plugin-data").rglob("sent.jsonl")).read_text().splitlines()]
        assert len(sent) == 1 and sent[0][1:3] == ["room", identity + ":complete"]
        recovered_report = asyncio.Event()
        append = OwnerTransaction.append
        def append_report(self, writer, message_id, body, **kwargs):
            message = append(self, writer, message_id, body, **kwargs)
            if message_id == identity + ":complete":
                recovered_report.set()
            return message
        monkeypatch.setattr(OwnerTransaction, "append", append_report)
        await restart()
        await asyncio.wait_for(recovered_report.wait(), 10)
        await host.terminate_all()
        assert len(reader.snapshot()) == len(rows)
        assert list((tmp_path / "workspace/runtime/plugin-update-validation").glob("*/workspace/sessions.db")) == databases
        sent_again = [json.loads(line) for line in next((tmp_path / "workspace/plugin-data").rglob("sent.jsonl")).read_text().splitlines()]
        assert sent_again == sent
