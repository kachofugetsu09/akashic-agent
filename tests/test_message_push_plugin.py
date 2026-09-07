import asyncio
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import yoyo

from agent.migrations.context import bind_migration_context
from agent.migrations.session_attributes import migrate as migrate_attributes
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition import ServiceKey
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.delivery.records import DeliveryRecords
from plugins.message_push.tool import message_id
from plugins.tools.execution import ToolExecution
from plugins.tools.plugin import TOOLS, open_tool
from agent.plugin_composition.tasks import Tasks
from session.log import MessageLog, OwnerTransaction
from session.store import SessionStore
from session.artifact_store import ArtifactStore
from tests.test_delivery_bindings import sources


def storage(workspace):
    """关闭旧连接，迁移后分别重开 Message 与附件 owner。"""
    workspace.mkdir()
    store = SessionStore(workspace / "sessions.db")
    store.close()
    migrations = Path(__file__).parents[1] / "migrations/yoyo"
    with bind_migration_context(workspace=workspace, config_path=workspace / "config.toml"), pytest.MonkeyPatch.context() as patch:
        patch.setattr(yoyo, "step", lambda callback: callback)
        for name, callback in (
            ("20260905_01_message_log", "migrate_message_log"),
            ("20260905_02_owner_records", "migrate_owner_records"),
            ("20260905_05_message_embeddings", "migrate_message_embeddings"),
            ("20260905_06_message_artifacts", "migrate_message_artifacts"),
        ):
            spec = importlib.util.spec_from_file_location(name, migrations / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            getattr(module, callback)(None)
    migrate_attributes(workspace / "sessions.db", workspace / "backups/attributes")
    return ArtifactStore(workspace / "sessions.db"), MessageLog(workspace / "sessions.db")


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [False, True])
async def test_push_keeps_artifacts_and_original_sender_after_crash_without_resending(tmp_path, monkeypatch, confirmed):
    source = tmp_path / "plugins"
    sources(source)
    for name in ("tools", "message_push"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    sender = source / "test_sender/plugin.py"
    code = sender.read_text().replace("idempotent = True", "idempotent = False").replace("idempotent=True", "idempotent=False")
    if not confirmed:
        code = code.replace('async def query(self, key, address):', 'async def query(self, key, address):\n            return None')
    sender.write_text(code)
    workspace = tmp_path / "workspace"
    store, log = storage(workspace)
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
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            tools = snapshot.composition_root.context.require(TOOLS)
            binding = tools.bind("message_push", bindings)
            activity = snapshot.composition_root.context.require(ServiceKey("fixture.delivery"))().activity("test", "room")
        # 原 Tool 的捕获状态已保存 Sender binding，归档不再依赖当前注册或源码。
        shutil.rmtree(source)
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
        restored = manager([])
        recovered = Bindings(log, restored._archive, restored.open_binding)
        async def no_new_authorization(*_):
            pytest.fail("query original send must not reauthorize or reprepare")
        execution = ToolExecution(log.owner("plugin:tools"), tasks, lambda key: open_tool(recovered, key),
                                  no_new_authorization, task_key="effects")
        answer = await execution.execute("push-once", binding, parameters)
        assert answer.outcome == ("success" if confirmed else "unknown")
        assert (await execution.execute("push-once", binding, parameters)) == answer
        assert len(log.reader("test:room").snapshot()) == 1
        record = DeliveryRecords(log.owner("plugin:delivery"), "message_push").read(identity, "test")[1]
        assert record.phase == ("delivered" if confirmed else "unknown")
        sent = [json.loads(line) for line in next(workspace.rglob("sent.jsonl")).read_text().splitlines()]
        assert len(sent) == 1 and sent[0][1:] == ["room", identity, "original-A"]
        assert next(workspace.rglob("receiver-starts")).read_text().splitlines() == ["started"]
    finally:
        await tasks.close()
        if restored is not None:
            await restored.terminate_all()
        await host.terminate_all()
        log.close()
        store.close()
