import asyncio
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.model_runtime.session_selection import read_session_model_selection
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition.models import DisableConnection, MODEL_SETTINGS, ModelUnavailableError
from agent.plugins.snapshot import lease_runtime_snapshot
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from plugins.conversation.plugin import CONVERSATION
from session.log import MessageLog, SessionAttributes, WriterExpired
from session.message import ContentPart, Control, Input, Output
from tests.test_default_reply import application
from tests.test_message_services import write_plugins


def model_provider(sources: Path) -> None:
    """沿用真实默认回复组合，仅替换网络 driver；选择校验使用真实 ModelsState。"""
    driver = sources / "fixture_driver"
    driver.mkdir()
    (driver / "plugin.py").write_text('''
from agent.plugin_composition import MODEL_DRIVERS
from plugins.openai_compatible.driver import definition
api_version = 3
name = "fixture_driver"
version = "1.0.0"
inject = (MODEL_DRIVERS,)
async def apply(ctx, config):
    await ctx.require(MODEL_DRIVERS).register(ctx, definition())
''')
    path = sources / "test_provider/plugin.py"
    text = path.read_text()
    text = text.replace("    class Driver:", '''    from agent.plugin_composition import MODEL_CATALOG, MODEL_DRIVERS, MODEL_SETTINGS, SNAPSHOT_SEALING
    from agent.plugin_composition.models import AddConnection, AddModel, ModelKind, ChatModelSelection
    from plugins.models.state import ModelsState
    if store.read_snapshot().revision == 0:
        store.add_connection(AddConnection(0, "connection", "test", "openai-compatible", "https://example.test/v1", "fixture", {"api_key": "fixture"}))
        store.add_model(AddModel(1, "saved", "connection", ModelKind.CHAT, "fixture", ModelCapabilities(supported_reasoning_efforts=("low", "high")), CapabilitySources()))
    model_state = ModelsState(store, root_instance_token=ctx.root_instance_token)
    await ctx.provide(MODEL_DRIVERS, model_state.drivers)
    await ctx.provide(MODEL_CATALOG, model_state.catalog)
    await ctx.provide(MODEL_SETTINGS, model_state.settings)
    await ctx.on(SNAPSHOT_SEALING, model_state.seal)
    selected = []
    await ctx.provide(ServiceKey("fixture.selected"), selected)
    class Driver:''')
    text = text.replace("            yield SimpleNamespace(chat=lambda role: model)", '''            model_state.validate_chat_selection(ChatModelSelection(model_id, reasoning_effort))
            selected.append((model_id, reasoning_effort))
            yield SimpleNamespace(chat=lambda role: model)''')
    path.write_text(text)


def inbound(metadata=None):
    return ChannelInboundMessage("test", "user", "room", "question", datetime(2026, 9, 5, tzinfo=UTC),
                                 {} if metadata is None else metadata)


def saved_metadata(path: Path, raw: str | None) -> None:
    """建立真实持久 Session 后关库，后续应用从同一数据库重开。"""
    with closing(MessageLog(path)) as log:
        log.ensure_session("test:room", SessionAttributes())
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE sessions SET metadata=? WHERE key='test:room'", (raw,))


async def terminal(log, count):
    async def wait():
        async for _ in log.catalog().follow():
            messages = log.reader("test:room").snapshot()
            if sum(isinstance(m.body, Output) and m.body.finish == "complete" for m in messages) >= count:
                return
    await asyncio.wait_for(wait(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    '{"model_runtime_override":"saved","other":{"keep":true}}',
    '{"model_selection":{"schema_version":1,"model_ref":"saved","reasoning_effort":"high"}}',
])
async def test_saved_choice_survives_reopen_and_drives_real_reply(tmp_path, raw):
    saved_metadata(tmp_path / "sessions.db", raw)
    async with application(tmp_path, replying=True, extra_sources=model_provider) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            context = snapshot.composition_root.context
            await context.require(CHANNEL_INPUT)("test:room", "u1", inbound())
            await terminal(log, 1)
            effort = "high" if "model_selection" in raw else None
            assert context.require(ServiceKey("fixture.selected")) == [("saved", effort)]
        with closing(sqlite3.connect(tmp_path / "sessions.db")) as connection:
            assert connection.execute("SELECT metadata FROM sessions").fetchone()[0] == raw
        assert not any(part.kind == "model.selection" for m in log.reader("test:room").snapshot()
                       if isinstance(m.body, Input) for part in m.body.parts)


@pytest.mark.asyncio
async def test_explicit_switch_clear_and_replay_share_one_session_fact(tmp_path):
    raw = '{"model_runtime_override":"saved","other":{"keep":true}}'
    saved_metadata(tmp_path / "sessions.db", raw)
    async with application(tmp_path, replying=True, extra_sources=model_provider) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            context = snapshot.composition_root.context
            accept = context.require(CHANNEL_INPUT)
            first = inbound({"model_runtime_id": "saved", "model_reasoning_effort": "low"})
            message = await accept("test:room", "u1", first)
            await terminal(log, 1)
            assert read_session_model_selection(log.reader("test:room").metadata()).reasoning_effort == "low"
            assert "model_runtime_override" not in log.reader("test:room").metadata()
            await accept("test:room", "u2", inbound({"model_runtime_id": ""}))
            await terminal(log, 2)
            before = log.reader("test:room").snapshot()
            assert await accept("test:room", "u1", first) == message
            assert log.reader("test:room").snapshot() == before
            assert log.reader("test:room").metadata() == {"other": {"keep": True}}
            await accept("test:room", "u3", inbound())
            await terminal(log, 3)
            assert context.require(ServiceKey("fixture.selected")) == [("saved", "low"), (None, None), (None, None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [
    {"model_runtime_id": "missing"},
    {"model_runtime_id": "saved", "model_reasoning_effort": "unsupported"},
    {"model_runtime_id": "", "model_reasoning_effort": "high"},
])
async def test_invalid_selection_leaves_no_input_or_session(tmp_path, metadata):
    async with application(tmp_path, replying=False, extra_sources=model_provider) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            with pytest.raises((ModelUnavailableError, ValueError)):
                await snapshot.composition_root.context.require(CHANNEL_INPUT)("test:room", "u1", inbound(metadata))
        assert log.catalog().snapshot_heads() == {}


@pytest.mark.asyncio
async def test_direct_conversation_uses_same_validation_and_atomic_metadata_write(tmp_path):
    async with application(tmp_path, replying=False, extra_sources=model_provider) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            conversation = snapshot.composition_root.context.require(CONVERSATION)("test:room")
            part = ContentPart("model.selection", {"model_id": "saved", "reasoning_effort": "high"})
            with pytest.raises(ValueError, match="一次"):
                await conversation.accept("double", Input((part, part)))
            assert log.catalog().snapshot_heads() == {}
            await conversation.accept("u1", Input((part,)))
            assert read_session_model_selection(log.reader("test:room").metadata()).model_ref == "saved"
            # SQLite 在真正 metadata UPDATE 处失败，整批正文与 Session 字段必须不变。
            with closing(sqlite3.connect(tmp_path / "sessions.db")) as connection, connection:
                connection.execute("CREATE TRIGGER reject_selection BEFORE UPDATE OF metadata ON sessions BEGIN SELECT RAISE(ABORT, 'selection fault'); END")
                before = tuple(connection.iterdump())
            clear = ContentPart("model.selection", {"model_id": None, "reasoning_effort": None})
            with pytest.raises(sqlite3.IntegrityError, match="selection fault"):
                await conversation.accept("clear", Input((clear,)))
            with closing(sqlite3.connect(tmp_path / "sessions.db")) as connection:
                assert tuple(connection.iterdump()) == before


@pytest.mark.asyncio
async def test_replay_survives_unavailable_catalog_and_does_not_restore_selection(tmp_path):
    async with application(tmp_path, replying=False, extra_sources=model_provider) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            context = snapshot.composition_root.context
            accept = context.require(CHANNEL_INPUT)
            first = inbound({"model_runtime_id": "saved"})
            message = await accept("test:room", "u1", first)
            await accept("test:room", "u2", inbound({"model_runtime_id": ""}))
            await context.require(MODEL_SETTINGS).apply(DisableConnection(2, "connection"))
            assert await accept("test:room", "u1", first) == message
            assert log.reader("test:room").metadata() == {}
            with pytest.raises(ModelUnavailableError):
                await accept("test:room", "u3", first)
            assert [m.message_id for m in log.reader("test:room").snapshot()] == ["u1", "u2"]


@pytest.mark.asyncio
async def test_saved_disabled_model_fails_reply_without_falling_back(tmp_path):
    saved_metadata(tmp_path / "sessions.db", '{"model_runtime_override":"saved"}')
    async with application(tmp_path, replying=True, extra_sources=model_provider) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            context = snapshot.composition_root.context
            await context.require(MODEL_SETTINGS).apply(DisableConnection(2, "connection"))
            await context.require(CHANNEL_INPUT)("test:room", "u1", inbound())
            async def failed():
                async for _ in log.catalog().follow():
                    controls = [m.body for m in log.reader("test:room").snapshot() if isinstance(m.body, Control)]
                    if controls:
                        return controls[-1]
            control = await asyncio.wait_for(failed(), 5)
            assert control.action == "failure"
            assert context.require(ServiceKey("fixture.selected")) == []
            assert read_session_model_selection(log.reader("test:room").metadata()).model_ref == "saved"


@pytest.mark.parametrize("raw", ['[]', '{"x":1,"x":2}', '{"x":NaN}', '{broken'])
def test_metadata_damage_is_explicit_and_unknown_read_never_creates(tmp_path, raw):
    saved_metadata(tmp_path / "sessions.db", raw)
    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        before = log.catalog().snapshot_heads()
        assert log.reader("unknown").metadata() is None
        assert log.catalog().snapshot_heads() == before
        with pytest.raises(ValueError, match="test:room.*metadata"):
            log.reader("test:room").metadata()


@pytest.mark.asyncio
async def test_clear_without_saved_selection_preserves_sql_null(tmp_path):
    saved_metadata(tmp_path / "sessions.db", None)
    async with application(tmp_path, replying=False, extra_sources=model_provider) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            await snapshot.composition_root.context.require(CHANNEL_INPUT)(
                "test:room", "u1", inbound({"model_runtime_id": ""}),
            )
        with closing(sqlite3.connect(tmp_path / "sessions.db")) as connection:
            assert connection.execute("SELECT metadata FROM sessions").fetchone()[0] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate", [False, True])
async def test_metadata_grants_belong_to_one_registered_plugin(tmp_path, duplicate):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    for name in (("one", "two") if duplicate else ("one",)):
        path = sources / name / "plugin.py"
        path.write_text(path.read_text() + '''
    def update(body):
        return {"preference": {"choice": 1}}
    registration = await ctx.require(MESSAGE_WRITERS).register_metadata(ctx, keys=frozenset({"preference"}), update=update)
    await ctx.provide(ServiceKey("update." + name), update)
    await ctx.provide(ServiceKey("registration." + name), registration)
''')
    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                             installed_cache_root=tmp_path / "home", message_log=log)
        try:
            if duplicate:
                with pytest.raises(RuntimeError, match="metadata 已有 owner"):
                    await host.load_all()
                assert log.catalog().snapshot_heads() == {}
                return
            await host.load_all()
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                context = snapshot.composition_root.context
                one = context.require(ServiceKey("probe.one"))
                two = context.require(ServiceKey("probe.two"))
                update = context.require(ServiceKey("update.one"))
                writers = context.require(MESSAGE_WRITERS)
                for owner, callback in ((two, update), (one, lambda body: {"preference": 2})):
                    with pytest.raises(PermissionError, match="未登记"):
                        writers.bind(owner, author="user", source="test", body_types=(Input,), content={},
                                     update_metadata=callback)
                writer = writers.bind(one, author="user", source="test", body_types=(Input,), content={},
                                      update_metadata=update)("s")
                message = writer.append("u1", Input(()))
                assert log.reader("s").metadata() == {"preference": {"choice": 1}}
                await context.require(ServiceKey[Effect]("registration.one")).aclose()
                await writers.register_metadata(two, keys=frozenset({"preference"}), update=update)
                with pytest.raises(WriterExpired, match="授权已释放"):
                    writer.append("u2", Input(()))
                assert writer.append("u1", Input(())) == message
                assert log.reader("s").snapshot() == (message,)
        finally:
            await host.terminate_all()
