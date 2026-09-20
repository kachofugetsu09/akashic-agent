"""真实验证宿主的输入与停止边界；不启动外部 listener。"""
import asyncio
from contextlib import closing, nullcontext
from datetime import UTC, datetime
import sqlite3

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.admission import SOURCE_ADMISSION
from agent.plugin_composition.channel_io import INPUT_CUSTODY, CHANNEL_IDENTITY
from agent.plugin_composition.channels import ChannelInboundMessage, RawInbound
from bus.queue import MessageBus
from session.inbound_store import InboundHandoffStore
from tests.test_plugin_business_validation import MODULE, prepare
from tests.test_plugin_install import _commit


def inbound(number):
    return RawInbound(f"message-{number}", ChannelInboundMessage(
        "probe", "user", "chat", "validation input", datetime.now(UTC),
        {"durable_inbound": True, "durable_handoff_id": f"handoff-{number}",
         "provider_message_id": f"message-{number}", "session_key_override": "validation",
         "require_existing_session": False},
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_validation_custody_owns_inputs_and_retains_failed_close(tmp_path, monkeypatch, fail_close):
    """真实接纳、普通 program 与停止均不改正式数据库；失败保留原宿主。"""
    source, workspace, _, log, host = prepare(tmp_path)
    validation = None
    try:
        await host.load_all()
        (source / "plugin.py").write_text(MODULE + "\nmarker = 'channel-validation'\n")
        _commit(source)
        result, _ = await host.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )
        # 完整 Manager 只能由正式宿主创建；隔离调用必须使用实际资源 owner。
        def reject_child_manager(*args, **kwargs):
            raise AssertionError("验证不得构造第二个 Manager")
        monkeypatch.setattr(type(host), "__init__", reject_child_manager)
        before = log.reader("formal").snapshot()
        with closing(sqlite3.connect(workspace / "sessions.db")) as db:
            formal_before = tuple(db.iterdump())
        expected = pytest.raises(OSError, match="input owner still open") if fail_close else nullcontext()
        with expected:
            async with host.open_validation(result.update_id) as scope:
                validation = next(iter(host._validation_hosts.values()))
                assert isinstance(validation.message_bus, MessageBus)
                assert validation.message_bus is not validation.bus
                custody = scope.require(INPUT_CUSTODY)
                identity = scope.require(CHANNEL_IDENTITY)
                admission = scope.require(SOURCE_ADMISSION)
                assert admission.validation
                with pytest.raises(RuntimeError, match="候选装配不能启动正式来源"):
                    admission.require_starting(validation.root.context)

                # 1. 验证程序仍真实运行；输入日志和路由只落验证库。
                entered, release = asyncio.Event(), asyncio.Event()
                release.set()
                output = await scope.require(ServiceKey("test.validation"))(entered, release)
                assert output.body.finish == "complete"
                await identity.remember("probe", "user", "chat")
                assert identity.resolve("probe", "user") == "chat"
                assert await custody.reserve_durable_inbound(inbound(1))
                assert custody.has_pending_durable_inbound(
                    channel="probe", session_key="validation", provider_message_id="message-1",
                )
                await custody.settle_rejected_inbound(
                    channel="probe", session_key="validation", provider_message_id="message-1",
                )
                assert not custody.has_pending_durable_inbound(
                    channel="probe", session_key="validation", provider_message_id="message-1",
                )
                assert await custody.reserve_durable_inbound(inbound(2))
                original_close = validation.message_bus.aclose
                if fail_close:
                    async def stop_failed():
                        raise OSError("input owner still open")
                    monkeypatch.setattr(validation.message_bus, "aclose", stop_failed)

        if fail_close:
            # 2. 离开 runtime scope 后才关闭；失败保留原宿主和连接。
            assert not validation.closed
            assert host._validation_hosts[validation.identity] is validation
            assert validation.inbound_store.list_inbound_handoffs()
            assert validation.identities.resolve("probe", "user") == "chat"
            monkeypatch.setattr(validation.message_bus, "aclose", original_close)
            await host.retry_validation_cleanup(validation.identity)
        assert validation.closed
        with pytest.raises(RuntimeError, match="已关闭"):
            await custody.reserve_durable_inbound(inbound(3))
        with closing(sqlite3.connect(validation.workspace / "sessions.db")) as db:
            assert db.execute("SELECT count(*) FROM session_admissions").fetchone()[0] == 0
        with closing(InboundHandoffStore(validation.workspace / "sessions.db")) as store:
            assert [row["handoff_id"] for row in store.list_inbound_handoffs()] == ["handoff-2"]
        assert log.reader("formal").snapshot() == before
        with closing(sqlite3.connect(workspace / "sessions.db")) as db:
            assert tuple(db.iterdump()) == formal_before
    finally:
        await host.terminate_all()
        log.close()
