import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import shutil

import pytest

from akashic_sdk import AsyncAkashic, RemoteError
from infra.control.socket import SocketAppServer
from plugins.content.plugin import check_text
from session.log import SessionAttributes
from session.message import ContentPart, Input, Output
from tests.test_message_control import runtime


@asynccontextmanager
async def endpoint(tmp_path, monkeypatch):
    async with runtime(tmp_path, monkeypatch, programmatic=True) as (core, service):
        server = SocketAppServer(tmp_path / "control.sock", service)
        await server.start()
        try:
            yield str(server.endpoint), core
        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_programmatic_admission_is_immutable_and_ack_retries_recover_same_input(tmp_path, monkeypatch):
    async with endpoint(tmp_path, monkeypatch) as (address, core):
        session = "programmatic:test"
        async with await AsyncAkashic.connect(address) as client:
            for _ in range(2):
                admission = await client.request("programmatic/session/admit", {"session_id": session})
                assert admission["learning"] == "excluded"
            with pytest.raises(RemoteError):
                await client.request("programmatic/session/admit", {"session_id": session, "persist_memory": True})
            with pytest.raises(RemoteError):
                await client.request("programmatic/message/send", {
                    "session_id": session, "message_id": "one", "text": "original", "persist_memory": True,
                })
            ack = await client.request("programmatic/message/send", {
                "session_id": session, "message_id": "one", "text": "original",
            })
            assert not (await client.session_list())["items"]
        async with await AsyncAkashic.connect(address) as client:
            assert ack == await client.request("programmatic/message/send", {
                "session_id": session, "message_id": "one", "text": "original",
            })
            with pytest.raises(RemoteError):
                await client.request("programmatic/message/send", {
                    "session_id": session, "message_id": "one", "text": "changed",
                })
            query = {"session_id": session, "input_id": "one"}
            assert (await client.request("programmatic/message/result", query))["status"] == "open"
            await client.request("programmatic/message/pause", {"session_id": session, "message_id": "pause"})
            assert (await client.request("programmatic/message/result", query))["status"] == "pause"
            await client.request("programmatic/message/resume", {**query, "message_id": "resume"})
            assert (await client.request("programmatic/message/result", query))["status"] == "open"
        assert core.message_log.catalog().attributes(session) == SessionAttributes("internal", "excluded")
        assert sum(isinstance(row.body, Input) for row in core.message_log.reader(session).snapshot()) == 1


@pytest.mark.asyncio
async def test_exec_cli_reads_exact_completed_message_over_real_socket(tmp_path, monkeypatch, capsys):
    from main import run_exec

    async with endpoint(tmp_path, monkeypatch) as (address, core):
        session = "programmatic:cli"

        async def model_output():
            async for message in core.message_log.reader(session).follow():
                if isinstance(message.body, Input):
                    core.message_log.writer(session, author="fixture", source="programmatic",
                        body_types=(Output,), content={"text": check_text}).append("final", Output((
                            ContentPart("text", "完整结果"),), "complete"))
                    return

        producer = asyncio.create_task(model_output())
        try:
            result = await asyncio.wait_for(run_exec(["exec", "--new", "--session", session,
                "--message-id", "original", "--endpoint", address, "--final-only", "原输入"],
                "unused.toml", tmp_path), 5)
            assert result == 0
            assert capsys.readouterr().out == "完整结果\n"
            assert core.message_log.reader(session).get("original").body.parts[0].value == "原输入"
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)


@pytest.mark.asyncio
async def test_programmatic_source_uses_real_default_reply_and_tool_settlement(tmp_path):
    from agent.plugins.snapshot import lease_runtime_snapshot
    from plugins.programmatic.control import PROGRAMMATIC, AdmitParams, SendParams, ResultParams
    from tests.test_default_reply import application

    def add_source(root):
        shutil.copytree(Path(__file__).parents[1] / "plugins/programmatic", root / "programmatic")

    async with application(tmp_path, replying=True, extra_sources=add_source) as (log, host):
        session = "programmatic:reply"
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            api = snapshot.composition_root.context.require(PROGRAMMATIC)
            await api.call("programmatic/session/admit", AdmitParams(session_id=session))
            await api.call("programmatic/message/send", SendParams(session_id=session, message_id="input", text="do work"))
        async with asyncio.timeout(5):
            async for _ in log.reader(session).follow():
                async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                    result = await snapshot.composition_root.context.require(PROGRAMMATIC).call(
                        "programmatic/message/result", ResultParams(session_id=session, input_id="input"))
                if result["status"] != "open":
                    break
        assert result["status"] == "complete"
        assert (tmp_path / "effect.txt").read_text() == "once\n"
        assert [type(row.body).__name__ for row in log.reader(session).snapshot()] == [
            "Input", "Output", "ToolResult", "Output",
        ]


@pytest.mark.asyncio
async def test_control_socket_never_removes_a_non_socket_or_replacement(tmp_path, monkeypatch):
    import os
    import stat

    async with runtime(tmp_path, monkeypatch) as (_, service):
        path = tmp_path / "control.sock"
        path.write_text("user file")
        server = SocketAppServer(path, service)
        with pytest.raises(RuntimeError, match="不是 socket"):
            await server.start()
        await server.stop()
        assert path.read_text() == "user file"
        path.unlink()
        await server.start()
        try:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            os.rename(path, tmp_path / "original.sock")
            path.write_text("replacement")
        finally:
            await server.stop()
        assert path.read_text() == "replacement"


@pytest.mark.asyncio
async def test_control_socket_stop_closes_clients_waiting_for_connection_slot(tmp_path, monkeypatch):
    async with runtime(tmp_path, monkeypatch) as (_, service):
        entered = asyncio.Event()
        class ObservedServer(SocketAppServer):
            async def _accept(self, reader, writer):
                # 调度屏障让 stop 在第二个 handler 已进入 slot 等待之后运行。
                entered.set()
                await super()._accept(reader, writer)
        server = ObservedServer(tmp_path / "slots.sock", service, max_connections=1)
        await server.start()
        writers = []
        try:
            first, one = await asyncio.open_unix_connection(str(server.endpoint))
            writers.append(one)
            await entered.wait()
            entered.clear()
            second, two = await asyncio.open_unix_connection(str(server.endpoint))
            writers.append(two)
            await entered.wait()
            await asyncio.wait_for(server.stop(), 2)
            assert await asyncio.wait_for(first.read(), 2) == b""
            assert await asyncio.wait_for(second.read(), 2) == b""
        finally:
            await server.stop()
            for writer in writers:
                writer.close()
                await writer.wait_closed()


@pytest.mark.asyncio
async def test_programmatic_requests_keep_exact_snapshot_while_follow_does_not_pin_it(tmp_path, monkeypatch):
    from agent.plugins.snapshot import RuntimeSnapshotCompiler, get_current_runtime_snapshot
    from plugins.programmatic.control import PROGRAMMATIC

    async with endpoint(tmp_path, monkeypatch) as (address, core):
        manager = core.plugin_manager
        old = manager.current_snapshot
        api = old.composition_root.context.require(PROGRAMMATIC)
        original = api.call
        entered, release = asyncio.Event(), asyncio.Event()
        observed = []

        async def blocked(method, params):
            snapshot = get_current_runtime_snapshot()
            observed.append(snapshot.snapshot_id)
            if params.session_id == "programmatic:old":
                entered.set()
                await release.wait()
                assert get_current_runtime_snapshot() is snapshot
            return await original(method, params)

        monkeypatch.setattr(api, "call", blocked)
        async with await AsyncAkashic.connect(address) as client:
            async with await client.session_follow("programmatic:new") as feed:
                stream = feed.events()
                assert (await asyncio.wait_for(anext(stream), 3))["type"] == "reply.status"
                assert old.lease_count == 0
                request = asyncio.create_task(client.request("programmatic/session/admit", {
                    "session_id": "programmatic:old",
                }))
                try:
                    await asyncio.wait_for(entered.wait(), 3)
                    replacement = RuntimeSnapshotCompiler().compile(old.generations,
                        snapshot_revision="programmatic-publication-proof", composition_root=old.composition_root,
                        core_channel_definitions=manager._core_channel_definitions)
                    await manager._publish_committed_snapshot(replacement)
                    assert manager.current_snapshot is replacement
                    assert not request.done() and old.lease_count == 1
                    release.set()
                    await asyncio.wait_for(request, 3)
                    await asyncio.wait_for(manager.snapshot_store.wait_for_snapshot_drained(old), 3)
                    await client.request("programmatic/session/admit", {"session_id": "programmatic:new"})
                    await client.request("programmatic/message/send", {
                        "session_id": "programmatic:new", "message_id": "new-input", "text": "new owner",
                    })
                    assert observed == [old.snapshot_id, replacement.snapshot_id, replacement.snapshot_id]
                    async with asyncio.timeout(3):
                        async for event in stream:
                            if event["type"] == "messages.appended":
                                assert event["items"][0]["id"] == "new-input"
                                break
                finally:
                    release.set()
                    request.cancel()
                    await asyncio.gather(request, return_exceptions=True)
                    await stream.aclose()
