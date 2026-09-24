import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import sqlite3
from contextlib import closing

import pytest

from akashic_sdk import AsyncAkashic, RemoteError
from infra.control.socket import SocketAppServer
from plugins.content.plugin import check_text
from session.log import SessionAttributes
from session.message import CallRef, ContentPart, Input, Output
from tests.test_message_control import runtime


def response_data(value: object) -> Mapping[str, object]:
    """Narrow one JSON-RPC result before asserting its protocol fields."""
    assert isinstance(value, Mapping)
    return value


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
async def test_programmatic_service_borrows_its_owner_for_another_fiber(tmp_path):
    """另一个真实 Fiber 的许可不能代替 Programmatic 自己的许可。"""
    from agent.plugin_composition import CompositionError, ServiceKey
    from agent.plugin_composition.messages import MESSAGE_WRITERS, SESSION_ADMISSION
    from agent.plugin_composition.tasks import TASKS
    from plugins.programmatic.control import (
        AdmitParams, PauseParams, PROGRAMMATIC, ResultParams, ResumeParams, SendParams,
    )
    from tests.test_default_reply import application

    caller_key = ServiceKey("fixture.programmatic.call")

    def add_sources(sources):
        shutil.copytree(Path(__file__).parents[1] / "plugins/programmatic", sources / "programmatic")
        caller = sources / "programmatic_caller"
        caller.mkdir()
        (caller / "plugin.py").write_text(
            "from agent.plugin_composition import ServiceKey\n"
            "from plugins.programmatic.control import PROGRAMMATIC\n"
            "api_version = 3\n"
            "name = 'programmatic_caller'\n"
            "version = '1.0.0'\n"
            "inject = (PROGRAMMATIC,)\n"
            "CALL = ServiceKey('fixture.programmatic.call')\n"
            "async def apply(ctx):\n"
            "    api = ctx.require(PROGRAMMATIC)\n"
            "    async def call(method, params):\n"
            "        async with ctx.runtime_scope():\n"
            "            return await api.call(method, params)\n"
            "    await ctx.provide(CALL, call)\n",
            encoding="utf-8",
        )

    api = None
    session = "programmatic:other-fiber"
    async with application(tmp_path, replying=False, extra_sources=add_sources) as (log, host):
        root = host.live_root
        assert root is not None
        call = root.service_value(caller_key)
        assert call is not None
        api = root.context.require(PROGRAMMATIC)
        ctx = api.ctx
        with pytest.raises(CompositionError, match="OwnerCall"):
            ctx.require(SESSION_ADMISSION).ensure(ctx, session, SessionAttributes("internal", "excluded"))
        with pytest.raises(CompositionError, match="OwnerCall"):
            ctx.require(MESSAGE_WRITERS).bind(ctx, author="user", source="programmatic",
                body_types=(Input,), content={})
        with pytest.raises(CompositionError, match="OwnerCall"):
            ctx.require(TASKS).open(ctx)

        admitted = await call("programmatic/session/admit", AdmitParams(session_id=session))
        assert admitted["learning"] == "excluded"
        sent = await call("programmatic/message/send", SendParams(
            session_id=session, message_id="input", text="真实 Fiber 输入"))
        assert sent["message_id"] == "input"
        assert [row.message_id for row in log.reader(session).snapshot()] == ["input"]
        result = await call("programmatic/message/result", ResultParams(
            session_id=session, input_id="input"))
        assert result["status"] == "open"
        await call("programmatic/message/pause", PauseParams(session_id=session, message_id="pause"))
        assert (await call("programmatic/message/result", ResultParams(
            session_id=session, input_id="input")))["status"] == "pause"
        await call("programmatic/message/resume", ResumeParams(
            session_id=session, message_id="resume", input_id="input"))
        assert (await call("programmatic/message/result", ResultParams(
            session_id=session, input_id="input")))["status"] == "open"
        assert [row.message_id for row in log.reader(session).snapshot()] == ["input", "pause", "resume"]

    assert api is not None
    with pytest.raises(CompositionError, match="当前 activation 不接纳新调用"):
        await api.call("programmatic/session/admit", AdmitParams(session_id=session))


@pytest.mark.asyncio
async def test_programmatic_admission_is_immutable_and_ack_retries_recover_same_input(tmp_path, monkeypatch):
    async with endpoint(tmp_path, monkeypatch) as (address, core):
        session = "programmatic:test"
        async with await AsyncAkashic.connect(address) as client:
            for _ in range(2):
                admission = await client.request("programmatic/session/admit", {"session_id": session})
                assert response_data(admission)["learning"] == "excluded"
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
            query: dict[str, object] = {"session_id": session, "input_id": "one"}
            assert response_data(await client.request("programmatic/message/result", query))["status"] == "open"
            await client.request("programmatic/message/pause", {"session_id": session, "message_id": "pause"})
            assert response_data(await client.request("programmatic/message/result", query))["status"] == "pause"
            await client.request("programmatic/message/resume", {**query, "message_id": "resume"})
            assert response_data(await client.request("programmatic/message/result", query))["status"] == "open"
        assert core.message_log.catalog().attributes(session) == SessionAttributes("internal", "excluded")
        assert sum(isinstance(row.body, Input) for row in core.message_log.reader(session).snapshot()) == 1


@pytest.mark.asyncio
async def test_public_programmatic_learning_policy_reaches_akasha_and_markdown(tmp_path, monkeypatch):
    """Public admission fixes eligibility before either real learning owner reads the log."""
    from agent.plugin_composition import CHAT_MODELS, ServiceKey
    from agent.plugin_composition.bindings import BINDINGS
    from plugins.akasha.learning import AKASHA_LEARNING, LearningConfig
    from plugins.akasha.infrastructure.persistence import load_consumption
    from plugins.markdown_memory.plugin import Config as MarkdownConfig, project
    from plugins.markdown_memory._boundaries import CONTENT, CONTEXT, TURN_PROJECTION, COMPACTION_READER
    from plugins.compaction.records import COMPACTION_SUMMARIES
    from tests.test_akasha_message_plugin import application as akasha_application
    from tests.test_message_markdown_memory import (
        application as markdown_application, profile_store, publish, record_use,
    )

    admission_root = tmp_path / "admission"
    admission_root.mkdir()
    sessions = (
        ("programmatic:default-excluded", None, "private default fact"),
        ("programmatic:false-excluded", False, "private false fact"),
        ("programmatic:eligible", True, "fact-one learned answer"),
    )
    # 1. Admit and send through the public control boundary.
    async with endpoint(admission_root, monkeypatch) as (address, core):
        async with await AsyncAkashic.connect(address) as client:
            for session, persist_memory, text in sessions:
                input_id = f"{session}:input"
                params: dict[str, object] = {"session_id": session}
                if persist_memory is not None:
                    params["persist_memory"] = persist_memory
                admitted = await client.request("programmatic/session/admit", params)
                assert admitted == await client.request("programmatic/session/admit", params)
                assert response_data(admitted)["learning"] == ("eligible" if persist_memory else "excluded")
                with pytest.raises(RemoteError):
                    await client.request("programmatic/session/admit", {
                        "session_id": session, "persist_memory": not bool(persist_memory),
                    })
                await client.request("programmatic/message/send", {
                    "session_id": session, "message_id": input_id, "text": text,
                })
                assert response_data(await client.request("programmatic/message/result", {
                    "session_id": session, "input_id": input_id,
                }))["status"] == "open"
                assert [type(row.body) for row in core.message_log.reader(session).snapshot()] == [Input]
                assert core.message_log.catalog().attributes(session) == SessionAttributes(
                    "internal", "eligible" if persist_memory else "excluded")

    source_db = admission_root / "workspace/sessions.db"

    def copy_log(source_db: Path, target: Path) -> None:
        target.mkdir()
        with closing(sqlite3.connect(source_db)) as source, closing(sqlite3.connect(target / "sessions.db")) as saved:
            source.backup(saved)

    akasha_root = tmp_path / "akasha"
    copy_log(source_db, akasha_root)
    # 2. Close the original turns while the actual Akasha owner watches this log.
    async with akasha_application(akasha_root) as (log, host):
        root = host.live_root
        assert root is not None
        for session, persist_memory, _ in sessions:
            log.writer(session, author="assistant", source="programmatic",
                body_types=(Output,), content={"text": check_text}).append(
                    f"{session}:answer", Output((ContentPart("text", "learned answer" if persist_memory else "private answer"),),
                                     "complete"))
        await asyncio.wait_for(root.context.require(ServiceKey("fixture.embedded")).wait(), 5)
        embedded = (akasha_root / "embedding-calls.txt").read_text()
        assert "fact-one" in embedded and "learned answer" in embedded
        assert "private default fact" not in embedded and "private false fact" not in embedded
        learning = root.context.require(AKASHA_LEARNING)
        blocked = LearningConfig(embedding_model="fixture", dimension=2, sources=("conversation",))
        assert learning.samples(log.catalog(), blocked, heads=log.catalog().snapshot_heads()) == ()
        for session, _, _ in sessions:
            assert [type(row.body) for row in log.reader(session).snapshot()] == [Input, Output]
    learned = load_consumption(akasha_root / "workspace/memory/akasha.db")
    assert learned is not None
    assert [(entry.session_id, entry.ending[1]) for entry in learned.applied] == [
        ("programmatic:eligible", "programmatic:eligible:answer")]

    markdown_root = tmp_path / "markdown"
    copy_log(akasha_root / "sessions.db", markdown_root)
    # 3. Use real summary receipts to drive Markdown's source and session gates.
    async with markdown_application(markdown_root) as (log, host):
        root = host.live_root
        assert root is not None
        ctx = root.context
        store = profile_store(markdown_root)

        async def consume(session: str, identity: str, sources: tuple[str, ...]):
            summary = publish(log, identity + "-summary", session_id=session)
            used = await record_use(log, host, summary, identity + "-use", source="programmatic",
                                    session_id=session)
            await project(used, reader=log.reader(session), bindings=ctx.require(BINDINGS), store=store,
                models=ctx.require(CHAT_MODELS), lock_path=markdown_root / "workspace/memory/markdown-profile.lock",
                sources=sources, projection=ctx.require(TURN_PROJECTION), content=ctx.require(CONTENT),
                context=ctx.require(CONTEXT), summaries=ctx.require(COMPACTION_SUMMARIES),
                compaction=ctx.require(COMPACTION_READER))
            return summary.reference, used

        for session, _, _ in sessions[:2]:
            reference, _ = await consume(session, session.split(":")[-1], MarkdownConfig().sources)
            assert not store.is_applied(reference)
        assert not (markdown_root / "requests.jsonl").exists()
        eligible = sessions[2][0]
        blocked_ref, eligible_use = await consume(eligible, "eligible", ("conversation",))
        assert not store.is_applied(blocked_ref)
        assert not (markdown_root / "requests.jsonl").exists()
        await project(eligible_use, reader=log.reader(eligible), bindings=ctx.require(BINDINGS), store=store,
            models=ctx.require(CHAT_MODELS), lock_path=markdown_root / "workspace/memory/markdown-profile.lock",
            sources=MarkdownConfig().sources, projection=ctx.require(TURN_PROJECTION), content=ctx.require(CONTENT),
            context=ctx.require(CONTEXT), summaries=ctx.require(COMPACTION_SUMMARIES),
            compaction=ctx.require(COMPACTION_READER))
        assert store.is_applied(blocked_ref)
        assert "fact-one" in store.read_memory()
        payload = (markdown_root / "requests.jsonl").read_text()
        assert "private default fact" not in payload and "private false fact" not in payload
        for session, _, _ in sessions:
            assert isinstance(log.reader(session).get(f"{session}:input").body, Input)


@pytest.mark.asyncio
async def test_programmatic_committed_output_releases_route_without_result_read(tmp_path, monkeypatch):
    async with endpoint(tmp_path, monkeypatch) as (address, core):
        session = "programmatic:route-settle"
        async with await AsyncAkashic.connect(address) as client:
            await client.request("programmatic/session/admit", {"session_id": session})
            await client.request("programmatic/message/send", {
                "session_id": session, "message_id": "input", "text": "finish without read",
            })
        writer = core.message_log.writer(
            session, author="assistant", source="programmatic",
            body_types=(Output,), content={"text": check_text},
        )
        writer.append("final", Output((ContentPart("text", "done"),), "complete"))
        async with asyncio.timeout(3):
            while core.control_frames._routes:  # type: ignore[attr-defined]
                await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_programmatic_resume_rebinds_output_to_new_connection_after_disconnect(
    tmp_path, monkeypatch,
):
    """旧连接断开后，显式 resume 必须把最终 Output 观察交给新连接。"""
    from plugins.programmatic.control import PROGRAMMATIC
    from plugins.turn_projection.plugin import TURN_PROJECTION

    async with endpoint(tmp_path, monkeypatch) as (address, core):
        session = "programmatic:resume"
        async with await AsyncAkashic.connect(address) as first:
            await first.request("programmatic/session/admit", {"session_id": session})
            await first.request("programmatic/message/send", {
                "session_id": session, "message_id": "input", "text": "pause me",
            })
            await first.request("programmatic/message/pause", {
                "session_id": session, "message_id": "pause",
            })

        async with await AsyncAkashic.connect(address) as second:
            await second.request("programmatic/message/resume", {
                "session_id": session, "message_id": "resume", "input_id": "input",
            })
            claim = core.control_frames.arm_claim(session, "input", CallRef("resume-call", 0))
            writer = core.message_log.writer(
                session, author="assistant", source="programmatic",
                body_types=(Output,), content={"text": check_text},
            )
            writer.append("final", Output((ContentPart("text", "恢复结果"),), "complete"))

            root = core.plugin_manager.live_root
            assert root is not None
            context = root.context
            reader = core.message_log.reader(session)
            projection = context.require(TURN_PROJECTION)
            turn = projection.project(reader.snapshot(), "programmatic")[-1]
            waiter = asyncio.create_task(
                context.require(PROGRAMMATIC).wait(reader, turn),
            )
            page = await second.message_read(session)
            await asyncio.wait_for(waiter, 3)
            claim.consume()

            assert [item["id"] for item in page["items"]][-1] == "final"


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
    from plugins.programmatic.control import PROGRAMMATIC, AdmitParams, SendParams, ResultParams
    from tests.test_default_reply import application

    def add_source(root):
        shutil.copytree(Path(__file__).parents[1] / "plugins/programmatic", root / "programmatic")

    async with application(tmp_path, replying=True, extra_sources=add_source) as (log, host):
        session = "programmatic:reply"
        root = host.live_root
        assert root is not None
        api = root.context.require(PROGRAMMATIC)
        generation = host.generation("programmatic")
        assert generation is not None and generation.fiber is not None
        context = generation.fiber.context
        async with context.runtime_scope():
            await api.call("programmatic/session/admit", AdmitParams(session_id=session))
            await api.call("programmatic/message/send", SendParams(session_id=session, message_id="input", text="do work"))
        async with asyncio.timeout(5):
            async for _ in log.reader(session).follow():
                async with context.runtime_scope():
                    result = await api.call(
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
