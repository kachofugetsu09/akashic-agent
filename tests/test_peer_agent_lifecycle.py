import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent.config_models import PeerAgentConfig
from agent.peer_agent.card_resolver import (
    AgentCard,
    AgentCardSchemaError,
    fetch_agent_card,
)
from agent.peer_agent.poller import PeerAgentPoller, PeerTaskDuplicateError
from agent.peer_agent.process_manager import (
    PeerProcessConfig,
    PeerProcessManager,
    PeerReady,
)
from agent.peer_agent.registry import PeerAgentRegistry
from agent.peer_agent.tool import PeerAgentTool
from bus.queue import MessageBus
from core.net.http import HttpRequester


class _ConfigBase(TypedDict):
    base_url: str
    cwd: str
    log_dir: str


def _as_requester(fake: object) -> HttpRequester:
    return cast(HttpRequester, fake)


def _as_bus(fake: object) -> MessageBus:
    return cast(MessageBus, fake)


def _as_process_manager(fake: object) -> PeerProcessManager:
    return cast(PeerProcessManager, fake)


def _as_poller(fake: object) -> PeerAgentPoller:
    return cast(PeerAgentPoller, fake)


def _new_poller(
    bus: object, process_manager: object, requester: object
) -> PeerAgentPoller:
    return PeerAgentPoller(
        _as_bus(bus),
        _as_process_manager(process_manager),
        _as_requester(requester),
    )


def _as_process(fake: object) -> asyncio.subprocess.Process:
    return cast(asyncio.subprocess.Process, fake)


class _Response:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://peer.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("http error", request=request, response=response)

    def json(self) -> object:
        return self._payload


def _valid_card() -> dict[str, object]:
    return {
        "name": "research",
        "url": "http://peer.test",
        "description": "deep research",
        "skills": [
            {
                "id": "research",
                "name": "Research",
                "description": "Research a topic",
                "tags": ["web"],
                "examples": ["compare tools"],
            }
        ],
    }


@pytest.mark.asyncio
async def test_agent_card_validates_root_and_skill_fields() -> None:
    requester = SimpleNamespace(
        get=AsyncMock(return_value=_Response(_valid_card()))
    )

    card = await fetch_agent_card("http://peer.test", _as_requester(requester))

    assert card.name == "research"
    assert card.url == "http://peer.test"
    assert card.primary_skill is not None
    assert card.primary_skill.tags == ["web"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"name": "", "url": "http://peer.test"},
        {"name": "research", "url": ""},
        {"name": "research", "url": "http://peer.test", "skills": {}},
        {
            "name": "research",
            "url": "http://peer.test",
            "skills": [{"id": "research", "name": "Research"}],
        },
    ],
)
async def test_agent_card_bad_schema_is_not_offline_fallback(payload: object) -> None:
    requester = SimpleNamespace(get=AsyncMock(return_value=_Response(payload)))

    with pytest.raises(AgentCardSchemaError):
        await fetch_agent_card("http://peer.test", _as_requester(requester))


@pytest.mark.asyncio
async def test_registry_falls_back_only_for_unreachable_server() -> None:
    requester = SimpleNamespace(
        get=AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    )
    registry = PeerAgentRegistry(
        process_manager=_as_process_manager(SimpleNamespace()),
        poller=_as_poller(SimpleNamespace()),
        requester=_as_requester(requester),
    )
    config = PeerAgentConfig(
        name="research",
        base_url="http://peer.test",
        launcher=["python", "-m", "research"],
        description="from config",
    )

    tools = await registry.discover_all([config])

    assert len(tools) == 1
    assert tools[0].description.startswith("from config")


@pytest.mark.asyncio
async def test_registry_does_not_hide_bad_card_schema() -> None:
    requester = SimpleNamespace(
        get=AsyncMock(return_value=_Response({"name": "research", "url": ""}))
    )
    registry = PeerAgentRegistry(
        process_manager=_as_process_manager(SimpleNamespace()),
        poller=_as_poller(SimpleNamespace()),
        requester=_as_requester(requester),
    )
    config = PeerAgentConfig(
        name="research",
        base_url="http://peer.test",
        launcher=["python", "-m", "research"],
    )

    with pytest.raises(AgentCardSchemaError):
        await registry.discover_all([config])


@pytest.mark.asyncio
async def test_registry_keeps_config_identity_and_route_with_live_card() -> None:
    live_card = _valid_card()
    live_card.update(
        name="remote-card-name",
        url="http://remote-card.test",
        description="live capability description",
    )
    response = _Response(live_card)
    submit_response = _Response({"result": {"id": "task-1"}})
    requester = SimpleNamespace(
        get=AsyncMock(return_value=response),
        post=AsyncMock(return_value=submit_response),
    )
    process_manager = SimpleNamespace(
        ensure_ready=AsyncMock(return_value=PeerReady(started_by_call=False)),
        terminate=AsyncMock(),
    )
    poller = SimpleNamespace(register=MagicMock(), has_pending=MagicMock(return_value=False))

    @asynccontextmanager
    async def submission_lease(agent_name: str):
        yield

    poller.submission_lease = submission_lease
    registry = PeerAgentRegistry(
        _as_process_manager(process_manager),
        _as_poller(poller),
        _as_requester(requester),
    )
    config = PeerAgentConfig(
        name="configured-agent",
        base_url="http://configured-peer.test",
        launcher=["python", "-m", "research"],
        description="config fallback description",
    )

    tools = await registry.discover_all([config])
    tool = tools[0]
    result = await tool.execute(goal="research topic", channel="telegram", chat_id="42")

    assert "submitted" in result
    assert tool._card.name == "configured-agent"
    assert tool._card.url == "http://configured-peer.test"
    assert tool._card.description == "live capability description"
    assert tool._card.primary_skill is not None
    process_manager.ensure_ready.assert_awaited_once_with("configured-agent")
    post_call = requester.post.call_args
    assert post_call.args[0] == "http://configured-peer.test"
    assert post_call.kwargs["json"]["method"] == "message/send"
    poller.register.assert_called_once()
    assert poller.register.call_args.kwargs["agent_name"] == "configured-agent"
    assert poller.register.call_args.kwargs["agent_url"] == "http://configured-peer.test"


def _task_response(payload: object) -> _Response:
    return _Response(payload)


def _submit_response(task_id: str) -> _Response:
    return _Response({"result": {"id": task_id}})


def _completed_payload() -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": "poll-1",
        "result": {
            "status": {
                "state": "completed",
                "message": {"parts": [{"text": "done"}]},
            },
            "artifacts": [
                {"name": "report", "parts": [{"root": {"text": "/tmp/report.md"}}]}
            ],
        },
    }


def _build_poller(
    response: _Response,
    *,
    bus: object | None = None,
    process_manager: object | None = None,
) -> tuple[PeerAgentPoller, AsyncMock, AsyncMock]:
    requester = SimpleNamespace(post=AsyncMock(return_value=response))
    bus_mock = bus or SimpleNamespace(publish_inbound=AsyncMock())
    pm_mock = process_manager or SimpleNamespace(terminate=AsyncMock())
    poller = _new_poller(bus_mock, pm_mock, requester)
    poller.register(
        task_id="task-1",
        agent_name="research",
        agent_url="http://peer.test",
        channel="telegram",
        chat_id="42",
        goal="research topic",
    )
    return poller, bus_mock.publish_inbound, pm_mock.terminate


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"error": "bad"},
        {"result": {}},
        {"result": {"status": {}}},
        {"result": {"status": {"state": "completed", "message": {}}}},
        {
            "result": {
                "status": {"state": "completed"},
                "artifacts": [{"name": "", "parts": []}],
            }
        },
        {
            "result": {
                "status": {"state": "completed"},
                "artifacts": [{"name": "report", "parts": [{"text": ""}]}],
            }
        },
    ],
)
async def test_tasks_get_bad_schema_is_rejected(payload: object) -> None:
    poller, _, _ = _build_poller(_task_response(payload))

    with pytest.raises((ValueError, RuntimeError)):
        await poller._get_task_status("http://peer.test", "task-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [AssertionError("bus invariant"), RuntimeError("bus bug")])
@pytest.mark.asyncio
async def test_message_bus_internal_error_fails_loud_and_keeps_pending(
    error: BaseException,
) -> None:
    bus = SimpleNamespace(publish_inbound=AsyncMock(side_effect=error))
    pm = SimpleNamespace(terminate=AsyncMock())
    poller, publish, terminate = _build_poller(
        _task_response(_completed_payload()), bus=bus, process_manager=pm
    )
    meta = poller._pending[("research", "task-1")]

    with pytest.raises(type(error), match=str(error)):
        await poller._check("task-1", meta)
    assert ("research", "task-1") in poller._pending
    publish.assert_awaited_once()
    terminate.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_retry_after_terminate_error_does_not_repeat_notification() -> None:
    bus = SimpleNamespace(publish_inbound=AsyncMock())
    pm = SimpleNamespace(
        terminate=AsyncMock(side_effect=[OSError("terminate failed"), None])
    )
    poller, publish, terminate = _build_poller(
        _task_response(_completed_payload()), bus=bus, process_manager=pm
    )
    meta = poller._pending[("research", "task-1")]

    with pytest.raises(OSError, match="terminate failed"):
        await poller._check("task-1", meta)
    await poller._check("task-1", meta)

    publish.assert_awaited_once()
    assert terminate.await_count == 2
    assert ("research", "task-1") not in poller._pending


@pytest.mark.asyncio
async def test_terminate_ownership_error_fails_loud_and_keeps_pending() -> None:
    bus = SimpleNamespace(publish_inbound=AsyncMock())
    pm = SimpleNamespace(terminate=AsyncMock(side_effect=RuntimeError("ownership lost")))
    poller, publish, terminate = _build_poller(
        _task_response(_completed_payload()), bus=bus, process_manager=pm
    )
    meta = poller._pending[("research", "task-1")]

    with pytest.raises(RuntimeError, match="ownership lost"):
        await poller._check("task-1", meta)

    publish.assert_awaited_once()
    terminate.assert_awaited_once_with("research")
    assert ("research", "task-1") in poller._pending


@pytest.mark.asyncio
async def test_pending_task_ids_are_scoped_by_agent() -> None:
    bus = SimpleNamespace(publish_inbound=AsyncMock())
    pm = SimpleNamespace(terminate=AsyncMock())
    requester = SimpleNamespace(
        post=AsyncMock(
            side_effect=[
                _task_response(_completed_payload()),
                _task_response(_completed_payload()),
            ]
        )
    )
    poller = _new_poller(bus, pm, requester)
    for agent_name in ("research-a", "research-b"):
        poller.register(
            task_id="task-1",
            agent_name=agent_name,
            agent_url=f"http://{agent_name}.test",
            channel="telegram",
            chat_id="42",
            goal="research topic",
        )

    await poller._check("task-1", poller._pending[("research-a", "task-1")])
    await poller._check("task-1", poller._pending[("research-b", "task-1")])

    assert pm.terminate.await_count == 2
    assert not poller._pending
    assert [call.args[0] for call in pm.terminate.await_args_list] == [
        "research-a",
        "research-b",
    ]


@pytest.mark.asyncio
async def test_same_agent_tasks_keep_process_until_last_completion() -> None:
    bus = SimpleNamespace(publish_inbound=AsyncMock())
    pm = SimpleNamespace(terminate=AsyncMock())
    response = _task_response(_completed_payload())
    requester = SimpleNamespace(post=AsyncMock(side_effect=[response, response]))
    poller = _new_poller(bus, pm, requester)
    for task_id in ("task-1", "task-2"):
        poller.register(
            task_id=task_id,
            agent_name="research",
            agent_url="http://research.test",
            channel="telegram",
            chat_id="42",
            goal=task_id,
        )

    await poller._check("task-1", poller._pending[("research", "task-1")])

    assert pm.terminate.await_count == 0
    assert ("research", "task-1") not in poller._pending
    assert ("research", "task-2") in poller._pending

    await poller._check("task-2", poller._pending[("research", "task-2")])

    pm.terminate.assert_awaited_once_with("research")
    assert not poller._pending


def test_duplicate_task_id_keeps_existing_pending_entry() -> None:
    poller = _new_poller(
        SimpleNamespace(publish_inbound=AsyncMock()),
        SimpleNamespace(terminate=AsyncMock()),
        SimpleNamespace(post=AsyncMock()),
    )
    poller.register(
        task_id="task-1",
        agent_name="research",
        agent_url="http://research.test",
        channel="telegram",
        chat_id="42",
        goal="first",
    )

    with pytest.raises(PeerTaskDuplicateError, match="重复注册"):
        poller.register(
            task_id="task-1",
            agent_name="research",
            agent_url="http://research.test",
            channel="telegram",
            chat_id="42",
            goal="duplicate",
        )

    assert poller._pending[("research", "task-1")].goal == "first"


@pytest.mark.asyncio
async def test_notification_does_not_block_submit_or_terminate_shared_process() -> None:
    publish_started = asyncio.Event()
    release_publish = asyncio.Event()
    ensure_started = asyncio.Event()

    async def publish(_message: object) -> None:
        publish_started.set()
        await release_publish.wait()

    async def ensure_ready(_name: str) -> PeerReady:
        ensure_started.set()
        return PeerReady(started_by_call=False)

    bus = SimpleNamespace(publish_inbound=publish)
    pm = SimpleNamespace(
        ensure_ready=ensure_ready,
        terminate=AsyncMock(),
    )
    requester = SimpleNamespace(
        post=AsyncMock(
            side_effect=[
                _task_response(_completed_payload()),
                _submit_response("task-new"),
            ]
        )
    )
    poller = _new_poller(bus, pm, requester)
    poller.register(
        task_id="task-old",
        agent_name="research",
        agent_url="http://research.test",
        channel="telegram",
        chat_id="42",
        goal="old",
    )
    old_check = asyncio.create_task(
        poller._check("task-old", poller._pending[("research", "task-old")])
    )
    await asyncio.wait_for(publish_started.wait(), timeout=1)

    tool = PeerAgentTool(
        AgentCard(name="research", url="http://research.test"),
        _as_process_manager(pm),
        _as_poller(poller),
        _as_requester(requester),
    )
    new_submit = asyncio.create_task(tool.execute(goal="new"))
    await asyncio.wait_for(ensure_started.wait(), timeout=1)
    result = await new_submit

    release_publish.set()
    await old_check

    assert '"status": "submitted"' in result
    pm.terminate.assert_not_awaited()
    assert ("research", "task-new") in poller._pending


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["queued", "running", "submitted", "working"])
async def test_in_progress_states_remain_pending(state: str) -> None:
    payload = {
        "result": {"status": {"state": state}},
    }
    poller, publish, terminate = _build_poller(_task_response(payload))
    meta = poller._pending[("research", "task-1")]

    await poller._check("task-1", meta)

    assert ("research", "task-1") in poller._pending
    publish.assert_not_awaited()
    terminate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"result": {"status": {"state": "mystery"}}},
        {"error": {"code": -32000, "message": "remote failure"}},
    ],
)
async def test_poll_loop_turns_protocol_error_into_failure_notification(
    payload: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages: list[object] = []
    notified = asyncio.Event()

    async def publish(message: object) -> None:
        messages.append(message)
        notified.set()

    bus = SimpleNamespace(publish_inbound=publish)
    pm = SimpleNamespace(terminate=AsyncMock())
    requester = SimpleNamespace(post=AsyncMock(return_value=_task_response(payload)))
    poller = _new_poller(bus, pm, requester)
    poller.register(
        task_id="task-1",
        agent_name="research",
        agent_url="http://research.test",
        channel="telegram",
        chat_id="42",
        goal="research topic",
    )
    monkeypatch.setattr("agent.peer_agent.poller._POLL_INTERVAL_S", 0)

    poller.start()
    await asyncio.wait_for(notified.wait(), timeout=1)
    await poller.stop()

    assert not poller._pending
    pm.terminate.assert_awaited_once_with("research")
    assert "协议错误" in messages[0].content


class _MalformedJsonResponse(_Response):
    def json(self) -> object:
        raise ValueError("malformed json")


@pytest.mark.asyncio
async def test_poll_loop_turns_malformed_json_into_failure_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[object] = []
    notified = asyncio.Event()

    async def publish(message: object) -> None:
        messages.append(message)
        notified.set()

    bus = SimpleNamespace(publish_inbound=publish)
    pm = SimpleNamespace(terminate=AsyncMock())
    requester = SimpleNamespace(
        post=AsyncMock(return_value=_MalformedJsonResponse(None))
    )
    poller = _new_poller(bus, pm, requester)
    poller.register(
        task_id="task-1",
        agent_name="research",
        agent_url="http://research.test",
        channel="telegram",
        chat_id="42",
        goal="research topic",
    )
    monkeypatch.setattr("agent.peer_agent.poller._POLL_INTERVAL_S", 0)

    poller.start()
    await asyncio.wait_for(notified.wait(), timeout=1)
    await poller.stop()

    assert "非法 JSON" in messages[0].content
    pm.terminate.assert_awaited_once_with("research")


@pytest.mark.asyncio
async def test_start_reraises_done_poller_exception() -> None:
    poller = _new_poller(
        SimpleNamespace(publish_inbound=AsyncMock()),
        SimpleNamespace(terminate=AsyncMock()),
        SimpleNamespace(post=AsyncMock()),
    )

    async def fail() -> None:
        raise RuntimeError("poller crashed")

    failed_task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    poller._task = failed_task

    with pytest.raises(RuntimeError, match="poller crashed"):
        poller.start()
    assert poller._task is None


class _FakeProcess:
    _next_pid = 1000

    def __init__(self, returncode: int | None = None) -> None:
        self.pid = _FakeProcess._next_pid
        _FakeProcess._next_pid += 1
        self.returncode = returncode
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        assert self.returncode is not None
        return self.returncode


class _HangingProcess(_FakeProcess):
    def terminate(self) -> None:
        self.terminate_calls += 1

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            await asyncio.Event().wait()
        assert self.returncode is not None
        return self.returncode


class _CloseFaultFile(io.BytesIO):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def close(self) -> None:
        if self.failures:
            self.failures -= 1
            raise OSError("log close failed")
        super().close()


class _FakeOwnedProcessGroup:
    def __init__(self, process: _FakeProcess) -> None:
        self.process = process

    async def terminate(self, *, timeout_s: float) -> None:
        if self.process.returncode is not None:
            await self.process.wait()
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=timeout_s)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()


@pytest.fixture(autouse=True)
def peer_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent.peer_agent.process_manager.OwnedProcessGroup.from_process",
        lambda process: _FakeOwnedProcessGroup(process),
    )


def _manager(
    tmp_path: Path,
    requester: object,
    *,
    startup_timeout_s: int = 1,
) -> PeerProcessManager:
    config = PeerProcessConfig(
        name="research",
        base_url="http://peer.test",
        launcher=["python", "-m", "research"],
        cwd=str(tmp_path),
        startup_timeout_s=startup_timeout_s,
        shutdown_timeout_s=1,
        log_dir=str(tmp_path / "logs"),
    )
    return PeerProcessManager([config], _as_requester(requester))


@pytest.mark.asyncio
async def test_process_manager_reaps_process_and_log_on_terminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-peer")
    proc = _FakeProcess()
    captured: dict[str, object] = {}
    requester = SimpleNamespace(
        get=AsyncMock(
            side_effect=[
                SimpleNamespace(status_code=503),
                SimpleNamespace(status_code=200),
            ]
        )
    )
    manager = _manager(tmp_path, requester)

    async def spawn(*args: object, **kwargs: object) -> _FakeProcess:
        captured.update(kwargs)
        return proc

    monkeypatch.setattr(
        "agent.peer_agent.process_manager.asyncio.create_subprocess_exec", spawn
    )
    ready = await manager.ensure_ready("research")
    assert ready.started_by_call
    await manager.terminate("research")

    assert proc.terminate_calls == 1
    assert proc.wait_calls == 1
    assert not manager._procs
    assert captured["stdout"].closed  # type: ignore[union-attr]
    assert captured["start_new_session"] is True
    assert captured["env"]["PATH"] == os.environ["PATH"]  # type: ignore[index]
    assert captured["env"]["AKASHIC_BOOT_ID"] == "boot-peer"  # type: ignore[index]


@pytest.mark.asyncio
async def test_process_manager_cleans_early_exit_and_create_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exited = _FakeProcess(returncode=3)
    requester = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status_code=503)))
    manager = _manager(tmp_path, requester)

    async def spawn_exited(*args: object, **kwargs: object) -> _FakeProcess:
        return exited

    monkeypatch.setattr(
        "agent.peer_agent.process_manager.asyncio.create_subprocess_exec", spawn_exited
    )
    with pytest.raises(RuntimeError, match="立即退出"):
        await manager.ensure_ready("research")
    assert exited.wait_calls == 1
    assert not manager._procs

    failed_capture: dict[str, object] = {}

    async def spawn_failed(*args: object, **kwargs: object) -> _FakeProcess:
        failed_capture.update(kwargs)
        raise OSError("exec failed")

    monkeypatch.setattr(
        "agent.peer_agent.process_manager.asyncio.create_subprocess_exec", spawn_failed
    )
    with pytest.raises(OSError, match="exec failed"):
        await manager.ensure_ready("research")
    assert not manager._procs
    assert failed_capture["stdout"].closed  # type: ignore[union-attr]
    assert (tmp_path / "logs" / "research.log").exists()


@pytest.mark.asyncio
async def test_process_manager_aggregates_create_and_log_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fault_file = _CloseFaultFile(failures=1)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: fault_file)
    requester = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status_code=503)))
    manager = _manager(tmp_path, requester)

    async def spawn_failed(*args: object, **kwargs: object) -> _FakeProcess:
        raise OSError("exec failed")

    monkeypatch.setattr(
        "agent.peer_agent.process_manager.asyncio.create_subprocess_exec", spawn_failed
    )

    with pytest.raises(BaseExceptionGroup) as error:
        await manager.ensure_ready("research")

    assert any("exec failed" in str(exc) for exc in error.value.exceptions)
    assert any("log close failed" in str(exc) for exc in error.value.exceptions)
    assert fault_file.closed
    assert not manager._procs


@pytest.mark.asyncio
async def test_process_manager_reaps_process_when_spawn_log_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fault_file = _CloseFaultFile(failures=1)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: fault_file)
    proc = _FakeProcess()
    requester = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status_code=503)))
    manager = _manager(tmp_path, requester)

    async def spawn(*args: object, **kwargs: object) -> _FakeProcess:
        return proc

    monkeypatch.setattr(
        "agent.peer_agent.process_manager.asyncio.create_subprocess_exec", spawn
    )

    with pytest.raises(OSError, match="log close failed"):
        await manager.ensure_ready("research")

    assert proc.terminate_calls == 1
    assert proc.wait_calls == 1
    assert fault_file.closed
    assert not manager._procs


@pytest.mark.asyncio
async def test_process_manager_cleans_spawn_on_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _FakeProcess()
    requester = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status_code=503)))
    manager = _manager(tmp_path, requester)

    async def spawn(*args: object, **kwargs: object) -> _FakeProcess:
        return proc

    async def cancelled(*args: object, **kwargs: object) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        "agent.peer_agent.process_manager.asyncio.create_subprocess_exec", spawn
    )
    monkeypatch.setattr(manager, "_wait_until_healthy", cancelled)

    with pytest.raises(asyncio.CancelledError):
        await manager.ensure_ready("research")
    assert proc.terminate_calls == 1
    assert proc.wait_calls == 1
    assert not manager._procs


@pytest.mark.asyncio
async def test_process_manager_spawn_timeout_releases_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _FakeProcess()
    requester = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status_code=503)))
    manager = _manager(tmp_path, requester)

    async def spawn(*args: object, **kwargs: object) -> _FakeProcess:
        return proc

    async def timed_out(*args: object, **kwargs: object) -> None:
        raise RuntimeError("启动超时")

    monkeypatch.setattr(
        "agent.peer_agent.process_manager.asyncio.create_subprocess_exec", spawn
    )
    monkeypatch.setattr(manager, "_wait_until_healthy", timed_out)

    with pytest.raises(RuntimeError, match="启动超时"):
        await manager.ensure_ready("research")
    assert proc.terminate_calls == 1
    assert proc.wait_calls == 1
    assert not manager._procs


@pytest.mark.asyncio
async def test_process_manager_kills_after_terminate_timeout() -> None:
    proc = _HangingProcess()

    await PeerProcessManager._kill(_as_process(proc), timeout_s=0.001)

    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1
    assert proc.wait_calls == 2


@pytest.mark.asyncio
async def test_process_manager_health_does_not_hide_programming_error(tmp_path: Path) -> None:
    requester = SimpleNamespace(get=AsyncMock(side_effect=RuntimeError("bad fake")))
    manager = _manager(tmp_path, requester)

    with pytest.raises(RuntimeError, match="bad fake"):
        await manager._is_healthy(manager._configs["research"])


@pytest.mark.asyncio
async def test_process_manager_reports_external_health_without_ownership(
    tmp_path: Path,
) -> None:
    requester = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status_code=200)))
    manager = _manager(tmp_path, requester)

    ready = await manager.ensure_ready("research")

    assert not ready.started_by_call
    assert not manager._procs


def test_process_manager_rejects_invalid_config(tmp_path: Path) -> None:
    requester = SimpleNamespace()
    base: _ConfigBase = {
        "base_url": "http://peer.test",
        "cwd": str(tmp_path),
        "log_dir": str(tmp_path / "logs"),
    }
    with pytest.raises(ValueError, match="launcher"):
        PeerProcessManager(
            [PeerProcessConfig(name="research", launcher=[], **base)],
            _as_requester(requester),
        )
    with pytest.raises(ValueError, match="重复"):
        PeerProcessManager(
            [
                PeerProcessConfig(name="research", launcher=["python"], **base),
                PeerProcessConfig(name="research", launcher=["python"], **base),
            ],
            _as_requester(requester),
        )
    with pytest.raises(ValueError, match="cwd"):
        PeerProcessManager(
            [
                PeerProcessConfig(
                    name="research",
                    launcher=["python"],
                    cwd=str(tmp_path / "missing"),
                    log_dir=str(tmp_path / "logs"),
                    base_url="http://peer.test",
                )
            ],
            _as_requester(requester),
        )


@pytest.mark.asyncio
async def test_process_manager_shutdown_preserves_all_failures(tmp_path: Path) -> None:
    requester = SimpleNamespace()
    base: _ConfigBase = {
        "base_url": "http://peer.test",
        "cwd": str(tmp_path),
        "log_dir": str(tmp_path / "logs"),
    }
    manager = PeerProcessManager(
        [
            PeerProcessConfig(name="one", launcher=["python"], **base),
            PeerProcessConfig(name="two", launcher=["python"], **base),
        ],
        _as_requester(requester),
    )
    first = _FakeProcess()
    second = _FakeProcess()
    first.terminate = lambda: (_ for _ in ()).throw(RuntimeError("first"))
    second.terminate = lambda: (_ for _ in ()).throw(RuntimeError("second"))
    manager._procs.update(one=_as_process(first), two=_as_process(second))

    with pytest.raises(BaseExceptionGroup) as error:
        await manager.shutdown_all()

    messages = [str(exc) for exc in error.value.exceptions]
    assert any("first" in message for message in messages)
    assert any("second" in message for message in messages)


@pytest.mark.asyncio
async def test_process_manager_shutdown_runs_terminations_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requester = SimpleNamespace()
    base: _ConfigBase = {
        "base_url": "http://peer.test",
        "cwd": str(tmp_path),
        "log_dir": str(tmp_path / "logs"),
    }
    manager = PeerProcessManager(
        [
            PeerProcessConfig(name="one", launcher=["python"], **base),
            PeerProcessConfig(name="two", launcher=["python"], **base),
        ],
        _as_requester(requester),
    )
    started: list[str] = []
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def terminate(name: str) -> None:
        started.append(name)
        if len(started) == 2:
            all_started.set()
        await release.wait()

    monkeypatch.setattr(manager, "terminate", terminate)
    shutdown = asyncio.create_task(manager.shutdown_all())
    await asyncio.wait_for(all_started.wait(), timeout=1)
    assert set(started) == {"one", "two"}
    assert not shutdown.done()

    release.set()
    await shutdown
