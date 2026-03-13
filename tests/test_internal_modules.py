from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.looping.consolidation import (
    AgentLoopConsolidationMixin,
    _build_consolidation_source_ref,
    _format_conversation_for_consolidation,
    _format_pending_items,
    _parse_consolidation_payload,
    _select_consolidation_window,
)
from memory2.post_response_worker import PostResponseMemoryWorker
from proactive.fitbit_sleep import (
    FitbitSleepProvider,
    SleepContext,
    _bootstrap_monitor_data,
    _monitor_wait_ready_and_refresh,
    run_fitbit_monitor,
)
class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class _ConsolidationHarness(AgentLoopConsolidationMixin):
    def __init__(self, payload: str) -> None:
        self._post_mem_failures = 0
        self._consolidating = set()
        self.memory_window = 4
        self._memory_port = SimpleNamespace(
            read_long_term=MagicMock(return_value="MEM"),
            append_history_once=MagicMock(return_value=True),
            append_pending_once=MagicMock(return_value=True),
            save_from_consolidation=AsyncMock(),
        )
        self.provider = SimpleNamespace(chat=AsyncMock(return_value=_Resp(payload)))
        self.model = "lm"
        self.session_manager = SimpleNamespace(save_async=AsyncMock())


@pytest.mark.asyncio
async def test_consolidation_helpers(
    monkeypatch: pytest.MonkeyPatch,
):
    assert _format_pending_items("x") == ""
    assert _format_pending_items(
        [
            {"tag": "preference", "content": "喜欢 A"},
            {"tag": "preference", "content": "喜欢 A"},
            {"tag": "bad", "content": "忽略"},
            "x",
        ]
    ) == "- [preference] 喜欢 A"
    assert _parse_consolidation_payload('{"x":1}') == {"x": 1}

    session = SimpleNamespace(
        key="telegram:1",
        last_consolidated=0,
        messages=[
            {"role": "user", "content": "u1", "timestamp": "2025-01-01T10:00:00"},
            {"role": "assistant", "content": "a1", "timestamp": "2025-01-01T10:01:00"},
            {"role": "tool", "content": "skip", "timestamp": "2025-01-01T10:02:00"},
            {
                "role": "assistant",
                "content": "skip proactive",
                "timestamp": "2025-01-01T10:03:00",
                "proactive": True,
            },
        ],
    )
    assert _select_consolidation_window(session, memory_window=10, archive_all=False) is None
    window = _select_consolidation_window(session, memory_window=4, archive_all=True)
    assert window and window.consolidate_up_to == 4
    assert _build_consolidation_source_ref(session, consolidate_up_to=2, archive_all=False) == "telegram:1@0-2"
    assert _build_consolidation_source_ref(session, consolidate_up_to=2, archive_all=True) == "telegram:1@archive_all"
    assert _format_conversation_for_consolidation(session.messages).count("USER") == 1

    harness = _ConsolidationHarness(
        json.dumps(
            {
                "history_entries": ["[2025-01-01 10:00] 主题A", "[2025-01-01 10:02] 主题B"],
                "pending_items": [{"tag": "preference", "content": "喜欢 A"}],
            }
        )
    )
    scheduled = []
    real_create_task = asyncio.create_task

    def _capture_task(coro):
        task = real_create_task(coro)
        scheduled.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", _capture_task)
    session._channel = "telegram"
    session._chat_id = "1"
    await harness._consolidate_memory(session, archive_all=True)
    if scheduled:
        await asyncio.gather(*scheduled)
    harness._memory_port.append_history_once.assert_called_once()
    harness._memory_port.append_pending_once.assert_called_once()
    assert harness._memory_port.save_from_consolidation.await_count == 2
    assert session.last_consolidated == 0

    scheduled.clear()
    awaited = _ConsolidationHarness(
        json.dumps({"history_entries": ["[2025-01-01 10:00] 主题A"]})
    )
    awaited_session = SimpleNamespace(
        key="telegram:2",
        last_consolidated=0,
        messages=session.messages,
        _channel="telegram",
        _chat_id="2",
    )
    await awaited._consolidate_memory(
        awaited_session,
        archive_all=True,
        await_vector_store=True,
    )
    assert awaited._memory_port.save_from_consolidation.await_count == 1
    assert scheduled and all(task.done() for task in scheduled)

    empty = _ConsolidationHarness("")
    short_session = SimpleNamespace(key="s", messages=[{"role": "user", "content": "u"}], last_consolidated=0)
    await empty._consolidate_memory(short_session)
    assert empty._memory_port.append_history_once.call_count == 0


@pytest.mark.asyncio
async def test_consolidation_background_and_error_accounting(monkeypatch: pytest.MonkeyPatch):
    harness = _ConsolidationHarness('{"history_entry":"[2025-01-01 10:00] old"}')
    harness._consolidating.add("telegram:1")
    session = SimpleNamespace(
        key="telegram:1",
        messages=[{"role": "user", "content": "u1", "timestamp": "2025-01-01T10:00:00"}] * 5,
        last_consolidated=0,
    )
    await harness._consolidate_memory_bg(session, "telegram:1")
    assert "telegram:1" not in harness._consolidating
    harness.session_manager.save_async.assert_awaited_once()

    ok = asyncio.create_task(asyncio.sleep(0))
    await ok
    harness._on_post_mem_task_done(ok, "s1")
    bad = asyncio.create_task(asyncio.sleep(0))
    bad.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bad
    harness._on_post_mem_task_done(bad, "s1")

    class _BrokenTask:
        def exception(self):
            raise RuntimeError("inspect failed")

    harness._on_post_mem_task_done(_BrokenTask(), "s2")
    assert harness._post_mem_failures == 1


@pytest.mark.asyncio
async def test_post_response_worker_extract_save_and_supersede_paths():
    memorizer = SimpleNamespace(
        save_item=AsyncMock(return_value="new:1"),
        supersede_batch=MagicMock(),
    )
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            side_effect=[
                [{"id": "x1", "score": 0.9, "summary": "旧规则"}],
                [{"id": "x1", "score": 0.9, "summary": "旧规则"}],
            ]
        )
    )
    provider = SimpleNamespace(chat=AsyncMock(return_value=_Resp('["topic"]')))
    tagger = SimpleNamespace(tag=AsyncMock(return_value={"scope": "steam"}))
    worker = PostResponseMemoryWorker(memorizer, retriever, provider, "lm", tagger=tagger)

    assert worker._consume_budget(10, 3) == (True, 7)
    assert worker._normalize_text(" A，B ") == "ab"
    assert worker._merge_extracted_items([{"summary": "A"}], [{"summary": "a"}]) == [{"summary": "A"}]
    assert worker._collect_explicit_memorized(
        [{"calls": [{"name": "memorize", "arguments": {"summary": "规则A"}, "result": "已记住（new:AbCDef12_34567890）：规则A"}]}]
    ) == (["规则A"], {"AbCDef12_34567890"})

    topics, remain = await worker._extract_invalidation_topics("你之前这个流程错了", 700)
    assert topics == ["topic"]

    provider.chat = AsyncMock(return_value=_Resp('["x1"]'))
    ids, remain = await worker._check_invalidate("topic", [{"id": "x1", "summary": "旧规则"}], remain)
    assert ids == ["x1"]

    provider.chat = AsyncMock(
        return_value=_Resp('[{"summary":"新规则","memory_type":"procedure","tool_requirement":null,"steps":[]}]')
    )
    items, remain = await worker._extract_implicit("以后这样做", "ok", [], remain)
    assert items[0]["summary"] == "新规则"

    provider.chat = AsyncMock(return_value=_Resp('["x1"]'))
    remain = await worker._save_with_supersede(
        {"summary": "新规则", "memory_type": "procedure", "tool_requirement": None, "steps": []},
        "src",
        token_budget=remain,
    )
    assert remain >= 0
    memorizer.supersede_batch.assert_called_once_with(["x1"])
    kwargs = memorizer.save_item.await_args.kwargs
    assert kwargs["memory_type"] == "procedure"
    assert kwargs["extra"]["trigger_tags"] == {"scope": "steam"}

    provider.chat = AsyncMock(return_value=_Resp('["x1"]'))
    superseded, remain = await worker._check_supersede(
        "新规则",
        [{"id": "x1", "summary": "旧规则"}],
        200,
    )
    assert superseded == ["x1"]


@pytest.mark.asyncio
async def test_post_response_worker_fallback_and_preference_helpers():
    memorizer = SimpleNamespace(save_item=AsyncMock(return_value="new:2"), supersede_batch=MagicMock())
    retriever = SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("boom")))
    provider = SimpleNamespace(chat=AsyncMock(return_value=_Resp("bad json")))
    worker = PostResponseMemoryWorker(memorizer, retriever, provider, "lm")

    topics, remain = await worker._extract_invalidation_topics("也许这个流程不对", 0)
    assert topics == []
    assert remain == 0

    items = await worker._dedupe_against_explicit(
        [{"summary": "以后查 Steam 必须用 MCP", "memory_type": "procedure"}],
        ["以后查 Steam 必须用 MCP 工具"],
        {"x1"},
    )
    assert items == []

    items = await worker._dedupe_against_explicit(
        [{"summary": "回复结尾要主动追问用户最关心的点", "memory_type": "preference"}],
        [],
        set(),
    )
    assert len(items) == 1

@pytest.mark.asyncio
async def test_fitbit_sleep_provider_and_bootstrap_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    assert SleepContext("sleeping", 0.9, "ml", 1, time.time(), True).sleep_modifier == 0.15
    assert SleepContext("uncertain", 0.7, "ml", 5, time.time(), True).sleep_modifier == 0.20
    assert SleepContext("uncertain", 0.2, "ml", 20, time.time(), True).sleep_modifier == 0.50
    assert SleepContext("awake", 0.1, "ml", 1, time.time(), True).sleep_modifier == 1.0
    assert SleepContext("unknown", None, "x", None, time.time(), True).sleep_modifier == 0.88
    assert SleepContext("unknown", None, "x", None, time.time(), False).sleep_modifier == 1.0
    assert SleepContext("unknown", None, "x", None, time.time(), True, health_events=[{"severity": "high"}]).has_urgent_health_event is True

    monkeypatch.setattr("threading.Thread.start", lambda self: None)
    provider = FitbitSleepProvider(url="http://x", poll_interval=60, sleeping_modifier=2)
    provider._cached = SleepContext("awake", 0.1, "ml", 1, 0.0, True)
    assert provider.get().available is False
    provider._cached = SleepContext("sleeping", 0.9, "ml", 1, time.time(), True)
    assert provider.get().state == "sleeping"

    class _Req:
        def __init__(self):
            self.calls = []
            self.fail_post = False

        def get(self, url: str, timeout: float):
            self.calls.append(("get", url, timeout))
            if url.endswith("/api/refresh"):
                raise RuntimeError("refresh failed")
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "sleep": {
                        "state": "sleeping",
                        "prob": 0.9,
                        "prob_source": "ml",
                        "data_lag_min": 2,
                    },
                    "health_events": {"bad": True},
                },
            )

        def post(self, url: str, timeout: float):
            self.calls.append(("post", url, timeout))
            if self.fail_post:
                raise RuntimeError("ack failed")
            return SimpleNamespace(raise_for_status=lambda: None)

    req = _Req()
    monkeypatch.setitem(__import__("sys").modules, "requests", req)
    provider._fetch_once(timeout=1)
    assert provider.get().health_events == []
    assert provider.refresh_now() is True
    req.fail_post = True
    provider.acknowledge_events(["e1"])
    _monitor_wait_ready_and_refresh("http://x", 1)
    assert any(call[0] == "post" for call in req.calls)

    sleep_calls = []
    provider2 = FitbitSleepProvider(url="http://x", poll_interval=5)
    provider2.STARTUP_GRACE_SECONDS = 0
    provider2._fetch_once = MagicMock(side_effect=[RuntimeError("boom"), None])

    def _sleep(sec: float):
        sleep_calls.append(sec)
        if len(sleep_calls) >= 2:
            raise RuntimeError("stop loop")

    monkeypatch.setattr("proactive.fitbit_sleep.time.sleep", _sleep)
    with pytest.raises(RuntimeError, match="stop loop"):
        provider2._loop()
    assert sleep_calls[0] == 5

    monkeypatch.setattr("proactive.fitbit_sleep.asyncio.to_thread", AsyncMock(return_value=None))
    await _bootstrap_monitor_data("http://x", max_wait_sec=1)

    async def _fail_to_thread(*args, **kwargs):
        raise RuntimeError("not ready")

    monkeypatch.setattr("proactive.fitbit_sleep.asyncio.to_thread", _fail_to_thread)
    monkeypatch.setattr("proactive.fitbit_sleep.asyncio.sleep", AsyncMock(return_value=None))
    monotonic_values = [0.0, 0.2, 1.2]
    monkeypatch.setattr(
        "proactive.fitbit_sleep.time.monotonic",
        lambda: monotonic_values.pop(0) if monotonic_values else 1.2,
    )
    await _bootstrap_monitor_data("http://x", max_wait_sec=1)

    assert await run_fitbit_monitor(tmp_path / "missing") is None


@pytest.mark.asyncio
async def test_run_fitbit_monitor_restart_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    server_dir = tmp_path / "monitor"
    server_dir.mkdir()
    (server_dir / "server.py").write_text("print('ok')", encoding="utf-8")

    rc_values = iter([1, 0])
    created = []

    class _Proc:
        def __init__(self, rc: int):
            self._rc = rc

        async def wait(self):
            return self._rc

    async def _create_proc(*args, **kwargs):
        created.append((args, kwargs))
        return _Proc(next(rc_values))

    async def _never_done(*args, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr("proactive.fitbit_sleep.asyncio.create_subprocess_exec", _create_proc)
    monkeypatch.setattr("proactive.fitbit_sleep._bootstrap_monitor_data", _never_done)
    monkeypatch.setattr("proactive.fitbit_sleep.asyncio.sleep", AsyncMock(return_value=None))

    await run_fitbit_monitor(server_dir)

    assert len(created) == 2
    assert (server_dir / "monitor.runtime.log").exists()
