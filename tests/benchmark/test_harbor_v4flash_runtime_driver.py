import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from akashic_sdk import SlowConsumerError

from benchmark.harbor_v4flash.runtime_driver import (
    AgentTurnFailed,
    ProviderAccountLimited,
    ProviderRateLimited,
    ProviderTransientFailure,
    TurnDeadlineExceeded,
    _driver_error_status,
    _observe_terminal,
    _turn_was_empty_provider_response,
    _turn_was_account_limited,
    _turn_was_rate_limited,
    _turn_was_transient_provider_failure,
)


class _HandleWithoutTerminalEvent:
    thread_id = "programmatic:test"
    id = "turn:test"

    async def events(self):
        yield {"method": "turn/started", "params": {"turnId": self.id}}
        await asyncio.Event().wait()


class _PersistedTerminalClient:
    async def turn_read(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        return {
            "id": turn_id,
            "threadId": thread_id,
            "status": "completed",
        }


class _DelayedPersistedTerminalClient(_PersistedTerminalClient):
    async def turn_read(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return await super().turn_read(thread_id, turn_id)


class _BurstHandle:
    thread_id = "programmatic:burst"
    id = "turn:burst"

    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(512)

    async def events(self):
        yield {"method": "turn/started", "params": {"turnId": self.id}}
        while True:
            event = await self.queue.get()
            yield event
            if event.get("method") == "turn/completed":
                return


class _BurstBeforeReadResponseClient:
    def __init__(self, handle: _BurstHandle) -> None:
        self.handle = handle
        self.terminal = {
            "id": handle.id,
            "threadId": handle.thread_id,
            "status": "completed",
        }

    async def turn_read(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        assert thread_id == self.handle.thread_id
        assert turn_id == self.handle.id
        for index in range(600):
            try:
                self.handle.queue.put_nowait(
                    {
                        "method": "turn/output_delta",
                        "params": {"turnId": turn_id, "delta": str(index)},
                    }
                )
            except asyncio.QueueFull as error:
                raise SlowConsumerError("turn notification queue overflow") from error
            await asyncio.sleep(0)
        self.handle.queue.put_nowait(
            {
                "method": "turn/completed",
                "params": {"turnId": turn_id, "turn": self.terminal},
            }
        )
        await asyncio.sleep(0)
        return self.terminal


class _TerminalThenReadErrorClient:
    def __init__(self, handle: _BurstHandle) -> None:
        self.handle = handle

    async def turn_read(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        self.handle.queue.put_nowait(
            {
                "method": "turn/completed",
                "params": {
                    "turnId": turn_id,
                    "turn": {
                        "id": turn_id,
                        "threadId": thread_id,
                        "status": "completed",
                    },
                },
            }
        )
        await asyncio.sleep(0)
        raise RuntimeError("turn read failed")


class _EndedStreamHandle:
    thread_id = "programmatic:ended"
    id = "turn:ended"

    async def events(self):
        yield {"method": "turn/started", "params": {"turnId": self.id}}


class _FailingStreamHandle:
    thread_id = "programmatic:failed-stream"
    id = "turn:failed-stream"

    async def events(self):
        yield {"method": "turn/started", "params": {"turnId": self.id}}
        raise RuntimeError("event stream failed")


class _CancellingHandle:
    thread_id = "programmatic:cancel"
    id = "turn:cancel"

    def __init__(self) -> None:
        self.closed = asyncio.Event()

    async def events(self):
        try:
            yield {"method": "turn/started", "params": {"turnId": self.id}}
            await asyncio.Event().wait()
        finally:
            self.closed.set()


class _NeverReturningClient:
    async def turn_read(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def test_driver_error_status_keeps_readiness_failure_out_of_valid_timeout() -> None:
    assert _driver_error_status(TimeoutError("readiness")) == "infra_failed"
    assert _driver_error_status(TurnDeadlineExceeded("budget")) == "timed_out"
    assert _driver_error_status(AgentTurnFailed("turn")) == "agent_failed"
    assert _driver_error_status(ProviderRateLimited("429")) == "rate_limited"
    assert _driver_error_status(ProviderTransientFailure("500")) == "provider_transient"
    assert _driver_error_status(ProviderAccountLimited("quota")) == "account_limited"


def test_rate_limit_detection_only_reads_structured_turn_error() -> None:
    assert _turn_was_rate_limited(
        {
            "status": "failed",
            "input": "unrelated",
            "error": {
                "type": "RateLimitError",
                "message": "Error code: 429 - Too Many Requests",
                "retryable": True,
            },
        }
    )
    assert not _turn_was_rate_limited(
        {
            "status": "failed",
            "input": "please explain HTTP 429 rate limits",
            "error": {"type": "RuntimeError", "message": "tool failed"},
        }
    )


def test_provider_transient_detection_requires_provider_type_and_explicit_5xx() -> None:
    assert _turn_was_transient_provider_failure(
        {
            "error": {
                "type": "InternalServerError",
                "message": "Error code: 500 - Router.Unavailable",
            }
        }
    )
    assert _turn_was_transient_provider_failure(
        {
            "error": {
                "type": "provider_error",
                "message": (
                    "Error code: 500 - {'type': 'Router.Unavailable', "
                    "'modelID': 'deepseek-v4-flash'}"
                ),
                "retryable": True,
            }
        }
    )
    assert not _turn_was_transient_provider_failure(
        {
            "error": {
                "type": "RuntimeError",
                "message": "tool returned status code: 500",
            }
        }
    )


def test_provider_transient_detection_accepts_incomplete_response_body() -> None:
    assert _turn_was_transient_provider_failure(
        {
            "error": {
                "type": "RemoteProtocolError",
                "message": (
                    "peer closed connection without sending complete message body "
                    "(incomplete chunked read)"
                ),
                "retryable": False,
            }
        }
    )
    assert not _turn_was_transient_provider_failure(
        {
            "error": {
                "type": "RuntimeError",
                "message": "incomplete chunked read",
            }
        }
    )


def test_empty_provider_response_requires_runtime_fallback_without_tool_calls() -> None:
    terminal = {
        "status": "completed",
        "finalResponse": "模型未返回可用回复，请重试。",
        "error": None,
        "items": [
            {"type": "userMessage", "data": {"content": "task"}},
            {
                "type": "assistantMessage",
                "data": {
                    "content": "模型未返回可用回复，请重试。",
                    "metadata": {"streamed_reply": False},
                },
            },
        ],
    }

    assert _turn_was_empty_provider_response(terminal)
    terminal["items"].insert(1, {"type": "toolCall", "data": {"name": "shell"}})
    assert not _turn_was_empty_provider_response(terminal)


def test_go_usage_limit_is_not_treated_as_ordinary_rate_limit() -> None:
    turn = {
        "error": {
            "type": "RateLimitError",
            "message": "GoUsageLimitError: 5 hour usage limit reached",
        }
    }

    assert _turn_was_account_limited(turn)
    assert _turn_was_rate_limited(turn)


@pytest.mark.asyncio
async def test_observer_recovers_persisted_terminal_after_delivery_gap(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"

    terminal, source, event_count = await _observe_terminal(
        _PersistedTerminalClient(),
        _HandleWithoutTerminalEvent(),
        trace_path=trace,
        turn_timeout_s=1,
        poll_interval_s=0.001,
        terminal_grace_s=0.003,
    )

    records = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    assert terminal["status"] == "completed"
    assert source == "turn/read_recovery"
    assert event_count == 1
    assert records[-1]["phase"] == "terminal_recovered"
    assert records[-1]["delivery_gap"] is True


@pytest.mark.asyncio
async def test_observer_continuously_drains_burst_while_turn_read_is_pending(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    handle = _BurstHandle()

    terminal, source, event_count = await _observe_terminal(
        _BurstBeforeReadResponseClient(handle),
        handle,
        trace_path=trace,
        turn_timeout_s=1,
        poll_interval_s=0.001,
    )

    records = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    assert terminal["status"] == "completed"
    assert source == "event"
    assert event_count == 602
    assert [record["event"]["method"] for record in records] == [
        "turn/started",
        *(["turn/output_delta"] * 600),
        "turn/completed",
    ]


@pytest.mark.asyncio
async def test_observer_recovers_terminal_after_event_stream_ends(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"

    terminal, source, event_count = await _observe_terminal(
        _DelayedPersistedTerminalClient(),
        _EndedStreamHandle(),
        trace_path=trace,
        turn_timeout_s=1,
        poll_interval_s=0.001,
    )

    assert terminal["status"] == "completed"
    assert source == "turn/read_recovery"
    assert event_count == 1


@pytest.mark.asyncio
async def test_observer_preserves_event_stream_error_priority(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="event stream failed"):
        await _observe_terminal(
            _PersistedTerminalClient(),
            _FailingStreamHandle(),
            trace_path=tmp_path / "trace.jsonl",
            turn_timeout_s=1,
            poll_interval_s=0.001,
        )


@pytest.mark.asyncio
async def test_observer_preserves_turn_read_error_priority(tmp_path: Path) -> None:
    handle = _BurstHandle()

    with pytest.raises(RuntimeError, match="turn read failed"):
        await _observe_terminal(
            _TerminalThenReadErrorClient(handle),
            handle,
            trace_path=tmp_path / "trace.jsonl",
            turn_timeout_s=1,
            poll_interval_s=0.001,
        )


@pytest.mark.asyncio
async def test_observer_cancellation_closes_event_drain(tmp_path: Path) -> None:
    handle = _CancellingHandle()
    observer = asyncio.create_task(
        _observe_terminal(
            _NeverReturningClient(),
            handle,
            trace_path=tmp_path / "trace.jsonl",
            turn_timeout_s=10,
            poll_interval_s=0.001,
        )
    )
    await asyncio.sleep(0.01)

    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer

    assert handle.closed.is_set()
