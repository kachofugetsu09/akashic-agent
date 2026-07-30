import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from akashic_sdk import SlowConsumerError

from benchmark.harbor_v4flash.runtime_driver import _observe_terminal


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
        json.loads(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
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
        json.loads(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
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
