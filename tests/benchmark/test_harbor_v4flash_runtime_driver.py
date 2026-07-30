import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.harbor_v4flash.runtime_driver import _observe_terminal


class _HandleWithoutTerminalEvent:
    thread_id = "programmatic:test"
    id = "turn:test"

    async def events(self):
        yield {"method": "turn/started", "params": {"turnId": self.id}}
        await asyncio.Event().wait()


class _PersistedTerminalClient:
    async def turn_read(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        assert thread_id == _HandleWithoutTerminalEvent.thread_id
        assert turn_id == _HandleWithoutTerminalEvent.id
        return {
            "id": turn_id,
            "threadId": thread_id,
            "status": "completed",
        }


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
