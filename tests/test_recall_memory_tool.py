import json
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

import agent.tools.recall_memory as recall_memory_module
from agent.tools.recall_memory import RecallMemoryTool
from core.memory.engine import (
    EvidenceRef,
    MemoryQueryResult,
    MemoryRecord,
    MemoryToolSpec,
)


class _CaptureMemory:
    request = None

    async def query(self, request):
        self.request = request
        return MemoryQueryResult()


@pytest.mark.asyncio
async def test_recall_memory_passes_current_timestamp_to_engine() -> None:
    memory = _CaptureMemory()
    tool = RecallMemoryTool(
        cast(Any, memory),
        MemoryToolSpec(description="", parameters={"type": "object", "properties": {}}),
    )
    timestamp = datetime(2026, 4, 4, 22, 0, 0)

    _ = await tool.execute(query="Akasha", current_timestamp=timestamp.isoformat())

    assert memory.request.timestamp == timestamp


def test_parse_time_filter_supports_presets_and_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timezone = ZoneInfo("Asia/Shanghai")
    monkeypatch.setattr(
        recall_memory_module,
        "_now_local",
        lambda: datetime(2026, 4, 25, 15, 30, tzinfo=timezone),
    )

    assert recall_memory_module._parse_time_filter("today") == (
        datetime(2026, 4, 25, 0, 0, tzinfo=timezone),
        datetime(2026, 4, 26, 0, 0, tzinfo=timezone),
    )
    assert recall_memory_module._parse_time_filter("recent_3d") == (
        datetime(2026, 4, 22, 15, 30, tzinfo=timezone),
        datetime(2026, 4, 25, 15, 30, tzinfo=timezone),
    )
    assert recall_memory_module._parse_time_filter("2026-04-20") == (
        datetime(2026, 4, 20, 0, 0, tzinfo=timezone),
        datetime(2026, 4, 21, 0, 0, tzinfo=timezone),
    )
    assert recall_memory_module._parse_time_filter("2026-04-20~2026-04-25") == (
        datetime(2026, 4, 20, 0, 0, tzinfo=timezone),
        datetime(2026, 4, 26, 0, 0, tzinfo=timezone),
    )


def test_recall_memory_response_preserves_activation_metadata() -> None:
    payload = json.loads(
        recall_memory_module._render_records(
            [
                MemoryRecord(
                    id="mem:1",
                    kind="event",
                    summary="用户提到 Falcons 比赛",
                    score=0.704,
                    engine_kind="akasha",
                    evidence=[EvidenceRef(refs=["msg:1"], source_ref="msg:1")],
                    signals={
                        "cosine": 0.81,
                        "lambda_before": 0.2,
                        "lambda_after": 0.9,
                        "activation": 0.9,
                        "activated": True,
                    },
                )
            ],
            trace={},
        )
    )

    assert payload["items"][0]["signals"] == {
        "cosine": 0.81,
        "lambda_before": 0.2,
        "lambda_after": 0.9,
        "activation": 0.9,
        "activated": True,
    }
