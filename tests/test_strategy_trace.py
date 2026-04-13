import json
from pathlib import Path
from types import SimpleNamespace

from agent.looping.memory_gate import _trace_memory_retrieve
from core.common.strategy_trace import build_strategy_trace_envelope
from proactive_v2.loop import ProactiveLoop


def test_build_strategy_trace_envelope_uses_subject_scope():
    payload = build_strategy_trace_envelope(
        trace_type="spawn",
        source="agent.spawn",
        subject_kind="job",
        subject_id="abcd1234",
        payload={"status": "completed"},
        timestamp="2026-03-09T00:00:00+00:00",
    )

    assert payload["trace_type"] == "spawn"
    assert payload["source"] == "agent.spawn"
    assert payload["subject"] == {"kind": "job", "id": "abcd1234"}
    assert payload["payload"] == {"status": "completed"}


def test_route_trace_writes_strategy_envelope(tmp_path: Path):
    _trace_memory_retrieve(
        tmp_path,
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        user_msg="你好",
        items=[],
        injected_block="",
    )

    trace_path = tmp_path / "memory" / "memory2_retrieve_trace.jsonl"
    line = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert line["trace_type"] == "route"
    assert line["subject"] == {"kind": "session", "id": "telegram:1"}
    assert line["payload"]["session_key"] == "telegram:1"
    assert line["session_key"] == "telegram:1"


class _ProactiveTraceLoop(ProactiveLoop):
    def __init__(self, workspace: Path) -> None:
        self._sessions = SimpleNamespace(workspace=workspace)
        self._cfg = SimpleNamespace(
            enabled=True,
            threshold=0.7,
            score_llm_threshold=0.6,
            tick_interval_s0=30,
            tick_interval_s1=60,
            tick_interval_s2=120,
            tick_interval_s3=240,
            tick_jitter=0.1,
            anyaction_enabled=True,
            anyaction_min_interval_seconds=60,
            anyaction_probability_min=0.1,
            anyaction_probability_max=0.5,
            memory_history_gate_enabled=True,
        )


def test_proactive_trace_accepts_global_subject(tmp_path: Path):
    loop = _ProactiveTraceLoop(tmp_path)
    loop._trace_proactive_rate_decision(base_score=0.5, interval=60, mode="adaptive")

    trace_path = tmp_path / "memory" / "proactive_rate_trace.jsonl"
    line = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert line["trace_type"] == "proactive_rate"
    assert line["subject"]["kind"] == "global"
    assert line["payload"]["mode"] == "adaptive"
