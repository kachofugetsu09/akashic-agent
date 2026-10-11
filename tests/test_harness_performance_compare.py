from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/harness-performance/compare.py"


@pytest.fixture(scope="module")
def compare() -> ModuleType:
    spec = importlib.util.spec_from_file_location("harness_performance_compare", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pairs(compare: ModuleType, key: str, a: list[float], b: list[float]) -> list[tuple[object, object]]:
    return [(compare.Run("A", i, True, "", {key: x}), compare.Run("B", i, True, "", {key: y}))
            for i, (x, y) in enumerate(zip(a, b))]


def _metric(compare: ModuleType, key: str) -> object:
    return next(metric for metric in compare.METRICS if metric.key == key)


def test_consistent_slowdown_is_regression(compare: ModuleType) -> None:
    a = [4.0, 4.4, 3.9, 4.2, 4.1, 4.3]
    row = compare.compare(_pairs(compare, "cu", a, [x * 1.2 for x in a]), _metric(compare, "cu"), 0)
    assert row.verdict == "衰退"
    assert row.low > 1.03 and row.ratio == pytest.approx(1.2)


def test_noise_within_threshold_is_not_reported(compare: ModuleType) -> None:
    a = [4.0, 4.4, 3.9, 4.2, 4.1, 4.3]
    b = [4.04, 4.36, 3.93, 4.17, 4.14, 4.28]
    assert compare.compare(_pairs(compare, "cu", a, b), _metric(compare, "cu"), 0).verdict == "无可分辨变化"


def test_request_volume_reports_small_but_material_growth(compare: ModuleType) -> None:
    metric = _metric(compare, "request_kib")
    jitter = compare.compare(_pairs(compare, "request_kib", [100.0] * 3, [100.03] * 3), metric, 0)
    grown = compare.compare(_pairs(compare, "request_kib", [100.0] * 3, [100.2] * 3), metric, 0)
    assert jitter.verdict == "无可分辨变化"
    assert grown.verdict == "衰退"


def _write_run(out: Path, outputs: list[str], script: Path) -> None:
    """写出基准 runner 与 mock 的文件协议，供既有校验覆盖使用。"""
    run = out / "akashic/20261011-000000"
    run.mkdir(parents=True)
    _ = (run / "run.json").write_text(json.dumps({"status": "ok"}))
    _ = (out / "wait4.json").write_text(json.dumps({"exec_exit": 0, "gateway_exit": 0}))
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Complete the scripted tool work."}]
    rows = [{"bytes": 1, "messages": copy.deepcopy(messages)}]
    responses = json.loads(script.read_text())["responses"]
    for response, text in zip(responses, outputs):
        assistant = copy.deepcopy(response["message"])
        call = assistant["tool_calls"][0]
        call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        messages.extend([assistant, {"role": "tool", "tool_call_id": call["id"],
                                    "content": [{"type": "text", "text": json.dumps({"output": text})}]}])
        rows.append({"bytes": 1, "messages": copy.deepcopy(messages)})
    _ = (out / "wire.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_validate_requires_every_tool_output(compare: ModuleType, tmp_path: Path) -> None:
    script = tmp_path / "script.json"
    _ = script.write_text(json.dumps({"responses": [
        {"message": {"role": "assistant", "content": f"Work for round {n}",
                     "tool_calls": [{"id": f"call_{n}", "type": "function",
                                     "function": {"name": "shell", "arguments": {
                                         "command": f"printf 'bench-tick-{n:04d}\\n'",
                                     }}}]}}
        for n in range(1, 3)
    ]}))
    outputs = ["bench-tick-0001\n", "bench-tick-0002\n"]
    _write_run(tmp_path / "ok", outputs, script)
    _write_run(tmp_path / "skipped", [outputs[0], "bench-tick-0003\n"], script)
    _write_run(tmp_path / "short", outputs[:1], script)
    assert compare.validate(tmp_path / "ok", 2, script) == ""
    assert "第 2 轮" in compare.validate(tmp_path / "skipped", 2, script)
    assert "请求数" in compare.validate(tmp_path / "short", 2, script)

    # 已有覆盖继续检查完整文件输入；最后一个结果正确也不能掩盖历史损坏。
    for case in ("failed", "truncated", "body", "arguments", "call_id", "old_result", "reordered", "prompt"):
        out = tmp_path / case
        _write_run(out, outputs, script)
        wire = out / "wire.jsonl"
        rows = [json.loads(line) for line in wire.read_text().splitlines()]
        messages = rows[-1]["messages"]
        if case == "failed":
            messages[-1]["content"].insert(0, {"type": "text", "text": "工具状态: error"})
        elif case == "truncated":
            rows[-1]["messages"] = messages[-1:]
        elif case == "body":
            messages[-2]["content"] = "truncated body"
        elif case == "arguments":
            messages[-2]["tool_calls"][0]["function"]["arguments"] = '{"command": "true"}'
        elif case == "call_id":
            messages[-1]["tool_call_id"] = "wrong_call"
        elif case == "old_result":
            messages[2]["content"] = "missing output"
        elif case == "reordered":
            messages[1:3] = reversed(messages[1:3])
        else:
            rows[0]["messages"][0]["content"] = "wrong prompt"
        _ = wire.write_text("".join(json.dumps(row) + "\n" for row in rows))
        assert compare.validate(out, 2, script), case
