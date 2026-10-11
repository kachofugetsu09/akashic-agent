from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

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


def _write_run(out: Path, outputs: list[str], *, status: str = "ok") -> None:
    run = out / "akashic/20261011-000000"
    run.mkdir(parents=True)
    _ = (run / "run.json").write_text(json.dumps({"status": status}))
    _ = (out / "wait4.json").write_text(json.dumps({"exec_exit": 0, "gateway_exit": 0}))
    rows = [{"bytes": 1, "messages": [{"role": "user", "content": "go"}]}]
    rows += [{"bytes": 1, "messages": [{"role": "tool", "content": [{"type": "text", "text": json.dumps({"output": text})}]}]}
             for text in outputs]
    _ = (out / "wire.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_validate_requires_every_tool_output(compare: ModuleType, tmp_path: Path) -> None:
    _write_run(tmp_path / "ok", ["bench-tick-0001\n", "bench-tick-0002\n"])
    _write_run(tmp_path / "skipped", ["bench-tick-0001\n", "bench-tick-0003\n"])
    _write_run(tmp_path / "short", ["bench-tick-0001\n"])
    _write_run(tmp_path / "failed", ["bench-tick-0001\n", "bench-tick-0002\n"])
    wire = tmp_path / "failed/wire.jsonl"
    rows = [json.loads(line) for line in wire.read_text().splitlines()]
    rows[1]["messages"][-1]["content"].insert(0, {"type": "text", "text": "工具状态: error"})
    _ = wire.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert compare.validate(tmp_path / "ok", 2) == ""
    assert "第 2 轮" in compare.validate(tmp_path / "skipped", 2)
    assert "请求数" in compare.validate(tmp_path / "short", 2)
    assert "第 1 轮" in compare.validate(tmp_path / "failed", 2)
