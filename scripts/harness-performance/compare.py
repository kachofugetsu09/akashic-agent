"""在同一台机器上交替运行 A（基线）与 B（候选），按 README 的 harness 性能定义报告是否衰退。

只报告、不判失败：退出码只反映测量本身是否有效（剧本跑完且结果正确）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT = "Complete the scripted tool work."
EXPECTED_OUTPUT = "bench-tick-{:04d}\n"
BOOTSTRAP_ROUNDS = 4000


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    # 小于该相对变化即视为分辨不出（即使置信区间不跨 0）。
    threshold: float
    digits: int


METRICS: tuple[Metric, ...] = (
    Metric("cu", "CU（主指标）", "", 0.03, 3),
    Metric("cpu_s", "CPU 核·秒", "s", 0.03, 3),
    Metric("gib_s", "内存 GiB·秒", "", 0.03, 3),
    Metric("peak_rss_mib", "进程树 RSS 峰值", "MiB", 0.05, 0),
    Metric("startup_s", "启动段", "s", 0.05, 3),
    Metric("ready_cpu_s", "gateway 就绪 CPU", "s", 0.05, 3),
    Metric("turn_ms", "每轮工具循环", "ms", 0.05, 1),
    Metric("tail_s", "收尾段", "s", 0.10, 3),
    Metric("exec_cpu_s", "exec 客户端 CPU", "s", 0.05, 3),
    Metric("request_kib", "模型请求总量", "KiB", 0.001, 1),
)


@dataclass(frozen=True)
class Run:
    side: str
    index: int
    valid: bool
    problem: str
    values: dict[str, float]


@dataclass(frozen=True)
class Comparison:
    metric: Metric
    a: float
    b: float
    ratio: float
    low: float
    high: float
    verdict: str


# 上游生成器产出 100 轮剧本，只把命令换成输出可校验的 printf，正文与体积保持上游原样。
def generate_script(benchmark: Path, turns: int, out: Path) -> None:
    _ = subprocess.run(["bun", "scripts/perf/gen-long-run.ts", "--turns", str(turns), "--tool", "shell",
                        "--args", "command+description", "--out", str(out)],
                       cwd=benchmark, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    script = json.loads(out.read_text())
    tick = 0
    for response in script["responses"]:
        for call in response["message"].get("tool_calls") or []:
            tick += 1
            call["function"]["arguments"]["command"] = f"printf 'bench-tick-{tick:04d}\\n'"
    if tick != turns:
        raise RuntimeError(f"剧本工具调用数 {tick} 与轮数 {turns} 不符")
    _ = out.write_text(json.dumps(script, ensure_ascii=False))


def prepare_side(root: Path, side: Path, benchmark: Path, python: str) -> None:
    if (side / "template").exists():
        return
    if side.exists():
        shutil.rmtree(side)
    _ = subprocess.run([python, str(HERE / "prepare.py"), "--root", str(root), "--side", str(side),
                        "--benchmark", str(benchmark)], check=True)


def install_playground(benchmark: Path) -> None:
    target = benchmark / "playground/akashic"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("perf-demo.ts", "mock.ts"):
        _ = shutil.copyfile(HERE / "playground" / name, target / name)


def measure(benchmark: Path, python: str, root: Path, side: Path, script: Path, out: Path) -> int:
    out.mkdir(parents=True)
    (out / "tool-cwd").mkdir()
    env = {**os.environ, "AK_ROOT": str(root), "AK_SIDE": str(side), "AK_TOOLS": str(HERE), "AK_PYTHON": python}
    with (out / "runner.log").open("wb") as log:
        result = subprocess.run(["bun", "playground/akashic/perf-demo.ts", "--script", str(script),
                                 "--port", "18457", "--interval-ms", "100", "--timeout-ms", "600000",
                                 "--work-dir", str(out / "tool-cwd"), "--out-dir", str(out),
                                 "--prompt", PROMPT], cwd=benchmark, env=env, stdout=log, stderr=log)
    return result.returncode


# 有效运行 = 进程正常退出 + 每个请求都按序收到上一轮工具的精确输出；否则不计入统计。
def validate(out: Path, turns: int) -> str:
    meta = json.loads(next(out.glob("akashic/*/run.json")).read_text())
    usage = json.loads((out / "wait4.json").read_text())
    wire = [json.loads(line) for line in (out / "wire.jsonl").read_text().splitlines()]
    if meta["status"] != "ok" or usage["exec_exit"] != 0 or usage["gateway_exit"] != 0:
        return f"退出异常: status={meta['status']} exec={usage['exec_exit']} gateway={usage['gateway_exit']}"
    if len(wire) != turns + 1:
        return f"请求数 {len(wire)}，应为 {turns + 1}"
    for n, request in enumerate(wire[1:], 1):
        message = request["messages"][-1]
        if message.get("role") != "tool":
            return f"第 {n} 个请求的最后一条不是工具结果: role={message.get('role')!r}"
        if _tool_output(message["content"]) != EXPECTED_OUTPUT.format(n):
            return f"第 {n} 轮工具结果不符: {json.dumps(message['content'], ensure_ascii=False)[:400]}"
    return ""


# 工具消息可能是纯文本或多个 block；失败时前面会有状态 block，只有一个可解析的输出才算成功。
def _tool_output(content: str | list[dict[str, str]]) -> str | None:
    texts = [content] if isinstance(content, str) else [block.get("text", "") for block in content]
    if len(texts) != 1:
        return None
    text = texts[0]
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    return parsed.get("output") if isinstance(parsed, dict) else text


def read_values(out: Path, turns: int) -> dict[str, float]:
    meta = json.loads(next(out.glob("akashic/*/run.json")).read_text())
    usage = json.loads((out / "wait4.json").read_text())
    wire = [json.loads(line) for line in (out / "wire.jsonl").read_text().splitlines()]
    timing = meta["timing"]
    gib_s = float(meta["cost"]["gbSeconds"])
    return {
        "cu": float(usage["cpu_s"]) + gib_s,
        "cpu_s": float(usage["cpu_s"]),
        "gib_s": gib_s,
        "peak_rss_mib": meta["peaks"]["treeRssBytes"] / 2**20,
        "startup_s": (timing["firstRequestAtMs"] - timing["harnessStartedAtMs"]) / 1000,
        "ready_cpu_s": float(usage["ready_cpu_s"]),
        "turn_ms": (timing["lastRequestAtMs"] - timing["firstRequestAtMs"]) / turns,
        "tail_s": (timing["harnessExitedAtMs"] - timing["lastRequestAtMs"]) / 1000,
        "exec_cpu_s": float(usage["exec_cpu_s"]),
        "request_kib": sum(int(row["bytes"]) for row in wire) / 1024,
    }


def run_once(args: argparse.Namespace, side: str, index: int, script: Path) -> Run:
    root, home = (args.a_root, args.work / "side-a") if side == "A" else (args.b_root, args.work / "side-b")
    out = args.work / "runs" / f"{index:02d}-{side}"
    code = measure(args.benchmark, args.python, root, home, script, out)
    wire = out / "wire.jsonl"
    if wire.exists():
        with wire.open() as rows:
            _ = (out / "wire-head.jsonl").write_text("".join(next(rows, "") for _ in range(3)))
    try:
        problem = validate(out, args.turns) if code == 0 else f"runner 退出码 {code}"
        values = read_values(out, args.turns) if not problem else {}
    except (OSError, StopIteration, KeyError, ValueError) as error:
        problem, values = f"结果缺失: {error!r}", {}
    print(f"[harness-perf] {out.name}: {problem or 'ok'} {values.get('cu', 0):.3f} CU", flush=True)
    return Run(side=side, index=index, valid=not problem, problem=problem, values=values)


# Hodges-Lehmann 估计：配对 log(B/A) 的 Walsh 平均取中位数；小样本下比中位数有效，仍抗离群。
def _hodges_lehmann(logs: Sequence[float]) -> float:
    return statistics.median((x + y) / 2 for i, x in enumerate(logs) for y in logs[i:])


# 变化量取 HL 估计，置信区间对配对重抽样。
def compare(pairs: Sequence[tuple[Run, Run]], metric: Metric, seed: int) -> Comparison:
    a_values = [a.values[metric.key] for a, _ in pairs]
    b_values = [b.values[metric.key] for _, b in pairs]
    a, b = statistics.median(a_values), statistics.median(b_values)
    logs = [math.log(y / x) for x, y in zip(a_values, b_values)]
    rng = random.Random(seed)
    boot = sorted(_hodges_lehmann(rng.choices(logs, k=len(logs))) for _ in range(BOOTSTRAP_ROUNDS))
    low = math.exp(boot[int(0.025 * BOOTSTRAP_ROUNDS)])
    high = math.exp(boot[int(0.975 * BOOTSTRAP_ROUNDS) - 1])
    if low > 1 + metric.threshold:
        verdict = "衰退"
    elif high < 1 - metric.threshold:
        verdict = "改善"
    else:
        verdict = "无可分辨变化"
    return Comparison(metric=metric, a=a, b=b, ratio=math.exp(_hodges_lehmann(logs)), low=low, high=high,
                      verdict=verdict)


def _percent(value: float) -> str:
    change = (value - 1) * 100
    return f"{change:+.1f}%" if abs(change) >= 0.1 or change == 0 else f"{change:+.3f}%"


def render(rows: Sequence[Comparison], runs: Sequence[Run], args: argparse.Namespace, valid_pairs: int) -> str:
    regressions = [row.metric.label for row in rows if row.verdict == "衰退"]
    headline = (f"**发现衰退**：{'、'.join(regressions)}" if regressions
                else "**未发现可分辨的衰退**")
    lines = [
        "<!-- harness-performance -->",
        "## Harness 性能对比（只报告）",
        "",
        f"{headline}。A = `{args.a_label}`，B = `{args.b_label}`；"
        f"{valid_pairs} 组有效交替配对，{args.turns} 轮工具调用剧本（harness-perf-benchmark `{args.benchmark_rev}`）。",
        "",
        "| 指标 | A 中位 | B 中位 | 变化（配对 HL 估计） | 95% 置信区间 | 判定 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        unit, d = row.metric.unit, row.metric.digits
        mark = {"衰退": "🔴 衰退", "改善": "🟢 改善"}.get(row.verdict, "⚪ 无可分辨变化")
        lines.append(f"| {row.metric.label} | {row.a:.{d}f}{unit} | {row.b:.{d}f}{unit} | {_percent(row.ratio)} "
                     f"| {_percent(row.low)} ~ {_percent(row.high)} | {mark} |")
    invalid = [run for run in runs if not run.valid]
    lines += ["", "判定：置信区间整体超出该指标的分辨阈值（CU/CPU/内存 3%，分段 5%，收尾 10%，"
              "请求总量 0.1%）才算衰退或改善。定义与口径见 "
              "`scripts/harness-performance/README.md`。"]
    if invalid:
        lines += ["", "无效运行（未计入）：" + "；".join(f"{r.index:02d}-{r.side} {r.problem}" for r in invalid)]
    lines += ["", f"<sub>runner: {platform.platform()}，{os.cpu_count()} CPU，Python {platform.python_version()}</sub>"]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--benchmark", type=Path, required=True, help="harness-perf-benchmark checkout")
    _ = parser.add_argument("--benchmark-rev", default="unknown")
    _ = parser.add_argument("--a-root", type=Path, required=True)
    _ = parser.add_argument("--b-root", type=Path, required=True)
    _ = parser.add_argument("--a-label", default="base")
    _ = parser.add_argument("--b-label", default="head")
    _ = parser.add_argument("--work", type=Path, required=True)
    _ = parser.add_argument("--pairs", type=int, default=8)
    _ = parser.add_argument("--turns", type=int, default=100)
    _ = parser.add_argument("--python", default=sys.executable)
    _ = parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("benchmark", "a_root", "b_root", "work", "report"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    return args


def run_pairs(count: int, step: Callable[[str, int], Run]) -> list[tuple[Run, Run]]:
    # 1. 每侧先空跑一次，让插件 home 的宿主缓存进入稳定状态（结果丢弃）。
    for side in ("A", "B"):
        _ = step(side, 0)
    # 2. 按 AB、BA 交替排序，抵消机器状态随时间的漂移。
    pairs: list[tuple[Run, Run]] = []
    for index in range(1, count + 1):
        order = ("A", "B") if index % 2 else ("B", "A")
        first, second = (step(side, index) for side in order)
        a, b = (first, second) if order[0] == "A" else (second, first)
        if a.valid and b.valid:
            pairs.append((a, b))
    return pairs


def main() -> int:
    args = parse_args(sys.argv[1:])
    if (args.work / "runs").exists():
        shutil.rmtree(args.work / "runs")
    args.work.mkdir(parents=True, exist_ok=True)
    install_playground(args.benchmark)
    script = args.work / "script.json"
    generate_script(args.benchmark, args.turns, script)
    prepare_side(args.a_root, args.work / "side-a", args.benchmark, args.python)
    prepare_side(args.b_root, args.work / "side-b", args.benchmark, args.python)
    runs: list[Run] = []

    def step(side: str, index: int) -> Run:
        run = run_once(args, side, index, script)
        if index:
            runs.append(run)
        return run

    pairs = run_pairs(args.pairs, step)
    if len(pairs) < 3:
        _ = args.report.write_text("<!-- harness-performance -->\n## Harness 性能对比\n\n有效配对不足 3 组，"
                                   "无法给出结论：\n\n" + "\n".join(f"- {r.index:02d}-{r.side}: {r.problem}"
                                                                 for r in runs if not r.valid) + "\n")
        return 1
    rows = [compare(pairs, metric, seed) for seed, metric in enumerate(METRICS)]
    _ = args.report.write_text(render(rows, runs, args, len(pairs)))
    _ = (args.work / "results.json").write_text(json.dumps({
        "pairs": [[asdict(a), asdict(b)] for a, b in pairs],
        "comparisons": [{**asdict(row), "metric": row.metric.key} for row in rows],
    }, ensure_ascii=False, indent=2))
    print(args.report.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
