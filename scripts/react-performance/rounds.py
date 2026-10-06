"""对齐真实代理、客户端与 runtime.timing，输出请求间隔和未归因时间。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def rows(path: Path, *, client_events: bool = False) -> list[dict[str, Any]]:
    """逐行读取；跳过未完成行，客户端只保留计时所需事件。"""
    result = []
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n") or not line.startswith("{"):
                continue
            row = json.loads(line)
            event = row.get("event", {})
            if client_events and not (
                row.get("bench") == "input" or event.get("bench") == "input"
                or event.get("type") in {
                    "messages.appended", "session.following", "message_end", "agent_end",
                    "tool_execution_start", "tool_execution_end",
                }
            ):
                continue
            result.append(row)
    return result


def merged_ms(intervals: list[tuple[int, int]]) -> float:
    """并行工具计算区间并集，不能把重叠 wall time 相加。"""
    end = 0
    total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total / 1e6


def analyze(
    wire: list[dict[str, Any]],
    events: list[dict[str, Any]],
    timing: list[dict[str, Any]],
    label: str,
    harness: str,
) -> dict[str, Any]:
    """以同机 monotonic_ns 对齐；缺少边界时保留 unknown，不填零。"""
    inputs = [
        row for row in events if row.get("bench") == "input" or row.get("event", {}).get("bench") == "input"
    ]
    if len(inputs) != 1:
        raise ValueError("每份客户端记录必须恰好有一次 benchmark input")
    started = inputs[0]["mono_ns"]
    session = inputs[0].get("event", {}).get("session_id")
    events = [row for row in events if row["mono_ns"] >= started]
    wire = [row for row in wire if row["label"] == label and row["mono_ns"] >= started]
    timing = [
        row
        for row in timing
        if row.get("event") == "runtime.timing"
        and (not row.get("session_id") or row["session_id"] == session)
    ]
    requests = [row for row in wire if row["event"] == "request.received"]
    output_times: list[int] = []
    final_received: int | None = None
    tools: list[tuple[int, int]] = []
    committed: list[int] = []
    tool_starts: dict[str, int] = {}
    physical_starts: dict[str, int] = {}
    physical: list[tuple[int, int]] = []
    if harness == "akashic":
        input_id = inputs[0].get("event", {}).get("message_id")
        input_seq = next((item["seq"] for row in events for item in row.get("event", {}).get("items", [])
                          if item.get("id") == input_id and item.get("body", {}).get("kind") == "input"), None)
        if input_seq is None:
            raise ValueError("Akashic input 标记须带 message_id，并观察到对应已提交 Input")
        outputs: dict[str, int] = {}
        for row in timing:
            at = row["measurement_value"]
            if at < started:
                continue
            phase, key = row["phase"], row.get("operation_id", "")
            if phase == "output.committed":
                outputs[row["parent_operation_id"]] = at
            elif phase == "tool.task.begin":
                tool_starts[key] = at
            elif phase == "tool.invoke.begin":
                physical_starts[key] = at
            elif phase in {"tool.invoke.end", "tool.invoke.failed"} and key in physical_starts:
                physical.append((physical_starts.pop(key), at))
            elif phase == "tool.committed":
                committed.append(at)
                if key in tool_starts:
                    tools.append((tool_starts.pop(key), at))
        for row in events:
            for item in row.get("event", {}).get("items", []):
                body = item.get("body", {})
                if body.get("kind") == "output" and item["seq"] > input_seq:
                    # 日志订阅可先于 append await 返回醒来；两者均证明已经落库。
                    outputs[item["id"]] = min(outputs.get(item["id"], row["mono_ns"]), row["mono_ns"])
                    if body.get("finish") == "complete":
                        final_received = min(final_received or row["mono_ns"], row["mono_ns"])
        output_times = sorted(outputs.values())
    else:
        for row in events:
            event, at = row.get("event", {}), row["mono_ns"]
            kind, key = event.get("type"), event.get("toolCallId", "")
            if kind == "message_end" and event.get("message", {}).get("role") == "assistant":
                if event["message"].get("stopReason") not in {"error", "aborted"}:
                    output_times.append(at)
            elif kind == "tool_execution_start":
                tool_starts[key] = at
            elif kind == "tool_execution_end" and key in tool_starts:
                tools.append((tool_starts.pop(key), at))
            elif kind == "message_end" and event.get("message", {}).get("role") == "toolResult":
                committed.append(at)
            elif kind == "agent_end":
                messages = event.get("messages", [])
                assistants = [m for m in messages if m.get("role") == "assistant"]
                if assistants and assistants[-1].get("stopReason") == "stop":
                    final_received = min(final_received or at, at)
        physical = tools[:]
    output_times = [t for t in output_times if final_received is None or t <= final_received]
    result = []
    requests = [row for row in requests if final_received is None or row["mono_ns"] <= final_received]
    logical_round = 1
    for index, request in enumerate(requests):
        start = request["mono_ns"]
        stop = requests[index + 1]["mono_ns"] if index + 1 < len(requests) else final_received
        marks = {row["event"]: row for row in wire if row["n"] == request["n"]}
        first = marks.get("provider.first_delta", {}).get("mono_ns")
        done = marks.get("provider.done", {}).get("mono_ns")
        failed = marks.get("transport.failed", {}).get("mono_ns")
        disconnected = marks.get("client.disconnected", {}).get("mono_ns")
        failed = failed if failed is not None else disconnected
        http_status = marks.get("upstream.headers", {}).get("status")
        if failed is None and http_status is not None and http_status >= 400:
            failed = marks.get("response.closed", marks["upstream.headers"])["mono_ns"]
        output = next((t for t in output_times if t >= start and (stop is None or t <= stop)), None)
        batch = [(a, b) for a, b in tools if a >= start and (stop is None or b <= stop)]
        last_commit = max((t for t in committed if t >= start and (stop is None or t <= stop)), default=None)
        first_tool = min((a for a, _b in batch), default=None)
        entry: dict[str, Any] = {
            "request": request["n"],
            "round": logical_round,
            "input_bytes": request["bytes"],
            "reasoning_rows": request["reasoning_rows"],
            "http_status": http_status,
            "status": "output" if output is not None else "disconnected" if disconnected else "failed" if failed else "incomplete",
        }
        segments: list[tuple[str, int | None, int | None]] = [
            ("wait_first_delta_ms", start, first),
            ("generation_ms", first, done),
            ("response_commit_ms", done, output),
        ]
        if batch:
            segments += [
                ("tool_admission_ms", output, first_tool),
                ("tool_batch_ms", first_tool, last_commit),
                ("next_prepare_ms", last_commit, stop),
            ]
        elif output is not None:
            segments.append(("delivery_or_next_ms", output, stop))
        elif failed is not None:
            segments = [("failed_attempt_ms", start, failed), ("retry_wait_ms", failed, stop)]
        known: list[tuple[int, int]] = []
        for name, a, b in segments:
            entry[name] = None
            if a is not None and b is not None and start <= a <= b and (stop is None or b <= stop):
                entry[name] = round((b - a) / 1e6, 3)
                known.append((a, b))
        entry["request_interval_ms"] = None if stop is None else round((stop - start) / 1e6, 3)
        entry["unattributed_ms"] = None if stop is None else round((stop - start) / 1e6 - merged_ms(known), 3)
        entry["tool_invoke_union_ms"] = round(
            merged_ms([(a, b) for a, b in physical if a >= start and (stop is None or b <= stop)]), 3
        )
        entry["usage"] = marks.get("provider.usage", {}).get("usage")
        result.append(entry)
        if output is not None:
            logical_round += 1
    tenth = output_times[9] if len(output_times) >= 10 else None
    return {
        "harness": harness,
        "label": label,
        "completed_model_rounds": len(output_times),
        "provider_attempts": len(requests),
        "final_received": final_received is not None,
        "input_to_first_request_ms": None if not requests else (requests[0]["mono_ns"] - started) / 1e6,
        "input_to_tenth_output_ms": None if tenth is None else (tenth - started) / 1e6,
        "input_to_final_output_ms": None if final_received is None else (final_received - started) / 1e6,
        "rounds": result,
    }


def main() -> None:
    """只读原始证据；新建 JSON 与逐轮 CSV，不覆盖已有报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--harness", choices=("akashic", "pi"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        rows(args.wire),
        rows(args.events, client_events=True),
        [] if args.timing is None else rows(args.timing),
        args.label,
        args.harness,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    flat = [{k: v for k, v in row.items() if k != "usage"} for row in result["rounds"]]
    if flat:
        fields = list(dict.fromkeys(key for row in flat for key in row))
        with args.output.with_suffix(".csv").open("x") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
    print(json.dumps({key: value for key, value in result.items() if key != "rounds"}))


if __name__ == "__main__":
    main()
