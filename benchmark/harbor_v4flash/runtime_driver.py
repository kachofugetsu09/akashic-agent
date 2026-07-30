from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from akashic_sdk import AsyncAkashic

TERMINAL_STATUSES = {"completed", "failed", "interrupted", "cancelled"}


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")


async def _connect(endpoint: str, deadline_s: float) -> AsyncAkashic:
    """在总 deadline 内连接当前 trial 独占的 app-server。"""

    # 1. 只重试尚未 ready 的本地 socket。
    deadline = time.monotonic() + deadline_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            return await AsyncAkashic.connect(endpoint)
        except (ConnectionError, FileNotFoundError, OSError) as error:
            last_error = f"{type(error).__name__}: {error}"
            await asyncio.sleep(0.2)

    # 2. readiness 超时必须让 trial 明确失败。
    raise TimeoutError(f"app-server readiness 超时：{last_error}")


async def _observe_terminal(
    client: Any,
    handle: Any,
    *,
    trace_path: Path,
    turn_timeout_s: float,
    poll_interval_s: float = 0.5,
    terminal_grace_s: float = 5.0,
) -> tuple[dict[str, Any], str, int]:
    """记录事件流，并在 terminal 通知丢失时用权威 turn/read 明示恢复。"""

    # 1. 正常路径逐帧记录；空闲时才读取持久化投影，不干扰事件吞吐。
    deadline = time.monotonic() + turn_timeout_s
    terminal_seen_at: float | None = None
    event_count = 0
    stream = handle.events().__aiter__()
    next_event: asyncio.Task[Any] | None = asyncio.create_task(stream.__anext__())
    try:
        while time.monotonic() < deadline:
            if next_event is not None:
                done, _ = await asyncio.wait(
                    {next_event},
                    timeout=min(poll_interval_s, max(0.0, deadline - time.monotonic())),
                )
                if done:
                    try:
                        event = next_event.result()
                    except StopAsyncIteration:
                        next_event = None
                    else:
                        event_count += 1
                        _append_jsonl(
                            trace_path,
                            {
                                "kind": "sdk_event",
                                "timestamp": time.time(),
                                "event": event,
                            },
                        )
                        if event.get("method") == "turn/completed":
                            params = event.get("params")
                            if isinstance(params, dict) and isinstance(
                                params.get("turn"),
                                dict,
                            ):
                                return params["turn"], "event", event_count
                        next_event = asyncio.create_task(stream.__anext__())
                        continue

            # 2. terminal 已持久化但通知在 grace 内仍未到达时，保留缺口证据并收束。
            persisted = await client.turn_read(handle.thread_id, handle.id)
            status = str(persisted.get("status") or "")
            if status in TERMINAL_STATUSES:
                if terminal_seen_at is None:
                    terminal_seen_at = time.monotonic()
                if next_event is None or time.monotonic() - terminal_seen_at >= terminal_grace_s:
                    _append_jsonl(
                        trace_path,
                        {
                            "kind": "driver",
                            "phase": "terminal_recovered",
                            "timestamp": time.time(),
                            "status": status,
                            "source": "turn/read",
                            "delivery_gap": True,
                            "grace_s": terminal_grace_s,
                        },
                    )
                    return persisted, "turn/read_recovery", event_count
            elif next_event is None:
                raise RuntimeError("SDK 事件流结束但 turn/read 尚未终止")
        raise TimeoutError(f"turn 未在 {turn_timeout_s:.1f}s 内终止")
    finally:
        if next_event is not None and not next_event.done():
            next_event.cancel()
            _ = await asyncio.gather(next_event, return_exceptions=True)


async def run_turn(
    *,
    endpoint: str,
    instruction_path: Path,
    trace_path: Path,
    result_path: Path,
    readiness_timeout_s: float,
    turn_timeout_s: float = 840.0,
) -> dict[str, Any]:
    """通过公开 SDK 完成一轮任务并保存完整事件与持久化投影。"""

    # 1. 连接独占 runtime，并建立本 task 唯一 thread。
    instruction = instruction_path.read_text(encoding="utf-8")
    client = await _connect(endpoint, readiness_timeout_s)
    try:
        thread = await client.thread_start(
            {
                "benchmark": "terminal-bench-2.1",
                "harness": "akasic-v4flash",
            }
        )
        handle = await thread.turn(instruction)
        _append_jsonl(
            trace_path,
            {
                "kind": "driver",
                "phase": "turn_started",
                "timestamp": time.time(),
                "thread_id": thread.id,
                "turn_id": handle.id,
            },
        )

        # 2. 正常记录完整通知；若 terminal delivery 丢失则显式标记并读权威终态。
        terminal, terminal_source, event_count = await _observe_terminal(
            client,
            handle,
            trace_path=trace_path,
            turn_timeout_s=turn_timeout_s,
        )

        # 3. 用公开读取接口核对终态已经持久化，再写结果。
        persisted = await client.turn_read(thread.id, handle.id)
        status = str(terminal.get("status") or "")
        if status not in TERMINAL_STATUSES:
            raise RuntimeError(f"非法 turn 终态：{status!r}")
        if persisted != terminal:
            raise RuntimeError("turn/read 与 terminal event 不一致")
        result = {
            "thread_id": thread.id,
            "turn_id": handle.id,
            "status": status,
            "terminal_source": terminal_source,
            "event_count": event_count,
            "terminal": terminal,
            "persisted": persisted,
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _append_jsonl(
            trace_path,
            {
                "kind": "driver",
                "phase": "turn_persisted",
                "timestamp": time.time(),
                "status": status,
            },
        )
        if status != "completed":
            raise RuntimeError(f"Akasic turn 未正常完成：{status}")
        return result
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--readiness-timeout", type=float, default=180.0)
    parser.add_argument("--turn-timeout", type=float, default=840.0)
    args = parser.parse_args()
    try:
        result = asyncio.run(
            run_turn(
                endpoint=args.endpoint,
                instruction_path=args.instruction_file,
                trace_path=args.trace,
                result_path=args.result,
                readiness_timeout_s=args.readiness_timeout,
                turn_timeout_s=args.turn_timeout,
            )
        )
    except Exception as error:
        _append_jsonl(
            args.trace,
            {
                "kind": "driver_error",
                "timestamp": time.time(),
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        raise
    print(
        json.dumps(
            {
                "thread_id": result["thread_id"],
                "turn_id": result["turn_id"],
                "status": result["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
