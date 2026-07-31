from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from akashic_sdk import AsyncAkashic

TERMINAL_STATUSES = {"completed", "failed", "interrupted", "cancelled"}


@dataclass(slots=True)
class _EventDrainState:
    event_count: int = 0


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


async def _drain_events(
    handle: Any,
    *,
    trace_path: Path,
    state: _EventDrainState,
) -> dict[str, Any] | None:
    """持续记录事件流，并返回其中唯一的 terminal turn。"""

    # 1. 按 SDK 到达顺序持续消费并封存每一帧。
    async for event in handle.events():
        state.event_count += 1
        _append_jsonl(
            trace_path,
            {
                "kind": "sdk_event",
                "timestamp": time.time(),
                "event": event,
            },
        )

        # 2. terminal 仍由协议事件唯一提交；普通流结束留给 turn/read 判定。
        if event.get("method") == "turn/completed":
            params = event.get("params")
            if isinstance(params, dict) and isinstance(params.get("turn"), dict):
                return params["turn"]
    return None


def _recovered_terminal(
    persisted: dict[str, Any],
    *,
    trace_path: Path,
    terminal_grace_s: float,
    event_count: int,
) -> tuple[dict[str, Any], str, int]:
    _append_jsonl(
        trace_path,
        {
            "kind": "driver",
            "phase": "terminal_recovered",
            "timestamp": time.time(),
            "status": str(persisted.get("status") or ""),
            "source": "turn/read",
            "delivery_gap": True,
            "grace_s": terminal_grace_s,
        },
    )
    return persisted, "turn/read_recovery", event_count


async def _wait_for_drain(
    drain_task: asyncio.Task[dict[str, Any] | None],
    timeout_s: float,
) -> tuple[bool, dict[str, Any] | None]:
    done, _ = await asyncio.wait({drain_task}, timeout=max(0.0, timeout_s))
    if not done:
        return False, None
    return True, drain_task.result()


async def _read_while_draining(
    client: Any,
    handle: Any,
    *,
    drain_task: asyncio.Task[dict[str, Any] | None],
    deadline: float,
    timeout_message: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """并发读取持久化投影，并让事件 drain 始终保持活跃。"""

    # 1. 发出一次读取；drain 先结束时仍保留读取直到 deadline。
    read_task = asyncio.create_task(client.turn_read(handle.thread_id, handle.id))
    try:
        while True:
            if drain_task.done():
                terminal = drain_task.result()
                if terminal is not None:
                    return terminal, None
            waiters = {read_task}
            if not drain_task.done():
                waiters.add(drain_task)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(timeout_message)

            # 2. 同时完成时先暴露异常；两者成功则以 terminal event 收束。
            done, _ = await asyncio.wait(
                waiters,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError(timeout_message)
            terminal = None
            if drain_task in done:
                terminal = drain_task.result()
            if read_task in done:
                persisted = read_task.result()
                if terminal is not None:
                    return terminal, None
                return None, persisted
            if terminal is not None:
                return terminal, None
    finally:
        if not read_task.done():
            read_task.cancel()
            _ = await asyncio.gather(read_task, return_exceptions=True)
        elif not read_task.cancelled():
            _ = read_task.exception()


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

    # 1. 独立 drain 在所有 turn/read 和 grace 等待期间持续消费事件。
    deadline = time.monotonic() + turn_timeout_s
    timeout_message = f"turn 未在 {turn_timeout_s:.1f}s 内终止"
    terminal_seen_at: float | None = None
    state = _EventDrainState()
    drain_task = asyncio.create_task(
        _drain_events(handle, trace_path=trace_path, state=state)
    )
    try:
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            drain_done, terminal = await _wait_for_drain(
                drain_task,
                min(poll_interval_s, remaining),
            )
            if terminal is not None:
                return terminal, "event", state.event_count

            # 2. turn/read 与 drain 并发；响应排在通知 burst 后也不会阻塞消费。
            terminal, persisted = await _read_while_draining(
                client,
                handle,
                drain_task=drain_task,
                deadline=deadline,
                timeout_message=timeout_message,
            )
            if terminal is not None:
                return terminal, "event", state.event_count
            assert persisted is not None
            if drain_task.done():
                terminal = drain_task.result()
                if terminal is not None:
                    return terminal, "event", state.event_count
            drain_done = drain_task.done()
            status = str(persisted.get("status") or "")
            if status in TERMINAL_STATUSES:
                if terminal_seen_at is None:
                    terminal_seen_at = time.monotonic()
                grace_remaining = terminal_grace_s - (
                    time.monotonic() - terminal_seen_at
                )
                if not drain_done and grace_remaining > 0:
                    drain_done, terminal = await _wait_for_drain(
                        drain_task,
                        min(
                            grace_remaining,
                            deadline - time.monotonic(),
                        ),
                    )
                    if terminal is not None:
                        return terminal, "event", state.event_count
                if drain_done or time.monotonic() - terminal_seen_at >= terminal_grace_s:
                    return _recovered_terminal(
                        persisted,
                        trace_path=trace_path,
                        terminal_grace_s=terminal_grace_s,
                        event_count=state.event_count,
                    )
            elif drain_done:
                raise RuntimeError("SDK 事件流结束但 turn/read 尚未终止")
        raise TimeoutError(timeout_message)
    finally:
        if not drain_task.done():
            drain_task.cancel()
            _ = await asyncio.gather(drain_task, return_exceptions=True)


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
