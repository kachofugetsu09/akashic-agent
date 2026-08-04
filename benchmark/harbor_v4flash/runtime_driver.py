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
_EMPTY_PROVIDER_REPLY = "模型未返回可用回复，请重试。"


class TurnDeadlineExceeded(TimeoutError):
    """题目原始 agent 时限已耗尽。"""


class AgentTurnFailed(RuntimeError):
    """Akashic turn 已开始但未形成成功终态。"""


class ProviderRateLimited(AgentTurnFailed):
    """Provider 以明确的 429/rate-limit 终止本次 turn。"""


class ProviderTransientFailure(AgentTurnFailed):
    """Provider 以明确的临时 5xx 终止本次 turn。"""


class ProviderAccountLimited(AgentTurnFailed):
    """OpenCode Go 账户使用额度已耗尽，必须等待 reset 或启用余额。"""


@dataclass(slots=True)
class _EventDrainState:
    event_count: int = 0


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")


def _turn_was_rate_limited(turn: dict[str, Any]) -> bool:
    """只从结构化 turn error 识别 provider 限流，不扫描题目或模型正文。"""

    error = turn.get("error")
    if not isinstance(error, dict):
        return False
    error_type = str(error.get("type") or "").lower()
    message = str(error.get("message") or "").lower()
    data = error.get("data")
    data_status = data.get("status_code") if isinstance(data, dict) else None
    data_code = str(data.get("code") or "").lower() if isinstance(data, dict) else ""
    return (
        data_status == 429
        or str(data_status) == "429"
        or data_code in {"429", "rate_limit", "rate_limited", "rate_limit_exceeded"}
        or "ratelimit" in error_type
        or "rate limit" in message
        or "rate_limit" in message
        or "too many requests" in message
        or "error code: 429" in message
        or "status code: 429" in message
        or "status_code=429" in message
    )


def _turn_was_transient_provider_failure(turn: dict[str, Any]) -> bool:
    """识别结构化 5xx 与响应体中断这两类临时 provider 故障。"""

    error = turn.get("error")
    if not isinstance(error, dict):
        return False
    error_type = str(error.get("type") or "").lower()
    message = str(error.get("message") or "").lower()
    data = error.get("data")
    data_status = data.get("status_code") if isinstance(data, dict) else None
    provider_error_type = any(
        token in error_type
        for token in (
            "internalservererror",
            "apistatuserror",
            "serviceunavailable",
            "badgateway",
            "gatewaytimeout",
            "providererror",
            "provider_error",
        )
    )
    explicit_status = any(
        marker in message
        for code in (500, 502, 503, 504)
        for marker in (
            f"error code: {code}",
            f"status code: {code}",
            f"status_code={code}",
        )
    )
    incomplete_response = error_type == "remoteprotocolerror" and any(
        marker in message
        for marker in (
            "peer closed connection without sending complete message body",
            "incomplete chunked read",
        )
    )
    return (
        incomplete_response
        or data_status in {500, 502, 503, 504}
        or (provider_error_type and explicit_status)
    )


def _turn_was_account_limited(turn: dict[str, Any]) -> bool:
    """识别 OpenCode Go 的账户额度终态，避免当作瞬时 429 重试。"""

    error = turn.get("error")
    if not isinstance(error, dict):
        return False
    message = str(error.get("message") or "").lower()
    return "gousagelimiterror" in message or "usage limit reached" in message


def _turn_was_empty_provider_response(turn: dict[str, Any]) -> bool:
    """识别 provider 未产生任何可用 delta 却被包装成 completed 的终态。"""

    # 1. 只接受 runtime 自有 fallback，不根据普通模型正文猜测失败。
    if (
        turn.get("status") != "completed"
        or turn.get("finalResponse") != _EMPTY_PROVIDER_REPLY
        or turn.get("error") is not None
    ):
        return False

    # 2. 要求唯一 assistant item 明示未流出回复且没有任何工具调用。
    items = turn.get("items")
    if not isinstance(items, list):
        return False
    assistant_items = [
        item
        for item in items
        if isinstance(item, dict) and item.get("type") == "assistantMessage"
    ]
    tool_items = [
        item
        for item in items
        if isinstance(item, dict) and item.get("type") == "toolCall"
    ]
    if len(assistant_items) != 1 or tool_items:
        return False
    data = assistant_items[0].get("data")
    metadata = data.get("metadata") if isinstance(data, dict) else None
    return (
        isinstance(data, dict)
        and data.get("content") == _EMPTY_PROVIDER_REPLY
        and isinstance(metadata, dict)
        and metadata.get("streamed_reply") is False
    )


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
                raise TurnDeadlineExceeded(timeout_message)

            # 2. 同时完成时先暴露异常；两者成功则以 terminal event 收束。
            done, _ = await asyncio.wait(
                waiters,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TurnDeadlineExceeded(timeout_message)
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
                if (
                    drain_done
                    or time.monotonic() - terminal_seen_at >= terminal_grace_s
                ):
                    return _recovered_terminal(
                        persisted,
                        trace_path=trace_path,
                        terminal_grace_s=terminal_grace_s,
                        event_count=state.event_count,
                    )
            elif drain_done:
                raise RuntimeError("SDK 事件流结束但 turn/read 尚未终止")
        raise TurnDeadlineExceeded(timeout_message)
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

    # 1. readiness 与 turn 共享同一份 task 官方 agent 预算。
    deadline = time.monotonic() + turn_timeout_s
    instruction = instruction_path.read_text(encoding="utf-8")
    client = await _connect(endpoint, min(readiness_timeout_s, turn_timeout_s))
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
        remaining_sec = deadline - time.monotonic()
        if remaining_sec <= 0:
            raise TurnDeadlineExceeded(f"agent 未在 {turn_timeout_s:.1f}s 内终止")
        try:
            terminal, terminal_source, event_count = await _observe_terminal(
                client,
                handle,
                trace_path=trace_path,
                turn_timeout_s=remaining_sec,
            )
        except TurnDeadlineExceeded:
            raise
        except Exception as error:
            raise AgentTurnFailed(str(error)) from error

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
        if status != "completed" and _turn_was_account_limited(terminal):
            raise ProviderAccountLimited("OpenCode Go usage limit 已耗尽")
        if status != "completed" and _turn_was_rate_limited(terminal):
            raise ProviderRateLimited("provider 返回明确的 rate-limit 终态")
        if status != "completed" and _turn_was_transient_provider_failure(terminal):
            raise ProviderTransientFailure("provider 返回明确的临时 5xx 终态")
        if _turn_was_empty_provider_response(terminal):
            raise ProviderTransientFailure("provider 未返回任何可用响应 delta")
        if status != "completed":
            raise AgentTurnFailed(f"Akasic turn 未正常完成：{status}")
        return result
    finally:
        await client.close()


def _write_driver_outcome(
    path: Path,
    *,
    status: str,
    error: BaseException | None = None,
) -> None:
    """原子记录 driver 终态，供 Harbor 区分题目超时与基础设施故障。"""

    payload: dict[str, object] = {"status": status, "timestamp": time.time()}
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _driver_error_status(error: BaseException) -> str:
    """把 driver 异常映射为 verifier 与有效性 Gate 使用的终态。"""

    if isinstance(error, TurnDeadlineExceeded):
        return "timed_out"
    if isinstance(error, ProviderRateLimited):
        return "rate_limited"
    if isinstance(error, ProviderTransientFailure):
        return "provider_transient"
    if isinstance(error, ProviderAccountLimited):
        return "account_limited"
    if isinstance(error, AgentTurnFailed):
        return "agent_failed"
    return "infra_failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True)
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
        _write_driver_outcome(
            args.outcome,
            status=_driver_error_status(error),
            error=error,
        )
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
    _write_driver_outcome(args.outcome, status="completed")
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
