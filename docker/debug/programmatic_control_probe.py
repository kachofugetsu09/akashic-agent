#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROTOCOL_VERSION = "1.0"
READINESS_DEADLINE_S = 30.0
SCENARIO_DEADLINE_S = 15.0
_PC09_COMPACTION_SUMMARY = """## Goal
验证大 tool batch 后连接仍可继续工作。
## Constraints & Preferences
保持当前会话和工具结果可重放。
## Progress
### Done
大 tool batch 已执行。
### In Progress
恢复下一次模型调用。
### Blocked
无。
## Key Decisions
使用当前模型生成 Pi-mono 六段摘要。
## Next Steps
继续处理 overflow complete，然后验证健康连接。
## Critical Context
这是 PC-09 的自动 compaction fixture；摘要只作为模型响应，不改变原始消息。
"""
_MEMORY_CONTEXT_SESSION = "programmatic:context-ledger"
_MEMORY_CONTEXT_INPUT = "ledger business query"
_MEMORY_CONTEXT_RESPONSE = "ledger business response"
_MEMORY_CONTEXT_THINKING = "ledger business reasoning"
_MEMORY_CONTEXT_TOKEN_REPEAT = 5_000


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    passed: bool
    evidence: object


class GateFailure(RuntimeError):
    pass


class JsonRpcSocketClient:
    """通过 UDS 发送 JSON-RPC，并保留完整协议证据。"""

    def __init__(self, endpoint: Path, events_path: Path) -> None:
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.connect(str(endpoint))
        self._reader = self._socket.makefile("rb")
        self._events_path = events_path
        self._request_id = 0
        self._pending_notifications: list[dict[str, Any]] = []

    def close(self) -> None:
        self._reader.close()
        self._socket.close()

    def notify(self, method: str, params: dict[str, object]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(
        self,
        method: str,
        params: dict[str, object],
        *,
        timeout: float = SCENARIO_DEADLINE_S,
    ) -> dict[str, Any]:
        response = self.request_raw(method, params, timeout=timeout)
        if "error" in response:
            raise GateFailure(f"{method} 返回 JSON-RPC error：{response['error']}")
        return response

    def request_raw(
        self,
        method: str,
        params: dict[str, object],
        *,
        timeout: float = SCENARIO_DEADLINE_S,
    ) -> dict[str, Any]:
        """发送请求并返回原始 result/error envelope。"""

        self._request_id += 1
        request_id = self._request_id
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        deadline = time.monotonic() + timeout
        while True:
            message = self._receive(deadline)
            if message.get("id") == request_id:
                return message
            if "method" in message:
                self._pending_notifications.append(message)

    def wait_terminal(
        self,
        turn_id: str,
        *,
        timeout: float = SCENARIO_DEADLINE_S,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            for index, event in enumerate(self._pending_notifications):
                if _is_terminal_event(event, turn_id):
                    return self._pending_notifications.pop(index)
            event = self._receive(deadline)
            if _is_terminal_event(event, turn_id):
                return event
            if "method" in event:
                self._pending_notifications.append(event)

    def wait_notification(
        self,
        method: str,
        *,
        turn_id: str | None = None,
        timeout: float = SCENARIO_DEADLINE_S,
    ) -> dict[str, Any]:
        """等待指定 method/turn notification，并缓存其他事件。"""

        deadline = time.monotonic() + timeout
        while True:
            for index, event in enumerate(self._pending_notifications):
                if _matches_event(event, method, turn_id):
                    return self._pending_notifications.pop(index)
            event = self._receive(deadline)
            if _matches_event(event, method, turn_id):
                return event
            if "method" in event:
                self._pending_notifications.append(event)

    def _send(self, message: dict[str, object]) -> None:
        self._record("client", message)
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        self._socket.sendall(payload.encode() + b"\n")

    def _receive(self, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GateFailure("等待 JSON-RPC 消息超时")
        self._socket.settimeout(remaining)
        line = self._reader.readline()
        if not line:
            raise GateFailure("JSON-RPC 连接在收到预期消息前关闭")
        raw = json.loads(line)
        if not isinstance(raw, dict) or raw.get("jsonrpc") != "2.0":
            raise GateFailure(f"收到非法 JSON-RPC 帧：{raw!r}")
        message = cast(dict[str, Any], raw)
        self._record("server", message)
        return message

    def _record(self, direction: str, message: dict[str, object]) -> None:
        record = {
            "timestamp": time.time(),
            "direction": direction,
            "message": message,
        }
        with self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_terminal_event(event: dict[str, Any], turn_id: str) -> bool:
    return _matches_event(event, "turn/completed", turn_id)


def _matches_event(
    event: dict[str, Any], method: str, turn_id: str | None = None
) -> bool:
    if event.get("method") != method:
        return False
    if turn_id is None:
        return True
    params = event.get("params")
    if not isinstance(params, dict):
        return False
    event_turn_id = params.get("turnId")
    if event_turn_id is None and isinstance(params.get("turn"), dict):
        event_turn_id = params["turn"].get("id")
    return event_turn_id == turn_id


def _event_turn(event: dict[str, Any]) -> dict[str, Any]:
    params = event.get("params")
    if not isinstance(params, dict) or not isinstance(params.get("turn"), dict):
        raise GateFailure(f"turn event 缺少 turn payload：{event!r}")
    return cast(dict[str, Any], params["turn"])


def _recorded_turn_notifications(path: Path, turn_id: str) -> list[dict[str, Any]]:
    """从原始协议记录中提取指定 turn 的服务端通知。"""

    notifications: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        message = record.get("message")
        if (
            record.get("direction") == "server"
            and isinstance(message, dict)
            and "method" in message
            and _matches_event(message, str(message["method"]), turn_id)
        ):
            notifications.append(cast(dict[str, Any], message))
    return notifications


def _tool_lifecycle(
    notifications: list[dict[str, Any]],
    tool_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """返回指定工具同 ID 的 started/completed item。"""

    started = [
        cast(dict[str, Any], event.get("params", {}).get("item"))
        for event in notifications
        if event.get("method") == "item/started"
        and isinstance(event.get("params", {}).get("item"), dict)
        and event["params"]["item"].get("type") == "toolCall"
        and event["params"]["item"].get("data", {}).get("name") == tool_name
    ]
    if len(started) != 1:
        raise GateFailure(f"{tool_name} started item 数量异常：{len(started)}")
    item_id = started[0].get("id")
    completed = [
        cast(dict[str, Any], event.get("params", {}).get("item"))
        for event in notifications
        if event.get("method") == "item/completed"
        and isinstance(event.get("params", {}).get("item"), dict)
        and event["params"]["item"].get("id") == item_id
    ]
    if len(completed) != 1:
        raise GateFailure(f"{tool_name} completed item 数量异常：{len(completed)}")
    return started[0], completed[0]


def _wait_tool_started(
    client: JsonRpcSocketClient,
    turn_id: str,
    tool_name: str,
) -> dict[str, Any]:
    """等待指定 turn 的真实工具 started 通知。"""

    deadline = time.monotonic() + SCENARIO_DEADLINE_S
    while time.monotonic() < deadline:
        event = client.wait_notification(
            "item/started",
            turn_id=turn_id,
            timeout=deadline - time.monotonic(),
        )
        item = event.get("params", {}).get("item")
        if (
            isinstance(item, dict)
            and item.get("type") == "toolCall"
            and item.get("data", {}).get("name") == tool_name
        ):
            return cast(dict[str, Any], item)
    raise GateFailure(f"等待工具 started 超时：turn={turn_id} tool={tool_name}")


def _turn_projection(turn: dict[str, Any]) -> dict[str, object]:
    """移除随机标识与时间，只保留 channel parity 所需领域事实。"""

    raw_items = turn.get("items")
    if not isinstance(raw_items, list):
        raise GateFailure(f"turn items 非数组：{turn!r}")
    items = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise GateFailure(f"turn item 非对象：{raw_item!r}")
        data = raw_item.get("data")
        if isinstance(data, dict):
            data = dict(data)
            data.pop("timestamp", None)
            metadata = data.get("metadata")
            if isinstance(metadata, dict):
                stable_metadata = dict(metadata)
                stable_metadata.pop("client_request_id", None)
                data["metadata"] = stable_metadata
        if raw_item.get("type") == "assistantMessage" and isinstance(data, dict):
            session_message_id = data.get("sessionMessageId")
            if isinstance(session_message_id, str):
                data["sessionMessageId"] = "<session-message-id>"
            metadata = data.get("metadata")
            if isinstance(metadata, dict):
                stable_metadata = dict(metadata)
                for volatile_key in (
                    "client_request_id",
                    "control_turn_id",
                    "turn_duration_ms",
                    "context_retry",
                ):
                    stable_metadata.pop(volatile_key, None)
                persisted_id = stable_metadata.get("persisted_user_message_id")
                if isinstance(persisted_id, str):
                    stable_metadata["persisted_user_message_id"] = (
                        "<persisted-user-message-id>"
                    )
                persisted_ids = stable_metadata.get("persisted_user_message_ids")
                if isinstance(persisted_ids, list):
                    stable_metadata["persisted_user_message_ids"] = [
                        "<persisted-user-message-id>" for _ in persisted_ids
                    ]
                data["metadata"] = stable_metadata
        items.append({"type": raw_item.get("type"), "data": data})
    error = turn.get("error")
    error_class = None
    if isinstance(error, dict):
        error_class = {
            "type": error.get("type"),
            "retryable": error.get("retryable"),
        }
    return {
        "status": turn.get("status"),
        "finalResponse": turn.get("finalResponse"),
        "items": items,
        "usage": turn.get("usage"),
        "error": error_class,
    }


def _wait_database_turn(
    database: Path,
    thread_id: str,
    input_text: str,
    *,
    timeout: float = SCENARIO_DEADLINE_S,
) -> dict[str, Any]:
    """等待 channel adapter 写入指定输入的领域终态并返回 wire 投影。"""

    deadline = time.monotonic() + timeout
    last_status = "missing"
    while time.monotonic() < deadline:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, session_key, status, input_json, items_json,
                       usage_json, error_json, final_response
                FROM turns WHERE session_key = ? ORDER BY created_at DESC
                """,
                (thread_id,),
            ).fetchall()
        for row in rows:
            input_payload = json.loads(row["input_json"])
            if input_payload.get("input") != input_text:
                continue
            last_status = str(row["status"])
            if last_status not in {"completed", "failed", "interrupted", "cancelled"}:
                break
            return {
                "id": row["id"],
                "threadId": row["session_key"],
                "status": row["status"],
                "finalResponse": row["final_response"],
                "items": json.loads(row["items_json"]),
                "usage": json.loads(row["usage_json"]) if row["usage_json"] else None,
                "error": json.loads(row["error_json"]) if row["error_json"] else None,
            }
        threading.Event().wait(0.02)
    raise GateFailure(
        f"等待 channel turn 终态超时：thread={thread_id} input={input_text!r} status={last_status}"
    )


def _wait_database_turn_status(
    database: Path,
    thread_id: str,
    input_text: str,
    expected: set[str],
    *,
    timeout: float = SCENARIO_DEADLINE_S,
) -> dict[str, Any]:
    """等待 channel turn 进入指定状态并返回最小可审计证据。"""

    deadline = time.monotonic() + timeout
    last_status = "missing"
    while time.monotonic() < deadline:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT id, status, final_response
                FROM turns
                WHERE session_key = ? AND json_extract(input_json, '$.input') = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (thread_id, input_text),
            ).fetchone()
        if row is not None:
            last_status = str(row["status"])
            if last_status in expected:
                return {
                    "id": row["id"],
                    "status": last_status,
                    "finalResponse": row["final_response"],
                }
        threading.Event().wait(0.02)
    raise GateFailure(
        f"等待 channel turn 状态超时：thread={thread_id} input={input_text!r} "
        f"expected={sorted(expected)} actual={last_status}"
    )


def _wait_database_turn_inputs(
    database: Path,
    thread_id: str,
    input_text: str,
    expected_count: int,
    *,
    timeout: float = SCENARIO_DEADLINE_S,
) -> dict[str, object]:
    """等待 active channel turn 持久化指定数量的有序 user item。"""

    deadline = time.monotonic() + timeout
    last_inputs: list[object] = []
    while time.monotonic() < deadline:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT id, status, items_json
                FROM turns
                WHERE session_key = ? AND json_extract(input_json, '$.input') = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (thread_id, input_text),
            ).fetchone()
        if row is not None:
            items = json.loads(row["items_json"])
            last_inputs = [
                item.get("data", {}).get("content")
                for item in items
                if isinstance(item, dict) and item.get("type") == "userMessage"
            ]
            if len(last_inputs) == expected_count:
                return {
                    "id": row["id"],
                    "status": row["status"],
                    "userInputs": last_inputs,
                }
        threading.Event().wait(0.02)
    raise GateFailure(
        f"等待 channel turn 输入超时：thread={thread_id} input={input_text!r} "
        f"expected_count={expected_count} actual={last_inputs!r}"
    )


def _receive_web_final(
    web: Any, *, timeout: float = SCENARIO_DEADLINE_S
) -> dict[str, Any]:
    """忽略流式帧并返回下一条 Web channel 最终帧。"""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = json.loads(web.recv(timeout=deadline - time.monotonic()))
        if frame.get("type") == "message.final":
            return cast(dict[str, Any], frame)
    raise GateFailure("Web channel 未在 deadline 内返回 message.final")


def _extract_id(response: dict[str, Any], resource: str) -> str:
    result = response.get("result")
    if not isinstance(result, dict):
        raise GateFailure(f"{resource} response 缺少 result 对象")
    nested = result.get(resource)
    identifier = nested.get("id") if isinstance(nested, dict) else result.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise GateFailure(f"{resource} response 缺少稳定 id：{result!r}")
    return identifier


def _http_json(
    method: str,
    url: str,
    payload: object | None = None,
    *,
    timeout: float = SCENARIO_DEADLINE_S,
) -> object:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _model_requests(payload: object) -> list[object]:
    if not isinstance(payload, dict):
        raise GateFailure(f"model-gate requests 响应非法：{payload!r}")
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise GateFailure(f"model-gate requests 缺少数组：{payload!r}")
    return list(requests)


def _memory_context_seed_content(role: str, index: int) -> str:
    """Return one deterministic large seed message for the ledger gate."""

    if role not in {"user", "assistant"}:
        raise ValueError(f"memory context seed role 无效: {role}")
    return (f"seed {role} {index} " + "token " * _MEMORY_CONTEXT_TOKEN_REPEAT).strip()


def _memory_context_seed_rows(session_key: str) -> list[tuple[str, str, str]]:
    """Return expected seed IDs, roles, and bodies in durable seq order."""

    rows: list[tuple[str, str, str]] = []
    for index in range(4):
        for role in ("user", "assistant"):
            seq = len(rows)
            rows.append(
                (
                    f"{session_key}:{seq}",
                    role,
                    _memory_context_seed_content(role, index),
                )
            )
    return rows


def _memory_context_source_plan_digest(session_key: str) -> str:
    """Hash the three selected complete units exactly as ContextCompactor does."""

    selected: list[dict[str, object]] = []
    for unit_index in range(3):
        source_from_seq = unit_index * 2
        through_seq = source_from_seq + 1
        for offset, role in enumerate(("user", "assistant")):
            seq = source_from_seq + offset
            selected.append(
                {
                    "id": f"{session_key}:{seq}",
                    "seq": seq,
                    "unit_ref": f"{source_from_seq}:{through_seq}:{unit_index}",
                    "message": {
                        "role": role,
                        "content": _memory_context_seed_content(role, unit_index),
                    },
                }
            )
    encoded = json.dumps(
        selected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _memory_context_request_kinds(requests: Sequence[object]) -> list[str]:
    """Classify the exact three model requests and reject tool-boundary drift."""

    if len(requests) != 3:
        raise GateFailure(f"memory-context 模型请求数量异常：{len(requests)}")
    kinds: list[str] = []
    for raw_request in requests:
        if not isinstance(raw_request, dict):
            raise GateFailure(f"memory-context 模型请求非法：{raw_request!r}")
        payload = raw_request.get("payload")
        if not isinstance(payload, dict):
            raise GateFailure("memory-context 模型请求缺少 payload")
        serialized = json.dumps(payload.get("messages", []), ensure_ascii=False)
        if "Closed history to consolidate" in serialized:
            kind = "summary"
        elif "Memory Extraction Agent" in serialized:
            kind = "markdown"
        elif _MEMORY_CONTEXT_INPUT in serialized:
            kind = "business"
        else:
            raise GateFailure("memory-context 模型请求无法归类")
        if kind in {"summary", "markdown"} and payload.get("tools", []) not in (
            None,
            [],
        ):
            raise GateFailure(f"memory-context {kind} 请求不得携带 tools")
        kinds.append(kind)
    if kinds != ["summary", "markdown", "business"]:
        raise GateFailure(f"memory-context 模型请求顺序异常：{kinds!r}")
    return kinds


def _wait_http_ready(url: str, deadline_s: float) -> None:
    """在总 deadline 内等待 HTTP readiness，不把单次连接成功当业务成功。"""

    deadline = time.monotonic() + deadline_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = _http_json("GET", url, timeout=1.0)
            if response == {"status": "ready"}:
                return
            last_error = f"unexpected response: {response!r}"
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = f"{type(error).__name__}: {error}"
        threading.Event().wait(0.05)
    raise GateFailure(f"model-gate readiness 超时：{last_error}")


def _wait_socket(endpoint: Path, deadline_s: float) -> None:
    """等待 UDS 真实接受连接，忽略进程重启遗留的 socket 文件。"""

    # 1. 轮询真实连接，不能把遗留路径当作 readiness。
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(min(0.1, max(0.0, deadline - time.monotonic())))
            if endpoint.exists() and probe.connect_ex(str(endpoint)) == 0:
                return
        finally:
            probe.close()
        threading.Event().wait(0.05)

    # 2. deadline 到期后显式暴露 readiness 失败。
    raise GateFailure(f"等待 UDS 文件超时：{endpoint}")


def _connect_client(endpoint: Path, events_path: Path) -> JsonRpcSocketClient:
    """建立连接并完成 initialize/initialized/status readiness。"""

    client = JsonRpcSocketClient(endpoint, events_path)
    client.request(
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "clientInfo": {"name": "docker-control-gate", "version": "1.0"},
            "capabilities": {"reasoningEvents": False},
        },
        timeout=READINESS_DEADLINE_S,
    )
    client.notify("initialized", {})
    status = client.request("server/status", {}, timeout=READINESS_DEADLINE_S)
    if status.get("result", {}).get("ready") is not True:
        client.close()
        raise GateFailure(f"server/status 未 ready：{status!r}")
    return client


def _create_barrier(model_url: str, name: str, script: dict[str, object]) -> None:
    _http_json("PUT", f"{model_url}/control/barriers/{name}")
    _http_json("PUT", f"{model_url}/control/script", {**script, "barrier": name})


def _wait_barrier(model_url: str, name: str) -> None:
    result = _http_json(
        "GET",
        f"{model_url}/control/barriers/{name}/wait?timeout=15",
        timeout=SCENARIO_DEADLINE_S + 1,
    )
    if not isinstance(result, dict) or result.get("reached") is not True:
        raise GateFailure(f"barrier 未到达：{name} {result!r}")


def _release_barrier(model_url: str, name: str) -> None:
    result = _http_json("POST", f"{model_url}/control/barriers/{name}/release")
    if not isinstance(result, dict) or result.get("released") is not True:
        raise GateFailure(f"barrier 释放失败：{name} {result!r}")


def _start_thread(client: JsonRpcSocketClient, check_id: str) -> str:
    return _extract_id(
        client.request("thread/start", {"metadata": {"gate": check_id}}),
        "thread",
    )


def _start_turn(
    client: JsonRpcSocketClient,
    thread_id: str,
    text: str,
    *,
    detached: bool = False,
) -> str:
    return _extract_id(
        client.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": text,
                "metadata": {},
                "detached": detached,
            },
        ),
        "turn",
    )


def _terminal_status(event: dict[str, Any]) -> str:
    status = _event_turn(event).get("status")
    if not isinstance(status, str):
        raise GateFailure(f"terminal event 缺少 status：{event!r}")
    return status


def _inside_smoke(report_dir: Path) -> int:
    """从独立 probe 容器验证真实 gateway、provider 和持久化路径。"""

    report_dir.mkdir(parents=True, exist_ok=True)
    events_path = report_dir / "events.jsonl"
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    endpoint = Path("/sandbox/akashic.sock")
    checks: list[CheckResult] = []
    client: JsonRpcSocketClient | None = None
    try:
        # 1. readiness 必须完成协议握手与 server/status
        _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        client = JsonRpcSocketClient(endpoint, events_path)
        initialized = client.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientInfo": {"name": "docker-control-gate", "version": "1.0"},
                "capabilities": {"reasoningEvents": False},
            },
            timeout=READINESS_DEADLINE_S,
        )
        client.notify("initialized", {})
        status = client.request("server/status", {}, timeout=READINESS_DEADLINE_S)
        mode = endpoint.stat().st_mode & 0o777
        checks.append(
            CheckResult(
                "PC-01",
                mode == 0o600,
                {"initialize": initialized, "status": status, "socketMode": oct(mode)},
            )
        )

        # 2. 基本 turn 必须真正穿过正式 HTTP provider wiring
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            {
                "mode": "stream",
                "deltas": ["control gate"],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 2,
                    "total_tokens": 13,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )
        thread_response = client.request(
            "thread/start", {"metadata": {"gate": "PC-03"}}
        )
        thread_id = _extract_id(thread_response, "thread")
        turn_response = client.request(
            "turn/start",
            {"threadId": thread_id, "input": "run control gate", "metadata": {}},
        )
        turn_id = _extract_id(turn_response, "turn")
        terminal = client.wait_terminal(turn_id)
        turn_read = client.request(
            "turn/read",
            {"threadId": thread_id, "turnId": turn_id},
        )
        _ = client.request("server/status", {})
        terminal_count = sum(
            event.get("method") == "turn/completed"
            for event in _recorded_turn_notifications(events_path, turn_id)
        )
        requests = _http_json("GET", f"{model_url}/control/requests")
        model_requests = _model_requests(requests)
        provider_called = len(model_requests) == 1
        terminal_turn = _event_turn(terminal)
        consistent = (
            terminal_turn.get("status") == "completed"
            and terminal_turn.get("finalResponse") == "control gate"
            and turn_read.get("result") == terminal_turn
        )
        checks.append(
            CheckResult(
                "PC-03",
                provider_called and consistent and terminal_count == 1,
                {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "terminal": terminal,
                    "turnRead": turn_read,
                    "terminalEventCount": terminal_count,
                    "modelRequestCount": len(model_requests or []),
                },
            )
        )

        # 3. 正式 streaming provider 的 tool/usage 必须投影到事件和同一 DB turn。
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            [
                {
                    "mode": "stream",
                    "deltas": [],
                    "tool_calls": [
                        {
                            "id": "call_pc04",
                            "name": "tool_search",
                            "arguments": {"query": "no-match-pc04"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                        "prompt_tokens_details": {"cached_tokens": 0},
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                },
                {
                    "mode": "stream",
                    "deltas": ["stream ", "complete"],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                        "prompt_tokens_details": {"cached_tokens": 0},
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            ],
        )
        stream_thread = _start_thread(client, "PC-04")
        stream_turn = _start_turn(client, stream_thread, "stream tool usage")
        stream_terminal = client.wait_terminal(stream_turn)
        stream_read = client.request(
            "turn/read", {"threadId": stream_thread, "turnId": stream_turn}
        )
        stream_payload = _event_turn(stream_terminal)
        notifications = _recorded_turn_notifications(events_path, stream_turn)
        deltas = [
            event["params"]
            for event in notifications
            if event.get("method") == "item/assistantMessage/delta"
        ]
        sequences = [delta.get("sequence") for delta in deltas]
        delta_text = "".join(str(delta.get("delta") or "") for delta in deltas)
        tool_items = [
            item
            for item in stream_payload.get("items", [])
            if isinstance(item, dict) and item.get("type") == "toolCall"
        ]
        started_ids = [
            event.get("params", {}).get("item", {}).get("id")
            for event in notifications
            if event.get("method") == "item/started"
        ]
        completed_ids = [
            event.get("params", {}).get("item", {}).get("id")
            for event in notifications
            if event.get("method") == "item/completed"
        ]
        usage = stream_payload.get("usage")
        pc04_passed = (
            stream_payload.get("status") == "completed"
            and stream_payload.get("finalResponse") == "stream complete"
            and delta_text == stream_payload.get("finalResponse")
            and sequences == list(range(len(sequences)))
            and started_ids == completed_ids
            and len(tool_items) == 1
            and tool_items[0].get("data", {}).get("callId") == "call_pc04"
            and usage
            == {
                "inputTokens": 12,
                "cachedInputTokens": 0,
                "outputTokens": 5,
                "reasoningOutputTokens": 0,
                "requestCount": 2,
                "coveredRequestCount": 2,
                "coverage": "exact",
            }
            and stream_read.get("result") == stream_payload
        )
        checks.append(
            CheckResult(
                "PC-04",
                pc04_passed,
                {
                    "threadId": stream_thread,
                    "turnId": stream_turn,
                    "itemMethods": [event.get("method") for event in notifications],
                    "deltaSequences": sequences,
                    "deltaText": delta_text,
                    "toolItems": tool_items,
                    "usage": usage,
                    "terminalEqualsRead": stream_read.get("result") == stream_payload,
                },
            )
        )
        final_requests = _http_json("GET", f"{model_url}/control/requests")
        _write_jsonl(
            report_dir / "model-requests.jsonl", _model_requests(final_requests)
        )
    except Exception as error:
        checks.append(
            CheckResult(
                "controller",
                False,
                {"type": type(error).__name__, "message": str(error)},
            )
        )
    finally:
        if client is not None:
            client.close()

    passed = bool(checks) and all(check.passed for check in checks)
    report = {
        "gate": "smoke",
        "status": "passed" if passed else "failed",
        "checks": [asdict(check) for check in checks],
    }
    _write_json(report_dir / "inside-gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def _inside_memory_context(report_dir: Path) -> int:
    """验证真实 session compaction ledger、Markdown side effects 和 append-only 语义。"""

    report_dir.mkdir(parents=True, exist_ok=True)
    events_path = report_dir / "events.jsonl"
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    endpoint = Path("/sandbox/akashic.sock")
    checks: list[CheckResult] = []
    client: JsonRpcSocketClient | None = None
    try:
        # 1. 按固定顺序提供 compaction summary、Markdown extraction 和业务响应。
        _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            [
                {
                    "mode": "complete",
                    "content": _PC09_COMPACTION_SUMMARY,
                },
                {
                    "mode": "complete",
                    "content": '{"history_entries":[],"pending_items":[]}',
                },
                {
                    "mode": "complete",
                    "content": (
                        f"<think>{_MEMORY_CONTEXT_THINKING}</think>"
                        f"{_MEMORY_CONTEXT_RESPONSE}"
                    ),
                },
            ],
        )
        client = _connect_client(endpoint, events_path)
        turn_id = _start_turn(client, _MEMORY_CONTEXT_SESSION, _MEMORY_CONTEXT_INPUT)
        terminal = client.wait_terminal(turn_id)
        payload = _event_turn(terminal)
        database = Path("/sandbox/workspace/sessions.db")
        seed_rows = _memory_context_seed_rows(_MEMORY_CONTEXT_SESSION)
        expected_seed_hashes = {
            message_id: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for message_id, _, content in seed_rows
        }
        connection = sqlite3.connect(database)
        try:
            connection.row_factory = sqlite3.Row
            session_row = connection.execute(
                "SELECT last_consolidated FROM sessions WHERE key = ?",
                (_MEMORY_CONTEXT_SESSION,),
            ).fetchone()
            message_rows = connection.execute(
                "SELECT id, seq, role, content FROM messages "
                "WHERE session_key = ? ORDER BY seq",
                (_MEMORY_CONTEXT_SESSION,),
            ).fetchall()
            compaction_row = connection.execute(
                "SELECT * FROM session_compactions "
                "WHERE session_key = ? AND generation = 1",
                (_MEMORY_CONTEXT_SESSION,),
            ).fetchone()
            prepare_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM session_compaction_prepares "
                    "WHERE session_key = ?",
                    (_MEMORY_CONTEXT_SESSION,),
                ).fetchone()[0]
            )
        finally:
            connection.close()

        if session_row is None or compaction_row is None:
            raise GateFailure("memory-context ledger row 缺失")
        actual_hashes = {
            str(row["id"]): hashlib.sha256(
                str(row["content"]).encode("utf-8")
            ).hexdigest()
            for row in message_rows
            if int(row["seq"]) < 8
        }
        seed_hashes_unchanged = actual_hashes == expected_seed_hashes
        source_ids = json.loads(compaction_row["source_message_ids_json"])
        retained_tail = json.loads(compaction_row["retained_tail_json"])
        source_digest = str(compaction_row["source_plan_digest"])
        expected_source_ids = [message_id for message_id, _, _ in seed_rows[:6]]
        expected_retained_ids = [message_id for message_id, _, _ in seed_rows[6:]]
        retained_ids = [str(item.get("id")) for item in retained_tail]
        final_messages_only_append = (
            len(message_rows) == 10
            and [str(row["id"]) for row in message_rows[:8]]
            == [message_id for message_id, _, _ in seed_rows]
            and [str(row["role"]) for row in message_rows[8:]] == ["user", "assistant"]
            and str(message_rows[8]["content"]) == _MEMORY_CONTEXT_INPUT
            and str(message_rows[9]["content"]) == _MEMORY_CONTEXT_RESPONSE
        )
        retained_tail_exact = (
            retained_ids == expected_retained_ids
            and [str(item.get("unit_ref")) for item in retained_tail]
            == ["6:7:0", "6:7:0"]
            and [str(item.get("message", {}).get("content")) for item in retained_tail]
            == [content for _, _, content in seed_rows[6:]]
        )
        ledger_passed = (
            session_row["last_consolidated"] == 1
            and compaction_row["context_window"] == 100_000
            and compaction_row["threshold_tokens"] == 74_000
            and source_ids == expected_source_ids
            and retained_tail_exact
            and source_digest
            == _memory_context_source_plan_digest(_MEMORY_CONTEXT_SESSION)
            and prepare_count == 0
            and seed_hashes_unchanged
            and final_messages_only_append
        )
        receipt_connection = sqlite3.connect(
            "/sandbox/workspace/memory/consolidation_writes.db"
        )
        try:
            receipt_row = receipt_connection.execute(
                "SELECT payload FROM consolidation_writes "
                "WHERE source_ref = ? AND kind = 'session_compaction_receipt'",
                (str(compaction_row["source_ref"]),),
            ).fetchone()
        finally:
            receipt_connection.close()
        pending_path = Path("/sandbox/workspace/memory/PENDING.md")
        pending_empty = (
            pending_path.exists()
            and not pending_path.read_text(encoding="utf-8").strip()
        )
        final_requests = _model_requests(
            _http_json("GET", f"{model_url}/control/requests")
        )
        request_kinds = _memory_context_request_kinds(final_requests)
        scripts = [
            request.get("script")
            for request in final_requests
            if isinstance(request, dict)
        ]
        scripts_boundary = (
            scripts[0] == {"mode": "complete", "content": _PC09_COMPACTION_SUMMARY}
            and scripts[1]
            == {
                "mode": "complete",
                "content": '{"history_entries":[],"pending_items":[]}',
            }
            and isinstance(scripts[2], dict)
            and "<think>" in str(scripts[2].get("content"))
        )
        projected = _turn_projection(payload)
        assistant_items = [
            item
            for item in projected["items"]
            if item.get("type") == "assistantMessage"
        ]
        thinking_boundary = (
            len(assistant_items) == 1
            and assistant_items[0]["data"].get("thinking") == _MEMORY_CONTEXT_THINKING
            and not any(item.get("type") == "toolCall" for item in projected["items"])
        )
        checks.append(
            CheckResult(
                "MC-01",
                payload.get("status") == "completed"
                and payload.get("finalResponse") == _MEMORY_CONTEXT_RESPONSE
                and request_kinds == ["summary", "markdown", "business"]
                and scripts_boundary
                and thinking_boundary
                and ledger_passed
                and receipt_row is not None
                and pending_empty,
                {
                    "terminal": payload,
                    "requestKinds": request_kinds,
                    "ledger": {
                        "lastConsolidated": session_row["last_consolidated"],
                        "sourceIds": source_ids,
                        "retainedIds": retained_ids,
                        "sourceDigest": source_digest,
                        "seedHashesUnchanged": seed_hashes_unchanged,
                        "finalMessagesOnlyAppend": final_messages_only_append,
                        "prepareCount": prepare_count,
                        "retainedTailExact": retained_tail_exact,
                    },
                    "receiptExists": receipt_row is not None,
                    "pendingEmpty": pending_empty,
                    "scriptsBoundary": scripts_boundary,
                    "thinkingBoundary": thinking_boundary,
                    "modelRequestCount": len(final_requests),
                },
            )
        )
        _write_jsonl(report_dir / "model-requests.jsonl", final_requests)
    except Exception as error:
        checks.append(
            CheckResult(
                "controller",
                False,
                {"type": type(error).__name__, "message": str(error)},
            )
        )
    finally:
        if client is not None:
            client.close()

    passed = bool(checks) and all(check.passed for check in checks)
    report = {
        "gate": "memory-context",
        "status": "passed" if passed else "failed",
        "checks": [asdict(check) for check in checks],
    }
    _write_json(report_dir / "inside-gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def _inside_failure_matrix(report_dir: Path) -> int:
    """以真实 barrier 和多连接驱动 PR 必选故障矩阵。"""

    report_dir.mkdir(parents=True, exist_ok=True)
    events_path = report_dir / "events.jsonl"
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    endpoint = Path("/sandbox/akashic.sock")
    checks: list[CheckResult] = []
    clients: list[JsonRpcSocketClient] = []
    restart_state: dict[str, str] = {}
    try:
        _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        first = _connect_client(endpoint, events_path)
        second = _connect_client(endpoint, events_path)
        clients.extend((first, second))

        # 1. 两连接两 thread：第二个 turn 保持 queued，释放后事件不串线。
        _create_barrier(
            model_url,
            "isolation-first",
            {"mode": "complete", "content": "isolation first"},
        )
        _create_barrier(
            model_url,
            "isolation-second",
            {"mode": "complete", "content": "isolation second"},
        )
        thread_a = _start_thread(first, "PC-05")
        thread_b = _start_thread(second, "PC-05")
        turn_a = _start_turn(first, thread_a, "isolation first")
        _wait_barrier(model_url, "isolation-first")
        turn_b = _start_turn(second, thread_b, "isolation second")
        queued_b = second.wait_notification("turn/queued", turn_id=turn_b)
        before_release = _http_json("GET", f"{model_url}/control/requests")
        before_requests = _model_requests(before_release)
        _release_barrier(model_url, "isolation-first")
        terminal_a = first.wait_terminal(turn_a)
        _wait_barrier(model_url, "isolation-second")
        _release_barrier(model_url, "isolation-second")
        terminal_b = second.wait_terminal(turn_b)
        isolated = (
            len(before_requests) == 1
            and _terminal_status(terminal_a) == "completed"
            and _terminal_status(terminal_b) == "completed"
            and queued_b.get("params", {}).get("threadId") == thread_b
            and terminal_a.get("params", {}).get("threadId") == thread_a
            and terminal_b.get("params", {}).get("threadId") == thread_b
        )
        checks.append(
            CheckResult(
                "PC-05",
                isolated,
                {
                    "requestsBeforeFirstRelease": len(before_requests),
                    "threadA": thread_a,
                    "threadB": thread_b,
                    "turnA": turn_a,
                    "turnB": turn_b,
                },
            )
        )

        # 2. 同 thread 的第二个 start 必须明确 busy，不能注入 owner turn。
        _create_barrier(
            model_url,
            "thread-conflict",
            {"mode": "complete", "content": "intermediate candidate"},
        )
        conflict_thread = _start_thread(first, "PC-06")
        conflict_turn = _start_turn(first, conflict_thread, "conflict owner")
        _wait_barrier(model_url, "thread-conflict")
        rejected = first.request_raw(
            "turn/start",
            {
                "threadId": conflict_thread,
                "input": "must conflict",
                "metadata": {},
            },
        )
        _release_barrier(model_url, "thread-conflict")
        conflict_terminal = first.wait_terminal(conflict_turn)
        rejected_error = rejected.get("error")
        terminal_turn = _event_turn(conflict_terminal)
        user_inputs = [
            item.get("data", {}).get("content")
            for item in terminal_turn.get("items", [])
            if isinstance(item, dict) and item.get("type") == "userMessage"
        ]
        checks.append(
            CheckResult(
                "PC-06",
                isinstance(rejected_error, dict)
                and rejected_error.get("code") == -32011
                and rejected_error.get("data") == {"retryable": True}
                and _terminal_status(conflict_terminal) == "completed"
                and terminal_turn.get("finalResponse") == "intermediate candidate"
                and user_inputs == ["conflict owner"],
                {
                    "rejected": rejected_error,
                    "ownerTerminal": conflict_terminal,
                    "userInputs": user_inputs,
                },
            )
        )

        # 3. 工具已 started 后精确 interrupt，owner 必须闭合同 ID item。
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            [
                {
                    "mode": "stream",
                    "deltas": [],
                    "tool_calls": [
                        {
                            "id": "call_pc07_unlock",
                            "name": "tool_search",
                            "arguments": {"query": "select:shell"},
                        }
                    ],
                },
                {
                    "mode": "stream",
                    "deltas": [],
                    "tool_calls": [
                        {
                            "id": "call_pc07_shell",
                            "name": "shell",
                            "arguments": {
                                "command": "sleep 300",
                                "description": "阻塞中断探针",
                                "timeout": 300,
                                "yield_time_ms": 30_000,
                            },
                        }
                    ],
                },
            ],
        )
        interrupt_thread = _start_thread(first, "PC-07")
        interrupted_turn = _start_turn(first, interrupt_thread, "interrupt me")
        shell_started = _wait_tool_started(first, interrupted_turn, "shell")
        interrupt_started = time.monotonic()
        interrupt_result = first.request(
            "turn/interrupt",
            {"threadId": interrupt_thread, "turnId": interrupted_turn},
        )
        interrupted_terminal = first.wait_terminal(interrupted_turn, timeout=2.0)
        interrupt_duration = time.monotonic() - interrupt_started
        interrupted_payload = _event_turn(interrupted_terminal)
        interrupted_read = first.request(
            "turn/read",
            {"threadId": interrupt_thread, "turnId": interrupted_turn},
        )
        _ = first.request("server/status", {})
        interrupted_events = _recorded_turn_notifications(events_path, interrupted_turn)
        interrupted_terminal_count = sum(
            event.get("method") == "turn/completed" for event in interrupted_events
        )
        _, shell_completed = _tool_lifecycle(interrupted_events, "shell")
        persisted_shell = next(
            (
                item
                for item in interrupted_payload.get("items", [])
                if isinstance(item, dict) and item.get("id") == shell_started.get("id")
            ),
            None,
        )

        _create_barrier(
            model_url,
            "interrupt-fresh",
            {"mode": "complete", "content": "fresh survives"},
        )
        fresh_turn = _start_turn(first, interrupt_thread, "fresh turn")
        _wait_barrier(model_url, "interrupt-fresh")
        stale_interrupt = first.request_raw(
            "turn/interrupt",
            {"threadId": interrupt_thread, "turnId": interrupted_turn},
        )
        fresh_before_release = first.request(
            "turn/read",
            {"threadId": interrupt_thread, "turnId": fresh_turn},
        )
        _release_barrier(model_url, "interrupt-fresh")
        fresh_terminal = first.wait_terminal(fresh_turn)
        stale_error = stale_interrupt.get("error")
        stale_result = stale_interrupt.get("result")
        stale_safe = (
            isinstance(stale_error, dict) and stale_error.get("code") == -32012
        ) or (
            isinstance(stale_result, dict)
            and stale_result.get("id") == interrupted_turn
            and stale_result.get("status") == "interrupted"
        )
        checks.append(
            CheckResult(
                "PC-07",
                _terminal_status(interrupted_terminal) == "interrupted"
                and interrupt_duration <= 2.0
                and shell_completed.get("id") == shell_started.get("id")
                and shell_completed.get("data", {}).get("status") == "interrupted"
                and persisted_shell == shell_completed
                and interrupted_terminal_count == 1
                and interrupted_read.get("result") == interrupted_payload
                and stale_safe
                and fresh_before_release.get("result", {}).get("status")
                == "in_progress"
                and _terminal_status(fresh_terminal) == "completed",
                {
                    "interrupt": interrupt_result,
                    "durationSeconds": interrupt_duration,
                    "toolStarted": shell_started,
                    "toolCompleted": shell_completed,
                    "terminalEqualsRead": interrupted_read.get("result")
                    == interrupted_payload,
                    "terminalEventCount": interrupted_terminal_count,
                    "staleError": stale_error,
                    "staleResult": stale_result,
                    "freshBeforeRelease": fresh_before_release,
                },
            )
        )

        # 4. 显式 detached turn 在连接断开后继续；重连后读取持久终态。
        disconnecting = _connect_client(endpoint, events_path)
        clients.append(disconnecting)
        _create_barrier(
            model_url,
            "disconnect-held",
            {"mode": "complete", "content": "survived disconnect"},
        )
        recovery_thread = _start_thread(disconnecting, "PC-08")
        recovery_turn = _start_turn(
            disconnecting,
            recovery_thread,
            "disconnect me",
            detached=True,
        )
        _wait_barrier(model_url, "disconnect-held")
        disconnecting.close()
        clients.remove(disconnecting)
        _release_barrier(model_url, "disconnect-held")
        resumed = _connect_client(endpoint, events_path)
        clients.append(resumed)
        deadline = time.monotonic() + SCENARIO_DEADLINE_S
        recovery_read: dict[str, Any] = {}
        while time.monotonic() < deadline:
            recovery_read = resumed.request(
                "turn/read",
                {"threadId": recovery_thread, "turnId": recovery_turn},
            )
            if recovery_read.get("result", {}).get("status") == "completed":
                break
            threading.Event().wait(0.02)
        checks.append(
            CheckResult(
                "PC-08",
                recovery_read.get("result", {}).get("finalResponse")
                == "survived disconnect",
                recovery_read,
            )
        )
        restart_state = {"threadId": recovery_thread, "turnId": recovery_turn}

        # 5. 不读取事件的客户端只能影响自身，另一连接仍须在 deadline 内完成。
        slow = _connect_client(endpoint, events_path)
        clients.append(slow)
        slow._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        overflow_calls = [
            {
                "id": f"call_pc09_{index}",
                "name": "tool_search",
                "arguments": {"query": f"no-match-pc09-{index}-" + "x" * (1024)},
            }
            for index in range(80)
        ]
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            [
                {
                    "mode": "stream",
                    "deltas": [],
                    "tool_calls": [
                        {
                            "id": "call_pc09_seed",
                            "name": "tool_search",
                            "arguments": {"query": "no-match-pc09-seed"},
                        }
                    ],
                },
                {
                    "mode": "stream",
                    "deltas": [],
                    "tool_calls": [
                        {
                            "id": "call_pc09_seed_2",
                            "name": "shell",
                            "arguments": {
                                "command": "python -c 'print(\"x\" * 80000)'",
                                "description": "生成可压缩的已闭合工具上下文",
                                "timeout": 5,
                                "yield_time_ms": 1000,
                            },
                        }
                    ],
                },
                # 保留两个已闭合批次；默认 keep_recent_tokens=20k 需要从首个
                # 批次切出可压缩前缀，不能只放一个很小的 seed。
                {
                    "mode": "stream",
                    "deltas": [],
                    "tool_calls": overflow_calls,
                },
                # 大 batch 闭合后下一次 business payload 过 compaction gate；
                # 先放合法摘要，避免把业务响应消费到 summary 请求。
                {"mode": "complete", "content": _PC09_COMPACTION_SUMMARY},
                {"mode": "stream", "deltas": ["overflow complete"]},
                {"mode": "complete", "content": "healthy after overflow"},
            ],
        )
        slow_thread = _start_thread(slow, "PC-09-slow")
        slow_turn = _start_turn(slow, slow_thread, "overflow this connection")
        slow_deadline = time.monotonic() + SCENARIO_DEADLINE_S
        slow_read: dict[str, Any] = {}
        while time.monotonic() < slow_deadline:
            slow_read = second.request(
                "turn/read", {"threadId": slow_thread, "turnId": slow_turn}
            )
            if slow_read.get("result", {}).get("status") == "completed":
                break
            threading.Event().wait(0.02)
        healthy_thread = _start_thread(second, "PC-09-healthy")
        healthy_turn = _start_turn(second, healthy_thread, "healthy connection")
        healthy_terminal = second.wait_terminal(healthy_turn, timeout=5.0)
        slow_closed = False
        close_deadline = time.monotonic() + SCENARIO_DEADLINE_S
        slow._socket.settimeout(0.25)
        while time.monotonic() < close_deadline:
            try:
                chunk = slow._socket.recv(256 * 1024)
            except TimeoutError:
                continue
            if not chunk:
                slow_closed = True
                break
        checks.append(
            CheckResult(
                "PC-09",
                slow_read.get("result", {}).get("status") == "completed"
                and _terminal_status(healthy_terminal) == "completed"
                and _event_turn(healthy_terminal).get("finalResponse")
                == "healthy after overflow",
                {
                    "slowConnectionClosed": slow_closed,
                    "slowTurnStatus": slow_read.get("result", {}).get("status"),
                    "overflowToolCalls": len(overflow_calls),
                    "healthyTurn": _event_turn(healthy_terminal),
                },
            )
        )
        slow.close()
        clients.remove(slow)

        # 6. 可达 gate failure 发生在 tool started 后，owner 闭合 failed item。
        failed_thread = _start_thread(second, "PC-10")
        failed_turn = _start_turn(second, failed_thread, "pc10 fail after tool started")
        failed_terminal = second.wait_terminal(failed_turn)
        failed_read = second.request(
            "turn/read", {"threadId": failed_thread, "turnId": failed_turn}
        )
        _ = second.request("server/status", {})
        failed_payload = _event_turn(failed_terminal)
        failed_events = _recorded_turn_notifications(events_path, failed_turn)
        failed_terminal_count = sum(
            event.get("method") == "turn/completed" for event in failed_events
        )
        failed_error = failed_payload.get("error")
        pc10_started, pc10_completed = _tool_lifecycle(
            failed_events, "pc10_failure_probe"
        )
        persisted_failed_item = next(
            (
                item
                for item in failed_payload.get("items", [])
                if isinstance(item, dict) and item.get("id") == pc10_started.get("id")
            ),
            None,
        )
        checks.append(
            CheckResult(
                "PC-10",
                failed_payload.get("status") == "failed"
                and isinstance(failed_error, dict)
                and failed_error.get("type") == "RuntimeError"
                and failed_error.get("retryable") is False
                and pc10_completed.get("id") == pc10_started.get("id")
                and pc10_completed.get("data", {}).get("status") == "failed"
                and persisted_failed_item == pc10_completed
                and failed_terminal_count == 1
                and failed_read.get("result") == failed_payload,
                {
                    "terminal": failed_terminal,
                    "read": failed_read,
                    "terminalEventCount": failed_terminal_count,
                    "toolStarted": pc10_started,
                    "toolCompleted": pc10_completed,
                    "failureSource": "gate before_reasoning fixture",
                },
            )
        )

        # 7. WebSocket channel adapter 保留领域投影、完整出站字段和 lane 语义。
        from websockets.sync.client import connect as connect_websocket

        websocket_url = "ws://akashic-control-gate:2236/ws"
        with connect_websocket(websocket_url, open_timeout=READINESS_DEADLINE_S) as web:
            web.send(
                json.dumps({"type": "session.create", "request_id": "pc16-create"})
            )
            created = json.loads(web.recv(timeout=SCENARIO_DEADLINE_S))
            web_thread = str(created["session_id"])

            fixtures = (
                (
                    "parity success",
                    {
                        "mode": "complete",
                        "content": "<think>channel reasoning</think>parity result",
                    },
                ),
                (
                    "parity failure",
                    [
                        {"mode": "error", "status": 500},
                        {"mode": "error", "status": 500},
                    ],
                ),
            )
            parity_evidence: list[dict[str, object]] = []
            parity_passed = True
            for index, (input_text, script) in enumerate(fixtures):
                _http_json("PUT", f"{model_url}/control/script", script)
                program_thread = _start_thread(first, f"PC-16-{index}")
                program_turn = _start_turn(first, program_thread, input_text)
                program_terminal = _event_turn(first.wait_terminal(program_turn))

                _http_json("PUT", f"{model_url}/control/script", script)
                web.send(
                    json.dumps(
                        {
                            "type": "message.send",
                            "request_id": f"pc16-{index}",
                            "session_id": web_thread,
                            "text": input_text,
                            "media": [],
                        }
                    )
                )
                final_frame = _receive_web_final(web)
                channel_turn = _wait_database_turn(
                    Path("/sandbox/workspace/sessions.db"), web_thread, input_text
                )
                program_projection = _turn_projection(program_terminal)
                channel_projection = _turn_projection(channel_turn)
                frame_projection = {
                    "content": final_frame.get("content"),
                    "thinking": final_frame.get("thinking"),
                    "media": final_frame.get("media"),
                    "metadata": final_frame.get("metadata"),
                    "duration_ms": final_frame.get("duration_ms"),
                }
                frame_fields_passed = (
                    isinstance(frame_projection["thinking"], str)
                    and isinstance(frame_projection["media"], list)
                    and isinstance(frame_projection["metadata"], dict)
                    and frame_projection["duration_ms"]
                    == cast(dict[str, object], frame_projection["metadata"]).get(
                        "turn_duration_ms"
                    )
                )
                if input_text == "parity success":
                    frame_fields_passed = (
                        frame_fields_passed
                        and frame_projection["content"] == "parity result"
                        and frame_projection["thinking"] == "channel reasoning"
                    )
                fixture_passed = (
                    program_projection == channel_projection and frame_fields_passed
                )
                parity_passed = parity_passed and fixture_passed
                parity_evidence.append(
                    {
                        "input": input_text,
                        "passed": fixture_passed,
                        "programmatic": program_projection,
                        "channel": channel_projection,
                        "channelFrame": frame_projection,
                    }
                )

        lane_evidence: dict[str, object] = {}
        database = Path("/sandbox/workspace/sessions.db")
        with (
            connect_websocket(
                websocket_url, open_timeout=READINESS_DEADLINE_S
            ) as slow_web,
            connect_websocket(
                websocket_url, open_timeout=READINESS_DEADLINE_S
            ) as fast_web,
        ):
            slow_web.send(
                json.dumps({"type": "session.create", "request_id": "pc16-slow"})
            )
            fast_web.send(
                json.dumps({"type": "session.create", "request_id": "pc16-fast"})
            )
            slow_thread = str(
                json.loads(slow_web.recv(timeout=SCENARIO_DEADLINE_S))["session_id"]
            )
            fast_thread = str(
                json.loads(fast_web.recv(timeout=SCENARIO_DEADLINE_S))["session_id"]
            )
            _create_barrier(
                model_url,
                "pc16-channel-slow",
                {"mode": "complete", "content": "slow complete"},
            )
            _http_json(
                "PUT",
                f"{model_url}/control/script",
                {"mode": "complete", "content": "fast complete"},
            )
            slow_web.send(
                json.dumps(
                    {
                        "type": "message.send",
                        "request_id": "pc16-slow-turn",
                        "session_id": slow_thread,
                        "text": "slow lane",
                        "media": [],
                    }
                )
            )
            _wait_barrier(model_url, "pc16-channel-slow")
            fast_web.send(
                json.dumps(
                    {
                        "type": "message.send",
                        "request_id": "pc16-fast-turn",
                        "session_id": fast_thread,
                        "text": "fast lane",
                        "media": [],
                    }
                )
            )
            fast_completed = _wait_database_turn_status(
                database, fast_thread, "fast lane", {"completed"}
            )
            fast_final = _receive_web_final(fast_web)
            _release_barrier(model_url, "pc16-channel-slow")
            slow_final = _receive_web_final(slow_web)
            lane_evidence["differentThreads"] = {
                "fastCompletedBeforeRelease": fast_completed,
                "slowFinal": slow_final.get("content"),
                "fastFinal": fast_final.get("content"),
            }

        with connect_websocket(
            websocket_url, open_timeout=READINESS_DEADLINE_S
        ) as lane_web:
            lane_web.send(
                json.dumps({"type": "session.create", "request_id": "pc16-lane"})
            )
            lane_thread = str(
                json.loads(lane_web.recv(timeout=SCENARIO_DEADLINE_S))["session_id"]
            )
            _create_barrier(
                model_url,
                "pc16-strict-lane",
                {"mode": "complete", "content": "order one final"},
            )
            _http_json(
                "PUT",
                f"{model_url}/control/script",
                [
                    {"mode": "complete", "content": "order two final"},
                    {"mode": "complete", "content": "order three final"},
                    {"mode": "complete", "content": "order four final"},
                ],
            )
            lane_web.send(
                json.dumps(
                    {
                        "type": "message.send",
                        "request_id": "pc16-order-1",
                        "session_id": lane_thread,
                        "text": "order one",
                        "media": [],
                    }
                )
            )
            _wait_barrier(model_url, "pc16-strict-lane")
            for request_id, text in (
                ("pc16-order-2", "order two"),
                ("pc16-order-3", "order three"),
                ("pc16-order-4", "order four"),
            ):
                lane_web.send(
                    json.dumps(
                        {
                            "type": "message.send",
                            "request_id": request_id,
                            "session_id": lane_thread,
                            "text": text,
                            "media": [],
                        }
                    )
                )
            active_inputs = _wait_database_turn_inputs(
                database, lane_thread, "order one", 1
            )
            _release_barrier(model_url, "pc16-strict-lane")
            ordered_finals = [_receive_web_final(lane_web) for _ in range(4)]
            ordered_turns = [
                _wait_database_turn(database, lane_thread, f"order {name}")
                for name in ("one", "two", "three", "four")
            ]

            _http_json(
                "PUT",
                f"{model_url}/control/script",
                [{"mode": "error", "status": 500} for _ in range(4)],
            )
            lane_web.send(
                json.dumps(
                    {
                        "type": "message.send",
                        "request_id": "pc16-fail",
                        "session_id": lane_thread,
                        "text": "lane failure",
                        "media": [],
                    }
                )
            )
            failed_final = _receive_web_final(lane_web)
            failed_state = _wait_database_turn_status(
                database, lane_thread, "lane failure", {"failed"}
            )

            _http_json(
                "PUT",
                f"{model_url}/control/script",
                {"mode": "complete", "content": "recovered"},
            )
            lane_web.send(
                json.dumps(
                    {
                        "type": "message.send",
                        "request_id": "pc16-recover",
                        "session_id": lane_thread,
                        "text": "lane recovery",
                        "media": [],
                    }
                )
            )
            recovered_final = _receive_web_final(lane_web)
            recovered_state = _wait_database_turn_status(
                database, lane_thread, "lane recovery", {"completed"}
            )
            lane_evidence["sameThread"] = {
                "activeInputs": active_inputs,
                "orderedTurns": [_turn_projection(turn) for turn in ordered_turns],
                "finals": [
                    *[frame.get("content") for frame in ordered_finals],
                    failed_final.get("content"),
                    recovered_final.get("content"),
                ],
                "statuses": [
                    *[turn["status"] for turn in ordered_turns],
                    failed_state["status"],
                    recovered_state["status"],
                ],
            }

        different_threads = cast(dict[str, object], lane_evidence["differentThreads"])
        same_thread = cast(dict[str, object], lane_evidence["sameThread"])
        lane_passed = (
            cast(
                dict[str, object],
                different_threads["fastCompletedBeforeRelease"],
            )["status"]
            == "completed"
            and different_threads["slowFinal"] == "slow complete"
            and different_threads["fastFinal"] == "fast complete"
            and cast(dict[str, object], same_thread["activeInputs"])["userInputs"]
            == ["order one"]
            and same_thread["finals"]
            == [
                "order one final",
                "order two final",
                "order three final",
                "order four final",
                "处理消息时出错，请稍后再试。",
                "recovered",
            ]
            and same_thread["statuses"]
            == [
                "completed",
                "completed",
                "completed",
                "completed",
                "failed",
                "completed",
            ]
        )
        checks.append(
            CheckResult(
                "PC-16",
                parity_passed and lane_passed,
                {
                    "projection": parity_evidence,
                    "lanes": lane_evidence,
                    "mediaCoverage": "tests/test_web_chat_channel.py adapter fixture",
                },
            )
        )

        # 8. 非法 JSON/params 返回稳定 error，随后新连接仍可 readiness。
        raw_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        raw_socket.connect(str(endpoint))
        raw_reader = raw_socket.makefile("rb")
        raw_socket.sendall(b"{invalid json\n")
        parse_error = json.loads(raw_reader.readline())
        raw_reader.close()
        raw_socket.close()
        invalid_params = second.request_raw(
            "thread/start", {"metadata": {}, "unexpected": True}
        )
        healthy = _connect_client(endpoint, events_path)
        clients.append(healthy)
        healthy_status = healthy.request("server/status", {})
        checks.append(
            CheckResult(
                "PC-11",
                parse_error.get("error", {}).get("code") == -32700
                and invalid_params.get("error", {}).get("code") == -32602
                and healthy_status.get("result", {}).get("ready") is True,
                {
                    "parseError": parse_error,
                    "invalidParams": invalid_params,
                    "healthyStatus": healthy_status,
                },
            )
        )
    except Exception as error:
        checks.append(
            CheckResult(
                "controller",
                False,
                {"type": type(error).__name__, "message": str(error)},
            )
        )
    finally:
        for client in clients:
            client.close()

    requests = _http_json("GET", f"{model_url}/control/requests")
    model_requests = _model_requests(requests)
    _write_jsonl(report_dir / "model-requests.jsonl", model_requests)
    _write_json(report_dir / "restart-state.json", restart_state)
    passed = bool(checks) and all(check.passed for check in checks)
    report = {
        "gate": "failure-matrix",
        "status": "passed" if passed else "failed",
        "checks": [asdict(check) for check in checks],
    }
    _write_json(report_dir / "inside-gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def _inside_restart_check(report_dir: Path) -> int:
    """重启后验证协议 readiness 与既有 turn 持久可读。"""

    endpoint = Path("/sandbox/akashic.sock")
    _wait_socket(endpoint, READINESS_DEADLINE_S)
    client = _connect_client(endpoint, report_dir / "events.jsonl")
    try:
        state = json.loads(
            (report_dir / "restart-state.json").read_text(encoding="utf-8")
        )
        turn = client.request(
            "turn/read",
            {"threadId": state["threadId"], "turnId": state["turnId"]},
        )
    finally:
        client.close()
    passed = turn.get("result", {}).get("status") == "completed"
    result = CheckResult("PC-13", passed, turn)
    _write_json(report_dir / "restart-check.json", asdict(result))
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if passed else 1


def _inside_soak(report_dir: Path) -> int:
    """执行 10 次预热和 100 次混合 turn，并记录稳定终态。"""

    report_dir.mkdir(parents=True, exist_ok=True)
    endpoint = Path("/sandbox/akashic.sock")
    events_path = report_dir / "events.jsonl"
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
    _wait_socket(endpoint, READINESS_DEADLINE_S)
    client = _connect_client(endpoint, events_path)
    counts = {"completed": 0, "failed": 0, "interrupted": 0, "reconnects": 0}
    turn_ids: list[str] = []

    def run_complete(index: int, *, warmup: bool = False) -> None:
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            {"mode": "complete", "content": f"soak-{index}"},
        )
        thread_id = _start_thread(client, "G5-warmup" if warmup else "G5")
        turn_id = _start_turn(client, thread_id, f"soak complete {index}")
        terminal = client.wait_terminal(turn_id)
        if _terminal_status(terminal) != "completed":
            raise GateFailure(f"soak complete turn 非 completed：{turn_id}")
        counts["completed"] += 1
        turn_ids.append(turn_id)

    try:
        # 1. 预热完成后等待 controller 采集资源基线。
        for index in range(10):
            run_complete(index, warmup=True)
        _write_json(
            report_dir / "soak-progress.json",
            {"phase": "warmup", "completed": 10, "counts": counts},
        )
        start_barrier = report_dir / "soak-start"
        deadline = time.monotonic() + READINESS_DEADLINE_S
        while not start_barrier.exists():
            if time.monotonic() >= deadline:
                raise GateFailure("controller 未释放 soak-start barrier")
            threading.Event().wait(0.02)

        # 2. 100 turns：10 reconnect、10 interrupt、10 provider failure。
        for index in range(100):
            if index % 10 == 0:
                client.close()
                client = _connect_client(endpoint, events_path)
                counts["reconnects"] += 1
            if index < 10:
                barrier = f"soak-interrupt-{index}"
                _create_barrier(
                    model_url,
                    barrier,
                    {"mode": "complete", "content": "must interrupt"},
                )
                thread_id = _start_thread(client, "G5-interrupt")
                turn_id = _start_turn(client, thread_id, f"soak interrupt {index}")
                _wait_barrier(model_url, barrier)
                client.request(
                    "turn/interrupt", {"threadId": thread_id, "turnId": turn_id}
                )
                terminal = client.wait_terminal(turn_id, timeout=2)
                _release_barrier(model_url, barrier)
                if _terminal_status(terminal) != "interrupted":
                    raise GateFailure(f"soak interrupt turn 非 interrupted：{turn_id}")
                counts["interrupted"] += 1
                turn_ids.append(turn_id)
            elif index < 20:
                _http_json(
                    "PUT",
                    f"{model_url}/control/script",
                    [
                        {"mode": "error", "status": 500},
                        {"mode": "error", "status": 500},
                    ],
                )
                thread_id = _start_thread(client, "G5-failure")
                turn_id = _start_turn(client, thread_id, f"soak failure {index}")
                terminal = client.wait_terminal(turn_id)
                if _terminal_status(terminal) != "failed":
                    raise GateFailure(f"soak failure turn 非 failed：{turn_id}")
                counts["failed"] += 1
                turn_ids.append(turn_id)
            else:
                run_complete(index)
            if (index + 1) % 10 == 0:
                _write_json(
                    report_dir / "soak-progress.json",
                    {
                        "phase": "run",
                        "completed": index + 1,
                        "counts": counts,
                    },
                )
    finally:
        client.close()

    expected = {
        "completed": 90,
        "failed": 10,
        "interrupted": 10,
        "reconnects": 10,
    }
    passed = counts == expected and len(set(turn_ids)) == 110
    result = CheckResult(
        "G5-turns",
        passed,
        {"counts": counts, "uniqueTurns": len(set(turn_ids)), "expected": expected},
    )
    _write_json(
        report_dir / "inside-gate.json",
        {
            "gate": "soak",
            "status": "passed" if passed else "failed",
            "checks": [asdict(result)],
        },
    )
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if passed else 1


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, items: list[object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def _snapshot_database(database: Path) -> dict[str, object]:
    """读取控制面相关 SQLite 终态，缺失数据库时明确记录。"""

    if not database.exists():
        return {"exists": False, "path": str(database)}
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        table_names = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        tables: dict[str, list[dict[str, object]]] = {}
        for name in ("sessions", "turns", "operations"):
            if name not in table_names:
                continue
            rows = connection.execute(f'SELECT * FROM "{name}"').fetchall()
            tables[name] = [dict(row) for row in rows]
    return {"exists": True, "path": str(database), "tables": tables}


def _repository_digest(repo: Path) -> dict[str, str]:
    """计算受 Git 管理且未 ignore 文件的内容摘要。"""

    output = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    result: dict[str, str] = {}
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        relative = os.fsdecode(raw_path)
        path = repo / relative
        if path.is_file() and not path.is_symlink():
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _write_config(
    sandbox: Path,
    *,
    context_window: int = 64_000,
    max_iterations: int = 2,
) -> None:
    """渲染只连接 compose 私网 model-gate 的隔离配置。"""

    config = f"""[llm]
main = "model_gate"

[llm.runtimes.model_gate]
provider = "openai"
model = "model-gate"
api_key = "model-gate-local"
base_url = "http://model-gate:8090/v1"
context_window = {context_window}

[agent]
system_prompt = "Return the deterministic model-gate response."
max_iterations = {max_iterations}
max_tokens = 64
spawn_enabled = false

[agent.context]
[agent.context.compaction]
keep_recent_tokens = 20000

[agent.maintenance]
memory_optimizer_enabled = false

[app_server]
enabled = true
listen = "/sandbox/akashic.sock"
max_connections = 8
ingress_queue_size = 32
outbound_queue_size = 64

[channels.chat]
enabled = true
channel_name = "web"

[channels.telegram]
enabled = false
token = ""

[channels.qq]
enabled = false
bot_uin = ""

[proactive]
enabled = false
profile = "quiet"
"""
    (sandbox / "config.toml").write_text(config, encoding="utf-8")


def _initialize_current_workspace(workspace: Path, source_root: Path) -> None:
    """为已标记 current 的 Gate workspace 写入当前版本必需资产。"""

    # 1. Gate fixture 必须使用候选源码中的版本化默认值。
    template = source_root / "prompts/VEDA.md"
    try:
        payload = template.read_bytes()
        content = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise GateFailure(f"无法读取 Gate Veda 模板: {template}") from exc
    if not content.strip():
        raise GateFailure(f"Gate Veda 模板为空: {template}")

    # 2. current cursor 禁止依赖 migration 补齐当前格式资产。
    target = workspace / "memory/VEDA.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _prepare_host_sandbox(
    sandbox: Path,
    source_root: Path,
    *,
    max_iterations: int = 2,
) -> None:
    """创建 control gate 独占的运行目录和可写静态目录。"""

    # 1. 复制当前工作树，确保 /app mountpoint 也完全归 sandbox 所有。
    source_root = source_root.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).resolve().relative_to(source_root)
        ignored = {
            name
            for name in names
            if name in {"__pycache__", ".pytest_cache", ".venv", "node_modules"}
            or name.endswith(".pyc")
        }
        if relative == Path("."):
            ignored.update({".git", "static"})
        if relative == Path("docker/debug"):
            ignored.add("reports")
        return ignored

    shutil.copytree(
        source_root,
        sandbox / "app",
        symlinks=True,
        ignore=ignore,
    )
    (sandbox / "app/static").mkdir()

    # 2. 所有运行时写入均归外部 sandbox，不依赖仓库 ignored 目录。
    (sandbox / "workspace").mkdir(parents=True)
    _initialize_current_workspace(sandbox / "workspace", sandbox / "app")
    (sandbox / "home").mkdir()
    (sandbox / "reports").mkdir()
    (sandbox / "static/dashboard").mkdir(parents=True)
    (sandbox / "static/chat").mkdir()

    # 3. 配置只引用同一 sandbox 内的路径。
    _write_config(sandbox, max_iterations=max_iterations)


def _install_control_failure_plugin(sandbox: Path) -> None:
    """安装只为 PC10 构造 started 后 gate failure 的隔离插件。"""

    cache = sandbox / "home/.akashic-plugin/cache/gate/control_failure/1.0.0"
    manifest = sandbox / "home/.akashic-plugin/manifest.toml"
    cache.mkdir(parents=True, exist_ok=True)
    _ = (cache / "plugin.py").write_text(
        "from agent.control.context import current_turn_id\n"
        "from agent.plugins import Plugin, on_before_reasoning\n"
        "from bus.events_lifecycle import ToolCallStarted\n"
        "class ControlFailurePlugin(Plugin):\n"
        "    name = 'control_failure'\n"
        "    version = '1.0.0'\n"
        "    @on_before_reasoning()\n"
        "    async def fail_after_started(self, event):\n"
        "        if event.content != 'pc10 fail after tool started': return\n"
        "        await self.context.event_bus.observe(ToolCallStarted(\n"
        "            session_key=event.session_key, channel=event.channel,\n"
        "            chat_id=event.chat_id, iteration=1, call_id='call_pc10_open',\n"
        "            tool_name='pc10_failure_probe', arguments={'probe': True},\n"
        "            turn_id=current_turn_id.get()))\n"
        "        raise RuntimeError('pc10 gate failure after tool started')\n",
        encoding="utf-8",
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _ = manifest.write_text(
        '[plugins."control_failure@gate"]\nenabled = true\n',
        encoding="utf-8",
    )


def _seed_memory_context_fixture(
    compose: list[str],
    repo: Path,
    env: dict[str, str],
) -> None:
    """在 gateway 启动前用生产 SessionManager 写入分页测试会话。"""

    script = """
from pathlib import Path
from session.manager import SessionManager

manager = SessionManager(Path("/sandbox/workspace"))
session = manager.get_or_create("programmatic:context-ledger")
for index in range(4):
    control_turn_id = f"memory-gate-seed-{index}"
    session.add_message(
        "user",
        (f"seed user {index} " + "token " * 5000).strip(),
        control_turn_id=control_turn_id,
    )
    session.add_message(
        "assistant",
        (f"seed assistant {index} " + "token " * 5000).strip(),
        control_turn_id=control_turn_id,
    )
manager.save(session)
manager.close()
"""
    seeded = subprocess.run(
        [
            *compose,
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "--entrypoint",
            "python",
            "akashic-control-gate",
            "-c",
            script,
        ],
        cwd=repo,
        env=env,
        check=False,
    )
    if seeded.returncode != 0:
        raise GateFailure(f"memory context fixture seed failed: {seeded.returncode}")


def _run_stdio_check(
    compose: list[str],
    repo: Path,
    env: dict[str, str],
    report_dir: Path,
) -> CheckResult:
    """以真实 compose run 驱动 stdio framing 并检查流隔离。"""

    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "clientInfo": {"name": "docker-stdio-gate", "version": "1.0"},
                "capabilities": {"reasoningEvents": False},
            },
        },
        {"jsonrpc": "2.0", "method": "initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "server/status", "params": {}},
    ]
    payload = "".join(json.dumps(item) + "\n" for item in messages)
    command = [
        *compose,
        "run",
        "--rm",
        "-T",
        "--no-deps",
        "akashic-control-gate",
        "app-server",
        "--stdio",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=env,
            input=payload,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=READINESS_DEADLINE_S,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return CheckResult(
            "PC-02", False, {"error": f"stdio deadline exceeded: {error}"}
        )
    (report_dir / "stdio.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (report_dir / "server.stderr.log").write_text(completed.stderr, encoding="utf-8")
    parsed: list[object] = []
    parse_error = ""
    for line in completed.stdout.splitlines():
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as error:
            parse_error = str(error)
            break
    response_ids = {
        item.get("id")
        for item in parsed
        if isinstance(item, dict) and item.get("jsonrpc") == "2.0"
    }
    stderr_protocol_frames: list[object] = []
    for line in completed.stderr.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("jsonrpc") == "2.0":
            stderr_protocol_frames.append(item)
    passed = (
        completed.returncode == 0
        and not parse_error
        and {1, 2} <= response_ids
        and not stderr_protocol_frames
    )
    return CheckResult(
        "PC-02",
        passed,
        {
            "returncode": completed.returncode,
            "frames": len(parsed),
            "responseIds": sorted(str(item) for item in response_ids),
            "parseError": parse_error,
            "stderrProtocolFrames": stderr_protocol_frames,
        },
    )


def _run_inside(
    compose: list[str],
    repo: Path,
    env: dict[str, str],
    *,
    gate: str,
    phase: str,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            *compose,
            "run",
            "--rm",
            "-T",
            "control-probe",
            "python",
            "docker/debug/programmatic_control_probe.py",
            "--gate",
            gate,
            "--inside-container",
            "--phase",
            phase,
            "--report-dir",
            "/sandbox/reports",
        ],
        cwd=repo,
        env=env,
        check=False,
    )


def _workspace_lock_check(
    compose: list[str], repo: Path, env: dict[str, str], report_dir: Path
) -> CheckResult:
    """启动第二个 workspace owner，要求 fail-loud 且不伤害 gateway。"""

    completed = subprocess.run(
        [
            *compose,
            "run",
            "--rm",
            "-T",
            "--no-deps",
            "akashic-control-gate",
            "app-server",
            "--stdio",
        ],
        cwd=repo,
        env=env,
        input=b"",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=READINESS_DEADLINE_S,
        check=False,
    )
    (report_dir / "workspace-lock.stderr.log").write_bytes(completed.stderr)
    return CheckResult(
        "PC-14",
        completed.returncode != 0,
        {
            "returncode": completed.returncode,
            "stderrTail": completed.stderr.decode(errors="replace")[-2000:],
        },
    )


def _non_terminal_turns(snapshot: dict[str, object]) -> list[dict[str, object]]:
    tables = snapshot.get("tables")
    if not isinstance(tables, dict):
        return []
    turns = tables.get("turns")
    if not isinstance(turns, list):
        return []
    return [
        turn
        for turn in turns
        if isinstance(turn, dict) and turn.get("status") in {"queued", "in_progress"}
    ]


def _sample_resources(
    compose: list[str], repo: Path, env: dict[str, str], milestone: int
) -> dict[str, int | float]:
    """从真实 gateway PID 1 读取 RSS、fd 和线程数。"""

    script = (
        "import json, pathlib; "
        "status=pathlib.Path('/proc/1/status').read_text(); "
        "rss=next(int(line.split()[1]) for line in status.splitlines() if line.startswith('VmRSS:')); "
        "print(json.dumps({'rssKiB':rss,'fdCount':len(list(pathlib.Path('/proc/1/fd').iterdir())),"
        "'threadCount':len(list(pathlib.Path('/proc/1/task').iterdir()))}))"
    )
    completed = subprocess.run(
        [
            *compose,
            "exec",
            "-T",
            "akashic-control-gate",
            "python",
            "-c",
            script,
        ],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise GateFailure(f"resource sample failed: {completed.stderr[-1000:]}")
    payload = json.loads(completed.stdout.splitlines()[-1])
    return {
        "timestamp": time.time(),
        "milestone": milestone,
        "rssKiB": int(payload["rssKiB"]),
        "fdCount": int(payload["fdCount"]),
        "threadCount": int(payload["threadCount"]),
    }


def _run_soak(
    compose: list[str],
    repo: Path,
    env: dict[str, str],
    sandbox: Path,
    report_dir: Path,
) -> list[CheckResult]:
    """并行采样 100-turn soak 资源，并执行公开增量阈值。"""

    command = [
        *compose,
        "run",
        "--rm",
        "-T",
        "control-probe",
        "python",
        "docker/debug/programmatic_control_probe.py",
        "--gate",
        "soak",
        "--inside-container",
        "--phase",
        "scenarios",
        "--report-dir",
        "/sandbox/reports",
    ]
    process = subprocess.Popen(command, cwd=repo, env=env)
    progress_path = sandbox / "reports/soak-progress.json"
    samples: list[dict[str, int | float]] = []
    sampled_milestones: set[int] = set()
    deadline = time.monotonic() + 600
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                raise GateFailure("soak 超过 10 分钟 deadline")
            if progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                phase = progress.get("phase")
                milestone = int(progress.get("completed", 0))
                sample_key = 0 if phase == "warmup" else milestone
                if sample_key not in sampled_milestones:
                    samples.append(_sample_resources(compose, repo, env, sample_key))
                    sampled_milestones.add(sample_key)
                    if phase == "warmup":
                        (sandbox / "reports/soak-start").touch()
            threading.Event().wait(0.05)
        returncode = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    shutil.copytree(sandbox / "reports", report_dir, dirs_exist_ok=True)
    with (report_dir / "resource.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample) + "\n")
    inside_payload = json.loads(
        (sandbox / "reports/inside-gate.json").read_text(encoding="utf-8")
    )
    checks = [CheckResult(**item) for item in inside_payload["checks"]]
    if returncode != 0:
        checks.append(CheckResult("G5-process", False, {"returncode": returncode}))
        return checks
    if len(samples) < 11:
        checks.append(CheckResult("G5-resources", False, {"samples": len(samples)}))
        return checks
    baseline = samples[0]
    final = samples[-1]
    rss_delta = int(final["rssKiB"]) - int(baseline["rssKiB"])
    fd_delta = int(final["fdCount"]) - int(baseline["fdCount"])
    thread_delta = int(final["threadCount"]) - int(baseline["threadCount"])
    snapshot = _snapshot_database(sandbox / "workspace/sessions.db")
    non_terminal = _non_terminal_turns(snapshot)
    checks.append(
        CheckResult(
            "G5-resources",
            rss_delta <= 64 * 1024
            and fd_delta <= 8
            and thread_delta <= 3
            and not non_terminal,
            {
                "samples": len(samples),
                "rssDeltaKiB": rss_delta,
                "fdDelta": fd_delta,
                "threadDelta": thread_delta,
                "nonTerminalTurns": non_terminal,
            },
        )
    )
    return checks


def _run_host(gate: str) -> int:
    """拥有完整 compose 生命周期、证据收集、清理和源码审计。"""

    repo = Path(__file__).resolve().parents[2]
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    report_dir = repo / "docker/debug/reports/programmatic-control" / run_id
    report_dir.mkdir(parents=True)
    sandbox = Path(tempfile.mkdtemp(prefix="akashic-control-gate-", dir="/tmp"))
    _prepare_host_sandbox(
        sandbox,
        repo,
        max_iterations=3 if gate == "failure-matrix" else 2,
    )
    if gate == "failure-matrix":
        _install_control_failure_plugin(sandbox)
    elif gate == "memory-context":
        _write_config(sandbox, context_window=100_000)
    before = _repository_digest(repo)
    _write_json(report_dir / "repo-digest.before.json", before)
    env = {
        **os.environ,
        "AKASHIC_CONTROL_SANDBOX": str(sandbox),
        "UID": str(os.getuid()),
        "GID": str(os.getgid()),
    }
    project = f"akashic-control-{run_id.lower()}"
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(repo / "docker/debug/docker-compose.control-gate.yml"),
    ]
    checks: list[CheckResult] = []
    cleanup_returncode = -1
    controller_error = ""
    try:
        build = subprocess.run(
            [*compose, "build", "model-gate"], cwd=repo, env=env, check=False
        )
        if build.returncode != 0:
            raise GateFailure(f"control-gate image build failed: {build.returncode}")
        if gate == "memory-context":
            _seed_memory_context_fixture(compose, repo, env)
        up = subprocess.run(
            [*compose, "up", "-d", "model-gate", "akashic-control-gate"],
            cwd=repo,
            env=env,
            check=False,
        )
        if up.returncode != 0:
            raise GateFailure(f"compose up failed: {up.returncode}")
        if gate == "soak":
            checks.extend(_run_soak(compose, repo, env, sandbox, report_dir))
            inside = None
        else:
            inside = _run_inside(compose, repo, env, gate=gate, phase="scenarios")
        inside_report = sandbox / "reports/inside-gate.json"
        if inside is not None and inside_report.exists():
            shutil.copytree(sandbox / "reports", report_dir, dirs_exist_ok=True)
            payload = json.loads(inside_report.read_text(encoding="utf-8"))
            checks.extend(CheckResult(**item) for item in payload["checks"])
        if inside is not None and inside.returncode != 0:
            raise GateFailure(f"inside {gate} failed: {inside.returncode}")
        if gate == "failure-matrix":
            checks.append(_workspace_lock_check(compose, repo, env, report_dir))
        stop_started = time.monotonic()
        gateway_stop = subprocess.run(
            [*compose, "stop", "-t", "15", "akashic-control-gate"],
            cwd=repo,
            env=env,
            check=False,
        )
        if gateway_stop.returncode != 0:
            raise GateFailure(f"gateway stop failed: {gateway_stop.returncode}")
        stop_duration = time.monotonic() - stop_started
        if gate == "smoke":
            checks.append(_run_stdio_check(compose, repo, env, report_dir))
        elif gate == "failure-matrix":
            stopped_snapshot = _snapshot_database(sandbox / "workspace/sessions.db")
            non_terminal = _non_terminal_turns(stopped_snapshot)
            checks.append(
                CheckResult(
                    "PC-12",
                    stop_duration <= 15 and not non_terminal,
                    {
                        "durationSeconds": stop_duration,
                        "nonTerminalTurns": non_terminal,
                    },
                )
            )
            restart = subprocess.run(
                [*compose, "start", "akashic-control-gate"],
                cwd=repo,
                env=env,
                check=False,
            )
            if restart.returncode != 0:
                raise GateFailure(f"gateway restart failed: {restart.returncode}")
            restart_check = _run_inside(
                compose,
                repo,
                env,
                gate=gate,
                phase="restart-check",
            )
            restart_payload = sandbox / "reports/restart-check.json"
            if restart_payload.exists():
                checks.append(CheckResult(**json.loads(restart_payload.read_text())))
            if restart_check.returncode != 0:
                raise GateFailure(
                    f"graceful restart check failed: {restart_check.returncode}"
                )

            crash = subprocess.run(
                [*compose, "kill", "-s", "SIGKILL", "akashic-control-gate"],
                cwd=repo,
                env=env,
                check=False,
            )
            restart_after_crash = subprocess.run(
                [*compose, "start", "akashic-control-gate"],
                cwd=repo,
                env=env,
                check=False,
            )
            crash_check = _run_inside(
                compose,
                repo,
                env,
                gate=gate,
                phase="restart-check",
            )
            checks.append(
                CheckResult(
                    "PC-13-crash",
                    crash.returncode == 0
                    and restart_after_crash.returncode == 0
                    and crash_check.returncode == 0,
                    {
                        "killReturncode": crash.returncode,
                        "restartReturncode": restart_after_crash.returncode,
                        "probeReturncode": crash_check.returncode,
                    },
                )
            )
    except Exception as error:
        controller_error = f"{type(error).__name__}: {error}"
    finally:
        logs = subprocess.run(
            [*compose, "logs", "--no-color"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (report_dir / "compose.log").write_text(logs.stdout, encoding="utf-8")
        cleanup = subprocess.run(
            [*compose, "down", "--remove-orphans", "--volumes"],
            cwd=repo,
            env=env,
            check=False,
        )
        cleanup_returncode = cleanup.returncode
        residual = subprocess.run(
            [*compose, "ps", "-aq"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        residual_containers = residual.stdout.split()

    _write_json(
        report_dir / "db-snapshot.json",
        _snapshot_database(sandbox / "workspace/sessions.db"),
    )

    after = _repository_digest(repo)
    _write_json(report_dir / "repo-digest.after.json", after)
    checks.append(
        CheckResult(
            "PC-15",
            cleanup_returncode == 0 and not residual_containers and before == after,
            {
                "cleanupReturncode": cleanup_returncode,
                "repositoriesUnchanged": before == after,
                "residualContainers": residual_containers,
                "sandbox": str(sandbox),
                "composeProject": project,
            },
        )
    )
    passed = not controller_error and checks and all(check.passed for check in checks)
    report = {
        "runId": run_id,
        "gate": gate,
        "status": "passed" if passed else "failed",
        "checks": [asdict(check) for check in checks],
        "controllerError": controller_error,
        "reportDir": str(report_dir),
    }
    _write_json(report_dir / "gate.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    shutil.rmtree(sandbox)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="程序化控制面 Docker 验收 controller")
    parser.add_argument(
        "--gate",
        required=True,
        choices=("smoke", "failure-matrix", "memory-context", "soak"),
    )
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--phase", default="scenarios")
    parser.add_argument("--report-dir", type=Path, default=Path("/sandbox/reports"))
    args = parser.parse_args()
    if args.inside_container:
        if args.phase == "restart-check":
            return _inside_restart_check(args.report_dir)
        if args.gate == "smoke":
            return _inside_smoke(args.report_dir)
        if args.gate == "failure-matrix":
            return _inside_failure_matrix(args.report_dir)
        if args.gate == "memory-context":
            return _inside_memory_context(args.report_dir)
        if args.gate == "soak":
            return _inside_soak(args.report_dir)
        raise GateFailure(f"未知 inside gate：{args.gate}")
    return _run_host(args.gate)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1) from error
