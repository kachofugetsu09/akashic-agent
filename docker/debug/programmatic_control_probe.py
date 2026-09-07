#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
import shutil
import select
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from docker.debug.model_plugin_fixture import add_openai_models
PROTOCOL_VERSION = "2.0"
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
_DEFAULT_SELF_MD = """# Akashic 的自我认知

## 人格与形象
- 我是 Akashic，一个直接、温暖、主动参与思考的长期协作伙伴。
- 我优先给出结论，再补充必要细节；不把自己伪装成没有立场的工具。

## 我对当前用户的理解
- 我会从长期记忆中逐步形成对当前用户的理解，不在缺少证据时编造画像。

## 我们关系的定义
- 我与当前用户的关系以透明、尊重边界和持续协作为基础。
"""
_MEMORY_CONTEXT_PROFILE_RESPONSE = json.dumps(
    {"memory": "", "self": _DEFAULT_SELF_MD},
    ensure_ascii=False,
)
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

    def request_result(
        self,
        method: str,
        params: dict[str, object],
        *,
        timeout: float = SCENARIO_DEADLINE_S,
    ) -> dict[str, Any]:
        """返回 v2 result，并保留旧场景需要的原始响应。"""

        response = self.request(method, params, timeout=timeout)
        result = response.get("result")
        if not isinstance(result, dict):
            raise GateFailure(f"{method} result 不是 object：{response!r}")
        return cast(dict[str, Any], result)

    def admit_programmatic(
        self, session_id: str, *, persist_memory: bool = False
    ) -> dict[str, Any]:
        """接纳一个属性不可变的程序 Session。"""

        return self.request_result(
            "programmatic/session/admit",
            {"session_id": session_id, "persist_memory": persist_memory},
        )

    def send_programmatic(
        self, session_id: str, message_id: str, text: str
    ) -> dict[str, Any]:
        """追加一个程序 Input，并返回持久 ACK。"""

        return self.request_result(
            "programmatic/message/send",
            {"session_id": session_id, "message_id": message_id, "text": text},
        )

    def read_messages(
        self, session_id: str, *, after_seq: int = -1, limit: int = 200
    ) -> dict[str, Any]:
        """读取 Session 的追加式 Message 页面。"""

        return self.request_result(
            "message/read",
            {"session_id": session_id, "after_seq": after_seq, "limit": limit},
        )

    def programmatic_result(
        self, session_id: str, input_id: str
    ) -> dict[str, Any]:
        """读取一个程序 Input 的结果投影。"""

        return self.request_result(
            "programmatic/message/result",
            {"session_id": session_id, "input_id": input_id},
        )

    def follow_session(
        self, session_id: str, subscription_id: str, *, after_seq: int = -1
    ) -> dict[str, Any]:
        """注册有界 v2 Session 订阅，并返回 ACK。"""

        return self.request_result(
            "session/follow",
            {
                "session_id": session_id,
                "subscription_id": subscription_id,
                "after_seq": after_seq,
            },
        )

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

    def wait_session_event(
        self,
        event_type: str,
        *,
        subscription_id: str | None = None,
        timeout: float = SCENARIO_DEADLINE_S,
    ) -> dict[str, Any]:
        """等待 v2 Session 事件，同时保留无关通知。"""

        deadline = time.monotonic() + timeout

        def matches(event: dict[str, Any]) -> bool:
            if event.get("method") != "session/event":
                return False
            params = event.get("params")
            if not isinstance(params, dict):
                return False
            if subscription_id is not None and params.get("subscription_id") != subscription_id:
                return False
            payload = params.get("event")
            return isinstance(payload, dict) and payload.get("type") == event_type

        while True:
            for index, event in enumerate(self._pending_notifications):
                if matches(event):
                    return self._pending_notifications.pop(index)
            event = self._receive(deadline)
            if matches(event):
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


def _turn_projection(turn: dict[str, Any]) -> dict[str, Any]:
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
                stable_metadata.pop("client_message_id", None)
                stable_metadata.pop("effects", None)
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
                SELECT id, status, final_response, error_json
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
                    "error": (
                        json.loads(row["error_json"]) if row["error_json"] else None
                    ),
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


def _model_call_records(database: Path, call_ids: Sequence[str]) -> list[dict[str, Any]]:
    """从 Models owner 读取指定调用的结算事实。"""

    if not call_ids:
        return []
    placeholders = ",".join("?" for _ in call_ids)
    uri = f"file:{database}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, state, binding_json, usage_json FROM model_calls "
                f"WHERE id IN ({placeholders}) ORDER BY rowid",
                tuple(call_ids),
            ).fetchall()
    except sqlite3.Error as error:
        raise GateFailure(f"读取 Models 调用账失败：{database}") from error
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "id": row["id"],
                "state": row["state"],
                "binding": json.loads(row["binding_json"]),
                "usage": (
                    None
                    if row["usage_json"] is None
                    else json.loads(row["usage_json"])
                ),
            }
        )
    return records


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
        elif "你维护两个长期 Markdown 档案" in serialized:
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
    # Committed fact 在 business response settle 后消费，顺序固定。
    if kinds != ["summary", "business", "markdown"]:
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


def _configure_model_gate(*, context_window: int = 64_000) -> None:
    """Configure the scripted model through the ordinary public plugin API."""

    add_openai_models(
        "http://akashic-control-gate:2236/api/settings/model",
        connection_id="model-gate",
        endpoint="http://model-gate:8090/v1",
        api_key="model-gate-local",
        chat_model="model-gate",
        context_window=context_window,
        allow_unverified_manual=True,
    )


def _connect_client(endpoint: Path, events_path: Path) -> JsonRpcSocketClient:
    """建立 v2 连接并完成 initialize/initialized/status readiness。"""

    client = JsonRpcSocketClient(endpoint, events_path)
    client.request(
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "clientInfo": {"name": "docker-control-gate", "version": "2.0"},
            "capabilities": {"reasoningEvents": False},
        },
        timeout=READINESS_DEADLINE_S,
    )
    client.notify("initialized", {})
    status = client.request("server/status", {}, timeout=READINESS_DEADLINE_S)
    status_result = status.get("result")
    if not isinstance(status_result, dict) or status_result.get("ready") is not True:
        client.close()
        raise GateFailure(f"server/status 未 ready：{status!r}")
    return client


def _create_barrier(model_url: str, name: str, script: dict[str, object]) -> None:
    _http_json("PUT", f"{model_url}/control/barriers/{name}")
    _http_json("PUT", f"{model_url}/control/script", {**script, "barrier": name})


def _create_chunk_barrier(
    model_url: str,
    name: str,
    scripts: list[dict[str, object]],
    *,
    script_index: int,
    after_chunk: int,
) -> None:
    """为指定 stream script 创建可控的 chunk 后 barrier。"""

    if not 0 <= script_index < len(scripts):
        raise ValueError("chunk barrier script_index 超出脚本范围")
    configured = [dict(script) for script in scripts]
    configured[script_index]["chunk_barrier"] = {
        "name": name,
        "after_chunk": after_chunk,
    }
    _http_json("PUT", f"{model_url}/control/barriers/{name}")
    _http_json("PUT", f"{model_url}/control/script", configured)


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


def _message_text(item: object) -> str:
    """Extract text parts from one wire Message row."""
    if not isinstance(item, dict):
        return ""
    body = item.get("body")
    if not isinstance(body, dict):
        return ""
    parts = body.get("parts")
    if not isinstance(parts, list):
        return ""
    return "".join(
        str(part.get("value", ""))
        for part in parts
        if isinstance(part, dict) and part.get("kind") == "text"
    )


def _message_items(page: object, kind: str) -> list[dict[str, Any]]:
    """Return wire rows of one body kind from a message page."""

    if not isinstance(page, dict) or not isinstance(page.get("items"), list):
        return []
    return [
        item
        for item in page["items"]
        if isinstance(item, dict)
        and isinstance(item.get("body"), dict)
        and item["body"].get("kind") == kind
    ]


def _wait_for_message_items(
    client: JsonRpcSocketClient,
    session_id: str,
    kind: str,
    *,
    minimum: int = 1,
    timeout: float = SCENARIO_DEADLINE_S,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Poll the durable page until it contains the requested body rows."""

    deadline = time.monotonic() + timeout
    page: dict[str, Any] = {}
    while time.monotonic() < deadline:
        page = client.read_messages(session_id)
        rows = _message_items(page, kind)
        if len(rows) >= minimum:
            return page, rows
        threading.Event().wait(0.05)
    raise GateFailure(
        f"{session_id} 未在 deadline 内得到 {minimum} 个 {kind}：{page!r}"
    )


def _pid_is_alive(pid: int) -> bool:
    """Check a process identity without treating a missing /proc entry as alive."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _drain_socket_until_eof(
    connection: socket.socket, *, timeout: float = SCENARIO_DEADLINE_S
) -> tuple[bool, int]:
    """Drain buffered frames before checking the peer's EOF."""

    connection.setblocking(False)
    deadline = time.monotonic() + timeout
    drained = 0
    while time.monotonic() < deadline:
        try:
            chunk = connection.recv(65_536)
        except BlockingIOError:
            readable, _, _ = select.select(
                [connection], [], [], min(0.1, max(0.0, deadline - time.monotonic()))
            )
            if not readable:
                continue
            continue
        if not chunk:
            return True, drained
        drained += len(chunk)
    return False, drained


def _wait_programmatic_result(
    client: JsonRpcSocketClient,
    session_id: str,
    input_id: str,
    *,
    timeout: float = SCENARIO_DEADLINE_S,
) -> dict[str, Any]:
    """轮询持久程序结果，直到状态离开 open。"""

    deadline = time.monotonic() + timeout
    result: dict[str, Any] = {}
    while time.monotonic() < deadline:
        result = client.programmatic_result(session_id, input_id)
        if result.get("status") != "open":
            return result
        threading.Event().wait(0.05)
    raise GateFailure(f"programmatic result 超时：{session_id}/{input_id} {result!r}")


def _inside_smoke(report_dir: Path) -> int:
    """从独立 probe 容器验证真实 v2 gateway、provider 和 Message 日志。"""

    report_dir.mkdir(parents=True, exist_ok=True)
    events_path = report_dir / "events.jsonl"
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    endpoint = Path("/sandbox/akashic.sock")
    checks: list[CheckResult] = []
    client: JsonRpcSocketClient | None = None
    try:
        # 1. readiness 必须完成 v2 握手与 server/status。
        _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
        _configure_model_gate()
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        client = JsonRpcSocketClient(endpoint, events_path)
        initialized = client.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientInfo": {"name": "docker-control-gate", "version": "2.0"},
                "capabilities": {"reasoningEvents": False},
            },
            timeout=READINESS_DEADLINE_S,
        )
        client.notify("initialized", {})
        status = client.request("server/status", {}, timeout=READINESS_DEADLINE_S)
        status_result = status.get("result")
        mode = endpoint.stat().st_mode & 0o777
        checks.append(
            CheckResult(
                "PC-01",
                mode == 0o600
                and isinstance(initialized.get("result"), dict)
                and initialized["result"].get("protocolVersion") == "2.0"
                and isinstance(status_result, dict)
                and status_result.get("ready") is True
                and status_result.get("protocolVersion") == "2.0",
                {"initialize": initialized, "status": status, "socketMode": oct(mode)},
            )
        )

        # 2. 程序 Input 穿过 provider，并从 Message 日志读取。
        pc03_session = "programmatic:pc03-smoke"
        admission = client.admit_programmatic(pc03_session)
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            {"mode": "complete", "content": "control gate"},
        )
        pc03_before = len(_model_requests(_http_json("GET", f"{model_url}/control/requests")))
        ack = client.send_programmatic(pc03_session, "pc03-input", "run control gate")
        result = _wait_programmatic_result(client, pc03_session, "pc03-input")
        page = client.read_messages(pc03_session)
        pc03_requests = _model_requests(_http_json("GET", f"{model_url}/control/requests"))[pc03_before:]
        rows = page.get("items")
        input_rows = [item for item in rows if isinstance(item, dict) and item.get("id") == "pc03-input"] if isinstance(rows, list) else []
        output_rows = [item for item in rows if isinstance(item, dict) and item.get("body", {}).get("kind") == "output"] if isinstance(rows, list) else []
        ending_id = result.get("ending_message_id")
        ending_seq = result.get("ending_seq")
        final_outputs = [
            item for item in output_rows
            if isinstance(item, dict)
            and item.get("id") == ending_id
            and isinstance(item.get("body"), dict)
            and item["body"].get("finish") == "complete"
        ]
        final_output = final_outputs[0] if len(final_outputs) == 1 else None
        checks.append(
            CheckResult(
                "PC-03",
                admission.get("session_id") == pc03_session
                and ack.get("message_id") == "pc03-input"
                and result.get("status") == "complete"
                and isinstance(ending_id, str)
                and type(ending_seq) is int
                and final_output is not None
                and final_output.get("seq") == ending_seq
                and len(pc03_requests) == 1
                and len(input_rows) == 1
                and any(
                    isinstance(item.get("body"), dict)
                    and any(
                        part.get("value") == "control gate"
                        for part in item["body"].get("parts", [])
                        if isinstance(part, dict)
                    )
                    for item in output_rows
                ),
                {
                    "admission": admission,
                    "ack": ack,
                    "result": result,
                    "messagePage": page,
                    "endingOutput": final_output,
                    "providerRequestCount": len(pc03_requests),
                },
            )
        )

        # 3. follow 与 provider stream 同时打开，先观察真实中间预览。
        pc04_session = "programmatic:pc04-smoke"
        client.admit_programmatic(pc04_session)
        follow_ack = client.follow_session(pc04_session, "pc04-follow")
        pc04_stream_barrier = f"pc04-stream-{uuid.uuid4().hex}"
        pc04_scripts = [
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
        ]
        _create_chunk_barrier(
            model_url,
            pc04_stream_barrier,
            pc04_scripts,
            script_index=1,
            after_chunk=0,
        )
        pc04_before = len(_model_requests(_http_json("GET", f"{model_url}/control/requests")))
        pc04_ack = client.send_programmatic(pc04_session, "pc04-input", "stream tool usage")
        intermediate_reply: dict[str, Any] | None = None
        intermediate_texts: list[str] = []
        reply_deadline = time.monotonic() + SCENARIO_DEADLINE_S
        while time.monotonic() < reply_deadline:
            event = client.wait_session_event(
                "reply.status",
                subscription_id="pc04-follow",
                timeout=reply_deadline - time.monotonic(),
            )
            payload = event.get("params", {}).get("event", {})
            if not isinstance(payload, dict) or payload.get("available") is not True:
                continue
            items = payload.get("items")
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                preview = item.get("preview")
                if isinstance(preview, dict) and isinstance(preview.get("text"), str):
                    intermediate_texts.append(preview["text"])
            if any(text and text != "stream complete" for text in intermediate_texts):
                intermediate_reply = event
                _release_barrier(model_url, pc04_stream_barrier)
                break
        if intermediate_reply is None:
            raise GateFailure("未观察到 chunk barrier 之前的中间 reply.status")
        pc04_result = _wait_programmatic_result(client, pc04_session, "pc04-input")
        pc04_page = client.read_messages(pc04_session)
        pc04_requests = _model_requests(_http_json("GET", f"{model_url}/control/requests"))[pc04_before:]
        pc04_barrier_status = _http_json(
            "GET", f"{model_url}/control/barriers/{pc04_stream_barrier}"
        )
        pc04_rows = pc04_page.get("items")
        tool_rows = [item for item in pc04_rows if isinstance(item, dict) and item.get("body", {}).get("kind") == "tool_result"] if isinstance(pc04_rows, list) else []
        output_rows = [item for item in pc04_rows if isinstance(item, dict) and item.get("body", {}).get("kind") == "output"] if isinstance(pc04_rows, list) else []
        final_id = pc04_result.get("ending_message_id")
        final_seq = pc04_result.get("ending_seq")
        final_output = next(
            (
                item for item in output_rows
                if isinstance(item, dict)
                and item.get("id") == final_id
                and isinstance(item.get("body"), dict)
                and item["body"].get("finish") == "complete"
            ),
            None,
        )
        final_text = "".join(
            str(part.get("value", ""))
            for part in final_output.get("body", {}).get("parts", [])
            if isinstance(part, dict) and part.get("kind") == "text"
        ) if isinstance(final_output, dict) else ""
        tool_calls: list[tuple[str, int, str, str]] = []
        for item in output_rows:
            if not isinstance(item, dict) or not isinstance(item.get("body"), dict):
                continue
            for index, part in enumerate(item["body"].get("parts", [])):
                if isinstance(part, dict) and part.get("kind") == "tool_call":
                    binding_id = part.get("binding_id")
                    name = part.get("name")
                    if isinstance(binding_id, str) and isinstance(name, str):
                        tool_calls.append((str(item.get("id")), index, binding_id, name))
        result_refs = [
            (
                cast(dict[str, Any], item["body"])["call_ref"].get("message_id"),
                cast(dict[str, Any], item["body"])["call_ref"].get("part_index"),
            )
            for item in tool_rows
            if isinstance(item, dict)
            and isinstance(item.get("body"), dict)
            and isinstance(item["body"].get("call_ref"), dict)
        ]
        call_ids = [
            cast(str, value["value"].get("call_record_id"))
            for item in output_rows
            if isinstance(item, dict) and isinstance(item.get("body"), dict)
            for value in item["body"].get("parts", [])
            if isinstance(value, dict)
            and value.get("kind") == "model.facts"
            and isinstance(value.get("value"), dict)
            and isinstance(value["value"].get("call_record_id"), str)
        ]
        model_calls = _model_call_records(
            Path("/sandbox/workspace/model-registry.sqlite3"), call_ids
        )
        model_calls_by_id = {
            str(record.get("id")): record for record in model_calls
        }
        actual_usage = [
            model_calls_by_id.get(call_id, {}).get("usage") for call_id in call_ids
        ]
        expected_usage = [
            {
                "cache_write_input_tokens": None,
                "cached_input_tokens": 0,
                "coverage": "exact",
                "covered_request_count": 1,
                "input_tokens": 7,
                "output_tokens": 3,
                "reasoning_output_tokens": 0,
                "request_count": 1,
            },
            {
                "cache_write_input_tokens": None,
                "cached_input_tokens": 0,
                "coverage": "exact",
                "covered_request_count": 1,
                "input_tokens": 5,
                "output_tokens": 2,
                "reasoning_output_tokens": 0,
                "request_count": 1,
            },
        ]
        usage_totals: dict[str, int | None] = {}
        for field in (
            "input_tokens",
            "output_tokens",
            "request_count",
            "covered_request_count",
        ):
            values = [
                usage.get(field)
                for usage in actual_usage
                if isinstance(usage, dict)
            ]
            usage_totals[field] = (
                sum(cast(int, value) for value in values)
                if len(values) == len(actual_usage)
                and all(type(value) is int for value in values)
                else None
            )
        message_events: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        event_deadline = time.monotonic() + SCENARIO_DEADLINE_S
        while time.monotonic() < event_deadline:
            event = client.wait_session_event(
                "messages.appended",
                subscription_id="pc04-follow",
                timeout=event_deadline - time.monotonic(),
            )
            message_events.append(event)
            payload = event.get("params", {}).get("event", {})
            items = payload.get("items") if isinstance(payload, dict) else None
            if isinstance(items, list):
                event_rows.extend(item for item in items if isinstance(item, dict))
            if any(item.get("id") == final_id for item in event_rows):
                break
        page_pairs = [
            (item.get("id"), item.get("seq"))
            for item in pc04_rows
            if isinstance(item, dict)
        ] if isinstance(pc04_rows, list) else []
        event_pairs = [(item.get("id"), item.get("seq")) for item in event_rows]
        seqs = [seq for _, seq in page_pairs]
        continuous_seqs = (
            bool(seqs)
            and all(type(seq) is int for seq in seqs)
            and seqs
            == list(
                range(
                    cast(int, seqs[0]),
                    cast(int, seqs[-1]) + 1,
                )
            )
        )
        valid_tool_ref = (
            len(tool_rows) == 1
            and len(tool_calls) == 1
            and len(result_refs) == 1
            and result_refs[0][:2] == tool_calls[0][:2]
            and tool_calls[0][2] != ""
            and tool_calls[0][3] == "tool_search"
        )
        valid_model_calls = (
            len(call_ids) == 2
            and len(set(call_ids)) == 2
            and len(model_calls) == 2
            and all(
                isinstance(record.get("binding"), dict)
                and record["binding"].get("model") == "model-gate"
                and record.get("state") == "success"
                for record in model_calls
            )
            and actual_usage == expected_usage
            and usage_totals
            == {
                "input_tokens": 12,
                "output_tokens": 5,
                "request_count": 2,
                "covered_request_count": 2,
            }
        )
        checks.append(
            CheckResult(
                "PC-04",
                follow_ack.get("subscription_id") == "pc04-follow"
                and pc04_ack.get("message_id") == "pc04-input"
                and pc04_result.get("status") == "complete"
                and len(pc04_requests) == 2
                and isinstance(final_id, str)
                and type(final_seq) is int
                and isinstance(final_output, dict)
                and final_output.get("seq") == final_seq
                and final_text == "stream complete"
                and valid_tool_ref
                and valid_model_calls
                and page_pairs == event_pairs
                and continuous_seqs
                and isinstance(pc04_barrier_status, dict)
                and pc04_barrier_status.get("reached") is True
                and pc04_barrier_status.get("released") is True
                and intermediate_reply is not None,
                {
                    "follow": follow_ack,
                    "ack": pc04_ack,
                    "result": pc04_result,
                    "messagePage": pc04_page,
                    "messageEvents": message_events,
                    "intermediateReply": intermediate_reply,
                    "intermediateTexts": intermediate_texts,
                    "chunkBarrier": {
                        "name": pc04_stream_barrier,
                        "status": pc04_barrier_status,
                    },
                    "providerRequestCount": len(pc04_requests),
                    "outputText": final_text,
                    "toolCalls": tool_calls,
                    "toolResultRefs": result_refs,
                    "modelCallIds": call_ids,
                    "modelCalls": model_calls,
                    "usage": actual_usage,
                    "usageTotals": usage_totals,
                },
            )
        )
        final_requests = _http_json("GET", f"{model_url}/control/requests")
        _write_jsonl(report_dir / "model-requests.jsonl", _model_requests(final_requests))
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
        # 1. 按固定顺序提供 summary、业务响应和 Markdown profile projection。
        _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
        _configure_model_gate(context_window=100_000)
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
                    "content": (
                        f"<think>{_MEMORY_CONTEXT_THINKING}</think>"
                        f"{_MEMORY_CONTEXT_RESPONSE}"
                    ),
                },
                {
                    "mode": "complete",
                    "content": _MEMORY_CONTEXT_PROFILE_RESPONSE,
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
        pending_retired = (
            not pending_path.exists()
            or not pending_path.read_text(encoding="utf-8").strip()
        )
        receipt_connection = sqlite3.connect(
            "/sandbox/workspace/memory/markdown-profile-writes.db"
        )
        try:
            memory_applied = receipt_connection.execute(
                "SELECT 1 FROM consolidation_writes "
                "WHERE source_ref = ? AND kind = 'markdown_memory_applied_v1'",
                (str(compaction_row["source_ref"]),),
            ).fetchone()
            self_applied = receipt_connection.execute(
                "SELECT 1 FROM consolidation_writes "
                "WHERE source_ref = ? AND kind = 'markdown_self_applied_v1'",
                (str(compaction_row["source_ref"]),),
            ).fetchone()
        finally:
            receipt_connection.close()
        final_requests = _model_requests(
            _http_json("GET", f"{model_url}/control/requests")
        )
        if len(final_requests) != 3:
            capabilities = _http_json(
                "GET",
                "http://akashic-control-gate:2236/api/chat/runtime/capabilities",
            )
            markdown_incidents = next(
                (
                    plugin.get("composition", {}).get("recent_incidents", [])
                    for plugin in capabilities.get("plugins", [])
                    if plugin.get("id") == "markdown_memory"
                ),
                [],
            )
            raise GateFailure(
                "memory-context 模型请求数量异常："
                f"{len(final_requests)} markdownIncidents="
                f"{json.dumps(markdown_incidents, ensure_ascii=False, sort_keys=True)}"
            )
        request_kinds = _memory_context_request_kinds(final_requests)
        scripts = [
            request.get("script")
            for request in final_requests
            if isinstance(request, dict)
        ]
        scripts_boundary = (
            scripts[0] == {"mode": "complete", "content": _PC09_COMPACTION_SUMMARY}
            and isinstance(scripts[1], dict)
            and "<think>" in str(scripts[1].get("content"))
            and scripts[2] == {"mode": "complete", "content": _MEMORY_CONTEXT_PROFILE_RESPONSE}
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
                and request_kinds == ["summary", "business", "markdown"]
                and scripts_boundary
                and thinking_boundary
                and ledger_passed
                and receipt_row is not None
                and pending_retired
                and memory_applied is not None
                and self_applied is not None,
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
                    "pendingRetired": pending_retired,
                    "memoryApplied": memory_applied is not None,
                    "selfApplied": self_applied is not None,
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
        _configure_model_gate(context_window=1_000_000)
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        first = _connect_client(endpoint, events_path)
        second = _connect_client(endpoint, events_path)
        clients.extend((first, second))

        # 1. 两个独立 programmatic Session 必须同时进入同一 provider barrier。
        pc05_a = "programmatic:pc05-a"
        pc05_b = "programmatic:pc05-b"
        first.admit_programmatic(pc05_a)
        second.admit_programmatic(pc05_b)
        pc05_a_barrier = "pc05-provider-a"
        pc05_b_barrier = "pc05-provider-b"
        _http_json(
            "PUT", f"{model_url}/control/barriers/{pc05_a_barrier}"
        )
        _http_json(
            "PUT", f"{model_url}/control/barriers/{pc05_b_barrier}"
        )
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            [
                {
                    "mode": "complete",
                    "content": "pc05 first complete",
                    "barrier": pc05_a_barrier,
                },
                {
                    "mode": "complete",
                    "content": "pc05 second complete",
                    "barrier": pc05_b_barrier,
                },
            ],
        )
        pc05_a_ack = first.send_programmatic(pc05_a, "pc05-input-a", "pc05 first")
        _wait_barrier(model_url, pc05_a_barrier)
        pc05_b_ack = second.send_programmatic(pc05_b, "pc05-input-b", "pc05 second")
        _wait_barrier(model_url, pc05_b_barrier)
        pc05_requests = _model_requests(
            _http_json("GET", f"{model_url}/control/requests")
        )
        pc05_blocked = [
            request
            for request in pc05_requests
            if isinstance(request, dict)
            and request.get("state") == "blocked"
            and isinstance(request.get("payload"), dict)
            and any(
                text in json.dumps(request["payload"], ensure_ascii=False)
                for text in ("pc05 first", "pc05 second")
            )
        ]
        _release_barrier(model_url, pc05_a_barrier)
        _release_barrier(model_url, pc05_b_barrier)
        pc05_a_result = _wait_programmatic_result(first, pc05_a, "pc05-input-a")
        pc05_b_result = _wait_programmatic_result(second, pc05_b, "pc05-input-b")
        pc05_a_page = first.read_messages(pc05_a)
        pc05_b_page = second.read_messages(pc05_b)
        pc05_a_output = next(
            (item for item in pc05_a_page.get("items", [])
             if isinstance(item, dict) and item.get("body", {}).get("kind") == "output"),
            None,
        )
        pc05_b_output = next(
            (item for item in pc05_b_page.get("items", [])
             if isinstance(item, dict) and item.get("body", {}).get("kind") == "output"),
            None,
        )
        pc05_input_rows = [
            item for page in (pc05_a_page, pc05_b_page)
            for item in page.get("items", [])
            if isinstance(item, dict) and item.get("body", {}).get("kind") == "input"
        ]
        pc05_passed = (
            pc05_a_ack.get("seq") == 0
            and pc05_b_ack.get("seq") == 0
            and len(pc05_blocked) == 2
            and pc05_a_result.get("status") == "complete"
            and pc05_b_result.get("status") == "complete"
            and pc05_a_output is not None
            and pc05_b_output is not None
            and _message_text(pc05_a_output) == "pc05 first complete"
            and _message_text(pc05_b_output) == "pc05 second complete"
            and len(pc05_input_rows) == 2
            and {item.get("session_id") for item in pc05_input_rows}
            == {pc05_a, pc05_b}
            and all(item.get("source") == "programmatic" for item in pc05_input_rows)
        )
        checks.append(
            CheckResult(
                "PC-05",
                pc05_passed,
                {
                    "sessions": [pc05_a, pc05_b],
                    "acks": [pc05_a_ack, pc05_b_ack],
                    "blockedProviderRequests": pc05_blocked,
                    "results": [pc05_a_result, pc05_b_result],
                    "messagePages": [pc05_a_page, pc05_b_page],
                },
            )
        )

        # 2. 同一来源再次提交时只追加新的 Input，并按 source head 撤掉旧回复。
        pc06 = "programmatic:pc06-source-head"
        first.admit_programmatic(pc06)
        pc06_barrier = "pc06-old-provider"
        _http_json("PUT", f"{model_url}/control/barriers/{pc06_barrier}")
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            [
                {
                    "mode": "complete",
                    "content": "pc06 stale response",
                    "barrier": pc06_barrier,
                },
                {"mode": "complete", "content": "pc06 latest response"},
            ],
        )
        pc06_first_ack = first.send_programmatic(pc06, "pc06-first", "pc06 first input")
        _wait_barrier(model_url, pc06_barrier)
        pc06_request_start = len(
            _model_requests(_http_json("GET", f"{model_url}/control/requests"))
        ) - 1
        pc06_second_ack = first.send_programmatic(
            pc06, "pc06-second", "pc06 same-source replacement"
        )
        pc06_before_release = first.read_messages(pc06)
        pc06_inputs_before = [
            item for item in pc06_before_release.get("items", [])
            if isinstance(item, dict) and item.get("body", {}).get("kind") == "input"
        ]
        _release_barrier(model_url, pc06_barrier)
        pc06_result = _wait_programmatic_result(first, pc06, "pc06-second")
        pc06_page = first.read_messages(pc06)
        pc06_inputs = [
            item for item in pc06_page.get("items", [])
            if isinstance(item, dict) and item.get("body", {}).get("kind") == "input"
        ]
        pc06_outputs = [
            item for item in pc06_page.get("items", [])
            if isinstance(item, dict) and item.get("body", {}).get("kind") == "output"
        ]
        pc06_requests = _model_requests(
            _http_json("GET", f"{model_url}/control/requests")
        )[max(0, pc06_request_start):]
        pc06_second_payload = next(
            (
                request.get("payload")
                for request in pc06_requests
                if isinstance(request, dict)
                and isinstance(request.get("payload"), dict)
                and "pc06 same-source replacement"
                in json.dumps(request["payload"], ensure_ascii=False)
            ),
            None,
        )
        pc06_second_messages = (
            pc06_second_payload.get("messages", [])
            if isinstance(pc06_second_payload, dict)
            else []
        )
        pc06_second_prompt = json.dumps(pc06_second_messages, ensure_ascii=False)
        pc06_passed = (
            pc06_first_ack.get("seq") == 0
            and pc06_second_ack.get("seq") == 1
            and len(pc06_inputs_before) == 2
            and [item.get("id") for item in pc06_inputs_before] == [
                "pc06-first", "pc06-second"
            ]
            and all(item.get("session_id") == pc06 for item in pc06_inputs_before)
            and all(item.get("source") == "programmatic" for item in pc06_inputs_before)
            and pc06_result.get("status") == "complete"
            and len(pc06_inputs) == 2
            and [item.get("id") for item in pc06_inputs] == [
                "pc06-first", "pc06-second"
            ]
            and len(pc06_outputs) == 1
            and _message_text(pc06_outputs[0]) == "pc06 latest response"
            and "pc06 first input" in pc06_second_prompt
            and "pc06 same-source replacement" in pc06_second_prompt
        )
        checks.append(
            CheckResult(
                "PC-06",
                pc06_passed,
                {
                    "firstAck": pc06_first_ack,
                    "secondAck": pc06_second_ack,
                    "inputsBeforeRelease": pc06_inputs_before,
                    "result": pc06_result,
                    "messagePage": pc06_page,
                    "providerRequests": pc06_requests,
                    "secondProviderPayload": pc06_second_payload,
                },
            )
        )

        # 3. 真实 shell 已 started 后暂停；清理物理进程，ToolResult 仍引用原 CallRef。
        pc07 = "programmatic:pc07-control"
        first.admit_programmatic(pc07)
        pc07_pause_barrier = "pc07-pause-provider"
        _http_json("PUT", f"{model_url}/control/barriers/{pc07_pause_barrier}")
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            [
                {
                    "mode": "complete",
                    "tool_calls": [
                        {
                            "id": "call_pc07_shell",
                            "name": "shell",
                            "arguments": {
                                "command": (
                                    "echo $$ > /sandbox/workspace/pc07-shell.pid; "
                                    "ls -l /sandbox/workspace/pc07-shell.pid; "
                                    "cat /sandbox/workspace/pc07-shell.pid; "
                                    "exec sleep 300"
                                ),
                                "description": "PC07 long running cleanup probe",
                                "yield_time_ms": 250,
                                "timeout": 300,
                            },
                        }
                    ],
                },
                {"mode": "timeout", "barrier": pc07_pause_barrier},
            ],
        )
        pc07_input_ack = first.send_programmatic(
            pc07, "pc07-input", "pc07 start controllable shell"
        )
        pc07_page_with_tool, pc07_tool_rows = _wait_for_message_items(
            first, pc07, "tool_result"
        )
        pc07_provider_count = len(
            _model_requests(_http_json("GET", f"{model_url}/control/requests"))
        )
        _wait_barrier(model_url, pc07_pause_barrier)
        _release_barrier(model_url, pc07_pause_barrier)
        pc07_outputs_with_tool = _message_items(pc07_page_with_tool, "output")
        pc07_tool_result = pc07_tool_rows[0]
        pc07_tool_ref = pc07_tool_result.get("body", {}).get("call_ref", {})
        try:
            pc07_tool_payload = json.loads(_message_text(pc07_tool_result))
            pc07_execution_id = int(pc07_tool_payload["execution_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise GateFailure(f"PC07 shell 未返回 execution_id：{pc07_tool_result!r}") from error
        pc07_tool_call: dict[str, Any] | None = None
        for output in pc07_outputs_with_tool:
            parts = output.get("body", {}).get("parts", [])
            if not isinstance(parts, list):
                continue
            for part_index, part in enumerate(parts):
                if (
                    isinstance(part, dict)
                    and part.get("kind") == "tool_call"
                    and part.get("name") == "shell"
                ):
                    pc07_tool_call = {
                        "message_id": output.get("id"),
                        "part_index": part_index,
                        "part": part,
                    }
                    break
            if pc07_tool_call is not None:
                break
        pid_path = Path("/sandbox/workspace/pc07-shell.pid")
        pid_deadline = time.monotonic() + SCENARIO_DEADLINE_S
        pc07_pid: int | None = None
        while time.monotonic() < pid_deadline:
            try:
                pc07_pid = int(pid_path.read_text(encoding="utf-8").strip())
            except (FileNotFoundError, ValueError):
                threading.Event().wait(0.05)
                continue
            if pc07_pid > 0:
                break
            pc07_pid = None
            threading.Event().wait(0.05)
        if pc07_pid is None:
            raise GateFailure("PC07 shell 没有留下可控 PID")

        # 第二次 provider 调用故意等待客户端断开，使 pause 发生在工具回执之后。
        pc07_pause_ack = first.request_result(
            "programmatic/message/pause",
            {"session_id": pc07, "message_id": "pc07-pause"},
        )
        pc07_paused_result = first.programmatic_result(pc07, "pc07-input")
        pc07_pid_gone = False
        pid_gone_deadline = time.monotonic() + SCENARIO_DEADLINE_S
        while time.monotonic() < pid_gone_deadline:
            if not _pid_is_alive(pc07_pid):
                pc07_pid_gone = True
                break
            threading.Event().wait(0.05)
        if not pc07_pid_gone:
            raise GateFailure(f"PC07 pause 后 shell PID 仍存活：{pc07_pid}")
        pc07_cancel_deadline = time.monotonic() + SCENARIO_DEADLINE_S
        while time.monotonic() < pc07_cancel_deadline:
            pc07_after_pause_requests = _model_requests(
                _http_json("GET", f"{model_url}/control/requests")
            )
            if any(
                isinstance(request, dict)
                and int(request.get("index", 0)) >= pc07_provider_count
                and request.get("state") == "client_disconnected"
                for request in pc07_after_pause_requests
            ):
                break
            threading.Event().wait(0.05)
        else:
            raise GateFailure("PC07 pause 后 provider 请求未确认断开")
        _http_json("PUT", f"{model_url}/control/barriers/pc07-new-input")
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            [
                {
                    "mode": "complete",
                    "tool_calls": [
                        {
                            "id": "call_pc07_after_cleanup",
                            "name": "write_stdin",
                            "arguments": {
                                "execution_id": pc07_execution_id,
                                "yield_time_ms": 250,
                            },
                        }
                    ],
                },
                {
                    "mode": "complete",
                    "content": "pc07 new input response",
                    "barrier": "pc07-new-input",
                },
            ],
        )
        pc07_new_ack = first.send_programmatic(
            pc07, "pc07-new-input", "pc07 new input after resume"
        )
        _wait_barrier(model_url, "pc07-new-input")
        pc07_stale_pause_ack = first.request_result(
            "programmatic/message/pause",
            {"session_id": pc07, "message_id": "pc07-pause"},
        )
        _release_barrier(model_url, "pc07-new-input")
        pc07_result = first.programmatic_result(pc07, "pc07-input")
        pc07_new_result = _wait_programmatic_result(first, pc07, "pc07-new-input")
        pc07_page = first.read_messages(pc07)
        pc07_controls = _message_items(pc07_page, "control")
        pc07_inputs = _message_items(pc07_page, "input")
        pc07_outputs = _message_items(pc07_page, "output")
        pc07_tool_results = _message_items(pc07_page, "tool_result")
        pc07_cleanup_tool_result = next(
            (
                item
                for item in pc07_tool_results
                if item.get("body", {}).get("call_ref") != pc07_tool_ref
            ),
            None,
        )
        pc07_shell_gone = (
            pc07_pid_gone
            and pc07_cleanup_tool_result is not None
            and pc07_cleanup_tool_result.get("body", {}).get("outcome") == "error"
            and "未知 execution_id"
            in _message_text(pc07_cleanup_tool_result)
        )
        pc07_passed = (
            pc07_input_ack.get("seq") == 0
            and pc07_tool_call is not None
            and pc07_tool_ref == {
                "message_id": pc07_tool_call.get("message_id"),
                "part_index": pc07_tool_call.get("part_index"),
            }
            and pc07_tool_result.get("body", {}).get("outcome") == "success"
            and pc07_pause_ack.get("seq") == 3
            and pc07_paused_result.get("status") == "pause"
            # Once a newer Input is appended, the old Input is open again;
            # the durable result before that append remains the pause proof.
            and pc07_result.get("status") == "open"
            and pc07_shell_gone
            and pc07_new_ack.get("seq") > pc07_pause_ack.get("seq", -1)
            # Repeating the same pause identity is idempotent: it returns the
            # original Control row and must not cancel the newer Input.
            and pc07_stale_pause_ack == pc07_pause_ack
            and pc07_new_result.get("status") == "complete"
            and [item.get("id") for item in pc07_inputs]
            == ["pc07-input", "pc07-new-input"]
            and [item.get("body", {}).get("action") for item in pc07_controls]
            == ["pause"]
            and pc07_controls[0].get("body", {}).get("through_seq") == 2
            and len(pc07_tool_results) == 2
            and pc07_cleanup_tool_result is not None
            and pc07_cleanup_tool_result.get("body", {}).get("call_ref", {}).get(
                "part_index"
            )
            == next(
                (
                    index
                    for output in pc07_outputs
                    for index, part in enumerate(output.get("body", {}).get("parts", []))
                    if isinstance(part, dict)
                    and part.get("kind") == "tool_call"
                    and part.get("name") == "write_stdin"
                ),
                None,
            )
            and any(
                _message_text(output) == "pc07 new input response"
                for output in pc07_outputs
            )
        )
        checks.append(
            CheckResult(
                "PC-07",
                pc07_passed,
                {
                    "inputAck": pc07_input_ack,
                    "pauseAck": pc07_pause_ack,
                    "newInputAck": pc07_new_ack,
                    "stalePauseAck": pc07_stale_pause_ack,
                    "pausedResult": pc07_paused_result,
                    "result": pc07_result,
                    "newInputResult": pc07_new_result,
                    "controls": pc07_controls,
                    "toolCall": pc07_tool_call,
                    "toolResult": pc07_tool_result,
                    "shellPid": pc07_pid,
                    "shellPidGone": pc07_pid_gone,
                    "shellGone": pc07_shell_gone,
                    "messagePage": pc07_page,
                },
            )
        )

        # 4. 连接断开只丢传输；重连后用 Input/result 读取同一持久前缀。
        pc08 = "programmatic:pc08-reconnect"
        disconnecting = _connect_client(endpoint, events_path)
        clients.append(disconnecting)
        disconnecting.admit_programmatic(pc08)
        pc08_barrier = "pc08-disconnect-provider"
        _http_json("PUT", f"{model_url}/control/barriers/{pc08_barrier}")
        _http_json(
            "PUT", f"{model_url}/control/script",
            {"mode": "complete", "content": "pc08 survived disconnect", "barrier": pc08_barrier},
        )
        pc08_ack = disconnecting.send_programmatic(pc08, "pc08-input", "pc08 disconnect")
        _wait_barrier(model_url, pc08_barrier)
        disconnecting.close()
        clients.remove(disconnecting)
        _release_barrier(model_url, pc08_barrier)
        resumed = _connect_client(endpoint, events_path)
        clients.append(resumed)
        resumed_result = _wait_programmatic_result(resumed, pc08, "pc08-input")
        resumed_page = resumed.read_messages(pc08)
        pc08_inputs = [
            item for item in resumed_page.get("items", [])
            if isinstance(item, dict) and item.get("body", {}).get("kind") == "input"
        ]
        pc08_outputs = [
            item for item in resumed_page.get("items", [])
            if isinstance(item, dict) and item.get("body", {}).get("kind") == "output"
        ]
        checks.append(
            CheckResult(
                "PC-08",
                pc08_ack.get("seq") == 0
                and resumed_result.get("status") == "complete"
                and len(pc08_inputs) == 1
                and pc08_inputs[0].get("id") == "pc08-input"
                and len(pc08_outputs) == 1
                and _message_text(pc08_outputs[0]) == "pc08 survived disconnect",
                {"ack": pc08_ack, "result": resumed_result, "messagePage": resumed_page},
            )
        )
        restart_state = {"sessionId": pc08, "inputId": "pc08-input"}

        # 5. 慢读者先制造有界队列压力；压力持续时健康 Session 仍须完成。
        pc09 = "programmatic:pc09-slow"
        pc09_healthy = "programmatic:pc09-healthy"
        first.admit_programmatic(pc09)
        slow = _connect_client(endpoint, events_path)
        clients.append(slow)
        slow._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256)
        slow.follow_session(pc09, "pc09-slow-follow")
        producer_count = 8
        burst_count = 4
        slow_content = "pc09 slow " + ("x" * (256 * 1024))
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            ([{"mode": "complete", "content": slow_content} for _ in range(burst_count)]
             + [{"mode": "complete", "content": "pc09 healthy response"}]
             + [{"mode": "complete", "content": slow_content}
                for _ in range(producer_count - burst_count)]),
        )
        producer_errors: list[str] = []
        producer_sent = 0
        burst_ready = threading.Event()
        continue_tail = threading.Event()

        def produce_slow_tail() -> None:
            """Append a bounded tail and expose all producer failures to the gate."""

            nonlocal producer_sent
            try:
                for index in range(producer_count):
                    first.send_programmatic(
                        pc09,
                        f"pc09-input-{index}",
                        f"pc09 slow input {index}",
                    )
                    result = _wait_programmatic_result(
                        first, pc09, f"pc09-input-{index}"
                    )
                    if result.get("status") != "complete":
                        raise GateFailure(
                            f"PC09 slow input {index} 未完成：{result!r}"
                        )
                    producer_sent = index + 1
                    if producer_sent == burst_count:
                        burst_ready.set()
                        if not continue_tail.wait(SCENARIO_DEADLINE_S):
                            raise GateFailure("PC09 slow producer 未收到继续信号")
            except BaseException as error:
                producer_errors.append(f"{type(error).__name__}: {error}")
            finally:
                burst_ready.set()

        producer = threading.Thread(
            target=produce_slow_tail, name="pc09-slow-producer", daemon=False
        )
        producer.start()
        if not burst_ready.wait(SCENARIO_DEADLINE_S):
            raise GateFailure("PC09 slow producer 未建立第一段压力")
        pressure_readable, _, _ = select.select([slow._socket], [], [], SCENARIO_DEADLINE_S)
        pressure_pending = bool(pressure_readable)
        second.admit_programmatic(pc09_healthy)
        second.send_programmatic(
            pc09_healthy, "pc09-healthy-input", "pc09 healthy input"
        )
        pc09_healthy_result = _wait_programmatic_result(
            second, pc09_healthy, "pc09-healthy-input"
        )
        continue_tail.set()
        producer.join()
        if producer.is_alive():
            raise GateFailure("PC09 slow producer 未完整 join")
        slow_closed, drained_bytes = _drain_socket_until_eof(slow._socket)
        checks.append(
            CheckResult(
                "PC-09",
                not producer_errors
                and producer_sent == producer_count
                and pressure_pending
                and slow_closed
                and pc09_healthy_result.get("status") == "complete",
                {
                    "slowConnectionClosed": slow_closed,
                    "healthyResult": pc09_healthy_result,
                    "producerSent": producer_sent,
                    "producerCount": producer_count,
                    "producerErrors": producer_errors,
                    "producerJoined": not producer.is_alive(),
                    "pressurePending": pressure_pending,
                    "drainedBytesBeforeEof": drained_bytes,
                    "healthyIsolation": pc09_healthy_result.get("status") == "complete",
                    "slowReceiveBuffer": slow._socket.getsockopt(
                        socket.SOL_SOCKET, socket.SO_RCVBUF
                    ),
                },
            )
        )
        slow.close()
        clients.remove(slow)

        # 6. 真实 TOOLS BoundTool 在 started 后失败，日志必须保留 ToolResult。
        pc10 = "programmatic:pc10-tool-failure"
        second.admit_programmatic(pc10)
        _http_json(
            "PUT", f"{model_url}/control/script",
            {
                "mode": "stream",
                "deltas": [],
                "tool_calls": [
                    {
                        "id": "call_pc10_failure",
                        "name": "pc10_failure_probe",
                        "arguments": {"probe": True},
                    }
                ],
            },
        )
        pc10_ack = second.send_programmatic(pc10, "pc10-input", "pc10 invoke failing tool")
        pc10_result = _wait_programmatic_result(second, pc10, "pc10-input")
        pc10_page = second.read_messages(pc10)
        pc10_outputs = [
            item for item in pc10_page.get("items", [])
            if isinstance(item, dict) and item.get("body", {}).get("kind") == "output"
        ]
        pc10_tools = [
            item for item in pc10_page.get("items", [])
            if isinstance(item, dict) and item.get("body", {}).get("kind") == "tool_result"
        ]
        pc10_controls = [
            item for item in pc10_page.get("items", [])
            if isinstance(item, dict) and item.get("body", {}).get("kind") == "control"
        ]
        pc10_passed = (
            pc10_ack.get("seq") == 0
            and pc10_result.get("status") == "failure"
            and len(pc10_outputs) == 1
            and len(pc10_tools) == 1
            and pc10_tools[0].get("body", {}).get("outcome") == "unknown"
            and pc10_tools[0].get("body", {}).get("call_ref", {}).get("message_id")
            == pc10_outputs[0].get("id")
            and pc10_tools[0].get("body", {}).get("call_ref", {}).get("part_index")
            == next(
                (
                    index
                    for index, part in enumerate(
                        pc10_outputs[0].get("body", {}).get("parts", [])
                    )
                    if isinstance(part, dict)
                    and part.get("kind") == "tool_call"
                    and part.get("name") == "pc10_failure_probe"
                ),
                None,
            )
            and any(
                isinstance(part, dict)
                and part.get("kind") == "tool_call"
                and part.get("name") == "pc10_failure_probe"
                and isinstance(part.get("binding_id"), str)
                and bool(part.get("binding_id"))
                for part in pc10_outputs[0].get("body", {}).get("parts", [])
            )
            and len(pc10_controls) == 1
            and pc10_controls[0].get("body", {}).get("action") == "failure"
            and "pc10 tool handler failure"
            in str(pc10_controls[0].get("body", {}).get("reason", ""))
        )
        checks.append(
            CheckResult(
                "PC-10",
                pc10_passed,
                {
                    "ack": pc10_ack,
                    "result": pc10_result,
                    "outputs": pc10_outputs,
                    "toolResults": pc10_tools,
                    "controls": pc10_controls,
                    "messagePage": pc10_page,
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
            failed_error = failed_state.get("error")
            if not isinstance(failed_error, dict) or not isinstance(
                failed_error.get("message"), str
            ):
                raise GateFailure(f"failed turn 缺少 error.message：{failed_state!r}")

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
                "failedError": failed_error["message"],
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
                same_thread["failedError"],
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
    _configure_model_gate()
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
    outbound_queue_size: int = 64,
) -> None:
    """渲染只连接 compose 私网 model-gate 的隔离配置。"""

    config = f"""[agent.plugins]
disabled_builtin = ["subagent"]

[app_server]
enabled = true
listen = "/sandbox/akashic.sock"
max_connections = 8
ingress_queue_size = 32
outbound_queue_size = {outbound_queue_size}

[channels.chat]
enabled = true

[channels.telegram]
enabled = false
token = ""

[channels.qq]
enabled = false
bot_uin = ""

"""
    (sandbox / "config.toml").write_text(config, encoding="utf-8")
    reply_config = sandbox / "workspace/plugin-data/reply-builtin/config.local.toml"
    reply_config.parent.mkdir(parents=True, exist_ok=True)
    reply_config.write_text(f"max_steps = {max_iterations}\n", encoding="utf-8")
    compaction_config = sandbox / "workspace/plugin-data/compaction-builtin/config.local.toml"
    compaction_config.parent.mkdir(parents=True, exist_ok=True)
    compaction_config.write_text("keep_recent_tokens = 20000\n", encoding="utf-8")


def _initialize_current_workspace(workspace: Path, source_root: Path) -> None:
    """在候选源码进程中调用 workspace 初始化 owner。"""

    config_path = workspace.parent / "config.toml"
    script = """
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
workspace = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(source_root))
from bootstrap import init_workspace as init_module

module_path = Path(init_module.__file__).resolve()
if source_root not in module_path.parents:
    raise RuntimeError(f"workspace init imported outside candidate source: {module_path}")
init_module.init_workspace(config_path=config_path, workspace=workspace, force=False)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(source_root),
                str(config_path),
                str(workspace),
            ],
            cwd=source_root,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        raise GateFailure(f"候选 workspace 初始化失败: {details}") from error


def _prepare_host_sandbox(
    sandbox: Path,
    source_root: Path,
    *,
    max_iterations: int = 2,
    outbound_queue_size: int = 64,
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
    (sandbox / "home").mkdir()
    (sandbox / "reports").mkdir()
    (sandbox / "static/dashboard").mkdir(parents=True)
    (sandbox / "static/chat").mkdir()

    # 3. 配置只引用同一 sandbox 内的路径。
    _write_config(
        sandbox,
        max_iterations=max_iterations,
        outbound_queue_size=outbound_queue_size,
    )
    _initialize_current_workspace(sandbox / "workspace", sandbox / "app")


def _install_control_failure_plugin(sandbox: Path) -> None:
    """安装只为 PC10 构造真实 BoundTool handler failure 的隔离插件。"""

    plugin_base = sandbox / "home/.akashic-plugin/cache/gate/control_failure"
    cache = plugin_base / ".artifacts/1.0.0"
    manifest = sandbox / "home/.akashic-plugin/manifest.toml"
    cache.mkdir(parents=True, exist_ok=True)
    _ = (cache / "plugin.py").write_text(
        "from contextlib import asynccontextmanager\n"
        "from plugins.tools.api import BoundTool, Result\n"
        "from plugins.tools.plugin import TOOLS\n"
        "api_version = 3\n"
        "name = 'control_failure'\n"
        "version = '1.0.0'\n"
        "inject = (TOOLS,)\n"
        "class FailureTool:\n"
        "    idempotent = False\n"
        "    async def prepare(self, arguments, source=None):\n"
        "        return arguments\n"
        "    async def invoke(self, key, arguments):\n"
        "        raise RuntimeError('pc10 tool handler failure')\n"
        "    async def query(self, key):\n"
        "        return None\n"
        "@asynccontextmanager\n"
        "async def open(_state):\n"
        "    yield FailureTool()\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(TOOLS).register(\n"
        "        ctx, name='pc10_failure_probe',\n"
        "        description='Fail inside the PC10 tool handler.',\n"
        "        parameters={'type': 'object', 'properties': {'probe': {'type': 'boolean'}},\n"
        "                     'required': ['probe'], 'additionalProperties': False},\n"
        "        open=open, risk='read-only', always_on=True, preloadable=True)\n",
        encoding="utf-8",
    )
    _ = (cache / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'control_failure'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    _ = (plugin_base / ".pointers.json").write_text(
        json.dumps(
            {"stable": ".artifacts/1.0.0", "latest": ".artifacts/1.0.0"},
            sort_keys=True,
        ),
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
                "clientInfo": {"name": "docker-stdio-gate", "version": "2.0"},
                "capabilities": {"reasoningEvents": False},
            },
        },
        {"jsonrpc": "2.0", "method": "initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "server/status", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "session/create", "params": {}},
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
    responses = [
        item for item in parsed
        if isinstance(item, dict) and item.get("jsonrpc") == "2.0" and "id" in item
    ]
    initialize_result = next((item.get("result") for item in responses if item.get("id") == 1), None)
    status_result = next((item.get("result") for item in responses if item.get("id") == 2), None)
    session_result = next((item.get("result") for item in responses if item.get("id") == 3), None)
    passed = (
        completed.returncode == 0
        and not parse_error
        and {1, 2, 3} <= response_ids
        and isinstance(initialize_result, dict)
        and initialize_result.get("protocolVersion") == "2.0"
        and isinstance(status_result, dict)
        and status_result.get("ready") is True
        and status_result.get("protocolVersion") == "2.0"
        and isinstance(session_result, dict)
        and session_result.get("session_id", "").startswith("akashic:")
        and not stderr_protocol_frames
    )
    return CheckResult(
        "PC-02",
        passed,
        {
            "returncode": completed.returncode,
            "frames": len(parsed),
            "responseIds": sorted(str(item) for item in response_ids),
            "initialize": initialize_result,
            "status": status_result,
            "sessionCreate": session_result,
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
        outbound_queue_size=4 if gate == "failure-matrix" else 64,
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
