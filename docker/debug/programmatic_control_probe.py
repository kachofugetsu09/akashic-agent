#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import closing
import array
import fcntl
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
import termios
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
_MEMORY_CONTEXT_TOKEN_REPEAT = 4_500


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


def _receive_web_follow_rows(
    web: Any,
    session_id: str,
    complete: Any,
    *,
    timeout: float = SCENARIO_DEADLINE_S,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从真实 session.follow 读取消息页和 reply.status，直到 predicate 成立。"""

    deadline = time.monotonic() + timeout
    rows_by_id: dict[str, dict[str, Any]] = {}
    statuses: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        frame = json.loads(web.recv(timeout=deadline - time.monotonic()))
        frame_type = frame.get("type")
        if frame_type == "messages.appended":
            if frame.get("session_id") != session_id:
                raise GateFailure(f"Web 消息页 Session 不匹配：{frame!r}")
            items = frame.get("items")
            if not isinstance(items, list):
                raise GateFailure(f"Web messages.appended 缺少 items：{frame!r}")
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    raise GateFailure(f"Web 消息页 item 非法：{frame!r}")
                rows_by_id[item["id"]] = item
        elif frame_type == "reply.status":
            if frame.get("session_id") != session_id:
                raise GateFailure(f"Web reply.status Session 不匹配：{frame!r}")
            statuses.append(frame)
        elif frame_type == "session.following":
            continue
        elif frame_type == "error":
            raise GateFailure(f"Web session.follow 返回 error：{frame!r}")
        else:
            raise GateFailure(f"Web session.follow 收到未知 frame：{frame!r}")
        rows = sorted(rows_by_id.values(), key=lambda item: int(item["seq"]))
        if complete(rows, statuses):
            return rows, statuses
    raise GateFailure(
        f"Web session.follow 未在 deadline 内达到条件：session={session_id} "
        f"rows={list(rows_by_id.values())!r} statuses={statuses!r}"
    )


def _wait_message_log_rows(
    database: Path,
    session_id: str,
    *,
    minimum: int = 1,
    required_ids: set[str] | frozenset[str] = frozenset(),
    timeout: float = SCENARIO_DEADLINE_S,
) -> list[dict[str, Any]]:
    """等待真实 Message 日志前缀，保留身份、seq、来源和完整 body。"""

    deadline = time.monotonic() + timeout
    last_rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, session_key, seq, ts, author, source, body "
                "FROM messages WHERE session_key = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        try:
            last_rows = [
                {
                    "id": row["id"],
                    "session_id": row["session_key"],
                    "seq": row["seq"],
                    "timestamp": row["ts"],
                    "author": row["author"],
                    "source": row["source"],
                    "body": json.loads(row["body"]),
                }
                for row in rows
            ]
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise GateFailure(
                f"Message 日志 body 不是有效 JSON：session={session_id}"
            ) from error
        if len(last_rows) >= minimum and required_ids <= {
            str(row["id"]) for row in last_rows
        }:
            return last_rows
        threading.Event().wait(0.02)
    raise GateFailure(
        f"Message 日志未在 deadline 内达到前缀：session={session_id} "
        f"minimum={minimum} required={sorted(required_ids)} rows={last_rows!r}"
    )


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


def _memory_context_seed_rows() -> list[tuple[str, str]]:
    """Return the expected historical role and text pairs in durable order."""

    rows: list[tuple[str, str]] = []
    for index in range(4):
        for role in ("user", "assistant"):
            rows.append((role, _memory_context_seed_content(role, index)))
    return rows


def _memory_context_source_plan_digest(
    rows: Sequence[sqlite3.Row], source_ids: Sequence[str]
) -> str:
    """Hash selected Message identity and complete encoded body facts."""

    by_id = {str(row["id"]): row for row in rows}
    if not source_ids:
        raise GateFailure("memory-context source digest 缺少 source IDs")
    try:
        selected_rows = [by_id[message_id] for message_id in source_ids]
    except KeyError as error:
        raise GateFailure(f"memory-context source digest 缺少 Message：{error}") from error
    selected: list[dict[str, object]] = []
    for row in selected_rows:
        body = json.loads(str(row["body"]))
        if not isinstance(body, dict):
            raise GateFailure("memory-context source body 不是 object")
        selected.append(
            {
                "id": str(row["id"]),
                "session_key": str(row["session_key"]),
                "seq": int(row["seq"]),
                "ts": str(row["ts"]),
                "author": str(row["author"]),
                "source": str(row["source"]),
                "body": body,
            }
        )
    encoded = json.dumps(
        selected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _memory_context_summary_source(requests: Sequence[object]) -> list[dict[str, object]]:
    """Extract the exact source rows submitted to the summary provider."""

    for raw_request in requests:
        if not isinstance(raw_request, dict):
            continue
        payload = raw_request.get("payload")
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) != 1:
            continue
        content = messages[0].get("content") if isinstance(messages[0], dict) else None
        if not isinstance(content, str) or "\n[Source messages]\n" not in content:
            continue
        source_text = content.split("\n[Source messages]\n", 1)[1]
        source_rows = json.loads(source_text)
        if not isinstance(source_rows, list) or not all(
            isinstance(row, dict) for row in source_rows
        ):
            raise GateFailure("摘要 provider source rows 不是 object 列表")
        return source_rows
    raise GateFailure("缺少摘要 provider 的 Source messages 输入")


def _memory_context_summary_body(body: object) -> object:
    """Remove private model replay parts exactly as summary source_text does."""

    if not isinstance(body, dict) or body.get("kind") == "control":
        return body
    normalized = dict(body)
    parts = normalized.get("parts")
    if isinstance(parts, list):
        normalized["parts"] = [
            part
            for part in parts
            if not isinstance(part, dict)
            or part.get("kind") not in {
                "model.facts", "context.summary", "model.selection", "tool.selection",
            }
        ]
    return normalized


def _memory_context_business_tail(
    messages: object, expected_rows: Sequence[sqlite3.Row], stop_text: str
) -> tuple[dict[str, object], list[tuple[str, str]], list[str]]:
    """Return the summary object, projected raw tail, and source text leaks."""

    if not isinstance(messages, list):
        raise GateFailure("业务 provider payload messages 不是 list")
    summary_object: dict[str, object] | None = None
    tail: list[tuple[str, str]] = []
    source_texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            try:
                value = json.loads(content)
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict) and "summary" in value and "reference" in value:
                summary_object = value
                continue
        if summary_object is None:
            continue
        if not isinstance(content, list):
            continue
        text_values = [
            str(part["text"])
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if text_values and text_values[0] == stop_text:
            break
        if text_values:
            tail.append((str(message.get("role")), text_values[0]))
    for row in expected_rows:
        body = json.loads(str(row["body"]))
        parts = body.get("parts", []) if isinstance(body, dict) else []
        source_texts.extend(
            str(part["value"])
            for part in parts
            if isinstance(part, dict) and part.get("kind") == "text"
        )
    return summary_object or {}, tail, source_texts


def _memory_context_request_kinds(requests: Sequence[object]) -> list[str]:
    """Classify four seed replies and the three expected memory requests."""

    if len(requests) != 7:
        raise GateFailure(f"memory-context 模型请求数量异常：{len(requests)}")
    kinds: list[str] = []
    for raw_request in requests:
        if not isinstance(raw_request, dict):
            raise GateFailure(f"memory-context 模型请求非法：{raw_request!r}")
        payload = raw_request.get("payload")
        if not isinstance(payload, dict):
            raise GateFailure("memory-context 模型请求缺少 payload")
        serialized = json.dumps(payload.get("messages", []), ensure_ascii=False)
        if (
            "更新当前长任务的上下文压缩摘要" in serialized
            and "[Source messages]" in serialized
        ):
            kind = "summary"
        elif "你维护两个长期 Markdown 档案" in serialized:
            kind = "markdown"
        elif _MEMORY_CONTEXT_INPUT in serialized:
            kind = "business"
        elif "seed user" in serialized:
            kind = "seed"
        else:
            raise GateFailure("memory-context 模型请求无法归类")
        if kind in {"summary", "markdown"} and payload.get("tools", []) not in (
            None,
            [],
        ):
            raise GateFailure(f"memory-context {kind} 请求不得携带 tools")
        kinds.append(kind)
    # Committed fact 在 business response settle 后消费，顺序固定。
    if kinds != ["seed", "seed", "seed", "seed", "summary", "business", "markdown"]:
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


def _execution_payload(item: object) -> dict[str, Any]:
    """Decode one standard shell ToolResult envelope."""

    try:
        payload = json.loads(_message_text(item))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise GateFailure(f"shell ToolResult payload 不是 JSON：{item!r}") from error
    if not isinstance(payload, dict):
        raise GateFailure(f"shell ToolResult payload 不是 object：{item!r}")
    return payload


def _execution_last_json(item: object) -> dict[str, Any]:
    """Decode the last JSON line emitted by one standard shell execution."""

    payload = _execution_payload(item)
    output = payload.get("output")
    if not isinstance(output, str):
        raise GateFailure(f"shell ToolResult 缺少 output：{item!r}")
    try:
        value = json.loads(output.strip().splitlines()[-1])
    except (IndexError, TypeError, ValueError) as error:
        raise GateFailure(f"shell ToolResult output 不是 JSON：{item!r}") from error
    if not isinstance(value, dict):
        raise GateFailure(f"shell ToolResult output JSON 不是 object：{item!r}")
    return value


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


def _socket_pending_bytes(connection: socket.socket) -> int:
    """Read queued bytes without consuming the slow subscriber's socket."""

    pending = array.array("I", [0])
    fcntl.ioctl(connection.fileno(), termios.FIONREAD, pending, True)
    return int(pending[0])


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


def _message_wire_projection(page: dict[str, Any]) -> list[dict[str, Any]]:
    """把 Message page 收敛为 messages 表的完整列集合。"""

    items = page.get("items")
    if not isinstance(items, list):
        raise GateFailure(f"message/read items 不是数组：{page!r}")
    projected: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise GateFailure(f"message/read item 不是对象：{item!r}")
        projected.append({
            "id": item.get("id"),
            "session_id": item.get("session_id"),
            "seq": item.get("seq"),
            "timestamp": item.get("timestamp"),
            "author": item.get("author"),
            "source": item.get("source"),
            "body": item.get("body"),
        })
    return projected


def _message_observable_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project wire and database rows onto the fields the Web contract exposes."""

    projected: list[dict[str, Any]] = []
    for row in rows:
        body = row.get("body")
        if not isinstance(body, dict):
            raise GateFailure(f"Message row body 不是 object：{row!r}")
        kind = body.get("kind")
        if not isinstance(kind, str):
            raise GateFailure(f"Message row body 缺少 kind：{row!r}")
        parts = body.get("parts", [])
        if not isinstance(parts, list):
            raise GateFailure(f"Message row parts 不是数组：{row!r}")
        item: dict[str, Any] = {
            "id": row.get("id"),
            "session_id": row.get("session_id"),
            "seq": row.get("seq"),
            "author": row.get("author"),
            "source": row.get("source"),
            "kind": kind,
            "text": [
                part.get("value")
                for part in parts
                if isinstance(part, dict) and part.get("kind") == "text"
            ],
        }
        if kind == "output":
            item["finish"] = body.get("finish")
            item["modelFacts"] = [
                {
                    "call_record_id": value.get("call_record_id"),
                    "thinking": value.get("thinking"),
                }
                for part in parts
                if isinstance(part, dict)
                and part.get("kind") == "model.facts"
                and isinstance(value := part.get("value"), dict)
            ]
        elif kind == "control":
            item["control"] = {
                "action": body.get("action"),
                "through_seq": body.get("through_seq"),
                "reason": body.get("reason"),
            }
        projected.append(item)
    return projected


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
    """验证显式 eligible 程序 Session 的 Message compaction 和 Markdown 投影。"""

    report_dir.mkdir(parents=True, exist_ok=True)
    events_path = report_dir / "events.jsonl"
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    endpoint = Path("/sandbox/akashic.sock")
    checks: list[CheckResult] = []
    client: JsonRpcSocketClient | None = None
    try:
        _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
        _configure_model_gate(context_window=100_000)
        seed_rows = _memory_context_seed_rows()
        scripts = [
            {"mode": "complete", "content": content}
            for role, content in seed_rows
            if role == "assistant"
        ]
        scripts.extend(
            [
                {"mode": "complete", "content": _PC09_COMPACTION_SUMMARY},
                {
                    "mode": "complete",
                    "content": (
                        f"<think>{_MEMORY_CONTEXT_THINKING}</think>"
                        f"{_MEMORY_CONTEXT_RESPONSE}"
                    ),
                },
                {"mode": "complete", "content": _MEMORY_CONTEXT_PROFILE_RESPONSE},
            ]
        )
        _http_json("PUT", f"{model_url}/control/script", scripts)
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        client = _connect_client(endpoint, events_path)
        admission = client.admit_programmatic(
            _MEMORY_CONTEXT_SESSION, persist_memory=True
        )
        if admission.get("learning") != "eligible":
            raise GateFailure(f"memory-context Session 未取得 eligible 准入：{admission!r}")
        database = Path("/sandbox/workspace/sessions.db")
        # 1. 通过正式程序来源显式声明 eligible，再追加四个已结算历史 Turn。
        seed_results: list[dict[str, Any]] = []
        for index in range(4):
            user_text = seed_rows[index * 2][1]
            assistant_text = seed_rows[index * 2 + 1][1]
            input_id = f"mc01-seed-{index}"
            ack = client.send_programmatic(
                _MEMORY_CONTEXT_SESSION, input_id, user_text
            )
            result = _wait_programmatic_result(
                client, _MEMORY_CONTEXT_SESSION, input_id
            )
            if (
                ack.get("message_id") != input_id
                or result.get("status") != "complete"
            ):
                raise GateFailure(
                    f"memory-context seed turn {index} 未完成：ack={ack!r} result={result!r}"
                )
            seed_results.append({"ack": ack, "result": result, "text": assistant_text})

        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            seed_message_rows = connection.execute(
                "SELECT id, session_key, seq, ts, author, source, body FROM messages "
                "WHERE session_key = ? ORDER BY seq",
                (_MEMORY_CONTEXT_SESSION,),
            ).fetchall()
        if len(seed_message_rows) != 8:
            raise GateFailure(
                f"memory-context seed Message 数量异常：{len(seed_message_rows)}"
            )
        # 2. 最终业务 Input 继续走同一程序来源，触发 compaction 后完成 Reply。
        business_id = "mc01-business"
        business_ack = client.send_programmatic(
            _MEMORY_CONTEXT_SESSION, business_id, _MEMORY_CONTEXT_INPUT
        )
        business_result = _wait_programmatic_result(
            client, _MEMORY_CONTEXT_SESSION, business_id
        )
        if business_ack.get("message_id") != business_id:
            raise GateFailure(f"memory-context business ACK 异常：{business_ack!r}")
        if business_result.get("status") != "complete":
            raise GateFailure(
                f"memory-context business turn 未完成：{business_result!r}"
            )

        # Markdown 投影由已提交 SummaryRecord 的普通插件任务完成；等待其
        # 可观察的 provider 请求，避免把 programmatic result 的完成 ACK 当成
        # 所有 post-commit side effect 已经落盘。
        final_requests: list[object] = []
        request_deadline = time.monotonic() + SCENARIO_DEADLINE_S
        while time.monotonic() < request_deadline:
            final_requests = _model_requests(
                _http_json("GET", f"{model_url}/control/requests")
            )
            if len(final_requests) == len(scripts):
                break
            threading.Event().wait(0.05)
        if len(final_requests) != len(scripts):
            raise GateFailure(
                f"memory-context Markdown 请求未完成：{len(final_requests)}/{len(scripts)}"
            )

        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            session_row = connection.execute(
                "SELECT attributes, next_seq FROM sessions WHERE key = ?",
                (_MEMORY_CONTEXT_SESSION,),
            ).fetchone()
            message_rows = connection.execute(
                "SELECT id, session_key, seq, ts, author, source, body FROM messages "
                "WHERE session_key = ? ORDER BY seq",
                (_MEMORY_CONTEXT_SESSION,),
            ).fetchall()
            owner_rows = connection.execute(
                "SELECT key, value FROM owner_records "
                "WHERE owner = 'plugin:compaction' ORDER BY key"
            ).fetchall()
            integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
            owner_records = {
                str(row["key"]): json.loads(str(row["value"])) for row in owner_rows
            }
        integrity_ok = [tuple(row) for row in integrity_rows] == [("ok",)]
        if session_row is None or not integrity_ok:
            raise GateFailure("memory-context Message ledger 缺失或完整性检查失败")
        head_record = owner_records.get(f"head:{_MEMORY_CONTEXT_SESSION}")
        summary_reference = (
            head_record.get("reference") if isinstance(head_record, dict) else None
        )
        summary_record = (
            owner_records.get(f"summary:{summary_reference}")
            if isinstance(summary_reference, str)
            else None
        )
        if not isinstance(summary_reference, str) or not isinstance(summary_record, dict):
            raise GateFailure("memory-context SummaryRecord/head 缺失")

        source_ids = summary_record.get("source_message_ids")
        source_ids = list(source_ids) if isinstance(source_ids, (list, tuple)) else []
        source_ids = [str(item) for item in source_ids]
        message_ids = {str(row["id"]) for row in message_rows}
        summary_record_valid = (
            summary_record.get("reference") == summary_reference
            and summary_record.get("session_id") == _MEMORY_CONTEXT_SESSION
            and summary_record.get("generation") == 1
            and summary_record.get("parent") is None
            and bool(source_ids)
            and len(source_ids) == len(set(source_ids))
            and set(source_ids) <= message_ids
            and summary_record.get("content") == _PC09_COMPACTION_SUMMARY.strip()
            and isinstance(summary_record.get("model_call_ids"), list)
            and bool(summary_record.get("model_call_ids"))
        )

        seed_pairs_match = True
        seed_full_shape: list[dict[str, object]] = []
        for row, (expected_role, expected_text) in zip(message_rows[:8], seed_rows):
            body = json.loads(str(row["body"]))
            parts = body.get("parts") if isinstance(body, dict) else None
            text_values = [
                part.get("value")
                for part in parts or ()
                if isinstance(part, dict) and part.get("kind") == "text"
            ]
            shape_ok = (
                isinstance(body, dict)
                and body.get("kind") == ("input" if expected_role == "user" else "output")
                and int(row["seq"]) == len(seed_full_shape)
                and str(row["author"]) == expected_role
                and str(row["source"]) == "programmatic"
                and text_values == [expected_text]
            )
            seed_pairs_match = seed_pairs_match and shape_ok
            seed_full_shape.append({
                "id": str(row["id"]), "session_key": str(row["session_key"]),
                "seq": int(row["seq"]), "ts": str(row["ts"]),
                "author": str(row["author"]), "source": str(row["source"]),
                "body": body,
            })
        seed_snapshot = [
            (str(row["id"]), str(row["session_key"]), int(row["seq"]),
             str(row["ts"]), str(row["author"]), str(row["source"]),
             str(row["body"]))
            for row in seed_message_rows
        ]
        final_snapshot = [
            (str(row["id"]), str(row["session_key"]), int(row["seq"]),
             str(row["ts"]), str(row["author"]), str(row["source"]),
             str(row["body"]))
            for row in message_rows
        ]
        seed_immutable = (
            len(seed_snapshot) == 8
            and final_snapshot[:8] == seed_snapshot
            and [int(row["seq"]) for row in message_rows] == list(range(10))
        )
        final_messages_only_append = (
            len(message_rows) == 10
            and [str(row["author"]) for row in message_rows[8:]] == ["user", "assistant"]
            and [str(row["source"]) for row in message_rows[8:]]
            == ["programmatic", "programmatic"]
        )
        final_texts = []
        for row in message_rows[8:]:
            body = json.loads(str(row["body"]))
            final_texts.append([
                part.get("value") for part in body.get("parts", [])
                if isinstance(part, dict) and part.get("kind") == "text"
            ])
        final_messages_only_append = final_messages_only_append and (
            final_texts == [[_MEMORY_CONTEXT_INPUT], [_MEMORY_CONTEXT_RESPONSE]]
        )

        final_output_body = json.loads(str(message_rows[9]["body"])) if len(message_rows) > 9 else {}
        final_output_facts = [
            part.get("value")
            for part in final_output_body.get("parts", [])
            if isinstance(part, dict) and part.get("kind") == "model.facts"
        ] if isinstance(final_output_body, dict) else []
        final_output_text = [
            part.get("value")
            for part in final_output_body.get("parts", [])
            if isinstance(part, dict) and part.get("kind") == "text"
        ] if isinstance(final_output_body, dict) else []

        memory_path = Path("/sandbox/workspace/memory/MEMORY.md")
        self_path = Path("/sandbox/workspace/memory/SELF.md")
        receipt_payloads: dict[str, dict[str, object]] = {}
        receipt_inventory: list[dict[str, object]] = []
        receipt_deadline = time.monotonic() + SCENARIO_DEADLINE_S
        while time.monotonic() < receipt_deadline:
            with sqlite3.connect(
                "/sandbox/workspace/memory/markdown-profile-writes.db"
            ) as connection:
                receipt_rows = connection.execute(
                    "SELECT source_ref, kind, payload FROM consolidation_writes "
                    "WHERE kind IN ('markdown_memory_applied_v1', 'markdown_self_applied_v1') "
                    "ORDER BY source_ref, kind"
                ).fetchall()
            receipt_inventory = [
                {"sourceRef": str(source_ref), "kind": str(kind)}
                for source_ref, kind, _payload in receipt_rows
            ]
            candidate: dict[str, dict[str, object]] = {}
            for source_ref, kind, payload in receipt_rows:
                if str(source_ref) != summary_reference:
                    continue
                value = json.loads(str(payload))
                if not isinstance(value, dict):
                    raise GateFailure(f"Markdown receipt 不是 object：{kind}")
                candidate[str(kind)] = value
            memory_content = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
            self_content = self_path.read_text(encoding="utf-8") if self_path.exists() else ""
            if (
                set(candidate) == {
                    "markdown_memory_applied_v1",
                    "markdown_self_applied_v1",
                }
                and memory_path.exists()
                and self_path.exists()
                and candidate["markdown_memory_applied_v1"].get("digest")
                == hashlib.sha256(memory_content.encode("utf-8")).hexdigest()
                and candidate["markdown_self_applied_v1"].get("digest")
                == hashlib.sha256(self_content.encode("utf-8")).hexdigest()
            ):
                receipt_payloads = candidate
                break
            threading.Event().wait(0.05)
        if len(receipt_payloads) != 2:
            raise GateFailure(
                "Markdown receipt 或目标文件未收敛："
                f"{sorted(receipt_payloads)} inventory={receipt_inventory!r} "
                f"summaryReference={summary_reference!r}"
            )

        pending_path = Path("/sandbox/workspace/memory/PENDING.md")
        pending_retired = (
            not pending_path.exists()
            or not pending_path.read_text(encoding="utf-8").strip()
        )
        memory_applied = receipt_payloads.get("markdown_memory_applied_v1")
        self_applied = receipt_payloads.get("markdown_self_applied_v1")

        request_kinds = _memory_context_request_kinds(final_requests)
        source_positions = [
            index for index, row in enumerate(message_rows)
            if str(row["id"]) in source_ids
        ]
        if len(source_positions) != len(source_ids):
            raise GateFailure("SummaryRecord 引用了不存在的 Message")
        source_contiguous = (
            bool(source_positions)
            and source_positions == list(range(source_positions[0], source_positions[-1] + 1))
            and source_positions[0] % 2 == 0
            and (source_positions[-1] - source_positions[0] + 1) % 2 == 0
        )
        source_digest = _memory_context_source_plan_digest(message_rows, source_ids)

        summary_source_rows = _memory_context_summary_source(final_requests)
        expected_summary_source = []
        for position in source_positions:
            row = message_rows[position]
            expected_summary_source.append(
                {
                    "message_id": str(row["id"]),
                    "source": str(row["source"]),
                    "seq": int(row["seq"]),
                    "body": _memory_context_summary_body(
                        json.loads(str(row["body"]))
                    ),
                }
            )
        summary_source_matches = summary_source_rows == expected_summary_source

        business_request = next(
            request for request in final_requests
            if isinstance(request, dict)
            and isinstance(request.get("payload"), dict)
            and _MEMORY_CONTEXT_INPUT in json.dumps(
                request["payload"].get("messages", []), ensure_ascii=False
            )
        )
        business_serialized = json.dumps(
            business_request["payload"].get("messages", []), ensure_ascii=False
        )
        summary_payload, business_tail, source_contents = _memory_context_business_tail(
            business_request["payload"].get("messages"),
            [message_rows[position] for position in source_positions],
            _MEMORY_CONTEXT_INPUT,
        )
        source_end = source_positions[-1] + 1
        expected_business_tail = []
        for row in message_rows[source_end:8]:
            body = json.loads(str(row["body"]))
            parts = body.get("parts", []) if isinstance(body, dict) else []
            text_values = [
                str(part["value"])
                for part in parts
                if isinstance(part, dict) and part.get("kind") == "text"
            ]
            if len(text_values) != 1:
                raise GateFailure("业务保留尾部 Message 缺少唯一原文 text")
            expected_business_tail.append((str(row["author"]), text_values[0]))
        summary_binding_ids = [
            str(part["value"]["reference"])
            for part in final_output_body.get("parts", [])
            if (
                isinstance(part, dict)
                and part.get("kind") == "context.summary"
                and isinstance(part.get("value"), dict)
                and isinstance(part["value"].get("reference"), str)
            )
        ] if isinstance(final_output_body, dict) else []
        if len(summary_binding_ids) != 1:
            raise GateFailure(
                "业务 Output 缺少唯一 context.summary binding："
                f"{summary_binding_ids!r}"
            )
        summary_binding_id = summary_binding_ids[0]
        with sqlite3.connect(database) as connection:
            binding_row = connection.execute(
                "SELECT descriptor FROM bindings WHERE binding_id = ?",
                (summary_binding_id,),
            ).fetchone()
        binding_descriptor = (
            json.loads(str(binding_row[0])) if binding_row is not None else None
        )
        binding_metadata = (
            binding_descriptor.get("metadata")
            if isinstance(binding_descriptor, dict)
            else None
        )
        summary_binding_metadata_valid = (
            isinstance(binding_descriptor, dict)
            and binding_descriptor.get("version") == 1
            and binding_descriptor.get("service") == "compaction.summaries.v1"
            and isinstance(binding_metadata, dict)
            and binding_metadata.get("record_ref") == summary_reference
            and binding_metadata.get("session_id") == _MEMORY_CONTEXT_SESSION
        )
        business_summary_replaced = (
            summary_payload.get("summary") == summary_record.get("content")
            and summary_payload.get("reference") == summary_binding_id
            and summary_binding_metadata_valid
            and business_tail == expected_business_tail
            and all(str(message_id) not in business_serialized for message_id in source_ids)
            and all(content not in business_serialized for content in source_contents)
        )

        owner_snapshot = [
            (str(row["key"]), str(row["value"])) for row in owner_rows
        ]
        receipts_database = Path(
            "/sandbox/workspace/memory/markdown-profile-writes.db"
        )
        with sqlite3.connect(receipts_database) as connection:
            receipt_snapshot = [
                tuple(str(value) for value in row)
                for row in connection.execute(
                    "SELECT source_ref, kind, payload, trailing_blank_line, done_at "
                    "FROM consolidation_writes ORDER BY source_ref, kind"
                ).fetchall()
            ]
        target_snapshot = {
            "memory": memory_path.read_text(encoding="utf-8"),
            "self": self_path.read_text(encoding="utf-8"),
        }
        pending_snapshot = (
            pending_path.read_text(encoding="utf-8")
            if pending_path.exists()
            else None
        )

        # 3. 断开原连接，再用同一 message_id/text 重试持久 send/result。
        client.close()
        client = None
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        client = _connect_client(endpoint, events_path)
        retry_ack = client.send_programmatic(
            _MEMORY_CONTEXT_SESSION, business_id, _MEMORY_CONTEXT_INPUT
        )
        retry_result = _wait_programmatic_result(
            client, _MEMORY_CONTEXT_SESSION, business_id
        )
        retry_requests = _model_requests(
            _http_json("GET", f"{model_url}/control/requests")
        )
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            retry_message_rows = connection.execute(
                "SELECT id, session_key, seq, ts, author, source, body "
                "FROM messages WHERE session_key = ? ORDER BY seq",
                (_MEMORY_CONTEXT_SESSION,),
            ).fetchall()
            retry_owner_rows = connection.execute(
                "SELECT key, value FROM owner_records "
                "WHERE owner = 'plugin:compaction' ORDER BY key"
            ).fetchall()
        retry_message_snapshot = [
            (str(row["id"]), str(row["session_key"]), int(row["seq"]),
             str(row["ts"]), str(row["author"]), str(row["source"]),
             str(row["body"]))
            for row in retry_message_rows
        ]
        retry_owner_snapshot = [
            (str(row["key"]), str(row["value"])) for row in retry_owner_rows
        ]
        with sqlite3.connect(receipts_database) as connection:
            retry_receipt_snapshot = [
                tuple(str(value) for value in row)
                for row in connection.execute(
                    "SELECT source_ref, kind, payload, trailing_blank_line, done_at "
                    "FROM consolidation_writes ORDER BY source_ref, kind"
                ).fetchall()
            ]
        retry_target_snapshot = {
            "memory": memory_path.read_text(encoding="utf-8"),
            "self": self_path.read_text(encoding="utf-8"),
        }
        retry_pending_snapshot = (
            pending_path.read_text(encoding="utf-8")
            if pending_path.exists()
            else None
        )
        retry_ack_same = retry_ack == business_ack
        retry_result_same = retry_result == business_result
        retry_messages_unchanged = retry_message_snapshot == final_snapshot
        retry_owner_unchanged = retry_owner_snapshot == owner_snapshot
        retry_receipts_unchanged = retry_receipt_snapshot == receipt_snapshot
        retry_targets_unchanged = retry_target_snapshot == target_snapshot
        retry_pending_unchanged = retry_pending_snapshot == pending_snapshot
        retry_transport_recovery = (
            retry_ack_same
            and retry_result_same
            and len(retry_message_snapshot) == 10
            and retry_messages_unchanged
            and retry_owner_unchanged
            and retry_receipts_unchanged
            and retry_targets_unchanged
            and retry_pending_unchanged
            and len(retry_requests) == len(scripts) == 7
        )
        scripts_boundary = (
            [request.get("script") for request in final_requests
             if isinstance(request, dict)] == scripts
        )
        thinking_boundary = (
            len(final_output_facts) == 1
            and isinstance(final_output_facts[0], dict)
            and final_output_facts[0].get("thinking") == _MEMORY_CONTEXT_THINKING
            and final_output_body.get("finish") == "complete"
        )
        attributes = json.loads(str(session_row["attributes"]))
        attributes_valid = (
            isinstance(attributes, dict)
            and attributes.get("visibility") == "internal"
            and attributes.get("learning") == "eligible"
        )
        checks.append(
            CheckResult(
                "MC-01",
                business_result.get("status") == "complete"
                and final_output_text == [_MEMORY_CONTEXT_RESPONSE]
                and request_kinds == ["seed", "seed", "seed", "seed", "summary", "business", "markdown"]
                and scripts_boundary
                and thinking_boundary
                and attributes_valid
                and session_row["next_seq"] == 10
                and seed_pairs_match
                and seed_immutable
                and summary_record_valid
                and summary_source_matches
                and source_contiguous
                and final_messages_only_append
                and summary_binding_metadata_valid
                and business_summary_replaced
                and retry_transport_recovery
                and pending_retired
                and memory_applied is not None
                and self_applied is not None,
                {
                    "admission": admission,
                    "seedResults": seed_results,
                    "businessAck": business_ack,
                    "businessResult": business_result,
                    "requestKinds": request_kinds,
                    "ledger": {
                        "sessionAttributes": attributes,
                        "nextSeq": session_row["next_seq"],
                        "summaryReference": summary_reference,
                        "sourceIds": source_ids,
                        "sourcePlanDigest": source_digest,
                        "sourcePositions": source_positions,
                        "sourceContiguous": source_contiguous,
                        "summarySourceMatches": summary_source_matches,
                        "summarySourceRows": summary_source_rows,
                        "summaryBinding": {
                            "bindingId": summary_binding_id,
                            "metadata": binding_metadata,
                            "metadataValid": summary_binding_metadata_valid,
                        },
                        "businessSummaryReplaced": business_summary_replaced,
                        "businessTail": business_tail,
                        "expectedBusinessTail": expected_business_tail,
                        "summaryRecordValid": summary_record_valid,
                        "seedPairsMatch": seed_pairs_match,
                        "seedImmutable": seed_immutable,
                        "finalMessagesOnlyAppend": final_messages_only_append,
                    },
                    "transportRetry": {
                        "disconnected": True,
                        "newConnection": True,
                        "ack": retry_ack,
                        "result": retry_result,
                        "ackSameIdSeq": retry_ack_same,
                        "resultUnchanged": retry_result_same,
                        "messageCount": len(retry_message_snapshot),
                        "messagesUnchanged": retry_messages_unchanged,
                        "summaryOwnerCount": len(owner_snapshot),
                        "summaryOwnerUnchanged": retry_owner_unchanged,
                        "receiptCount": len(receipt_snapshot),
                        "receiptsUnchanged": retry_receipts_unchanged,
                        "targetsUnchanged": retry_targets_unchanged,
                        "pendingUnchanged": retry_pending_unchanged,
                        "providerRequestCount": len(retry_requests),
                    },
                    "pendingRetired": pending_retired,
                    "memoryApplied": memory_applied is not None,
                    "selfApplied": self_applied is not None,
                    "markdownTargets": {
                        "memory": memory_path.read_text(encoding="utf-8"),
                        "self": self_path.read_text(encoding="utf-8"),
                    },
                    "scriptsBoundary": scripts_boundary,
                    "thinkingBoundary": thinking_boundary,
                    "modelRequestCount": len(final_requests),
                    "seedFullShape": seed_full_shape,
                    "finalOutputFacts": final_output_facts,
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
    restart_state: dict[str, object] = {}
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
        pc07_identity_command = (
            "pid=$(cat /sandbox/workspace/pc07-shell.pid) || "
            "{ printf 'PC07 identity: cannot read pid\\n' >&2; exit 43; }; "
            "expected=$(cat /sandbox/workspace/pc07-shell.starttime) || "
            "{ printf 'PC07 identity: cannot read expected starttime\\n' >&2; exit 43; }; "
            "stat_path=/proc/$pid/stat; alive=false; status=42; actual=null; "
            "if [ -e \"$stat_path\" ]; then "
            "actual=$(awk '{print $22}' \"$stat_path\") || "
            "{ printf 'PC07 identity: cannot read %s\\n' \"$stat_path\" >&2; exit 43; }; "
            "if [ -z \"$actual\" ]; then "
            "printf 'PC07 identity: empty starttime from %s\\n' \"$stat_path\" >&2; exit 43; fi; "
            "if [ \"$actual\" = \"$expected\" ]; then "
            "if kill -0 \"$pid\" 2>/dev/null; then alive=true; status=0; "
            "elif [ ! -e \"$stat_path\" ]; then actual=null; "
            "else printf 'PC07 identity: cannot verify pid %s\\n' \"$pid\" >&2; exit 43; fi; "
            "fi; fi; "
            "printf '{\"alive\":%s,\"pid\":%s,' \"$alive\" \"$pid\"; "
            "printf '\"expected_starttime\":%s,\"actual_starttime\":%s}\\n' "
            "\"$expected\" \"$actual\"; exit \"$status\""
        )
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
                                    "pid_file=/sandbox/workspace/pc07-shell.pid; "
                                    "start_file=/sandbox/workspace/pc07-shell.starttime; "
                                    "printf '%s\\n' \"$$\" > \"$pid_file\"; "
                                    "awk '{print $22}' /proc/$$/stat > \"$start_file\"; "
                                    "printf '{\"pid\":%s,\"starttime\":%s}\\n' "
                                    "\"$(cat \"$pid_file\")\" \"$(cat \"$start_file\")\"; "
                                    "exec sleep 300"
                                ),
                                "description": "PC07 long running cleanup probe",
                                "yield_time_ms": 250,
                                "timeout": 300,
                            },
                        }
                    ],
                },
                {
                    "mode": "complete",
                    "tool_calls": [
                        {
                            "id": "call_pc07_shell_identity_before_pause",
                            "name": "shell",
                            "arguments": {
                                "command": pc07_identity_command,
                                "description": "PC07 verify shell PID identity before pause",
                                "yield_time_ms": 250,
                                "timeout": 30,
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
            first, pc07, "tool_result", minimum=2
        )
        pc07_provider_count = len(
            _model_requests(_http_json("GET", f"{model_url}/control/requests"))
        )
        _wait_barrier(model_url, pc07_pause_barrier)
        _release_barrier(model_url, pc07_pause_barrier)
        pc07_outputs_with_tool = _message_items(pc07_page_with_tool, "output")
        pc07_tool_payloads = [
            _execution_payload(item)
            for item in pc07_tool_rows
        ]
        pc07_tool_result = next(
            (
                item
                for item, payload in zip(pc07_tool_rows, pc07_tool_payloads)
                if isinstance(payload, dict) and isinstance(payload.get("execution_id"), int)
            ),
            None,
        )
        pc07_pre_identity_result = next(
            (
                item
                for item, payload in zip(pc07_tool_rows, pc07_tool_payloads)
                if isinstance(payload, dict)
                and isinstance(payload.get("output"), str)
                and '"alive":true' in payload["output"].replace(" ", "")
            ),
            None,
        )
        if pc07_tool_result is None or pc07_pre_identity_result is None:
            raise GateFailure(
                f"PC07 shell identity ToolResult 缺失：{pc07_tool_rows!r}"
            )
        pc07_tool_ref = pc07_tool_result.get("body", {}).get("call_ref", {})
        try:
            pc07_tool_payload = _execution_payload(pc07_tool_result)
            pc07_execution_id = int(pc07_tool_payload["execution_id"])
            pc07_shell_identity = _execution_last_json(pc07_tool_result)
            pc07_pre_identity = _execution_last_json(pc07_pre_identity_result)
        except (KeyError, TypeError, ValueError) as error:
            raise GateFailure(
                f"PC07 shell 未返回可核验 PID identity：{pc07_tool_rows!r}"
            ) from error
        if not (
            isinstance(pc07_shell_identity, dict)
            and isinstance(pc07_pre_identity, dict)
            and pc07_shell_identity.get("pid") == pc07_pre_identity.get("pid")
            and pc07_shell_identity.get("starttime")
            == pc07_pre_identity.get("expected_starttime")
            and pc07_pre_identity.get("expected_starttime")
            == pc07_pre_identity.get("actual_starttime")
            and pc07_pre_identity.get("alive") is True
        ):
            raise GateFailure(
                f"PC07 pause 前 shell identity 未确认存活："
                f"{pc07_shell_identity!r} / {pc07_pre_identity!r}"
            )
        pc07_tool_call: dict[str, Any] | None = None
        for output in pc07_outputs_with_tool:
            if output.get("id") != pc07_tool_ref.get("message_id"):
                continue
            parts = output.get("body", {}).get("parts", [])
            if not isinstance(parts, list):
                continue
            part_index = pc07_tool_ref.get("part_index")
            if (
                type(part_index) is int
                and 0 <= part_index < len(parts)
                and isinstance(parts[part_index], dict)
                and parts[part_index].get("kind") == "tool_call"
                and parts[part_index].get("name") == "shell"
            ):
                pc07_tool_call = {
                    "message_id": output.get("id"),
                    "part_index": part_index,
                    "part": parts[part_index],
                }
                break
        if pc07_tool_call is None:
            raise GateFailure(f"PC07 shell ToolCall 缺失：{pc07_tool_ref!r}")

        # 后续 provider 调用故意等待客户端断开，使 pause 发生在工具回执之后。
        pc07_pause_ack = first.request_result(
            "programmatic/message/pause",
            {"session_id": pc07, "message_id": "pc07-pause"},
        )
        pc07_paused_result = first.programmatic_result(pc07, "pc07-input")
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
                        },
                        {
                            "id": "call_pc07_shell_identity_after_pause",
                            "name": "shell",
                            "arguments": {
                                "command": pc07_identity_command,
                                "description": "PC07 verify shell PID identity after pause",
                                "yield_time_ms": 250,
                                "timeout": 30,
                            },
                        },
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
        pc07_page_before_release = first.read_messages(pc07)
        pc07_result_before_release = first.programmatic_result(pc07, "pc07-input")
        pc07_stale_pause_ack = first.request_result(
            "programmatic/message/pause",
            {"session_id": pc07, "message_id": "pc07-pause"},
        )
        pc07_pre_release_tool_results = _message_items(
            pc07_page_before_release, "tool_result"
        )
        pc07_post_identity_result: dict[str, Any] | None = None
        for item in pc07_pre_release_tool_results:
            try:
                identity = _execution_last_json(item)
            except GateFailure:
                continue
            if identity.get("alive") is False:
                pc07_post_identity_result = item
                break
        pc07_write_stdin_result = next(
            (
                item
                for item in pc07_pre_release_tool_results
                if "未知 execution_id" in _message_text(item)
            ),
            None,
        )
        try:
            if pc07_post_identity_result is None:
                raise GateFailure("PC07 pause 后 shell identity ToolResult 缺失")
            pc07_post_identity = _execution_last_json(pc07_post_identity_result)
        except (GateFailure, KeyError, TypeError, ValueError) as error:
            raise GateFailure(
                f"PC07 pause 后 shell identity 未返回："
                f"{pc07_pre_release_tool_results!r}"
            ) from error
        _release_barrier(model_url, "pc07-new-input")
        pc07_result = first.programmatic_result(pc07, "pc07-input")
        pc07_new_result = _wait_programmatic_result(first, pc07, "pc07-new-input")
        pc07_page = first.read_messages(pc07)
        pc07_controls = _message_items(pc07_page, "control")
        pc07_inputs = _message_items(pc07_page, "input")
        pc07_outputs = _message_items(pc07_page, "output")
        pc07_tool_results = _message_items(pc07_page, "tool_result")
        pc07_cleanup_tool_result = pc07_write_stdin_result
        pc07_post_identity_ref = (
            pc07_post_identity_result.get("body", {}).get("call_ref", {})
            if pc07_post_identity_result is not None
            else {}
        )
        pc07_post_identity_call = any(
            output.get("id") == pc07_post_identity_ref.get("message_id")
            and isinstance(output.get("body", {}).get("parts"), list)
            and type(pc07_post_identity_ref.get("part_index")) is int
            and 0 <= pc07_post_identity_ref["part_index"]
            < len(output["body"]["parts"])
            and isinstance(output["body"]["parts"][pc07_post_identity_ref["part_index"]], dict)
            and output["body"]["parts"][pc07_post_identity_ref["part_index"]].get("kind")
            == "tool_call"
            and output["body"]["parts"][pc07_post_identity_ref["part_index"]].get("name")
            == "shell"
            for output in pc07_outputs
        )
        pc07_shell_gone = (
            pc07_post_identity.get("pid") == pc07_shell_identity.get("pid")
            and pc07_post_identity.get("expected_starttime")
            == pc07_shell_identity.get("starttime")
            and pc07_post_identity.get("actual_starttime") is None
            and pc07_post_identity.get("alive") is False
            and pc07_post_identity_call
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
            and pc07_pause_ack.get("seq") == pc07_pre_identity_result.get("seq", -1) + 1
            and pc07_paused_result.get("status") == "pause"
            # Once a newer Input is appended, the old Input is open again;
            # the durable result before that append remains the pause proof.
            and pc07_result_before_release.get("status") == "open"
            and pc07_page_before_release.get("items") is not None
            and pc07_post_identity_result is not None
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
            and pc07_controls[0].get("body", {}).get("through_seq")
            == pc07_pre_identity_result.get("seq")
            and len(pc07_tool_results) == 4
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
                    "resultBeforeNewInputRelease": pc07_result_before_release,
                    "result": pc07_result,
                    "newInputResult": pc07_new_result,
                    "controls": pc07_controls,
                    "toolCall": pc07_tool_call,
                    "toolResult": pc07_tool_result,
                    "prePauseShellIdentity": pc07_pre_identity,
                    "postPauseShellIdentity": pc07_post_identity,
                    "shellPid": pc07_shell_identity.get("pid"),
                    "shellPidGone": pc07_post_identity.get("actual_starttime") is None,
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
        producer_stop = threading.Event()
        producer_done = threading.Event()
        producer_client: JsonRpcSocketClient | None = None
        producer: threading.Thread | None = None
        pressure_pending_bytes = 0
        pressure_min_pending_bytes = 0
        pressure_observed_at: float | None = None
        healthy_started_at: float | None = None
        healthy_completed_at: float | None = None
        producer_tail_released_at: float | None = None
        slow_eof_at: float | None = None
        slow_closed = False
        drained_bytes = 0

        def produce_slow_tail() -> None:
            """Append a bounded tail and expose all producer failures to the gate."""

            nonlocal producer_sent
            try:
                for index in range(producer_count):
                    if producer_stop.is_set():
                        return
                    assert producer_client is not None
                    producer_client.send_programmatic(
                        pc09,
                        f"pc09-input-{index}",
                        f"pc09 slow input {index}",
                    )
                    result = _wait_programmatic_result(
                        producer_client, pc09, f"pc09-input-{index}"
                    )
                    if result.get("status") != "complete":
                        raise GateFailure(
                            f"PC09 slow input {index} 未完成：{result!r}"
                        )
                    producer_sent = index + 1
                    if producer_sent == burst_count:
                        burst_ready.set()
                        while not continue_tail.wait(0.05):
                            if producer_stop.is_set():
                                return
            except BaseException as error:
                if not producer_stop.is_set():
                    producer_errors.append(f"{type(error).__name__}: {error}")
            finally:
                burst_ready.set()
                producer_done.set()

        try:
            producer_client = _connect_client(endpoint, events_path)
            clients.append(producer_client)
            producer = threading.Thread(
                target=produce_slow_tail, name="pc09-slow-producer", daemon=False
            )
            producer.start()
            if not burst_ready.wait(SCENARIO_DEADLINE_S):
                raise GateFailure("PC09 slow producer 未建立第一段压力")
            slow_receive_buffer = slow._socket.getsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF
            )
            pressure_min_pending_bytes = max(64 * 1024, slow_receive_buffer * 2)
            pressure_deadline = time.monotonic() + SCENARIO_DEADLINE_S
            while time.monotonic() < pressure_deadline:
                pressure_pending_bytes = _socket_pending_bytes(slow._socket)
                if pressure_pending_bytes >= pressure_min_pending_bytes:
                    break
                if producer_done.is_set() and producer_sent < burst_count:
                    raise GateFailure(
                        f"PC09 slow producer 在 burst 内结束：{producer_sent}/{burst_count}"
                    )
                threading.Event().wait(0.05)
            pressure_pending_bytes = _socket_pending_bytes(slow._socket)
            pressure_observed_at = time.monotonic()
            if pressure_pending_bytes < pressure_min_pending_bytes:
                raise GateFailure(
                    "PC09 慢订阅未达到可观测 socket backlog："
                    f"{pressure_pending_bytes} < {pressure_min_pending_bytes}"
                )
            second.admit_programmatic(pc09_healthy)
            healthy_started_at = time.monotonic()
            second.send_programmatic(
                pc09_healthy, "pc09-healthy-input", "pc09 healthy input"
            )
            pc09_healthy_result = _wait_programmatic_result(
                second, pc09_healthy, "pc09-healthy-input"
            )
            healthy_completed_at = time.monotonic()
            continue_tail.set()
            producer_tail_released_at = time.monotonic()
            if not producer_done.wait(SCENARIO_DEADLINE_S):
                raise GateFailure("PC09 slow producer 未在 deadline 内完成")
            assert producer is not None
            producer.join(timeout=SCENARIO_DEADLINE_S)
            if producer.is_alive():
                raise GateFailure("PC09 slow producer join 超时")
            slow_closed, drained_bytes = _drain_socket_until_eof(slow._socket)
            slow_eof_at = time.monotonic()
            checks.append(
                CheckResult(
                    "PC-09",
                    not producer_errors
                    and producer_sent == producer_count
                    and pressure_pending_bytes >= pressure_min_pending_bytes
                    and slow_closed
                    and pc09_healthy_result.get("status") == "complete"
                    and pressure_observed_at < healthy_completed_at
                    and healthy_completed_at <= producer_tail_released_at
                    and producer_tail_released_at <= slow_eof_at,
                    {
                        "slowConnectionClosed": slow_closed,
                        "healthyResult": pc09_healthy_result,
                        "producerSent": producer_sent,
                        "producerCount": producer_count,
                        "producerErrors": producer_errors,
                        "producerJoined": not producer.is_alive(),
                        "pressurePending": pressure_pending_bytes >= pressure_min_pending_bytes,
                        "pressurePendingBytesBeforeHealthy": pressure_pending_bytes,
                        "pressureMinimumPendingBytes": pressure_min_pending_bytes,
                        "drainedBytesBeforeEof": drained_bytes,
                        "healthyIsolation": pc09_healthy_result.get("status") == "complete",
                        "healthyStartedAt": healthy_started_at,
                        "healthyCompletedAt": healthy_completed_at,
                        "pressureObservedAt": pressure_observed_at,
                        "producerTailReleasedAt": producer_tail_released_at,
                        "slowEofAt": slow_eof_at,
                        "pressureObservedBeforeHealthy": pressure_observed_at < healthy_completed_at,
                        "healthyCompletedBeforeTailRelease": healthy_completed_at <= producer_tail_released_at,
                        "tailReleasedBeforeSlowEof": producer_tail_released_at <= slow_eof_at,
                        "slowReceiveBuffer": slow_receive_buffer,
                    },
                )
            )
        finally:
            continue_tail.set()
            producer_stop.set()
            if producer is not None and producer.is_alive() and producer_client is not None:
                try:
                    producer_client.close()
                except OSError:
                    pass
            if producer is not None:
                producer.join(timeout=SCENARIO_DEADLINE_S)
                if producer.is_alive():
                    raise GateFailure("PC09 slow producer cleanup join 超时")
            if producer_client is not None:
                producer_client.close()
                if producer_client in clients:
                    clients.remove(producer_client)
            slow.close()
            if slow in clients:
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
            and pc10_tools[0].get("body", {}).get("outcome") == "error"
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

        # 7. 新 Message 合同下核对 Web/程序接口 parity、并发、撤销和恢复。
        from websockets.sync.client import connect as connect_websocket

        websocket_url = "ws://akashic-control-gate:2236/ws"
        database = Path("/sandbox/workspace/sessions.db")

        def output_projection(rows: list[dict[str, Any]]) -> dict[str, object]:
            outputs = [
                row for row in rows
                if isinstance(row.get("body"), dict)
                and row["body"].get("kind") == "output"
            ]
            final = next(
                (
                    row for row in outputs
                    if isinstance(row["body"].get("finish"), str)
                    and row["body"]["finish"] != "continue"
                ),
                None,
            )
            text = ""
            thinking: str | None = None
            if isinstance(final, dict) and isinstance(final.get("body"), dict):
                for part in final["body"].get("parts", []):
                    if not isinstance(part, dict):
                        continue
                    if part.get("kind") == "text":
                        text += str(part.get("value", ""))
                    if part.get("kind") == "model.facts" and isinstance(
                        part.get("value"), dict
                    ):
                        value = part["value"].get("thinking")
                        if isinstance(value, str):
                            thinking = value
            return {
                "outputCount": len(outputs),
                "final": final,
                "content": text,
                "thinking": thinking,
            }

        def programmatic_projection(
            page: dict[str, Any], result: dict[str, Any]
        ) -> dict[str, object]:
            rows = [item for item in page.get("items", []) if isinstance(item, dict)]
            return {
                "status": result.get("status"),
                "endingMessageId": result.get("ending_message_id"),
                "endingSeq": result.get("ending_seq"),
                "messages": rows,
                **output_projection(rows),
            }

        fixtures = (
            (
                "parity success",
                {
                    "mode": "complete",
                    "content": "<think>channel reasoning</think>parity result",
                },
                "complete",
            ),
            (
                "parity failure",
                [
                    {"mode": "error", "status": 500},
                    {"mode": "error", "status": 500},
                ],
                "failure",
            ),
        )
        parity_evidence: list[dict[str, object]] = []
        parity_passed = True
        for index, (input_text, script, expected_status) in enumerate(fixtures):
            program_session = f"programmatic:pc16-parity-{index}"
            program_input = f"pc16-program-{index}"
            first.admit_programmatic(program_session)
            _http_json("PUT", f"{model_url}/control/script", script)
            program_ack = first.send_programmatic(
                program_session, program_input, input_text
            )
            program_result = _wait_programmatic_result(
                first, program_session, program_input
            )
            program_page = first.read_messages(program_session)
            program_view = programmatic_projection(program_page, program_result)

            with connect_websocket(
                websocket_url, open_timeout=READINESS_DEADLINE_S
            ) as web:
                web.send(json.dumps({
                    "type": "session.create",
                    "request_id": f"pc16-create-{index}",
                }))
                web_session = str(
                    json.loads(web.recv(timeout=SCENARIO_DEADLINE_S))["session_id"]
                )
                web.send(json.dumps({
                    "type": "session.follow",
                    "version": 2,
                    "request_id": f"pc16-follow-{index}",
                    "session_id": web_session,
                    "after_seq": -1,
                }))
                following = json.loads(web.recv(timeout=SCENARIO_DEADLINE_S))
                if (
                    following.get("type") != "session.following"
                    or following.get("session_id") != web_session
                ):
                    raise GateFailure(f"Web session.follow ACK 非法：{following!r}")
                _http_json("PUT", f"{model_url}/control/script", script)
                web_request_id = f"pc16-web-{index}"
                web.send(json.dumps({
                    "type": "message.send",
                    "request_id": web_request_id,
                    "session_id": web_session,
                    "text": input_text,
                    "media": [],
                }))
                web_rows, web_status_frames = _receive_web_follow_rows(
                    web,
                    web_session,
                    lambda rows, _statuses: (
                        any(
                            isinstance(row.get("body"), dict)
                            and row["body"].get("kind") == "output"
                            and row["body"].get("finish") != "continue"
                            for row in rows
                        )
                        if expected_status == "complete"
                        else any(
                            isinstance(row.get("body"), dict)
                            and row["body"].get("kind") == "control"
                            and row["body"].get("action") == "failure"
                            for row in rows
                        )
                    ),
                )
            web_raw_rows = _wait_message_log_rows(
                database,
                web_session,
                minimum=len(web_rows),
                required_ids={str(row["id"]) for row in web_rows},
            )
            web_wire_observable = _message_observable_projection(web_rows)
            web_raw_observable = _message_observable_projection(web_raw_rows)
            web_raw_matches_wire = (
                len(web_raw_rows) == len(web_rows)
                and web_raw_observable == web_wire_observable
            )
            web_view = output_projection(web_rows)
            web_status = (
                "complete"
                if web_view["outputCount"] == 1
                else "failure"
                if any(
                    isinstance(row.get("body"), dict)
                    and row["body"].get("kind") == "control"
                    and row["body"].get("action") == "failure"
                    for row in web_rows
                )
                else "open"
            )
            wire_projection = {
                "following": following,
                "messages": web_rows,
                "replyStatus": web_status_frames,
            }
            fixture_passed = (
                program_ack.get("message_id") == program_input
                and program_result.get("status") == expected_status
                and web_status == expected_status
                and web_raw_matches_wire
                and any(
                    isinstance(frame, dict)
                    and frame.get("type") == "reply.status"
                    and isinstance(frame.get("available"), bool)
                    for frame in web_status_frames
                )
            )
            if expected_status == "complete":
                fixture_passed = fixture_passed and (
                    program_view["content"] == "parity result"
                    and program_view["thinking"] == "channel reasoning"
                    and web_view["content"] == "parity result"
                    and web_view["thinking"] == "channel reasoning"
                )
            else:
                fixture_passed = fixture_passed and (
                    web_view["outputCount"] == 0
                )
            parity_passed = parity_passed and fixture_passed
            parity_evidence.append({
                "input": input_text,
                "sessionId": program_session,
                "inputId": program_input,
                "passed": fixture_passed,
                "programmaticResult": program_result,
                "programmatic": program_view,
                "channel": {"status": web_status, "messages": web_rows, **web_view},
                "channelDatabase": {
                    "sessionId": web_session,
                    "rawMessages": web_raw_rows,
                    "observable": web_raw_observable,
                    "matchesWire": web_raw_matches_wire,
                },
                "channelWire": wire_projection,
            })

        lane_evidence: dict[str, object] = {}
        with (
            connect_websocket(websocket_url, open_timeout=READINESS_DEADLINE_S) as slow_web,
            connect_websocket(websocket_url, open_timeout=READINESS_DEADLINE_S) as fast_web,
        ):
            slow_web.send(json.dumps({"type": "session.create", "request_id": "pc16-slow"}))
            fast_web.send(json.dumps({"type": "session.create", "request_id": "pc16-fast"}))
            slow_session = str(json.loads(slow_web.recv(timeout=SCENARIO_DEADLINE_S))["session_id"])
            fast_session = str(json.loads(fast_web.recv(timeout=SCENARIO_DEADLINE_S))["session_id"])
            for web, session_id, request_id in (
                (slow_web, slow_session, "pc16-slow-follow"),
                (fast_web, fast_session, "pc16-fast-follow"),
            ):
                web.send(json.dumps({
                    "type": "session.follow",
                    "version": 2,
                    "request_id": request_id,
                    "session_id": session_id,
                    "after_seq": -1,
                }))
                following = json.loads(web.recv(timeout=SCENARIO_DEADLINE_S))
                if (
                    following.get("type") != "session.following"
                    or following.get("session_id") != session_id
                ):
                    raise GateFailure(f"Web Session follow ACK 非法：{following!r}")
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
            slow_request_id = "pc16-slow-input"
            slow_web.send(json.dumps({
                "type": "message.send",
                "request_id": slow_request_id,
                "session_id": slow_session,
                "text": "slow lane",
                "media": [],
            }))
            _wait_barrier(model_url, "pc16-channel-slow")
            fast_request_id = "pc16-fast-input"
            fast_web.send(json.dumps({
                "type": "message.send",
                "request_id": fast_request_id,
                "session_id": fast_session,
                "text": "fast lane",
                "media": [],
            }))
            fast_rows, fast_statuses = _receive_web_follow_rows(
                fast_web,
                fast_session,
                lambda rows, _statuses: output_projection(rows)["outputCount"] == 1,
            )
            fast_completed = {
                "status": "complete"
                if output_projection(fast_rows)["outputCount"] == 1
                else "open",
                "messages": fast_rows,
                "replyStatus": fast_statuses,
            }
            _release_barrier(model_url, "pc16-channel-slow")
            slow_rows, slow_statuses = _receive_web_follow_rows(
                slow_web,
                slow_session,
                lambda rows, _statuses: output_projection(rows)["outputCount"] == 1,
            )
            lane_evidence["differentSessions"] = {
                "slowSessionId": slow_session,
                "fastSessionId": fast_session,
                "fastCompletedBeforeRelease": fast_completed,
                "fastFinal": output_projection(fast_rows)["content"],
                "slowFinal": output_projection(slow_rows)["content"],
                "fastRows": fast_rows,
                "slowRows": slow_rows,
                "slowReplyStatus": slow_statuses,
            }

        def run_failure_recovery() -> dict[str, object]:
            # Web failure 通过 session.follow 暴露 Control.failure，随后同 Session 新 Input 可以恢复。
            with connect_websocket(
                websocket_url, open_timeout=READINESS_DEADLINE_S
            ) as recovery_web:
                recovery_web.send(json.dumps({
                    "type": "session.create",
                    "request_id": "pc16-recovery",
                }))
                recovery_session = str(
                    json.loads(recovery_web.recv(timeout=SCENARIO_DEADLINE_S))["session_id"]
                )
                recovery_web.send(json.dumps({
                    "type": "session.follow",
                    "version": 2,
                    "request_id": "pc16-recovery-follow",
                    "session_id": recovery_session,
                    "after_seq": -1,
                }))
                following = json.loads(recovery_web.recv(timeout=SCENARIO_DEADLINE_S))
                if (
                    following.get("type") != "session.following"
                    or following.get("session_id") != recovery_session
                ):
                    raise GateFailure(f"Web recovery follow ACK 非法：{following!r}")
                _http_json(
                    "PUT",
                    f"{model_url}/control/script",
                    {"mode": "error", "status": 500},
                )
                recovery_web.send(json.dumps({
                    "type": "message.send",
                    "request_id": "pc16-failure",
                    "session_id": recovery_session,
                    "text": "pc16 lane failure",
                    "media": [],
                }))
                failed_rows, failed_statuses = _receive_web_follow_rows(
                    recovery_web,
                    recovery_session,
                    lambda rows, _statuses: any(
                        isinstance(row.get("body"), dict)
                        and row["body"].get("kind") == "control"
                        and row["body"].get("action") == "failure"
                        for row in rows
                    ),
                )
                _http_json(
                    "PUT",
                    f"{model_url}/control/script",
                    {"mode": "complete", "content": "pc16 recovered"},
                )
                recovery_web.send(json.dumps({
                    "type": "message.send",
                    "request_id": "pc16-recovery-input",
                    "session_id": recovery_session,
                    "text": "pc16 lane recovery",
                    "media": [],
                }))
                recovered_rows, recovered_statuses = _receive_web_follow_rows(
                    recovery_web,
                    recovery_session,
                    lambda rows, _statuses: any(
                        isinstance(row.get("body"), dict)
                        and row["body"].get("kind") == "output"
                        and row["body"].get("finish") == "complete"
                        for row in rows
                    ),
                )
            recovery_passed = (
                any(
                    isinstance(row.get("body"), dict)
                    and row["body"].get("kind") == "input"
                    and row.get("id") == "pc16-failure"
                    for row in failed_rows
                )
                and any(
                    isinstance(row.get("body"), dict)
                    and row["body"].get("kind") == "control"
                    and row["body"].get("action") == "failure"
                    for row in failed_rows
                )
                and any(
                    isinstance(row.get("body"), dict)
                    and row["body"].get("kind") == "output"
                    and row["body"].get("finish") == "complete"
                    for row in recovered_rows
                )
                and _message_text(next(
                    row for row in recovered_rows
                    if isinstance(row.get("body"), dict)
                    and row["body"].get("kind") == "output"
                    and row["body"].get("finish") == "complete"
                )) == "pc16 recovered"
                and any(
                    isinstance(frame, dict)
                    and frame.get("type") == "reply.status"
                    and isinstance(frame.get("available"), bool)
                    for frame in (*failed_statuses, *recovered_statuses)
                )
            )
            return {
                "sessionId": recovery_session,
                "failedRows": failed_rows,
                "recoveredRows": recovered_rows,
                "failedReplyStatus": failed_statuses,
                "recoveredReplyStatus": recovered_statuses,
                "passed": recovery_passed,
            }


        lane_evidence["failureRecovery"] = run_failure_recovery()

        # 同 source 的四条 Input 全部先落日志；旧 provider 被取消后只提交一个新 Output。
        pc16_source = "programmatic:pc16-source-head"
        pc16_inputs = [
            (f"pc16-input-{index}", f"pc16 input {index}")
            for index in range(1, 5)
        ]
        first.admit_programmatic(pc16_source)
        pc16_old_barrier = "pc16-old-provider"
        pc16_shared_barrier = "pc16-shared-provider"
        _create_barrier(
            model_url,
            pc16_old_barrier,
            {"mode": "timeout"},
        )
        _http_json("PUT", f"{model_url}/control/barriers/{pc16_shared_barrier}")
        _http_json(
            "PUT",
            f"{model_url}/control/script",
            [
                {
                    "mode": "complete",
                    "content": "pc16 four input final",
                    "barrier": pc16_shared_barrier,
                }
                for _ in range(4)
            ],
        )
        first_ack = first.send_programmatic(pc16_source, *pc16_inputs[0])
        _wait_barrier(model_url, pc16_old_barrier)
        request_start = len(
            _model_requests(_http_json("GET", f"{model_url}/control/requests"))
        ) - 1
        input_acks = [first_ack]
        for message_id, text in pc16_inputs[1:]:
            input_acks.append(first.send_programmatic(pc16_source, message_id, text))
        before_release = first.read_messages(pc16_source)
        before_rows = [
            item for item in before_release.get("items", [])
            if isinstance(item, dict)
        ]
        input_rows = [
            item for item in before_rows
            if isinstance(item.get("body"), dict)
            and item["body"].get("kind") == "input"
        ]
        # 旧 provider 先独立进入 client_disconnected；共享 barrier 仍保持，
        # 因而后续请求只能在所有 Input 落日志后被观察和释放。
        _release_barrier(model_url, pc16_old_barrier)
        all_texts = [text for _, text in pc16_inputs]
        request_deadline = time.monotonic() + SCENARIO_DEADLINE_S
        old_request: dict[str, Any] | None = None
        new_request: dict[str, Any] | None = None
        while time.monotonic() < request_deadline:
            requests = _model_requests(
                _http_json("GET", f"{model_url}/control/requests")
            )
            for request in requests[max(0, request_start):]:
                payload = request.get("payload") if isinstance(request, dict) else None
                prompt = (
                    json.dumps(payload, ensure_ascii=False)
                    if isinstance(payload, dict)
                    else ""
                )
                if all(text in prompt for text in all_texts):
                    new_request = request
                elif all_texts[0] in prompt:
                    old_request = request
            if (
                new_request is not None
                and new_request.get("state") == "blocked"
                and old_request is not None
                and old_request.get("state") == "client_disconnected"
            ):
                break
            threading.Event().wait(0.05)
        pre_release_requests = {
            "old": old_request,
            "new": new_request,
            "barrier": _http_json(
                "GET", f"{model_url}/control/barriers/{pc16_shared_barrier}"
            ),
        }
        _release_barrier(model_url, pc16_shared_barrier)
        input_results = [
            _wait_programmatic_result(first, pc16_source, message_id)
            for message_id, _ in pc16_inputs
        ]
        post_release_requests = _model_requests(
            _http_json("GET", f"{model_url}/control/requests")
        )
        for request in post_release_requests[max(0, request_start):]:
            payload = request.get("payload") if isinstance(request, dict) else None
            prompt = (
                json.dumps(payload, ensure_ascii=False)
                if isinstance(payload, dict)
                else ""
            )
            if all(text in prompt for text in all_texts):
                new_request = request
            elif all_texts[0] in prompt:
                old_request = request
        source_page = first.read_messages(pc16_source)
        source_rows = [
            item for item in source_page.get("items", [])
            if isinstance(item, dict)
        ]
        source_outputs = [
            item for item in source_rows
            if isinstance(item.get("body"), dict)
            and item["body"].get("kind") == "output"
        ]
        final_output = source_outputs[0] if len(source_outputs) == 1 else None
        input_body_exact = all(
            item.get("id") == message_id
            and item.get("session_id") == pc16_source
            and item.get("source") == "programmatic"
            and item.get("seq") == index
            and isinstance(item.get("body"), dict)
            and item["body"].get("kind") == "input"
            and any(
                isinstance(part, dict)
                and part.get("kind") == "text"
                and part.get("value") == text
                for part in item["body"].get("parts", [])
            )
            for index, ((message_id, text), item) in enumerate(
                zip(pc16_inputs, input_rows)
            )
        )
        result_ids = {result.get("ending_message_id") for result in input_results}
        result_seqs = {result.get("ending_seq") for result in input_results}
        pre_release_passed = (
            isinstance(pre_release_requests.get("old"), dict)
            and isinstance(pre_release_requests.get("new"), dict)
            and pre_release_requests["old"].get("state") == "client_disconnected"
            and pre_release_requests["new"].get("state") == "blocked"
        )
        ending_refs_match = (
            isinstance(final_output, dict)
            and isinstance(final_output.get("id"), str)
            and final_output.get("seq") == 4
            and result_ids == {final_output["id"]}
            and result_seqs == {final_output["seq"]}
        )
        same_source_passed = (
            [ack.get("seq") for ack in input_acks] == [0, 1, 2, 3]
            and len(input_rows) == 4
            and input_body_exact
            and len(source_rows) == 5
            and len(source_outputs) == 1
            and isinstance(final_output, dict)
            and final_output.get("seq") == 4
            and final_output.get("source") == "programmatic"
            and final_output.get("body", {}).get("finish") == "complete"
            and _message_text(final_output) == "pc16 four input final"
            and all(result.get("status") == "complete" for result in input_results)
            and ending_refs_match
            and pre_release_passed
            and new_request is not None
            and old_request is not None
            and old_request.get("state") == "client_disconnected"
            and new_request.get("state") == "completed"
        )
        lane_evidence["sameSource"] = {
            "acks": input_acks,
            "inputsBeforeRelease": input_rows,
            "results": input_results,
            "messagePage": source_page,
            "oldProvider": old_request,
            "newProvider": new_request,
            "preReleaseRequests": pre_release_requests,
            "postReleaseRequests": post_release_requests[max(0, request_start):],
            "preReleasePassed": pre_release_passed,
            "endingReferencesMatch": ending_refs_match,
            "passed": same_source_passed,
        }
        different_sessions = cast(dict[str, object], lane_evidence["differentSessions"])
        same_source = cast(dict[str, object], lane_evidence["sameSource"])
        failure_recovery = cast(dict[str, object], lane_evidence["failureRecovery"])
        if input_results and source_outputs:
            pc06_first_result = first.programmatic_result(pc06, "pc06-first")
            pc07_current_result = first.programmatic_result(pc07, "pc07-input")
            pc09_results = [
                first.programmatic_result(pc09, f"pc09-input-{index}")
                for index in range(producer_count)
            ]
            raw_messages = _wait_message_log_rows(
                database,
                pc16_source,
                minimum=5,
                required_ids={message_id for message_id, _ in pc16_inputs}
                | {str(final_output["id"])},
            )

            def result_evidence(
                session_id: str,
                input_id: str,
                result: dict[str, Any],
                *,
                lifecycle: str,
                observations: list[dict[str, Any]] | None = None,
            ) -> dict[str, object]:
                status = result.get("status")
                settled = status in {"complete", "pause", "failure"}
                return {
                    "sessionId": session_id,
                    "inputId": input_id,
                    "status": status,
                    "endingMessageId": result.get("ending_message_id"),
                    "endingSeq": result.get("ending_seq"),
                    "lifecycle": lifecycle,
                    "settled": settled,
                    "result": result,
                    "observations": observations or [result],
                }

            programmatic_results = [
                result_evidence(pc05_a, "pc05-input-a", pc05_a_result, lifecycle="complete"),
                result_evidence(pc05_b, "pc05-input-b", pc05_b_result, lifecycle="complete"),
                result_evidence(
                    pc06, "pc06-first", pc06_first_result,
                    lifecycle="superseded_same_source",
                ),
                result_evidence(pc06, "pc06-second", pc06_result, lifecycle="complete"),
                result_evidence(
                    pc07, "pc07-input", pc07_current_result,
                    lifecycle="pause_then_resumed",
                    observations=[pc07_paused_result, pc07_current_result],
                ),
                result_evidence(pc07, "pc07-new-input", pc07_new_result, lifecycle="complete"),
                result_evidence(pc08, "pc08-input", resumed_result, lifecycle="complete"),
                *[
                    result_evidence(
                        pc09, f"pc09-input-{index}", result, lifecycle="complete"
                    )
                    for index, result in enumerate(pc09_results)
                ],
                result_evidence(
                    pc09_healthy, "pc09-healthy-input", pc09_healthy_result,
                    lifecycle="complete",
                ),
                result_evidence(pc10, "pc10-input", pc10_result, lifecycle="failure"),
                *[
                    result_evidence(
                        str(item["sessionId"]),
                        str(item["inputId"]),
                        cast(dict[str, Any], item["programmaticResult"]),
                        lifecycle=(
                            "complete"
                            if cast(dict[str, Any], item["programmaticResult"]).get("status")
                            == "complete"
                            else "failure"
                        ),
                    )
                    for item in parity_evidence
                    if isinstance(item.get("programmaticResult"), dict)
                ],
                *[
                    result_evidence(
                        pc16_source, message_id, result, lifecycle="complete"
                    )
                    for (message_id, _), result in zip(pc16_inputs, input_results)
                ],
            ]
            restart_state = {
                "sessionId": pc16_source,
                "inputId": pc16_inputs[-1][0],
                "inputText": pc16_inputs[-1][1],
                "inputSeq": 3,
                "endingMessageId": input_results[-1].get("ending_message_id"),
                "endingSeq": input_results[-1].get("ending_seq"),
                "rawMessages": raw_messages,
                "wireMessages": source_rows,
                "statusBefore": first.request("server/status", {}).get("result"),
                "providerRequestCount": len(
                    _model_requests(_http_json("GET", f"{model_url}/control/requests"))
                ),
                "programmaticResults": programmatic_results,
            }
            web_session_ids = {
                str(item["channelDatabase"]["sessionId"])
                for item in parity_evidence
                if isinstance(item.get("channelDatabase"), dict)
                and isinstance(item["channelDatabase"].get("sessionId"), str)
            }
            web_session_ids.update(
                str(different_sessions[key])
                for key in ("slowSessionId", "fastSessionId")
                if isinstance(different_sessions.get(key), str)
            )
            if isinstance(failure_recovery.get("sessionId"), str):
                web_session_ids.add(str(failure_recovery["sessionId"]))
            programmatic_session_ids = {
                str(item["sessionId"])
                for item in programmatic_results
                if isinstance(item, dict) and isinstance(item.get("sessionId"), str)
            }
            restart_state["relevantSessionIds"] = sorted(
                programmatic_session_ids | web_session_ids
            )

        if restart_state:
            restart_state["messageTerminalEvidence"] = {
                "parity": [
                    {
                        "status": item["channel"]["status"],
                        "settled": item["passed"],
                    }
                    for item in parity_evidence
                ],
                "differentSessions": {
                    "fast": different_sessions["fastCompletedBeforeRelease"]["status"]
                    == "complete",
                    "slow": different_sessions["slowFinal"] == "slow complete",
                },
                "sameSource": same_source.get("passed") is True,
                "failureRecovery": failure_recovery.get("passed") is True,
            }
        checks.append(
            CheckResult(
                "PC-16",
                parity_passed
                and same_source.get("passed") is True
                and failure_recovery.get("passed") is True
                and cast(
                    dict[str, object],
                    different_sessions["fastCompletedBeforeRelease"],
                )["status"] == "complete"
                and different_sessions["fastFinal"] == "fast complete"
                and different_sessions["slowFinal"] == "slow complete",
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
            "programmatic/message/send", {"session_id": "programmatic:pc11-invalid"}
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
    """重启后验证 Message 原文、结果 ACK 和 provider 请求均未改变。"""

    endpoint = Path("/sandbox/akashic.sock")
    _wait_socket(endpoint, READINESS_DEADLINE_S)
    client = _connect_client(endpoint, report_dir / "events.jsonl")
    try:
        state = json.loads(
            (report_dir / "restart-state.json").read_text(encoding="utf-8")
        )
        if not isinstance(state, dict):
            raise GateFailure("restart-state 不是 object")
        session_id = state.get("sessionId")
        input_id = state.get("inputId")
        input_text = state.get("inputText")
        input_seq = state.get("inputSeq")
        ending_id = state.get("endingMessageId")
        ending_seq = state.get("endingSeq")
        expected_messages = state.get("rawMessages")
        expected_wire_messages = state.get("wireMessages")
        before_status = state.get("statusBefore")
        phase = state.get("restartPhase")
        runtime = state.get("runtime")
        phase_runtime = runtime.get(phase) if isinstance(runtime, dict) else None
        if (
            not isinstance(session_id, str)
            or not isinstance(input_id, str)
            or not isinstance(input_text, str)
            or type(input_seq) is not int
            or not isinstance(ending_id, str)
            or type(ending_seq) is not int
            or not isinstance(expected_messages, list)
            or not isinstance(expected_wire_messages, list)
            or not isinstance(before_status, dict)
            or not isinstance(phase, str)
            or not isinstance(phase_runtime, dict)
        ):
            raise GateFailure(f"restart-state 字段不完整：{state!r}")
        status_response = client.request("server/status", {})
        current_status = status_response.get("result")
        if not isinstance(current_status, dict):
            raise GateFailure(f"重启后 server/status 非 object：{status_response!r}")
        ack = client.send_programmatic(session_id, input_id, input_text)
        result = client.programmatic_result(session_id, input_id)
        page = client.read_messages(session_id)
        actual_messages = _wait_message_log_rows(
            Path("/sandbox/workspace/sessions.db"),
            session_id,
            minimum=len(expected_messages),
            required_ids={str(row["id"]) for row in expected_messages if isinstance(row, dict)},
        )
        model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
        provider_request_count = len(
            _model_requests(_http_json("GET", f"{model_url}/control/requests"))
        )
    finally:
        client.close()
    runtime_before = phase_runtime.get("before")
    runtime_after = phase_runtime.get("after")
    boot_before = before_status.get("bootId")
    boot_after = current_status.get("bootId")
    runtime_changed = (
        isinstance(runtime_before, dict)
        and isinstance(runtime_after, dict)
        and isinstance(runtime_before.get("cmdline"), str)
        and runtime_before.get("cmdline") == runtime_after.get("cmdline")
        and type(runtime_before.get("starttime")) is int
        and type(runtime_after.get("starttime")) is int
        and type(runtime_before.get("pid")) is int
        and type(runtime_after.get("pid")) is int
        and (
            runtime_before.get("pid"), runtime_before.get("starttime")
        ) != (
            runtime_after.get("pid"), runtime_after.get("starttime")
        )
    )
    wire_messages = _message_wire_projection(page)
    wire_messages_before = _message_wire_projection({"items": expected_wire_messages})
    input_row = next(
        (
            row for row in expected_messages
            if isinstance(row, dict) and row.get("id") == input_id
        ),
        None,
    )
    passed = (
        current_status.get("ready") is True
        and current_status.get("protocolVersion") == PROTOCOL_VERSION
        and isinstance(boot_before, str)
        and isinstance(boot_after, str)
        and boot_before != boot_after
        and runtime_changed
        and ack.get("version") == 2
        and ack.get("session_id") == session_id
        and ack.get("message_id") == input_id
        and ack.get("seq") == input_seq
        and result.get("status") == "complete"
        and result.get("ending_message_id") == ending_id
        and result.get("ending_seq") == ending_seq
        and page.get("session_id") == session_id
        and page.get("through_seq") == ending_seq
        and actual_messages == expected_messages
        and wire_messages == wire_messages_before
        and provider_request_count == state.get("providerRequestCount")
        and isinstance(input_row, dict)
        and input_row.get("seq") == input_seq
        and any(
            isinstance(part, dict)
            and part.get("kind") == "text"
            and part.get("value") == input_text
            for part in (
                input_row.get("body", {}).get("parts", [])
                if isinstance(input_row.get("body"), dict)
                else []
            )
        )
    )
    evidence = {
        "phase": phase,
        "statusBefore": before_status,
        "statusAfter": current_status,
        "runtimeBefore": runtime_before,
        "runtimeAfter": runtime_after,
        "runtimeChanged": runtime_changed,
        "ack": ack,
        "result": result,
        "messagePage": page,
        "rawMessagesBefore": expected_messages,
        "rawMessagesAfter": actual_messages,
        "wireMessagesBefore": expected_wire_messages,
        "wireMessagesAfter": page.get("items"),
        "wireUnchanged": wire_messages == wire_messages_before,
        "messagesUnchanged": actual_messages == expected_messages,
        "providerRequestCountBefore": state.get("providerRequestCount"),
        "providerRequestCountAfter": provider_request_count,
    }
    result = CheckResult("PC-13", passed, evidence)
    _write_json(report_dir / "restart-check.json", asdict(result))
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if passed else 1


def _inside_soak(report_dir: Path) -> int:
    """执行 programmatic Message soak，并核对每个结果的日志引用。"""

    report_dir.mkdir(parents=True, exist_ok=True)
    endpoint = Path("/sandbox/akashic.sock")
    events_path = report_dir / "events.jsonl"
    model_url = os.environ.get("AKASHIC_MODEL_GATE_URL", "http://model-gate:8090")
    _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
    _configure_model_gate()
    _wait_socket(endpoint, READINESS_DEADLINE_S)
    client = _connect_client(endpoint, events_path)
    session_id = "programmatic:g5-soak"
    counts = {"complete": 0, "pause": 0, "failure": 0, "reconnects": 0}
    records: list[dict[str, object]] = []

    def wait_ack_barrier(path: Path) -> None:
        deadline = time.monotonic() + READINESS_DEADLINE_S
        while not path.exists():
            if time.monotonic() >= deadline:
                raise GateFailure(f"controller 未确认 soak milestone：{path.name}")
            threading.Event().wait(0.02)

    def run_message(
        index: int,
        *,
        phase: str,
        expected_status: str,
        mode: str = "complete",
    ) -> None:
        message_id = f"g5-{phase}-{index:03d}"
        text = f"soak {phase} {index}"
        barrier_name = f"g5-pause-{index}"
        if mode == "complete":
            _http_json(
                "PUT",
                f"{model_url}/control/script",
                {"mode": "complete", "content": f"reply {message_id}"},
            )
        elif mode == "failure":
            _http_json(
                "PUT",
                f"{model_url}/control/script",
                [
                    {"mode": "error", "status": 500},
                    {"mode": "error", "status": 500},
                ],
            )
        elif mode == "pause":
            _http_json("PUT", f"{model_url}/control/barriers/{barrier_name}")
            _http_json(
                "PUT",
                f"{model_url}/control/script",
                [
                    {"mode": "timeout", "barrier": barrier_name},
                ],
            )
        else:
            raise GateFailure(f"未知 soak mode：{mode}")

        ack = client.send_programmatic(session_id, message_id, text)
        if ack.get("message_id") != message_id or not isinstance(ack.get("seq"), int):
            raise GateFailure(f"soak Input ACK 异常：{ack!r}")
        if mode == "pause":
            _wait_barrier(model_url, barrier_name)
            _release_barrier(model_url, barrier_name)
            control_ack = client.request_result(
                "programmatic/message/pause",
                {
                    "session_id": session_id,
                    "message_id": f"g5-pause-control-{index:03d}",
                },
            )
            if control_ack.get("seq") != ack["seq"] + 1:
                raise GateFailure(f"soak pause Control ACK 序号异常：{control_ack!r}")
        result = _wait_programmatic_result(client, session_id, message_id)
        if result.get("status") != expected_status:
            raise GateFailure(
                f"soak {message_id} 结果状态异常：expected={expected_status!r} result={result!r}"
            )
        ending_seq = result.get("ending_seq")
        if not isinstance(ending_seq, int) or ending_seq <= int(ack["seq"]):
            raise GateFailure(f"soak {message_id} 结果缺少 terminal seq：{result!r}")
        records.append(
            {
                "index": index,
                "phase": phase,
                "messageId": message_id,
                "text": text,
                "ack": ack,
                "result": result,
                "expectedStatus": expected_status,
            }
        )
        counts[expected_status] += 1

    try:
        admission = client.admit_programmatic(session_id)
        if (
            admission.get("visibility") != "internal"
            or admission.get("learning") != "excluded"
        ):
            raise GateFailure(f"G5 programmatic Session 准入异常：{admission!r}")

        # 1. 预热完成后等待 controller 采集资源基线。
        for index in range(10):
            run_message(index, phase="warmup", expected_status="complete")
        _write_json(
            report_dir / "soak-progress.json",
            {"phase": "warmup", "completed": 10, "counts": counts},
        )
        start_barrier = report_dir / "soak-start"
        wait_ack_barrier(start_barrier)

        # 2. 100 个独立 Input：80 complete、10 pause、10 provider failure。
        for index in range(100):
            if index % 10 == 0:
                client.close()
                client = _connect_client(endpoint, events_path)
                counts["reconnects"] += 1
            if index < 10:
                run_message(index, phase="pause", expected_status="pause", mode="pause")
            elif index < 20:
                run_message(
                    index, phase="failure", expected_status="failure", mode="failure"
                )
            else:
                run_message(index, phase="complete", expected_status="complete")
            if (index + 1) % 10 == 0:
                milestone = index + 1
                _write_json(
                    report_dir / "soak-progress.json",
                    {
                        "phase": "run",
                        "completed": milestone,
                        "counts": counts,
                    },
                )
                wait_ack_barrier(report_dir / f"soak-ack-{milestone}")
    finally:
        client.close()

    expected = {
        "complete": 90,
        "pause": 10,
        "failure": 10,
        "reconnects": 10,
    }
    unique_inputs = len({str(item["messageId"]) for item in records})
    passed = counts == expected and unique_inputs == 110 and len(records) == 110
    result = CheckResult(
        "G5-turns",
        passed,
        {
            "counts": counts,
            "uniqueInputs": unique_inputs,
            "records": len(records),
            "expected": expected,
        },
    )
    # Controller 采样期间已关闭原连接；这里明确建立独立只读核验连接。
    ledger_client = _connect_client(endpoint, events_path)
    try:
        pages: list[dict[str, Any]] = []
        after_seq = -1
        while True:
            current_page = ledger_client.read_messages(
                session_id, after_seq=after_seq, limit=200
            )
            pages.append(current_page)
            if current_page.get("has_more") is not True:
                break
            next_after = current_page.get("next_after_seq")
            if not isinstance(next_after, int) or next_after <= after_seq:
                raise GateFailure(f"G5 Message 分页游标无进展：{current_page!r}")
            after_seq = next_after
    finally:
        ledger_client.close()
    rows = [
        item
        for page in pages
        for item in page.get("items", [])
        if isinstance(page, dict) and isinstance(page.get("items"), list)
    ]
    ledger_passed = isinstance(rows, list) and len(rows) == 220
    ledger_evidence: dict[str, object] = {
        "sessionId": session_id,
        "messageCount": len(rows) if isinstance(rows, list) else None,
        "records": records,
    }
    if ledger_passed:
        by_id = {str(item.get("id")): item for item in rows if isinstance(item, dict)}
        ledger_passed = len(by_id) == 220 and [
            item.get("seq") for item in rows if isinstance(item, dict)
        ] == list(range(220))
        for record in records:
            message_id = str(record["messageId"])
            ack = cast(dict[str, object], record["ack"])
            result_payload = cast(dict[str, object], record["result"])
            input_row = by_id.get(message_id)
            ending_id = result_payload.get("ending_message_id")
            ending_row = by_id.get(str(ending_id)) if ending_id is not None else None
            if not isinstance(input_row, dict) or not isinstance(ending_row, dict):
                ledger_passed = False
                break
            body = input_row.get("body")
            ending_body = ending_row.get("body")
            ledger_passed = ledger_passed and (
                input_row.get("seq") == ack.get("seq")
                and _message_text(input_row) == str(record["text"])
                and input_row.get("author") == "user"
                and input_row.get("source") == "programmatic"
                and isinstance(body, dict)
                and body.get("kind") == "input"
                and ending_row.get("seq") == result_payload.get("ending_seq")
                and ending_row.get("source") == "programmatic"
                and (
                    (
                        record["expectedStatus"] == "complete"
                        and ending_row.get("author") == "assistant"
                        and isinstance(ending_body, dict)
                        and ending_body.get("kind") == "output"
                        and ending_body.get("finish") == "complete"
                    )
                    or (
                        record["expectedStatus"] in {"pause", "failure"}
                        and ending_row.get("author") == "app"
                        and isinstance(ending_body, dict)
                        and ending_body.get("kind") == "control"
                        and ending_body.get("action") == record["expectedStatus"]
                    )
                )
            )
    from datetime import datetime

    from plugins.turn_projection.plugin import TurnProjection
    from session.message import Message
    from session.message_codec import decode_body

    database = Path("/sandbox/workspace/sessions.db")
    raw_rows: list[sqlite3.Row] = []
    if not database.exists():
        raise GateFailure(f"G5 raw Message ledger 不存在：{database}")
    try:
        with closing(
            sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        ) as connection:
            connection.row_factory = sqlite3.Row
            raw_rows = connection.execute(
                "SELECT id, session_key, seq, ts, author, source, body "
                "FROM messages WHERE session_key = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
    except sqlite3.Error as error:
        raise GateFailure(f"G5 raw Message ledger 只读读取失败：{error}") from error
    try:
        projected_messages = [
            Message(
                message_id=str(row["id"]),
                session_id=str(row["session_key"]),
                seq=int(row["seq"]),
                recorded_at=datetime.fromisoformat(str(row["ts"])),
                author=str(row["author"]),
                source=str(row["source"]),
                body=decode_body(str(row["body"])),
            )
            for row in raw_rows
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise GateFailure(
            f"G5 raw Message 无法还原为 TurnProjection 输入：{error}"
        ) from error
    raw_ids = [str(row["id"]) for row in raw_rows]
    raw_seqs = [int(row["seq"]) for row in raw_rows]
    wire_ids = [str(item["id"]) for item in rows if isinstance(item, dict)]
    ledger_evidence["rawDatabase"] = {
        "path": str(database),
        "messageCount": len(raw_rows),
        "seqs": raw_seqs,
        "idsDigest": hashlib.sha256("\n".join(raw_ids).encode()).hexdigest(),
    }
    ledger_evidence["wireIdsMatchRawIds"] = wire_ids == raw_ids
    ledger_passed = (
        ledger_passed
        and len(raw_rows) == 220
        and raw_seqs == list(range(220))
        and wire_ids == raw_ids
    )
    projected = TurnProjection().project(projected_messages, "programmatic")
    projection_open = [
        {
            "afterSeq": turn.after_seq,
            "throughSeq": turn.through_seq,
            "messageIds": list(turn.message_ids),
            "status": turn.status,
        }
        for turn in projected
        if turn.status == "open"
    ]
    ledger_evidence["projectionOpenTurns"] = projection_open
    ledger_evidence["activeOpenTurns"] = projection_open
    ledger_evidence["turnProjection"] = {
        "turnCount": len(projected),
        "activeOpenTurns": len(projection_open),
    }
    ledger_passed = ledger_passed and not projection_open
    ledger_evidence["allTerminalReferences"] = ledger_passed
    ledger_result = CheckResult("G5-ledger", ledger_passed, ledger_evidence)
    _write_json(report_dir / "soak-ledger.json", ledger_evidence)
    _write_json(
        report_dir / "inside-gate.json",
        {
            "gate": "soak",
            "status": "passed" if passed and ledger_passed else "failed",
            "checks": [asdict(result), asdict(ledger_result)],
        },
    )
    print(
        json.dumps(
            {"turns": asdict(result), "ledger": asdict(ledger_result)},
            ensure_ascii=False,
        )
    )
    return 0 if passed and ledger_passed else 1


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
    """读取控制面 SQLite 终态，并明确报告必需表是否存在。"""

    if not database.exists():
        return {
            "exists": False,
            "path": str(database),
            "tableNames": [],
            "missingRequiredTables": ["sessions", "messages"],
            "tables": {},
        }
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        table_names = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        tables: dict[str, list[dict[str, object]]] = {}
        for name in ("sessions", "messages", "turns", "operations"):
            if name not in table_names:
                continue
            order = " ORDER BY session_key, seq" if name == "messages" else ""
            rows = connection.execute(f'SELECT * FROM "{name}"{order}').fetchall()
            tables[name] = [dict(row) for row in rows]
    required_tables = ("sessions", "messages")
    return {
        "exists": True,
        "path": str(database),
        "tableNames": sorted(table_names),
        "missingRequiredTables": [
            name for name in required_tables if name not in table_names
        ],
        "tables": tables,
    }


def _snapshot_message_rows(
    snapshot: dict[str, object], session_ids: Sequence[str]
) -> tuple[dict[str, list[dict[str, object]]], list[str]]:
    """从数据库快照解码指定 Session 的完整 Message 行。"""

    wanted = list(dict.fromkeys(session_ids))
    rows_by_session = {session_id: [] for session_id in wanted}
    errors: list[str] = []
    tables = snapshot.get("tables")
    messages = tables.get("messages") if isinstance(tables, dict) else None
    if not isinstance(messages, list):
        return rows_by_session, ["messages table snapshot missing"]
    for index, raw_row in enumerate(messages):
        if not isinstance(raw_row, dict):
            errors.append(f"messages[{index}] is not an object")
            continue
        session_id = raw_row.get("session_key")
        if session_id not in rows_by_session:
            continue
        required = ("id", "session_key", "seq", "ts", "author", "source", "body")
        missing = [name for name in required if name not in raw_row]
        if missing:
            errors.append(
                f"messages[{index}] missing columns for {session_id}: {missing}"
            )
            continue
        raw_body = raw_row["body"]
        if not isinstance(raw_body, str):
            errors.append(f"messages[{index}] body is not text for {session_id}")
            continue
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as error:
            errors.append(f"messages[{index}] invalid body for {session_id}: {error}")
            continue
        rows_by_session[str(session_id)].append(
            {
                "id": raw_row["id"],
                "session_id": raw_row["session_key"],
                "seq": raw_row["seq"],
                "timestamp": raw_row["ts"],
                "author": raw_row["author"],
                "source": raw_row["source"],
                "body": body,
            }
        )
    return rows_by_session, errors


def _runtime_identity(
    compose: list[str], repo: Path, env: dict[str, str]
) -> dict[str, object]:
    """读取进程树中真实 gateway 的 PID、启动时间和命令行。"""

    deadline = time.monotonic() + READINESS_DEADLINE_S
    while True:
        try:
            sample = _sample_resources(compose, repo, env, milestone=-1)
            break
        except GateFailure as error:
            if time.monotonic() >= deadline:
                raise
            threading.Event().wait(0.05)
    try:
        pid = int(sample["gatewayPid"])
        starttime = int(str(sample["gatewayStarttime"]))
        cmdline = str(sample["gatewayCmdline"])
    except (KeyError, TypeError, ValueError) as error:
        raise GateFailure(f"gateway runtime identity 字段非法：{sample!r}") from error
    return {"pid": pid, "starttime": starttime, "cmdline": cmdline}


def _gateway_status(
    compose: list[str], repo: Path, env: dict[str, str]
) -> dict[str, Any]:
    """通过 gateway 自己的 UDS 证明现有 owner 仍然 ready。"""

    script = (
        "import json,socket; "
        "connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); "
        "connection.connect('/sandbox/akashic.sock'); "
        "reader=connection.makefile('rb'); "
        "connection.sendall((json.dumps({'jsonrpc':'2.0','id':1,'method':'initialize',"
        "'params':{'protocolVersion':'2.0','clientInfo':{'name':'workspace-lock-check',"
        "'version':'2.0'},'capabilities':{}}})+'\\n').encode()); "
        "reader.readline(); "
        "connection.sendall((json.dumps({'jsonrpc':'2.0','method':'initialized',"
        "'params':{}})+'\\n').encode()); "
        "connection.sendall((json.dumps({'jsonrpc':'2.0','id':2,'method':'server/status',"
        "'params':{}})+'\\n').encode()); "
        "print(reader.readline().decode().strip())"
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
        return {
            "ready": False,
            "error": completed.stderr[-1000:],
            "returncode": completed.returncode,
        }
    try:
        frame = json.loads(completed.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        return {
            "ready": False,
            "error": f"status output invalid: {error}",
            "stdout": completed.stdout[-1000:],
        }
    result = frame.get("result") if isinstance(frame, dict) else None
    return result if isinstance(result, dict) else {"ready": False, "frame": frame}


def _socket_unavailable(endpoint: Path) -> bool:
    """证明停止后共享 UDS 没有可接受连接的进程。"""

    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        return not endpoint.exists() or probe.connect_ex(str(endpoint)) != 0
    finally:
        probe.close()


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
    stderr = completed.stderr.decode(errors="replace")
    (report_dir / "workspace-lock.stderr.log").write_text(stderr, encoding="utf-8")
    gateway_status = _gateway_status(compose, repo, env)
    lock_reason = "workspace 已由其他 runtime 占用"
    return CheckResult(
        "PC-14",
        completed.returncode != 0
        and lock_reason in stderr
        and gateway_status.get("ready") is True,
        {
            "returncode": completed.returncode,
            "lockReason": lock_reason,
            "reasonMatched": lock_reason in stderr,
            "stderrTail": stderr[-2000:],
            "gatewayStatus": gateway_status,
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
) -> dict[str, object]:
    """从进程树中定位真实 gateway，并读取其 RSS、fd 和线程数。"""

    script = """
import json
import pathlib

def read(pid, name):
    return pathlib.Path('/proc', str(pid), name).read_text()

def children(pid):
    return [int(value) for value in read(pid, 'task/' + str(pid) + '/children').split()]

roots = [1, *children(1)]
candidates = []
seen = set()
stack = list(roots)
while stack:
    pid = stack.pop()
    if pid in seen:
        continue
    seen.add(pid)
    cmd = read(pid, 'cmdline').replace(chr(0), ' ').strip()
    if '/main.py' in cmd and ' gateway' in (' ' + cmd):
        candidates.append((pid, cmd))
    stack.extend(children(pid))
if len(candidates) != 1:
    raise RuntimeError('gateway identity ambiguous: roots=%r candidates=%r' % (roots, candidates))
pid, cmd = candidates[0]
status = read(pid, 'status')
rss = next(int(line.split()[1]) for line in status.splitlines() if line.startswith('VmRSS:'))
stat = read(pid, 'stat')
starttime = stat.rsplit(')', 1)[1].split()[19]
print(json.dumps({'pid': pid, 'cmdline': cmd, 'rssKiB': rss,
                  'starttime': starttime,
                  'fdCount': len(list(pathlib.Path('/proc', str(pid), 'fd').iterdir())),
                  'threadCount': len(list(pathlib.Path('/proc', str(pid), 'task').iterdir()))}))
"""
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
        "gatewayPid": int(payload["pid"]),
        "gatewayCmdline": str(payload["cmdline"]),
        "gatewayStarttime": str(payload["starttime"]),
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
    """并行采样 Message soak 资源，并执行公开增量阈值。"""

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
    samples: list[dict[str, object]] = []
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
                    else:
                        (sandbox / f"reports/soak-ack-{milestone}").touch()
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
    gateway_identities = [
        (int(sample["gatewayPid"]), str(sample["gatewayStarttime"]))
        for sample in samples
    ]
    ledger = json.loads(
        (sandbox / "reports/soak-ledger.json").read_text(encoding="utf-8")
    )
    all_terminal = ledger.get("allTerminalReferences") is True
    checks.append(
        CheckResult(
            "G5-resources",
            rss_delta <= 64 * 1024
            and fd_delta <= 8
            and thread_delta <= 3
            and all_terminal
            and len(set(gateway_identities)) == 1,
            {
                "samples": len(samples),
                "rssDeltaKiB": rss_delta,
                "fdDelta": fd_delta,
                "threadDelta": thread_delta,
                "gatewayIdentities": gateway_identities,
                "gatewayCmdline": baseline["gatewayCmdline"],
                "gatewayStarttime": baseline["gatewayStarttime"],
                "allTerminalReferences": all_terminal,
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
        max_iterations=4 if gate == "failure-matrix" else 2,
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
        restart_state_path = sandbox / "reports/restart-state.json"
        if gate == "failure-matrix":
            restart_state = json.loads(restart_state_path.read_text(encoding="utf-8"))
            restart_state["restartPhase"] = "graceful"
            restart_state["runtime"] = {
                "graceful": {
                    "before": _runtime_identity(compose, repo, env),
                }
            }
            relevant_session_ids = restart_state.get("relevantSessionIds")
            if not isinstance(relevant_session_ids, list) or not all(
                isinstance(session_id, str) and session_id
                for session_id in relevant_session_ids
            ):
                raise GateFailure(
                    f"restart-state 缺少 relevantSessionIds：{relevant_session_ids!r}"
                )
            pre_stop_snapshot = _snapshot_database(
                sandbox / "workspace/sessions.db"
            )
            pre_stop_messages, pre_stop_message_errors = _snapshot_message_rows(
                pre_stop_snapshot,
                cast(list[str], relevant_session_ids),
            )
            restart_state["preStopMessages"] = pre_stop_messages
            restart_state["preStopMessageErrors"] = pre_stop_message_errors
            restart_state["preStopMissingRequiredTables"] = pre_stop_snapshot.get(
                "missingRequiredTables"
            )
            _write_json(restart_state_path, restart_state)
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
            restart_state = json.loads(restart_state_path.read_text(encoding="utf-8"))
            programmatic_results = restart_state.get("programmaticResults")
            terminal_evidence = restart_state.get("messageTerminalEvidence")
            relevant_session_ids = restart_state.get("relevantSessionIds")
            pre_stop_messages = restart_state.get("preStopMessages")
            pre_stop_message_errors = restart_state.get("preStopMessageErrors")
            pre_stop_missing_tables = restart_state.get("preStopMissingRequiredTables")
            missing_tables = stopped_snapshot.get("missingRequiredTables")
            post_stop_messages, post_stop_message_errors = _snapshot_message_rows(
                stopped_snapshot,
                cast(list[str], relevant_session_ids)
                if isinstance(relevant_session_ids, list)
                else [],
            )
            session_message_exact_match = (
                isinstance(relevant_session_ids, list)
                and bool(relevant_session_ids)
                and all(isinstance(session_id, str) for session_id in relevant_session_ids)
                and isinstance(pre_stop_messages, dict)
                and pre_stop_missing_tables == []
                and not pre_stop_message_errors
                and not post_stop_message_errors
                and pre_stop_messages == post_stop_messages
            )
            open_results = (
                [
                    item
                    for item in programmatic_results
                    if isinstance(item, dict) and item.get("settled") is not True
                ]
                if isinstance(programmatic_results, list)
                else ["missing"]
            )
            different_sessions = (
                terminal_evidence.get("differentSessions")
                if isinstance(terminal_evidence, dict)
                else None
            )
            no_running_work = (
                isinstance(programmatic_results, list)
                and bool(programmatic_results)
                and not open_results
                and isinstance(terminal_evidence, dict)
                and isinstance(different_sessions, dict)
                and all(value is True for value in different_sessions.values())
                and terminal_evidence.get("sameSource") is True
                and terminal_evidence.get("failureRecovery") is True
            )
            socket_unavailable = _socket_unavailable(sandbox / "akashic.sock")
            checks.append(
                CheckResult(
                    "PC-12",
                    stop_duration <= 15
                    and stopped_snapshot.get("exists") is True
                    and missing_tables == []
                    and no_running_work
                    and session_message_exact_match
                    and socket_unavailable,
                    {
                        "durationSeconds": stop_duration,
                        "database": {
                            "exists": stopped_snapshot.get("exists"),
                            "tableNames": stopped_snapshot.get("tableNames"),
                            "missingRequiredTables": missing_tables,
                            "messageCount": len(
                                stopped_snapshot.get("tables", {}).get("messages", [])
                            )
                            if isinstance(stopped_snapshot.get("tables"), dict)
                            else None,
                        },
                        "sessionMessageParity": {
                            "relevantSessionIds": relevant_session_ids,
                            "preStopMessages": pre_stop_messages,
                            "preStopMissingRequiredTables": pre_stop_missing_tables,
                            "postStopMessages": post_stop_messages,
                            "preStopErrors": pre_stop_message_errors,
                            "postStopErrors": post_stop_message_errors,
                            "exactMatch": session_message_exact_match,
                        },
                        "programmaticResults": programmatic_results,
                        "openResults": open_results,
                        "messageTerminalEvidence": terminal_evidence,
                        "socketUnavailable": socket_unavailable,
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
            restart_state = json.loads(restart_state_path.read_text(encoding="utf-8"))
            restart_state["runtime"]["graceful"]["after"] = _runtime_identity(
                compose, repo, env
            )
            _write_json(restart_state_path, restart_state)
            restart_check = _run_inside(
                compose,
                repo,
                env,
                gate=gate,
                phase="restart-check",
            )
            restart_payload = sandbox / "reports/restart-check.json"
            if restart_payload.exists():
                shutil.copy2(restart_payload, report_dir / "restart-check-graceful.json")
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
            if restart_after_crash.returncode != 0:
                raise GateFailure(
                    f"gateway crash restart failed: {restart_after_crash.returncode}"
                )
            restart_state = json.loads(restart_state_path.read_text(encoding="utf-8"))
            restart_state["restartPhase"] = "crash"
            restart_state["runtime"]["crash"] = {
                "before": restart_state["runtime"]["graceful"]["after"],
                "after": _runtime_identity(compose, repo, env),
            }
            _write_json(restart_state_path, restart_state)
            crash_check = _run_inside(
                compose,
                repo,
                env,
                gate=gate,
                phase="restart-check",
            )
            if restart_payload.exists():
                shutil.copy2(restart_payload, report_dir / "restart-check-crash.json")
            crash_payload = (
                json.loads(restart_payload.read_text())
                if restart_payload.exists()
                else None
            )
            checks.append(
                CheckResult(
                    "PC-13-crash",
                    crash.returncode == 0
                    and restart_after_crash.returncode == 0
                    and crash_check.returncode == 0
                    and isinstance(crash_payload, dict)
                    and crash_payload.get("passed") is True,
                    {
                        "killReturncode": crash.returncode,
                        "restartReturncode": restart_after_crash.returncode,
                        "probeReturncode": crash_check.returncode,
                        "restartCheck": (
                            crash_payload
                        ),
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
