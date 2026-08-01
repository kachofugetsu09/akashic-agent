#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import select
import signal
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

DEFAULT_USAGE = {
    "prompt_tokens": 7,
    "completion_tokens": 3,
    "total_tokens": 10,
}


@dataclass
class Barrier:
    reached: threading.Event = field(default_factory=threading.Event)
    released: threading.Event = field(default_factory=threading.Event)


class ModelGateState:
    """保存脚本、barrier 和模型请求证据。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scripts: deque[dict[str, Any]] = deque()
        self._barriers: dict[str, Barrier] = {}
        self._requests: list[dict[str, Any]] = []

    def load_scripts(self, payload: object) -> int:
        """在控制边界校验并追加一个或多个响应脚本。"""

        # 1. 规范化控制请求
        raw_scripts = payload if isinstance(payload, list) else [payload]
        if not raw_scripts or not all(isinstance(item, dict) for item in raw_scripts):
            raise ValueError("script 必须是对象或非空对象数组")

        # 2. 一次性校验并提交，避免部分装载
        scripts = [
            self._validate_script(cast(dict[str, Any], item)) for item in raw_scripts
        ]
        with self._lock:
            self._scripts.extend(scripts)
        return len(scripts)

    def create_barrier(self, name: str) -> None:
        if not name:
            raise ValueError("barrier 名称不能为空")
        with self._lock:
            if name in self._barriers:
                raise ValueError(f"barrier 已存在：{name}")
            self._barriers[name] = Barrier()

    def release_barrier(self, name: str) -> bool:
        barrier = self._barrier(name)
        was_reached = barrier.reached.is_set()
        barrier.released.set()
        return was_reached

    def wait_barrier(self, name: str, timeout: float) -> bool:
        return self._barrier(name).reached.wait(timeout)

    def barrier_status(self, name: str) -> dict[str, bool]:
        barrier = self._barrier(name)
        return {
            "reached": barrier.reached.is_set(),
            "released": barrier.released.is_set(),
        }

    def begin_request(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], int]:
        """消费下一个脚本并记录请求关联证据。"""

        # 1. 原子消费脚本并登记请求
        with self._lock:
            if not self._scripts:
                raise LookupError("没有为本次模型请求装载响应脚本")
            script = self._scripts.popleft()
            request_index = len(self._requests) + 1
            record = {
                "index": request_index,
                "receivedAt": time.time(),
                "model": payload.get("model"),
                "stream": bool(payload.get("stream")),
                "metadata": payload.get("metadata"),
                "headers": {
                    key: value
                    for key, value in headers.items()
                    if key.startswith("x-akashic-") or key == "x-request-id"
                },
                "payload": payload,
                "script": script,
                "state": "received",
            }
            self._requests.append(record)

        # 2. barrier 精确标记 provider 已进入，再等待 controller 释放
        barrier_name = script.get("barrier")
        if isinstance(barrier_name, str):
            barrier = self._barrier(barrier_name)
            self._set_request_state(request_index, "blocked")
            barrier.reached.set()
            barrier.released.wait()
        self._set_request_state(request_index, "responding")
        return script, request_index

    def finish_request(self, request_index: int, state: str) -> None:
        self._set_request_state(request_index, state)

    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self._requests))

    def _barrier(self, name: str) -> Barrier:
        with self._lock:
            barrier = self._barriers.get(name)
        if barrier is None:
            raise KeyError(f"barrier 不存在：{name}")
        return barrier

    def _set_request_state(self, request_index: int, state: str) -> None:
        with self._lock:
            self._requests[request_index - 1]["state"] = state
            self._requests[request_index - 1]["updatedAt"] = time.time()

    @staticmethod
    def _validate_script(script: dict[str, Any]) -> dict[str, Any]:
        mode = script.get("mode", "complete")
        if mode not in {"complete", "stream", "error", "timeout", "truncate"}:
            raise ValueError(f"不支持的 script mode：{mode!r}")
        delay_ms = script.get("delay_ms", 0)
        if (
            isinstance(delay_ms, bool)
            or not isinstance(delay_ms, int)
            or not 0 <= delay_ms <= 5_000
        ):
            raise ValueError("script.delay_ms 必须是 0..5000 的整数")
        if "barrier" in script and not isinstance(script["barrier"], str):
            raise ValueError("script.barrier 必须是字符串")
        if mode == "error":
            status = script.get("status")
            if not isinstance(status, int) or status < 400 or status > 599:
                raise ValueError("error script.status 必须是 400..599")
        if mode in {"stream", "truncate"}:
            deltas = script.get("deltas", [])
            if not isinstance(deltas, list) or not all(
                isinstance(delta, (str, dict)) for delta in deltas
            ):
                raise ValueError("stream script.deltas 必须是字符串或对象数组")
        tool_calls = script.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list) or not all(
                isinstance(call, dict)
                and isinstance(call.get("name"), str)
                and bool(call["name"])
                for call in tool_calls
            ):
                raise ValueError("script.tool_calls 必须是包含 name 的对象数组")
        return json.loads(json.dumps(script))


class ModelGateServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: ModelGateState) -> None:
        super().__init__(address, ModelGateHandler)
        self.state = state


class ModelGateHandler(BaseHTTPRequestHandler):
    server: ModelGateServer
    protocol_version = "HTTP/1.1"

    def handle_one_request(self) -> None:
        """把控制边界的输入错误转换为明确 HTTP 4xx。"""

        try:
            super().handle_one_request()
        except (json.JSONDecodeError, KeyError, ValueError) as error:
            self.close_connection = True
            self._json_response(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "message": str(error)},
            )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/readyz":
            self._json_response(HTTPStatus.OK, {"status": "ready"})
            return
        if parsed.path == "/control/requests":
            self._json_response(
                HTTPStatus.OK,
                {"requests": self.server.state.requests()},
            )
            return
        barrier_name = self._barrier_name(parsed.path, suffix="/wait")
        if barrier_name is not None:
            timeout = self._wait_timeout(parsed.query)
            reached = self.server.state.wait_barrier(barrier_name, timeout)
            status = self.server.state.barrier_status(barrier_name)
            self._json_response(
                HTTPStatus.OK if reached else HTTPStatus.REQUEST_TIMEOUT,
                {"name": barrier_name, **status},
            )
            return
        barrier_name = self._barrier_name(parsed.path)
        if barrier_name is not None:
            self._json_response(
                HTTPStatus.OK,
                {
                    "name": barrier_name,
                    **self.server.state.barrier_status(barrier_name),
                },
            )
            return
        self._json_response(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/control/script":
            count = self.server.state.load_scripts(self._read_json())
            self._json_response(HTTPStatus.OK, {"loaded": count})
            return
        barrier_name = self._barrier_name(parsed.path)
        if barrier_name is not None:
            self.server.state.create_barrier(barrier_name)
            self._json_response(HTTPStatus.CREATED, {"name": barrier_name})
            return
        self._json_response(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        barrier_name = self._barrier_name(parsed.path, suffix="/release")
        if barrier_name is not None:
            reached = self.server.state.release_barrier(barrier_name)
            self._json_response(
                HTTPStatus.OK,
                {"name": barrier_name, "reached": reached, "released": True},
            )
            return
        if parsed.path == "/v1/chat/completions":
            self._chat_completion()
            return
        self._json_response(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _chat_completion(self) -> None:
        """按脚本生成 OpenAI-compatible completion 或 SSE 流。"""

        # 1. 在 HTTP 边界校验 payload，并进入可观察 barrier
        payload = self._read_json()
        if not isinstance(payload, dict):
            raise ValueError("模型请求必须是 JSON 对象")
        headers = {key.lower(): value for key, value in self.headers.items()}
        script, request_index = self.server.state.begin_request(payload, headers)

        # 2. 返回脚本化错误、完整响应或流
        mode = script.get("mode", "complete")
        try:
            if mode == "timeout":
                self.server.state.finish_request(
                    request_index, "waiting_for_client_timeout"
                )
                self._wait_for_client_disconnect()
                self.server.state.finish_request(request_index, "client_disconnected")
                return
            if mode == "error":
                status = cast(int, script["status"])
                body = script.get(
                    "body",
                    {
                        "error": {
                            "message": f"scripted model-gate HTTP {status}",
                            "type": "model_gate_error",
                        }
                    },
                )
                self._json_response(status, body)
                self.server.state.finish_request(request_index, "error_sent")
                return
            if mode == "complete":
                if payload.get("stream") is True:
                    stream_script = dict(script)
                    content = stream_script.pop("content", "model-gate response")
                    stream_script["deltas"] = [content] if content else []
                    self._stream_response(payload, stream_script, truncated=False)
                else:
                    self._json_response(
                        HTTPStatus.OK,
                        self._completion_payload(payload, script),
                    )
                self.server.state.finish_request(request_index, "completed")
                return
            self._stream_response(payload, script, truncated=mode == "truncate")
            self.server.state.finish_request(
                request_index,
                "truncated" if mode == "truncate" else "completed",
            )
        except (BrokenPipeError, ConnectionResetError):
            self.server.state.finish_request(request_index, "client_disconnected")

    def _completion_payload(
        self,
        request: dict[str, Any],
        script: dict[str, Any],
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": script.get("content", "model-gate response"),
        }
        tool_calls = script.get("tool_calls")
        if tool_calls is not None:
            message["tool_calls"] = self._tool_calls(tool_calls)
        return {
            "id": f"chatcmpl-gate-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "model-gate"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": script.get("usage", DEFAULT_USAGE),
        }

    def _stream_response(
        self,
        request: dict[str, Any],
        script: dict[str, Any],
        *,
        truncated: bool,
    ) -> None:
        completion_id = f"chatcmpl-gate-{uuid.uuid4().hex}"
        delay_seconds = cast(int, script.get("delay_ms", 0)) / 1_000
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for raw_delta in script.get("deltas", []):
            delta = {"content": raw_delta} if isinstance(raw_delta, str) else raw_delta
            self._write_sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.get("model", "model-gate"),
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            )
            if delay_seconds:
                time.sleep(delay_seconds)
        if truncated:
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        finish_reason = "tool_calls" if script.get("tool_calls") else "stop"
        if script.get("tool_calls"):
            self._write_sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.get("model", "model-gate"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": self._stream_tool_calls(
                                    script["tool_calls"]
                                )
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
        self._write_sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.get("model", "model-gate"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
        )
        self._write_sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.get("model", "model-gate"),
                "choices": [],
                "usage": script.get("usage", DEFAULT_USAGE),
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    @staticmethod
    def _tool_calls(raw_calls: object) -> list[dict[str, Any]]:
        if not isinstance(raw_calls, list):
            raise ValueError("tool_calls 必须是数组")
        calls: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, dict):
                raise ValueError("tool_calls 元素必须是对象")
            arguments = raw.get("arguments", {})
            calls.append(
                {
                    "id": raw.get("id", f"call_gate_{index}"),
                    "type": "function",
                    "function": {
                        "name": raw["name"],
                        "arguments": (
                            arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments, ensure_ascii=False)
                        ),
                    },
                }
            )
        return calls

    @classmethod
    def _stream_tool_calls(cls, raw_calls: object) -> list[dict[str, Any]]:
        return [
            {"index": index, **call}
            for index, call in enumerate(cls._tool_calls(raw_calls))
        ]

    def _read_json(self) -> object:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("请求缺少 Content-Length")
        length = int(content_length)
        if length > 16 * 1024 * 1024:
            raise ValueError("请求体超过 16 MiB")
        return json.loads(self.rfile.read(length))

    def _json_response(self, status: int, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _write_sse(self, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()

    def _wait_for_client_disconnect(self) -> None:
        """不发送响应，并由 socket 关闭事件判定 provider 客户端已取消。"""

        while True:
            readable, _, _ = select.select([self.connection], [], [])
            if not readable:
                continue
            data = self.connection.recv(1, socket.MSG_PEEK)
            if not data:
                return
            _ = self.connection.recv(len(data))

    @staticmethod
    def _barrier_name(path: str, *, suffix: str = "") -> str | None:
        prefix = "/control/barriers/"
        if not path.startswith(prefix):
            return None
        remainder = path[len(prefix) :]
        if suffix:
            if not remainder.endswith(suffix):
                return None
            remainder = remainder[: -len(suffix)]
        elif "/" in remainder:
            return None
        return remainder or None

    @staticmethod
    def _wait_timeout(query: str) -> float:
        values = parse_qs(query).get("timeout", ["30"])
        timeout = float(values[0])
        if timeout <= 0 or timeout > 60:
            raise ValueError("barrier wait timeout 必须在 0..60 秒")
        return timeout

    def log_message(self, format: str, *args: object) -> None:
        print(f"[model-gate] {self.address_string()} {format % args}", flush=True)


def build_server(host: str, port: int) -> ModelGateServer:
    return ModelGateServer((host, port), ModelGateState())


def main() -> int:
    parser = argparse.ArgumentParser(description="确定性 OpenAI-compatible model gate")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = build_server(args.host, args.port)
    stop_thread: threading.Thread | None = None

    def stop_server(signum: int, frame: object) -> None:
        """从非 serve_forever 线程触发收束，避免 PID 1 忽略 SIGTERM。"""

        nonlocal stop_thread
        if stop_thread is None:
            stop_thread = threading.Thread(
                target=server.shutdown, name="model-gate-stop"
            )
            stop_thread.start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    print(f"model-gate listening on {args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if stop_thread is not None:
            stop_thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
