#!/usr/bin/env python3
"""通过 Unix socket 调用已运行的 Akashic app-server。"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any, cast


class JsonRpcConnection:
    def __init__(self, endpoint: str, timeout: float) -> None:
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)
        self._socket.connect(endpoint)
        self._stream = self._socket.makefile("rwb")
        self._notifications: list[dict[str, Any]] = []

    def send(self, payload: dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        _ = self._stream.write(data.encode("utf-8") + b"\n")
        self._stream.flush()

    def read(self) -> dict[str, Any]:
        line = self._stream.readline()
        if not line:
            raise ConnectionError("Akashic app-server closed the connection")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("JSON-RPC frame must be an object")
        return cast(dict[str, Any], payload)

    def request(
        self,
        request_id: int,
        method: str,
        params: dict[str, object],
    ) -> dict[str, Any]:
        """发送 request，并跳过响应前到达的 notification。"""

        # 1. 发出带稳定 ID 的请求。
        self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )

        # 2. 等待对应 response；异步 notification 原样输出。
        while True:
            message = self.read()
            if message.get("id") != request_id:
                self._notifications.append(message)
                continue
            error = message.get("error")
            if error is not None:
                raise RuntimeError(json.dumps(error, ensure_ascii=False))
            result = message.get("result")
            if not isinstance(result, dict):
                raise ValueError(f"{method} result must be an object")
            return cast(dict[str, Any], result)

    def next_notification(self) -> dict[str, Any]:
        if self._notifications:
            return self._notifications.pop(0)
        return self.read()

    def close(self) -> None:
        self._stream.close()
        self._socket.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call an already-running Akashic gateway over its Unix socket.",
    )
    _ = parser.add_argument("endpoint", help="固定 workspace 的 akashic.sock 路径")
    thread = parser.add_mutually_exclusive_group(required=True)
    _ = thread.add_argument("--new", action="store_true", help="创建并打印新 thread ID")
    _ = thread.add_argument("--thread", help="复用明确的 thread ID")
    _ = parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        default=600.0,
        help="每次 socket 操作的超时秒数，默认 600",
    )
    _ = parser.add_argument("prompt", help="本轮输入；传 - 时从 stdin 读取")
    return parser.parse_args()


def _positive_timeout(value: str) -> float:
    timeout = float(value)
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout 必须是正数")
    return timeout


def run(connection: JsonRpcConnection, args: argparse.Namespace) -> int:
    """完成握手、发起一轮并等待唯一终态。"""

    # 1. 完成协议握手。
    _ = connection.request(
        1,
        "initialize",
        {
            "protocolVersion": "1.0",
            "clientInfo": {"name": "raw-jsonrpc-example", "version": "1.0"},
            "capabilities": {"reasoningEvents": False},
            "workspaceToken": None,
        },
    )
    connection.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    # 2. 创建或恢复调用方明确选择的 thread。
    if args.new:
        thread = connection.request(2, "thread/start", {"metadata": {"caller": "raw-example"}})
        thread_id = str(thread["id"])
        print(json.dumps({"threadId": thread_id}, ensure_ascii=False))
    else:
        thread_id = str(args.thread)
        _ = connection.request(2, "thread/resume", {"threadId": thread_id})

    # 3. 发起 turn，并消费到匹配的唯一终态。
    prompt = sys.stdin.read() if args.prompt == "-" else str(args.prompt)
    turn = connection.request(
        3,
        "turn/start",
        {
            "threadId": thread_id,
            "input": prompt,
            "metadata": {"caller": "raw-example"},
        },
    )
    turn_id = str(turn["id"])
    while True:
        event = connection.next_notification()
        print(json.dumps(event, ensure_ascii=False))
        params = event.get("params")
        if event.get("method") != "turn/completed" or not isinstance(params, dict):
            continue
        typed_params = cast(dict[str, Any], params)
        if typed_params.get("turnId") != turn_id:
            continue
        terminal = typed_params.get("turn")
        if not isinstance(terminal, dict):
            raise ValueError("turn/completed is missing turn")
        typed_terminal = cast(dict[str, Any], terminal)
        return 0 if typed_terminal.get("status") == "completed" else 1


def main() -> int:
    args = parse_args()
    connection = JsonRpcConnection(str(args.endpoint), float(args.timeout))
    try:
        return run(connection, args)
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
