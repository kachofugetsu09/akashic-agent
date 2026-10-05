"""用隔离插件运行时、真实 HTTP/SSE 和 SQLite 验证回复预算及截断边界。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.plugin_composition.channels import CHANNEL_INPUT_V2, ChannelInboundMessage
from agent.plugin_composition.config_input import save_config
from session.message import Control, Output, ToolResult
from tests.test_default_reply import application, live_root


class Provider(ThreadingHTTPServer):
    """协议夹具只接受合成请求；实际客户端、调用账、回复和工具均由产品执行。"""

    def __init__(self, mode: str):
        super().__init__(("127.0.0.1", 0), Handler)
        self.mode = mode
        self.requests: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    server: Provider

    def do_POST(self):
        """复现 4K 推理耗尽，以及正文或合法工具参数已返回但仍被截断的情况。"""
        # 1. 记录真实 wire 预算，按场景返回公开 SSE 协议。
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        budget = body["max_tokens"]
        mode = self.server.mode
        tool = {"index": 0, "id": "call-evidence", "type": "function",
                "function": {"name": "write_evidence", "arguments": "{}"}}
        if mode == "empty":
            delta, finish, used = {}, "stop", 0
        elif mode == "text-length":
            delta, finish, used = {"content": "unfinished"}, "length", budget
        elif mode == "tool-length":
            delta, finish, used = {"tool_calls": [tool]}, "length", budget
        elif mode == "after-tool-length" and len(self.server.requests) == 2:
            delta, finish, used = {"reasoning_content": "synthetic reasoning"}, "length", budget
        elif budget <= 4096:
            delta, finish, used = {"reasoning_content": "synthetic reasoning"}, "length", budget
        elif len(self.server.requests) == 1:
            delta, finish, used = {"tool_calls": [tool]}, "tool_calls", 5000
        else:
            delta, finish, used = {"content": "finished"}, "stop", 12
        usage = {"prompt_tokens": 100, "completion_tokens": used,
                 "completion_tokens_details": {"reasoning_tokens": used if budget <= 4096 else 0}}
        events = [{"choices": [{"delta": delta, "finish_reason": None}]},
                  {"choices": [{"delta": {}, "finish_reason": finish}], "usage": usage}]
        data = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        data += "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data.encode())))
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, *_args):
        pass


def install_provider(sources: Path, endpoint: str, cap: int | None, configured: int | None):
    """在一次性插件源码中接入真实 driver；不替换产品调用路径或正式凭据。"""
    save_config(sources.parent / "workspace/plugin-data/reply-builtin",
                {} if configured is None else {"max_output_tokens": configured})
    module = sources / "test_provider/plugin.py"
    text = module.read_text().replace(
        "ModelCapabilities(context_window=10000)",
        f"ModelCapabilities(context_window=1000000, max_output_tokens={cap!r})",
    )
    text = text.replace("    model = _BoundChat(descriptor, Driver(), store)", f'''
    import httpx
    from core.net.http import HttpClient
    from plugins.openai_compatible import driver
    class Credential:
        async def read(self):
            return {{"driver": "api_key", "api_key": "isolated-only"}}
    http = HttpClient(lambda: httpx.AsyncClient(base_url={endpoint!r}, trust_env=False))
    await ctx.effect(lambda: http.aclose, label="isolated-http")
    physical = driver._BoundChat(
        driver._ConnectionConfig({endpoint!r}, 5, 5, 0, False),
        Credential(), descriptor, driver._ModelConfig(None, 16), http)
    model = _BoundChat(descriptor, physical, store)
''')
    module.write_text(text)


async def check(directory: Path, *, mode: str, cap: int | None = None,
                configured: int | None = None, expected_budget: int, expected_error: str | None):
    """从 Channel Input 跑到终态，再查消息、工具副作用和模型账的持久结果。"""
    server = Provider(mode)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # 1. 原有完整插件装配接纳输入，HTTP server 是唯一合成的外部边界。
        endpoint = f"http://127.0.0.1:{server.server_port}"
        async with application(directory, replying=True, extra_sources=lambda path:
                               install_provider(path, endpoint, cap, configured)) as (log, host):
            async with live_root(host) as root:
                await root.context.require(CHANNEL_INPUT_V2)(
                    "test:room", "input-budget", ChannelInboundMessage(
                        "test", "user", "room", "Write evidence once, then finish.",
                        datetime.now(UTC), {}))
            reader = log.reader("test:room")
            original = reader.snapshot()[0]

            async def terminal():
                async for _ in log.catalog().follow():
                    rows = reader.snapshot()
                    if any((isinstance(m.body, Output) and m.body.finish == "complete")
                           or (isinstance(m.body, Control) and m.body.action == "failure") for m in rows):
                        return rows

            rows = await asyncio.wait_for(terminal(), 20)
            assert rows[0] == original
            outputs = [m for m in rows if isinstance(m.body, Output)]
            failures = [m.body.reason for m in rows if isinstance(m.body, Control)
                        and m.body.action == "failure"]
            assert all(body["max_tokens"] == expected_budget for body in server.requests)
            if expected_error is None:
                assert not failures, failures
                assert outputs[-1].body.finish == "complete"
                assert sum(isinstance(m.body, ToolResult) for m in rows) == 1
                assert (directory / "effect.txt").read_text() == "once\n"
                assert len(server.requests) == 2
            else:
                assert len(failures) == 1 and expected_error in failures[0], failures
                if mode == "after-tool-length":
                    assert len(outputs) == 1 and outputs[0].body.finish == "continue"
                    assert (directory / "effect.txt").read_text() == "once\n"
                    assert len(server.requests) == 2
                else:
                    assert not outputs and not (directory / "effect.txt").exists()
                    assert len(server.requests) == 1  # 不自动重放已计费请求。

        # 2. 关闭运行时后重开 DB，原始模型回执、usage 和 Input 仍可读。
        db = directory / "workspace/plugin-data/test_provider-builtin/models.db"
        with sqlite3.connect(db) as connection:
            receipts = connection.execute(
                "SELECT state,response_json,usage_json FROM model_calls ORDER BY started_at,id"
            ).fetchall()
            assert len(receipts) == len(server.requests)
            assert all(state == "success" for state, _, _ in receipts)
            assert all(json.loads(usage)["output_tokens"] is not None for _, _, usage in receipts)
            if expected_error is not None:
                reasons = [json.loads(response)["finish_reason"] for _, response, _ in receipts]
                assert ("stop" if mode == "empty" else "length") in reasons
        with sqlite3.connect(directory / "sessions.db") as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert connection.execute("SELECT id FROM messages ORDER BY seq LIMIT 1").fetchone()[0] == original.message_id
        return {"mode": mode, "budget": expected_budget, "requests": len(server.requests),
                "outcome": "complete" if expected_error is None else failures[0],
                "original_input_preserved": True, "receipts_preserved": True}
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


async def main(directory: Path):
    results = []
    cases = [
        dict(mode="reasoning", expected_budget=32768, expected_error=None),
        dict(mode="reasoning", cap=8192, expected_budget=8192, expected_error=None),
        dict(mode="reasoning", cap=65536, expected_budget=65536, expected_error=None),
        dict(mode="reasoning", configured=4096, expected_budget=4096, expected_error="长度限制"),
        dict(mode="reasoning", configured=16384, cap=65536, expected_budget=16384, expected_error=None),
        dict(mode="text-length", expected_budget=32768, expected_error="长度限制"),
        dict(mode="tool-length", expected_budget=32768, expected_error="长度限制"),
        dict(mode="after-tool-length", expected_budget=32768, expected_error="长度限制"),
        dict(mode="empty", expected_budget=32768, expected_error="空响应不是 quiet"),
    ]
    for index, case in enumerate(cases):
        results.append(await check(directory / str(index), **case))
    return results


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-reply-budget-") as path:
        print(json.dumps(asyncio.run(main(Path(path))), ensure_ascii=False, indent=2))
