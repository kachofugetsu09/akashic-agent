"""开发实例用的确定性 OpenAI 兼容上游：不花额度，回复里带模型名；可让每次用户消息先跑 N 步工具。

    python3 mock_model.py --port 44099 --steps 6 --tool shell --args '{"command": "echo step {n}", "description": "第 {n} 步"}'

只依赖标准库。--steps 为 0 时直接回复文字；请求里没有 --tool 这个工具时也直接回复。
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODELS = ("mock-main", "mock-mini")


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


# 1. 标题、压缩摘要这类旁路调用直接给固定文字；2. 本轮工具结果不足 N 条就再调一次工具；3. 否则收尾。
def decide(body: dict[str, Any], steps: int, tool: str, arguments: str) -> dict[str, Any]:
    model, messages = str(body["model"]), list(body["messages"])
    everything = " ".join(_text(message) for message in messages)
    if "拟一个简短会话标题" in everything:
        return {"role": "assistant", "content": f"{model} 起的标题"}
    if "[Source messages]" in everything:
        return {"role": "assistant", "content": f"## Goal\n- {model} 的摘要"}
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=0)
    done = sum(1 for m in messages[last_user:] if m.get("role") == "tool")
    tools = {item["function"]["name"] for item in body.get("tools", [])}
    if done < steps and tool in tools:
        call = {"id": f"call-{len(messages)}", "type": "function",
                "function": {"name": tool, "arguments": arguments.replace("{n}", str(done + 1))}}
        return {"role": "assistant", "content": None, "tool_calls": [call]}
    return {"role": "assistant", "content": f"{model} 回复：{_text(messages[last_user])[:80]}（{done} 步）"}


def handler(steps: int, tool: str, arguments: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: object) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._json({"object": "list", "data": [{"id": name, "object": "model"} for name in MODELS]})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            message = decide(body, steps, tool, arguments)
            finish = "tool_calls" if message.get("tool_calls") else "stop"
            chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in body["messages"])
            usage = {"prompt_tokens": 1000 + chars // 3, "completion_tokens": 50, "total_tokens": 1050 + chars // 3}
            if not body.get("stream"):
                self._json({"id": "mock", "object": "chat.completion", "usage": usage,
                            "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
                return
            delta = {key: value for key, value in message.items() if key != "role"}
            for index, call in enumerate(delta.get("tool_calls") or []):
                call["index"] = index
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in ({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
                          {"choices": [], "usage": usage}):
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=44099)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--tool", default="shell")
    parser.add_argument("--args", default='{"command": "echo step {n}", "description": "第 {n} 步"}', help="工具参数 JSON，{n} 替换为步号")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.steps, args.tool, args.args))
    print(f"mock model on http://127.0.0.1:{args.port} (steps={args.steps}, tool={args.tool})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
