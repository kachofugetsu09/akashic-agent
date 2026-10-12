"""真实 AppRuntime/Computer/账本/Provider 链路验证图片附件；本地脚本模型只驱动固定动作。"""
from __future__ import annotations

import asyncio
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import threading
from uuid import uuid4
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "sdk/python/src")]
from benchmark.computer_runtime.runtime import docker, runtime
from akashic_sdk import AsyncAkashic
from PIL import Image


class Handler(BaseHTTPRequestHandler):
    """只产生固定工具调用，检查真实 Provider 请求内的图像字节。"""
    def do_GET(self):
        body = json.dumps({"data": [{"id": "attachment-fixture"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        step = len(self.server.requests)
        probe = request["messages"] == [{"role": "user", "content": "Reply OK."}]
        if not probe:
            self.server.requests.append(request)
        if probe:
            name = None
        elif step == 0:
            name, arguments = "load_tools", {"plugin": "computer"}
        elif step == 1:
            name, arguments = "tool_call", {"name": "computer", "arguments": {"code": '''
var tab = await browser.tabs.new();
await tab.goto('data:text/html,<body style="margin:0;background:rgb(20,80,200)">');
nodeRepl.write('before');
for(var i=0;i<33;i++) await nodeRepl.emitImage(await tab.screenshot({clip:{x:0,y:0,width:20,height:20}}));
nodeRepl.write('after');
'''}}
        else:
            name = None
        delta = ({"tool_calls": [{"index": 0, "id": f"call-{step}", "type": "function",
                  "function": {"name": name, "arguments": json.dumps(arguments)}}]}
                 if name else {"content": "Fixture finished."})
        chunks = [{"choices": [{"index": 0, "delta": delta}]},
                  {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if name else "stop"}]}]
        body = (''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n').encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def images(value):
    """收集模型请求内的图像块，不依赖图像被投影到哪个 role。"""
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            yield value["image_url"]["url"]
        for item in value.values():
            yield from images(item)
    elif isinstance(value, list):
        for item in value:
            yield from images(item)


async def run(directory: Path):
    """从正式输入触发截图，跨缓存淘汰和宿主退出核对持久结果。"""
    # 1. 本地协议驱动没有真实凭据；浏览器、MCP、工具和 Ledger 都使用真实实现。
    key = directory / "key"
    key.write_text("local-fixture-only")
    key.chmod(0o600)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = docker("network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}")
    try:
        async with runtime(directory, max_steps=5, endpoint=f"http://{host}:{server.server_port}/v1",
                           model="attachment-fixture", key_file=key) as identity:
            async with await AsyncAkashic.connect(identity["rpc"], max_message_bytes=32 * 1024 * 1024) as client:
                session, message = "programmatic:" + uuid4().hex, uuid4().hex
                await client.request("programmatic/session/admit", {"session_id": session, "persist_memory": False})
                async with await client.session_follow(session) as feed:
                    await client.request("programmatic/message/send", {"session_id": session, "message_id": message, "text": "Run the attachment integration fixture."})
                    async with asyncio.timeout(120):
                        async for event in feed.events():
                            if event["type"] == "messages.appended":
                                result = cast(dict[str, Any], await client.request("programmatic/message/result", {"session_id": session, "input_id": message}))
                                if result["status"] != "open":
                                    assert result["status"] == "complete", result
                                    break
                rows = (await client.message_read(session, after_seq=-1, limit=100))["items"]
            (directory / "messages.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            (directory / "requests.json").write_text(json.dumps(server.requests, ensure_ascii=False))
            # 2. 下一次模型请求直接含图，不需要模型再调用文件工具。
            assert len(server.requests) == 3, len(server.requests)
            sent = list(images(server.requests[-1]["messages"]))
            assert len(sent) == 33, len(sent)
            for url in sent:
                picture = Image.open(BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert("RGB")
                assert picture.size == (20, 20)
                assert all(abs(a-b) <= 5 for a,b in zip(cast(tuple[int, ...], picture.getpixel((10,10))), (20,80,200)))
            result = next(r for r in rows if len(r["attachments"]) == 33)
            assert result["body"]["parts"][0]["value"] == "before"
            assert result["body"]["parts"][-1]["value"] == "after"
            assert not any(p.get("name") == "read_file" for r in rows for p in r["body"].get("parts", []))
            workspace = Path(identity["workspace"])
            cached = list((workspace / "plugin-data/computer-builtin/screenshots").glob("computer-*"))
            assert len(cached) == 32, len(cached)
            first_path = Path(result["body"]["parts"][1]["value"].removeprefix("Screenshot: "))
            assert not first_path.exists(), first_path
            saved = [(workspace / "uploads/artifacts" / (ref["artifact_id"] + ".bin"), ref["sha256"]) for ref in result["attachments"]]
            (directory / "messages.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        # 3. 宿主与截图缓存生命周期结束后，已提交附件仍保持原字节。
        for path, digest in saved:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        evidence = {"image_attachments": 33, "cache_files": 32, "next_request_images": 33,
                    "read_file_calls": 0, "source_evicted_artifact_preserved": True,
                    "text_order": ["before", "after"], "after_shutdown_bytes_match": True}
        (directory / "result.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence), flush=True)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    target = Path(tempfile.mkdtemp(prefix="akashic-computer-attachments-"))
    print(target, flush=True)
    asyncio.run(run(target))
