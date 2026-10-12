"""真实宿主验证原生看图、视觉转交和缺少视觉角色；脚本模型只驱动协议。"""
from __future__ import annotations

import asyncio
import base64
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


def images(value):
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            yield value["image_url"]["url"]
        for item in value.values():
            yield from images(item)
    elif isinstance(value, list):
        for item in value:
            yield from images(item)


class Handler(BaseHTTPRequestHandler):
    """严格区分只接收文本的主模型和接收图片的视觉模型。"""
    def do_GET(self):
        self.send_json({"data": [{"id": "main-fixture"}, {"id": "vision-fixture"}]})

    def send_json(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        probe = request["messages"] == [{"role": "user", "content": "Reply OK."}]
        if probe:
            name, text = None, "OK"
        elif request["model"] == "vision-fixture":
            self.server.vision.append(request)
            name, text = None, "The image is a blue square."
        else:
            step = len(self.server.main)
            self.server.main.append(request)
            text = "Fixture finished."
            if step == 0:
                name, arguments = "read_file", {"path": str(self.server.picture)}
            elif step == 1 and self.server.mode != "native":
                name, arguments = "read_image_vision", {
                    "path": str(self.server.picture), "prompt": "Describe the color and shape."}
            else:
                name = None
        tool = ({"id": "call-"+uuid4().hex, "type": "function",
                 "function": {"name": name, "arguments": json.dumps(arguments)}} if name else None)
        if not request.get("stream"):
            self.send_json({"choices": [{"message": {"role": "assistant", "content": text},
                                         "finish_reason": "stop"}]})
            return
        delta = {"tool_calls": [{"index": 0, **tool}]} if tool else {"content": text}
        chunks = [{"choices": [{"index": 0, "delta": delta}]},
                  {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool else "stop"}]}]
        body = (''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n').encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


async def configure(client, mode):
    """通过公开模型配置入口选择主模型模态和独立视觉角色。"""
    result = await client.request("models/catalog", {})
    revision = result["body"]["revision"]

    async def command(values):
        nonlocal revision
        result = await client.request("models/command", {**values, "expected_revision": revision})
        assert result["status"] == 200, result
        revision = result["body"]["revision"]

    if mode != "native":
        await command({"type": "update_model", "model_id": "benchmark", "image_input": False,
                       "context_window": 1000000, "max_output_tokens": 8192})
    if mode == "delegated":
        await command({"type": "add_model", "model_id": "vision", "connection_id": "benchmark",
                       "kind": "chat", "model": "vision-fixture", "capabilities": {
                           "context_window": 1000000, "max_output_tokens": 8192,
                           "input_modalities": ["text", "image"]}, "capability_sources": {}})
        await command({"type": "set_default", "role": "vision", "model_id": "vision"})


async def run(directory: Path, mode: str):
    """从正式 Session 调工具并检查真正发给两个模型的 HTTP 请求。"""
    # 1. 每种配置使用独立宿主、workspace 和本地 Provider；不接触正式状态。
    key = directory / "key"
    key.write_text("local-fixture-only")
    key.chmod(0o600)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.main, server.vision, server.mode = [], [], mode
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = docker("network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}")
    try:
        async with runtime(directory, max_steps=5, endpoint=f"http://{host}:{server.server_port}/v1",
                           model="main-fixture", key_file=key) as identity:
            server.picture = Path(identity["workspace"]) / "fixture.png"
            Image.new("RGB", (64,64), (20,80,200)).save(server.picture)
            async with await AsyncAkashic.connect(identity["rpc"], max_message_bytes=32*1024*1024) as client:
                await configure(client, mode)
                session, message = "programmatic:"+uuid4().hex, uuid4().hex
                await client.request("programmatic/session/admit", {"session_id":session,"persist_memory":False})
                async with await client.session_follow(session) as feed:
                    await client.request("programmatic/message/send", {"session_id":session,"message_id":message,
                                                                        "text":"Read the fixture image."})
                    async with asyncio.timeout(120):
                        async for event in feed.events():
                            if event["type"] == "messages.appended":
                                result = cast(dict[str,Any],await client.request("programmatic/message/result",
                                    {"session_id":session,"input_id":message}))
                                if result["status"] != "open":
                                    assert result["status"] == "complete",result
                                    break
                rows = (await client.message_read(session,after_seq=-1,limit=100))["items"]
            # 2. 原生模型直接得到附件；文本模型从未收到图片字节。
            main_images = [len(list(images(r["messages"]))) for r in server.main]
            assert main_images == ([0,1] if mode == "native" else [0,0,0]),main_images
            if mode != "native":
                assert "当前模型不接收图片" in json.dumps(server.main[1],ensure_ascii=False)
            if mode == "delegated":
                assert len(server.vision) == 1,len(server.vision)
                sent = list(images(server.vision[0]["messages"]))
                assert len(sent) == 1,sent
                picture = Image.open(BytesIO(base64.b64decode(sent[0].split(',',1)[1]))).convert("RGB")
                assert picture.size == (64,64) and picture.getpixel((32,32)) == (20,80,200)
                assert "The image is a blue square." in json.dumps(server.main[-1])
            else:
                assert not server.vision
            # 3. 缺少视觉角色明确报错，不调用文本模型冒充视觉结果。
            failures = [r for r in rows if r["body"]["kind"] == "tool_result" and r["body"]["outcome"] == "error"]
            if mode == "missing":
                assert len(failures) == 1,failures
                assert "模型角色不可用: vision" in json.dumps(failures,ensure_ascii=False)
            else:
                assert not failures,failures
            evidence = {"mode":mode,"main_request_images":main_images,
                        "vision_calls":len(server.vision),"tool_errors":len(failures)}
            (directory/"messages.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2))
            (directory/"requests.json").write_text(json.dumps({"main":server.main,"vision":server.vision}))
            (directory/"result.json").write_text(json.dumps(evidence,indent=2))
            print(json.dumps(evidence),flush=True)
    finally:
        server.shutdown()
        server.server_close()


async def main():
    directory = Path(tempfile.mkdtemp(prefix="akashic-computer-vision-"))
    print(directory,flush=True)
    for mode in ("native","delegated","missing"):
        target = directory/mode
        target.mkdir()
        await run(target,mode)


if __name__ == "__main__":
    asyncio.run(main())
