"""真实模型 driver 与 curl 重放同一组 HTTP 请求，全部使用本地服务。"""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import argparse
import hashlib

sys.path.insert(0, str(Path(__file__).parent))
import fixture as base
from plugins.openai_compatible.driver import _BoundChat, _ConnectionConfig, _ModelConfig
from plugins.openai_compatible import driver as compatible

WIRE = []
PORT = None
POOLS = []
LEGACY_DRIVER = False
MATERIALS = False


class Credential:
    async def read(self):
        return {"api_key": "local-fixture"}


def driver(descriptor, calls):
    """给实际 driver 加计时；旧版测量必须显式选择。"""

    class Driver(_BoundChat):
        async def complete(self, request):
            base.mark("provider_enter")
            calls.append(request)
            try:
                return await super().complete(request)
            finally:
                base.mark("provider_return")

    config = _ConnectionConfig(f"http://127.0.0.1:{PORT}/v1", 10, 30, 0, True)
    if LEGACY_DRIVER:
        return Driver(config, Credential(), descriptor, _ModelConfig(None, 128))
    from core.net.http import HttpClient

    pool = HttpClient(lambda: compatible._client(config))
    POOLS.append(pool)
    return Driver(config, Credential(), descriptor, _ModelConfig(None, 128), pool)


base.probe_module.driver = driver
original_extras = base.extras


def extras(mode):
    """接入 HTTP driver，并可选装入真实记忆插件与固定 embedding 边界。"""

    def setup(sources):
        original_extras(mode)(sources)
        if MATERIALS:
            import ast
            import shutil

            for name in ("akasha", "markdown_memory"):
                shutil.copytree(
                    base.ROOT / "plugins" / name,
                    sources / name,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            parsed = ast.parse(
                (base.ROOT / "tests/test_akasha_message_plugin.py").read_text()
            )
            source = next(
                node.value
                for node in ast.walk(parsed)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and 'name = "fixture_embeddings"' in node.value
            )
            directory = sources / "fixture_embeddings"
            directory.mkdir()
            (directory / "plugin.py").write_text(
                source.replace(
                    "LOG_PATH", repr(str(sources.parent / "embedding-calls.txt"))
                ).replace("EMBEDDING_AVAILABLE", "True")
            )
            settings = (
                sources.parent
                / "workspace/plugin-data/context-builtin/config.local.toml"
            )
            settings.parent.mkdir(parents=True, exist_ok=True)
            settings.write_text(
                '[prompt_sources]\nmarkdown_memory = "markdown_memory"\n'
            )
        p = sources / "test_provider/plugin.py"
        text = p.read_text()
        start = text.index("    class Driver:")
        stop = text.index("    descriptor =", start)
        text = text[:start] + text[stop:]
        text = "from latency_probe_marks import driver\n" + text
        text = text.replace(
            "model = _BoundChat(descriptor, Driver(), store)",
            "model = _BoundChat(descriptor, driver(descriptor, calls), store)",
        )
        text = text.replace("context_window=10000", "context_window=10000000")
        p.write_text(text)

    return setup


base.extras = extras


async def handle(reader, writer):
    """本地 provider 返回两次工具调用与一次结束响应，支持 HTTP keep-alive。"""
    try:
        while True:
            try:
                raw = await reader.readuntil(b"\r\n\r\n")
            except asyncio.IncompleteReadError:
                return
            lines = raw.decode().split("\r\n")
            headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
            headers = {key.lower(): value for key, value in headers.items()}
            if headers.get("expect", "").lower() == "100-continue":
                writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
                await writer.drain()
            body = await reader.readexactly(int(headers.get("content-length", 0)))
            request = json.loads(body)
            base.mark("wire_received")
            index = len(WIRE) % 3
            WIRE.append(body)
            delta = (
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "provider-call",
                            "type": "function",
                            "function": {"name": "write_evidence", "arguments": "{}"},
                        }
                    ]
                }
                if index < 2
                else {"content": "finished"}
            )
            chunks = [
                {
                    "id": "fixture-response",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "fixture",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                },
                {
                    "id": "fixture-response",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "fixture",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls" if index < 2 else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                },
            ]
            output = (
                "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            ).encode()
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {len(output)}\r\n\r\n".encode()
                + output
            )
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def run(history, output):
    """保存链路实测，再用一个 curl 进程重放三份完全相同的请求。"""
    global PORT
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    PORT = server.sockets[0].getsockname()[1]
    async with server:
        WIRE.clear()
        try:
            result = await base.run("actual", history, True)
        finally:
            for pool in POOLS:
                await pool.aclose()
            POOLS.clear()
        captured = WIRE[:]
        assert len(captured) == 3
        with tempfile.TemporaryDirectory(prefix="akashic-curl-") as folder:
            files = []
            for index, body in enumerate(captured):
                path = Path(folder) / f"{index}.json"
                path.write_bytes(body)
                files.append(path)
            curl_times = []
            for repeat in range(4):
                before = len(WIRE)
                start = time.perf_counter()
                args = ["curl", "--silent", "--show-error", "--fail", "--noproxy", "*"]
                for index, path in enumerate(files):
                    if index:
                        args += [
                            "--next",
                            "--silent",
                            "--show-error",
                            "--fail",
                            "--noproxy",
                            "*",
                        ]
                    args += [
                        "-H",
                        "Content-Type: application/json",
                        "-H",
                        "Expect:",
                        "--data-binary",
                        f"@{path}",
                        "-o",
                        os.devnull,
                        f"http://127.0.0.1:{PORT}/v1/chat/completions",
                    ]
                job = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await job.communicate()
                assert job.returncode == 0, stderr
                curl_times.append(round((time.perf_counter() - start) * 1000, 3))
                assert WIRE[before:] == captured
        result["curl_3_requests_ms"] = curl_times
        result["wire_bytes"] = [len(body) for body in captured]
        result["wire_digests"] = [hashlib.sha256(body).hexdigest() for body in captured]
        with Path(output).open("x") as target:
            target.write(json.dumps(result) + "\n")
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"reads", "request_shapes"}
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--materials", action="store_true")
    parser.add_argument(
        "--legacy-driver",
        action="store_true",
        help="Measure the pre-pool driver explicitly",
    )
    args = parser.parse_args()
    if args.history < 0:
        parser.error("--history must be nonnegative")
    if Path(args.output).exists() or (
        args.profile is not None and args.profile.exists()
    ):
        parser.error("output and profile paths must be new")
    LEGACY_DRIVER = args.legacy_driver
    MATERIALS = args.materials
    if args.profile is not None:
        import cProfile

        base.PROFILE = cProfile.Profile()
    asyncio.run(run(args.history, args.output))
    if args.profile is not None:
        base.PROFILE.dump_stats(args.profile)
