"""通过真实 TCP→UDS 测量代理复用、隔离、换代与流式释放。"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import statistics
import sys
import tempfile
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from uvicorn.protocols.http.h11_impl import H11Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.plugin_composition.endpoints import Endpoint, save_endpoint_plan


class Connections:
    def __init__(self) -> None:
        self.accepted = 0
        self.streams = 0
        self.live: set[asyncio.Transport] = set()

    def protocol(self):
        counts = self

        class ObservedH11(H11Protocol):
            def connection_made(self, transport):
                counts.accepted += 1
                counts.live.add(transport)
                super().connection_made(transport)

            def connection_lost(self, error):
                counts.live.discard(self.transport)
                super().connection_lost(error)

        return ObservedH11


@asynccontextmanager
async def serve(app, *, uds=None, counts=None):
    """等待真实监听就绪；停止时排空应用与连接。"""
    ready = asyncio.Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            ready.set()

    # TCP 明确使用 IPPROTO_TCP，与正式 create_server 相同；proto=0 会使 asyncio 跳过 TCP_NODELAY。
    with socket.socket(socket.AF_UNIX if uds else socket.AF_INET, socket.SOCK_STREAM,
                       0 if uds else socket.IPPROTO_TCP) as sock:
        sock.bind(str(uds) if uds else ("127.0.0.1", 0))
        server = Server(uvicorn.Config(app, log_level="error", lifespan="on",
            http=counts.protocol() if counts else "h11", timeout_graceful_shutdown=1))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            await asyncio.wait_for(ready.wait(), 10)
            yield sock.getsockname()
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)


def backend(generation: str, counts: Connections):
    """对端回显传输事实，并通过事件协调长流与取消。"""
    app = FastAPI()
    gate, stream_closed, headers_started, headers_release = (asyncio.Event() for _ in range(4))
    posts = []

    @app.get("/echo")
    async def echo(request: Request):
        return JSONResponse({"generation": generation, "cookie": request.headers.get("cookie"),
            "authorization": request.headers.get("authorization"), "query": request.url.query},
            headers={"Set-Cookie": "upstream_secret=must-not-leak; Path=/"})

    @app.get("/stream")
    async def stream():
        async def chunks():
            counts.streams += 1
            try:
                yield b"started\n"
                await gate.wait()
                yield b"finished\n"
            finally:
                counts.streams -= 1
                stream_closed.set()
        return StreamingResponse(chunks(), media_type="text/plain")

    @app.get("/headers")
    async def headers():
        headers_started.set()
        await headers_release.wait()
        return {"generation": generation}

    @app.post("/drop")
    async def drop(request: Request):
        posts.append(await request.body())
        for transport in tuple(counts.live):
            transport.abort()
        return {"committed": True}

    return app, gate, stream_closed, headers_started, headers_release, posts


async def measure(client, counts):
    """固定请求数、并发与查询；记录墙钟、进程 CPU 和事件循环调度延迟。"""
    rows = []
    for concurrency in (1, 10):
        for run in range(5):
            lag, latencies = [], []
            stopped = asyncio.Event()

            async def monitor():
                while not stopped.is_set():
                    start = time.perf_counter()
                    await asyncio.sleep(0.005)
                    lag.append(max(0, time.perf_counter() - start - 0.005))

            async def request():
                start = time.perf_counter()
                response = await client.get("/echo?q=a%2Fb&empty=")
                assert response.status_code == 200 and response.json()["generation"] == "one"
                latencies.append(time.perf_counter() - start)

            monitor_task = asyncio.create_task(monitor())
            accepted = counts.accepted
            start, cpu = time.perf_counter(), time.process_time()
            try:
                for _ in range(100 // concurrency):
                    await asyncio.gather(*(request() for _ in range(concurrency)))
            finally:
                stopped.set()
                await monitor_task
            rows.append({"concurrency": concurrency, "run": run, "requests": 100,
                "wallMs": (time.perf_counter() - start) * 1000,
                "cpuMs": (time.process_time() - cpu) * 1000,
                "connections": counts.accepted - accepted,
                "medianMs": statistics.median(latencies) * 1000,
                "p95Ms": sorted(latencies)[94] * 1000,
                "maxLoopLagMs": max(lag, default=0) * 1000})
    return rows


async def run(factory, workspace, pooled):
    """保持上游和网络不变；故障检查不计入性能样本。"""
    counts = Connections()
    app, gate, stream_closed, headers_started, headers_release, posts = backend("one", counts)
    uds = workspace / "backend.sock"
    plan = workspace / "runtime/endpoints.json"

    def publish(generation):
        save_endpoint_plan(plan, generation, (
            Endpoint("probe", "http+unix", str(uds), ("/",), "scenario", generation),))

    async with serve(app, uds=uds, counts=counts), serve(factory(workspace)) as address:
        publish("one")
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{address[1]}", trust_env=False,
            timeout=10, limits=httpx.Limits(max_connections=None)) as client:
            rows = await measure(client, counts)
            # 1. 响应 Cookie 不会污染另一个浏览器；各请求原有 Cookie/Authorization 原样传递。
            for identity in ("alice", "bob", None):
                headers = {} if identity is None else {"Cookie": f"user={identity}", "Authorization": identity}
                response = await client.get("/echo", headers=headers)
                assert response.json()["cookie"] == headers.get("Cookie")
                assert response.json()["authorization"] == identity

            # 2. 超过 HTTPX 默认总名额的长流不能挤掉短请求；取消后实际上游生成器结束。
            async with AsyncExitStack() as streams:
                for _ in range(101):
                    response = await streams.enter_async_context(client.stream("GET", "/stream"))
                    assert response.status_code == 200
                async with asyncio.timeout(3):
                    assert (await client.get("/echo")).status_code == 200
            await asyncio.wait_for(stream_closed.wait(), 3)
            async with asyncio.timeout(3):
                while counts.streams:
                    await asyncio.sleep(0)

            # 3. 响应头之前取消、上游已接纳 POST 后掉线，都不能隐式重发。
            pending = asyncio.create_task(client.get("/headers"))
            await asyncio.wait_for(headers_started.wait(), 3)
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass
            headers_release.set()
            response = await client.post("/drop", content=b"one-commit")
            assert response.status_code == 503 and posts == [b"one-commit"]
            assert (await client.get("/echo")).status_code == 200

            # 4. 同一路径发布新 generation；旧流排空，新请求只能到新 listener。
            stream_closed.clear()
            async with client.stream("GET", "/stream") as old:
                lines = old.aiter_lines()
                assert await anext(lines) == "started"
                uds.unlink()
                next_counts = Connections()
                replacement, *_ = backend("two", next_counts)
                async with serve(replacement, uds=uds, counts=next_counts):
                    publish("two")
                    assert (await client.get("/echo")).json()["generation"] == "two"
                    gate.set()
                    assert await anext(lines) == "finished"
                    assert [line async for line in lines] == []
                    await asyncio.wait_for(stream_closed.wait(), 3)
                    async with asyncio.timeout(3):
                        while counts.live:
                            await asyncio.sleep(0)
                    # 正式撤下计划后不接受新请求；池退休不会重新选择旧地址。
                    save_endpoint_plan(plan, "empty", ())
                    assert (await client.get("/echo")).status_code == 503
                    if pooled:
                        async with asyncio.timeout(3):
                            while next_counts.live:
                                await asyncio.sleep(0)
        assert not posts[1:]
    assert not counts.live
    return {"runs": rows, "cookies_isolated": True, "streams_do_not_starve": True,
            "cancelled_stream_closed": True, "post_not_replayed": True,
            "same_path_new_generation": True, "shutdown_closed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pooled", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("measured_web_shell", args.source.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="web-shell-http-") as folder:
        result = asyncio.run(run(module.create_web_shell_app, Path(folder), args.pooled))
    result["sourceSha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
