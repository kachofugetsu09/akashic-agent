"""即时关闭的 uvicorn Server 保留在途请求与 lifespan 的优雅关闭语义。"""
import asyncio
import threading
import time

import pytest
import uvicorn

from core.net.asgi_server import PromptShutdownServer


def _app(events: list[str]):
    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                else:
                    events.append("lifespan.shutdown")
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        await receive()
        await asyncio.sleep(0.2)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"done"})
        events.append("response")
    return app


async def _serve(tmp_path, events, server_type=PromptShutdownServer):
    server = server_type(uvicorn.Config(_app(events), uds=str(tmp_path / "s.sock"), log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    return server, task


async def test_idle_exit_from_other_thread_runs_lifespan_shutdown_promptly(tmp_path):
    events: list[str] = []
    server, task = await _serve(tmp_path, events)
    started = time.perf_counter()
    thread = threading.Thread(target=setattr, args=(server, "should_exit", True))
    thread.start()
    await asyncio.wait_for(task, 1)
    thread.join()
    assert time.perf_counter() - started < 0.08
    assert events == ["lifespan.shutdown"]


@pytest.mark.parametrize("server_type", [uvicorn.Server, PromptShutdownServer])
async def test_exit_waits_for_in_flight_response(tmp_path, server_type):
    events: list[str] = []
    server, task = await _serve(tmp_path, events, server_type)
    reader, writer = await asyncio.open_unix_connection(str(tmp_path / "s.sock"))
    writer.write(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
    await writer.drain()
    await asyncio.sleep(0.05)
    server.should_exit = True
    await asyncio.wait_for(task, 2)
    assert b"\r\ndone\r\n" in await reader.read()
    assert events == ["response", "lifespan.shutdown"]
    writer.close()
    await writer.wait_closed()
