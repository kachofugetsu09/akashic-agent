"""各 Web 服务器所有者的即时关闭保留在途请求与 lifespan 的优雅关闭语义。"""
import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol

from bootstrap.web_shell import WebShellServer
from plugins.akashic_clients.chat_api import ChatApiServer
from plugins.ui.dashboard_server import DashboardServer

PROMPT_SERVERS = [WebShellServer, DashboardServer, ChatApiServer]


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


async def _serve(tmp_path, events, server_type):
    server = server_type(uvicorn.Config(_app(events), uds=str(tmp_path / "s.sock"), log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    return server, task


@pytest.mark.parametrize("server_type", PROMPT_SERVERS)
async def test_idle_exit_from_other_thread_starts_graceful_shutdown_promptly(tmp_path, server_type, monkeypatch):
    events: list[str] = []
    server, task = await _serve(tmp_path, events, server_type)
    shutdown_started = asyncio.Event()
    shutdown = server.shutdown

    async def observe_shutdown(sockets=None):
        shutdown_started.set()
        await shutdown(sockets)

    monkeypatch.setattr(server, "shutdown", observe_shutdown)
    started = time.perf_counter()
    thread = threading.Thread(target=setattr, args=(server, "should_exit", True))
    thread.start()
    await asyncio.wait_for(shutdown_started.wait(), 1)
    # 唤醒不等待轮询；连接排空仍由 uvicorn 按自己的超时执行。
    assert time.perf_counter() - started < 0.08
    await asyncio.wait_for(task, 1)
    thread.join()
    assert events == ["lifespan.shutdown"]


@pytest.mark.parametrize("server_type", [uvicorn.Server, *PROMPT_SERVERS])
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


@pytest.mark.parametrize("server_type", [uvicorn.Server, *PROMPT_SERVERS])
async def test_accepted_connection_before_registration_keeps_shutdown_timeout(server_type):
    """真实连接已接受但尚未登记时退出，仍会超时并执行 lifespan 关闭。"""
    events: list[str] = []
    config = uvicorn.Config(_app(events), host="127.0.0.1", port=0, log_level="critical")
    # uvicorn 把该参数标注为 int，但只用于 asyncio.wait_for；测试用亚秒超时保持快速。
    setattr(config, "timeout_graceful_shutdown", 0.03)
    config.load()
    server = server_type(config)
    server.lifespan = config.lifespan_class(config)
    shutdown_task: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

    def protocol(*args, **kwargs):
        # 构造协议时连接尚未登记；关闭任务排在 connection_made 之前。
        connection = H11Protocol(*args, **kwargs)
        shutdown_task.set_result(asyncio.create_task(server.shutdown()))
        return connection

    config.http_protocol_class = protocol
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        await server.startup(sockets=[listener])
        _reader, writer = await asyncio.open_connection(*listener.getsockname())
        task = await asyncio.wait_for(shutdown_task, 1)
        try:
            # 客户端保持连接；不能依赖它断开才完成关闭。
            await asyncio.wait_for(asyncio.shield(task), 1)
            assert events == ["lifespan.shutdown"]
        finally:
            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(task, 1)
