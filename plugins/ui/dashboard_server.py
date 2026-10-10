"""UI 的 Effect 持有监听器；端点只能在实际就绪后发布。"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Generator
from contextlib import contextmanager
import logging
import socket
import signal
from pathlib import Path
from fastapi import FastAPI
import uvicorn

from agent.plugin_composition import Context
from core.common.unix_socket import prepare_unix_socket, socket_alias_path
from .dashboard_app import create_dashboard_app, _install_dashboard_access_log_filter

logger = logging.getLogger(__name__)


class DashboardServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config):
        super().__init__(config)
        self.ready = asyncio.Event()
        self.servers: list[asyncio.AbstractServer] = []

    @contextmanager
    def capture_signals(self) -> Generator[None]:
        # 进程信号归外壳；插件服务器只响应自己的 Effect 关闭。
        yield

    async def startup(self, sockets=None) -> None:
        await super().startup(sockets)
        self.ready.set()


async def start_dashboard_server(ctx: Context, app: FastAPI) -> None:
    """先保留清理责任，再等待实际启动，最后发布当前 owner 的端点。"""
    # 1. Unix 路径保持原 workspace 节点；代码与构建资产只读。
    path = socket_alias_path(ctx.runtime.workspace / "runtime", "dashboard.sock")
    server = DashboardServer(uvicorn.Config(app, log_level="info", lifespan="off",
                                           timeout_graceful_shutdown=10))
    _install_dashboard_access_log_filter()
    health = await ctx.health("dashboard")
    health.degrade("starting")
    task: asyncio.Task[None] | None = None
    stopping = False

    def setup():
        nonlocal task
        prepare_unix_socket(path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound = False
        try:
            listener.bind(str(path))
            bound = True
            listener.setblocking(False)
            node = path.lstat()
            task = asyncio.create_task(server.serve(sockets=[listener]), name="ui-dashboard-server")
        except BaseException:
            listener.close()
            if bound:
                path.unlink()
            raise
        running = task
        closed = False

        async def close() -> None:
            nonlocal closed, stopping
            if closed:
                return
            stopping = True
            server.should_exit = True
            try:
                await running
            finally:
                listener.close()
                if any(item.is_serving() for item in server.servers):
                    raise RuntimeError("UI listener 仍在接纳连接")
                current = path.lstat()
                if (current.st_dev, current.st_ino) != (node.st_dev, node.st_ino):
                    raise RuntimeError("UI socket 已被另一节点替换")
                path.unlink()
                closed = True
        return close

    # 2. setup 没有 await；任务第一次等待 RPC 前，Effect 已保留实际关闭责任。
    await ctx.effect(setup, label="dashboard.listener")
    assert task is not None
    ready = asyncio.create_task(server.ready.wait(), name="ui-dashboard-ready")
    try:
        await asyncio.wait((ready, task), return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            await task
            raise RuntimeError("UI server 在就绪前结束")
        if not server.started:
            raise RuntimeError("UI listener 没有完成启动")
    finally:
        ready.cancel()
        await asyncio.gather(ready, return_exceptions=True)

    def finished(done: asyncio.Task[None]) -> None:
        if stopping:
            return
        error = None if done.cancelled() else done.exception()
        detail = "UI listener 意外结束" if error is None else str(error) or type(error).__name__
        health.degrade(detail)
        ctx.report_incident("dashboard.listener.failed", detail)
        logger.error(detail, exc_info=error)

    task.add_done_callback(finished)
    # 3. Effect 反序关闭：撤下端点成功之后，才停止 listener 并排空 HTTP/WebSocket。
    await ctx.endpoint("dashboard", protocol="http+unix", address=str(path), routes=("/",))
    health.recover()


async def main(arguments: tuple[str, ...], *, workspace: Path, config_path: Path) -> int:
    """独立静态 Dashboard 命令不启动 Root、打开业务库或取得 workspace 锁。"""
    parser = argparse.ArgumentParser(description="Run the selected Dashboard assets")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2236)
    options = parser.parse_args(arguments)
    app = create_dashboard_app(Path(__file__).parent / "static/dashboard")
    server = DashboardServer(uvicorn.Config(app, host=options.host, port=options.port, lifespan="off"))
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, lambda: setattr(server, "should_exit", True))
    try:
        await server.serve()
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)
    return 0
