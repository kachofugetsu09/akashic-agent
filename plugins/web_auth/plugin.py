"""Web 入口门禁：发布入口守卫与登录页；卸载即恢复无门状态，需要重启实例才放行。"""
from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
import logging
from pathlib import Path
import socket

import uvicorn

from agent.plugin_composition import Context
from core.common.file_io import run_file_io
from core.common.unix_socket import prepare_unix_socket, socket_alias_path

from .app import create_app
from .state import USERNAME, OwnerAuth

api_version = 3
name = "web-auth"
version = "1.0.0"
desc = "Web 入口门禁：主人账号密码、记住设备与本机一次性登录"
workspace_roots = ("web-auth",)
# 选中即要求入口守卫在场：本插件 apply 失败时 Web Shell 拒绝而不是放行。
ingress_guard = True
entrypoints = {"web-login": "cli.login_main", "web-passwd": "cli.passwd_main"}

logger = logging.getLogger(__name__)


class _Server(uvicorn.Server):
    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.ready = asyncio.Event()

    @contextmanager
    def capture_signals(self) -> Generator[None]:
        # 进程信号归外壳；插件服务器只响应自己的 Effect 关闭。
        yield

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets)
        self.ready.set()


async def apply(ctx: Context) -> None:
    """账号先于监听准备好；守卫端点最后发布、最先撤下。"""
    # 1. 首次启动创建账号；初始密码只写日志与 0600 文件，不推送到任何渠道。
    auth = OwnerAuth(ctx.workspace_root("web-auth"))
    initial = await run_file_io(auth.ensure)
    if initial is not None:
        logger.warning("[web.auth] 尚无账号，已创建 %s。初始密码 %s（首次登录后需要改成自己的；"
                       "也可在本机运行 web-login 免密进入）", USERNAME, initial)
    app = create_app(auth, Path(__file__).with_name("login.html"))
    path = socket_alias_path(ctx.runtime.workspace / "runtime", "web-auth.sock")
    server = _Server(uvicorn.Config(app, log_level="warning", lifespan="off", timeout_graceful_shutdown=5))

    # 2. Effect 持有监听器；关闭时先停接纳，再删除本次创建的 socket 节点。
    def setup():
        prepare_unix_socket(path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(path))
            listener.setblocking(False)
        except BaseException:
            listener.close()
            raise
        task = asyncio.create_task(server.serve(sockets=[listener]), name="web-auth-server")

        async def close() -> None:
            server.should_exit = True
            try:
                await task
            finally:
                listener.close()
                path.unlink(missing_ok=True)
        return close

    await ctx.effect(setup, label="web-auth.listener")
    await asyncio.wait_for(server.ready.wait(), 10)
    # 3. 撤下顺序与发布相反：守卫先撤，Web Shell 在同一 boot 内随即拒绝而非放行。
    await ctx.endpoint("pages", protocol="http+unix", address=str(path), routes=("/_auth",))
    await ctx.endpoint("guard", protocol="guard+unix", address=str(path))
