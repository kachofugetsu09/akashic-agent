"""Uvicorn Server whose graceful shutdown starts as soon as exit is requested."""

from __future__ import annotations

import asyncio
import socket

import uvicorn


class PromptShutdownServer(uvicorn.Server):
    """保持 uvicorn 的优雅关闭语义，只去掉退出请求的轮询延迟和空闲固定等待。"""

    def __init__(self, config: uvicorn.Config) -> None:
        self._exit_loop: asyncio.AbstractEventLoop | None = None
        self._exit_wake: asyncio.Event | None = None
        self._exit_requested = False
        super().__init__(config)

    @property
    def should_exit(self) -> bool:
        return self._exit_requested

    # 调用方仍只写 should_exit；可能来自其他线程，所以经 loop 线程安全地唤醒。
    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._exit_requested = value
        loop, wake = self._exit_loop, self._exit_wake
        if value and loop is not None and wake is not None and not loop.is_closed():
            _ = loop.call_soon_threadsafe(wake.set)

    # 未请求退出时保持 uvicorn 每 0.1s 一次的 on_tick 节拍。
    async def main_loop(self) -> None:
        self._exit_wake = asyncio.Event()
        self._exit_loop = asyncio.get_running_loop()
        counter = 0
        should_exit = await self.on_tick(counter)
        while not should_exit:
            counter = (counter + 1) % 864000
            try:
                await asyncio.wait_for(self._exit_wake.wait(), timeout=0.1)
            except TimeoutError:
                pass
            should_exit = await self.on_tick(counter)

    # 1. 先停止接纳新连接。
    # 2. 仍有连接或后台任务时完全沿用 uvicorn 的等待与超时取消。
    # 3. 空闲时 uvicorn 的固定 0.1s 等待没有等待对象，直接完成其余关闭步骤。
    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        for server in self.servers:
            server.close()
        if self.server_state.connections or self.server_state.tasks:
            await super().shutdown(sockets)
            return
        for sock in sockets or []:
            sock.close()
        for server in self.servers:
            await server.wait_closed()
        if not self.force_exit:
            await self.lifespan.shutdown()
