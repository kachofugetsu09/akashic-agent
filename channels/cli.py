"""
CLI Channel（客户端）

连接到运行中的 agent 实例（通过 Unix socket），提供交互式命令行界面。
启动方式：python main.py cli
"""

import asyncio
import json
import sys

from agent.config import DEFAULT_SOCKET

_EXIT_CMDS = {"exit", "quit", "q"}


class CLIClient:
    """连接到已运行的 agent 实例"""

    def __init__(self, socket_path: str = DEFAULT_SOCKET) -> None:
        self._socket_path = socket_path

    async def run(self) -> None:
        try:
            reader, writer = await asyncio.open_unix_connection(self._socket_path)
        except FileNotFoundError, ConnectionRefusedError:
            print(
                f"无法连接到 agent（{self._socket_path}），请先启动主进程：python main.py"
            )
            return

        _print_banner()

        # 后台持续接收 agent 回复
        receive_task = asyncio.create_task(self._receive(reader))

        try:
            while True:
                text = await _read_line()
                stripped = text.strip()
                if stripped.lower() in _EXIT_CMDS:
                    break
                if not stripped:
                    continue
                payload = json.dumps({"content": stripped}, ensure_ascii=False) + "\n"
                writer.write(payload.encode())
                await writer.drain()
        except KeyboardInterrupt, EOFError:
            pass
        finally:
            receive_task.cancel()
            writer.close()
            print("\n再见！")

    @staticmethod
    async def _receive(reader: asyncio.StreamReader) -> None:
        while True:
            line = await reader.readline()
            if not line:
                print("\n连接已断开")
                break
            data = json.loads(line)
            print(f"\n{data['content']}\n> ", end="", flush=True)


# ── 工具函数 ──────────────────────────────────────────────────────


def _print_banner() -> None:
    print("Akasic Agent CLI  |  输入 exit 退出\n")


async def _read_line() -> str:
    loop = asyncio.get_event_loop()
    sys.stdout.write("> ")
    sys.stdout.flush()
    return await loop.run_in_executor(None, sys.stdin.readline)
