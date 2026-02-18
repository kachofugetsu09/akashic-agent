"""
CLI Channel
将标准输入/输出作为一个 channel 接入 MessageBus。
用法：python main.py
"""
import asyncio
import sys

from bus.events import InboundMessage
from bus.queue import MessageBus

_CHANNEL = "cli"
_CHAT_ID = "default"
_EXIT_CMDS = {"exit", "quit", "q"}


class CLIChannel:
    """
    CLI 是一种特殊的 channel：
    - stdin  → publish_inbound  → AgentLoop
    - AgentLoop → publish_outbound → stdout
    """

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus
        self._response_ready = asyncio.Event()
        self._last_response: str = ""
        bus.subscribe_outbound(_CHANNEL, self._on_response)

    async def run(self) -> None:
        _print_banner()
        while True:
            try:
                user_input = await _read_line()
            except (EOFError, KeyboardInterrupt):
                break

            text = user_input.strip()
            if not text:
                continue
            if text.lower() in _EXIT_CMDS:
                break

            # 发送消息，等待回复
            self._response_ready.clear()
            await self.bus.publish_inbound(InboundMessage(
                channel=_CHANNEL,
                sender="user",
                chat_id=_CHAT_ID,
                content=text,
            ))
            await self._response_ready.wait()
            print(f"\n{self._last_response}\n")

        print("再见！")

    async def _on_response(self, msg) -> None:
        self._last_response = msg.content
        self._response_ready.set()


# ── 工具函数 ──────────────────────────────────────────────────────

def _print_banner() -> None:
    print("Akasic Agent  |  输入 exit 退出\n")


async def _read_line() -> str:
    """在线程池中读取 stdin，不阻塞事件循环"""
    loop = asyncio.get_event_loop()
    sys.stdout.write("> ")
    sys.stdout.flush()
    return await loop.run_in_executor(None, sys.stdin.readline)
