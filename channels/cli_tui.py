"""
Textual CLI 客户端

连接到运行中的 agent 实例（通过 Unix socket），提供 TUI 交互界面。
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime

from agent.config import DEFAULT_SOCKET

try:
    from rich.markdown import Markdown
    from rich.text import Text
    from textual import on
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Input, RichLog, Static
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "缺少依赖 textual，请先安装：.venv/bin/pip install textual"
    ) from exc


@dataclass
class TUIStats:
    sent: int = 0
    received: int = 0
    tool_calls: int = 0


class HeaderBar(Horizontal):
    def compose(self) -> ComposeResult:
        yield Static("# Akasic Agent CLI / Textual", id="title")
        yield Static("0 sent  0 recv  0 tools  connected: no", id="meta")


class FooterBar(Horizontal):
    def compose(self) -> ComposeResult:
        yield Static("Akasic (Textual TUI)", id="footer-left")
        yield Static(
            "enter send   ctrl+c quit   ctrl+l clear   kitty: Shift+drag copy",
            id="footer-right",
        )


class CLITextualApp(App[None]):
    CSS = """
    Screen {
      background: #0b0f14;
      color: #d6d9df;
    }

    #root {
      height: 100%;
      width: 100%;
      padding: 1 2;
    }

    HeaderBar {
      height: 3;
      border: solid #2f3640;
      background: #11161d;
      padding: 0 1;
      align: center middle;
    }

    #title {
      width: 1fr;
      content-align: left middle;
      text-style: bold;
      color: #e8eaee;
    }

    #meta {
      width: auto;
      color: #838c99;
      content-align: right middle;
    }

    #log {
      height: 1fr;
      margin: 1 0;
      border: solid #2f3640;
      background: #0d1117;
      scrollbar-color: #3a4757 #0d1117;
    }

    Input {
      height: 3;
      border: solid #2f3640;
      background: #141922;
      color: #f5f7fb;
      padding: 0 1;
    }

    Input:focus {
      border: solid #16c7d9;
      background: #18222d;
    }

    FooterBar {
      height: 2;
      margin-top: 1;
      border-top: solid #2a313b;
    }

    #footer-left {
      width: 1fr;
      color: #32d5ff;
      text-style: bold;
    }

    #footer-right {
      width: auto;
      color: #7b8594;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear log"),
    ]

    def __init__(self, socket_path: str = DEFAULT_SOCKET) -> None:
        super().__init__()
        self.socket_path = socket_path
        self.stats = TUIStats()
        self.connected = False
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            yield HeaderBar()
            yield RichLog(
                id="log", highlight=False, markup=False, auto_scroll=True, wrap=True
            )
            yield Input(placeholder="输入消息并回车发送...", id="input")
            yield FooterBar()

    async def on_mount(self) -> None:
        self._write_system_message("正在连接 agent ...")
        self.run_worker(
            self._connect_and_receive(), name="socket-worker", exclusive=True
        )
        self.query_one(Input).focus()

    async def on_unmount(self) -> None:
        await self._close_stream()

    @on(Input.Submitted)
    async def _on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.lower() in {"q", "quit", "exit"}:
            self.exit()
            return
        if not self._writer:
            self._write_system_message("尚未连接到 agent，消息未发送。")
            return

        payload = json.dumps({"content": text}, ensure_ascii=False) + "\n"
        try:
            self._writer.write(payload.encode("utf-8"))
            await self._writer.drain()
        except Exception as exc:
            self._write_system_message(f"发送失败: {exc}")
            await self._close_stream()
            return

        self.stats.sent += 1
        self._write_user_message(text)
        self._refresh_header()

    def action_clear_log(self) -> None:
        self.query_one(RichLog).clear()
        self._write_system_message("日志已清空。")

    async def _connect_and_receive(self) -> None:
        try:
            reader, writer = await asyncio.open_unix_connection(self.socket_path)
        except FileNotFoundError, ConnectionRefusedError:
            self._write_system_message(f"无法连接到 agent: {self.socket_path}")
            self._write_system_message("请先启动主进程: python main.py")
            self.connected = False
            self._refresh_header()
            return
        except Exception as exc:
            self._write_system_message(f"连接失败: {exc}")
            self.connected = False
            self._refresh_header()
            return

        self._reader = reader
        self._writer = writer
        self.connected = True
        self._refresh_header()
        self._write_system_message(f"连接成功: {self.socket_path}")

        while True:
            try:
                line = await reader.readline()
            except Exception as exc:
                self._write_system_message(f"接收异常: {exc}")
                break
            if not line:
                self._write_system_message("连接已断开。")
                break

            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                raw = line.decode("utf-8", errors="ignore").strip()
                self._write_system_message(f"RAW: {raw}")
                continue

            metadata = (
                data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            )
            tool_chain = (
                metadata.get("tool_chain")
                if isinstance(metadata.get("tool_chain"), list)
                else []
            )
            if tool_chain:
                self._write_tool_chain(tool_chain)

            content = str(data.get("content", "")).strip()
            if content:
                self.stats.received += 1
                self._write_agent_message(content)
                self._refresh_header()

        await self._close_stream()

    async def _close_stream(self) -> None:
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer:
            writer.close()
            await writer.wait_closed()
        if self.connected:
            self.connected = False
            self._refresh_header()

    def _refresh_header(self) -> None:
        state = "yes" if self.connected else "no"
        self.query_one("#meta", Static).update(
            f"{self.stats.sent} sent  {self.stats.received} recv  {self.stats.tool_calls} tools  connected: {state}"
        )

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _write_user_message(self, message: str) -> None:
        log = self.query_one(RichLog)
        log.write(Text(f"[{self._ts()}] USER", style="bold #00d1ff"))
        log.write(Text(f"  {message}", style="#d8f8ff"))

    def _write_agent_message(self, message: str) -> None:
        log = self.query_one(RichLog)
        log.write(Text(f"[{self._ts()}] AGENT", style="bold #ffcd4a"))
        log.write(Markdown(message, code_theme="monokai"))

    def _write_system_message(self, message: str) -> None:
        log = self.query_one(RichLog)
        log.write(Text(f"[{self._ts()}] SYSTEM", style="bold #9aa4b2"))
        log.write(Text(f"  {message}", style="#bec6d2"))

    def _write_tool_chain(self, tool_chain: list[dict]) -> None:
        log = self.query_one(RichLog)
        for item in tool_chain:
            calls = item.get("calls", [])
            if not isinstance(calls, list):
                continue
            for call in calls:
                name = str(call.get("name", "unknown"))
                self.stats.tool_calls += 1
                log.write(
                    Text(
                        f"[{self._ts()}] TOOL {name}  (collapsed)", style="bold #7dff9f"
                    )
                )
        self._refresh_header()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def run_tui(socket_path: str = DEFAULT_SOCKET) -> None:
    # 默认启用鼠标（滚动条可用）；需要复制优先时可设置 AKASIC_TUI_MOUSE=0。
    mouse_enabled = _env_bool("AKASIC_TUI_MOUSE", True)
    CLITextualApp(socket_path).run(mouse=mouse_enabled)
