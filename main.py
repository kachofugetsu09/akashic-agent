"""
入口

两种模式：
  python main.py          启动 agent 服务（AgentLoop + 所有 channel + IPC server）
  python main.py cli      连接到运行中的 agent（CLI 客户端）
"""
import asyncio
import logging
import sys
from pathlib import Path

from bus.queue import MessageBus
from agent.config import Config
from agent.loop import AgentLoop
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from agent.tools.message_push import MessagePushTool
from agent.tools.shell import ShellTool
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from session.manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
# 降低第三方库的噪音
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ── 服务端 ────────────────────────────────────────────────────────

def _build_agent(config: Config, workspace: Path) -> tuple[AgentLoop, MessageBus, ToolRegistry, MessagePushTool, SessionManager]:
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(ShellTool())
    tools.register(WebSearchTool())
    tools.register(WebFetchTool())
    tools.register(ReadFileTool())
    tools.register(WriteFileTool())
    tools.register(EditFileTool())
    tools.register(ListDirTool())
    push_tool = MessagePushTool()
    tools.register(push_tool)
    provider = LLMProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        system_prompt=config.system_prompt,
        extra_body=config.extra_body,
    )
    session_manager = SessionManager(workspace)
    loop = AgentLoop(
        bus=bus, provider=provider, tools=tools,
        session_manager=session_manager,
        workspace=workspace,
        model=config.model,
        max_iterations=config.max_iterations,
        max_tokens=config.max_tokens,
    )
    return loop, bus, tools, push_tool, session_manager


async def serve(config_path: str = "config.json") -> None:
    config = Config.load(config_path)
    workspace = Path.home() / ".akasic" / "workspace"
    agent_loop, bus, tools, push_tool, session_manager = _build_agent(config, workspace)

    from channels.ipc_server import IPCServerChannel
    ipc = IPCServerChannel(bus, config.channels.socket)
    await ipc.start()
    print(f"Agent 已启动  |  CLI 连接地址: {config.channels.socket}")

    tg_channel = None
    if config.channels.telegram:
        from channels.telegram_channel import TelegramChannel
        tg = config.channels.telegram
        tg_channel = TelegramChannel(token=tg.token, bus=bus, session_manager=session_manager, allow_from=tg.allow_from)
        await tg_channel.start()
        push_tool.register_channel("telegram", text=tg_channel.send, file=tg_channel.send_file, image=tg_channel.send_image)
        print(f"Telegram Bot 已启动")

    qq_channel = None
    if config.channels.qq:
        from channels.qq_channel import QQChannel
        qq = config.channels.qq
        qq_channel = QQChannel(bot_uin=qq.bot_uin, bus=bus, session_manager=session_manager, allow_from=qq.allow_from, groups=qq.groups)
        await qq_channel.start()
        push_tool.register_channel("qq", text=qq_channel.send, file=qq_channel.send_file, image=qq_channel.send_image)
        print(f"QQ Bot 已启动  |  QQ 号: {qq.bot_uin}")

    try:
        await asyncio.gather(
            agent_loop.run(),
            bus.dispatch_outbound(),
        )
    finally:
        await ipc.stop()
        if tg_channel:
            await tg_channel.stop()
        if qq_channel:
            await qq_channel.stop()


# ── 客户端 ────────────────────────────────────────────────────────

def connect_cli(config_path: str = "config.json") -> None:
    socket_path = Config.load(config_path).channels.socket
    try:
        from channels.cli_tui import run_tui
    except RuntimeError as exc:
        print(exc)
        print("回退到纯文本 CLI。")
        from channels.cli import CLIClient
        asyncio.run(CLIClient(socket_path).run())
        return

    run_tui(socket_path)


# ── 入口 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    config_path = "config.json"

    # 支持 python main.py [cli] [--config path]
    if "--config" in args:
        idx = args.index("--config")
        config_path = args[idx + 1]

    if not Path(config_path).exists():
        print(f"找不到配置文件 {config_path!r}，请先复制 config.example.json 为 config.json。")
        sys.exit(1)

    if "cli" in args:
        connect_cli(config_path)
    else:
        asyncio.run(serve(config_path))
