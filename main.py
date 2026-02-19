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
from agent.tools.shell import ShellTool
from agent.tools.web_fetch import WebFetchTool
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

def _build_agent(config: Config, workspace: Path) -> tuple[AgentLoop, MessageBus, ToolRegistry]:
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(ShellTool())
    tools.register(WebFetchTool())
    provider = LLMProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        system_prompt=config.system_prompt,
    )
    session_manager = SessionManager(workspace)
    loop = AgentLoop(
        bus=bus, provider=provider, tools=tools,
        session_manager=session_manager,
        model=config.model,
        max_iterations=config.max_iterations,
        max_tokens=config.max_tokens,
    )
    return loop, bus, tools


async def serve(config_path: str = "config.json") -> None:
    config = Config.load(config_path)
    workspace = Path(config_path).parent / "workspace"
    agent_loop, bus, tools = _build_agent(config, workspace)

    from channels.ipc_server import IPCServerChannel
    ipc = IPCServerChannel(bus, config.channels.socket)
    await ipc.start()
    print(f"Agent 已启动  |  CLI 连接地址: {config.channels.socket}")

    tg_channel = None
    if config.channels.telegram:
        from channels.telegram_channel import TelegramChannel
        from agent.tools.telegram_push import TelegramPushTool
        tg = config.channels.telegram
        tg_channel = TelegramChannel(token=tg.token, bus=bus, allow_from=tg.allow_from)
        await tg_channel.start()
        # 注册推送工具，共享 bot 实例和 user_map 引用
        tools.register(TelegramPushTool(bot=tg_channel.bot, user_map=tg_channel.user_map))
        print(f"Telegram Bot 已启动")

    try:
        await asyncio.gather(
            agent_loop.run(),
            bus.dispatch_outbound(),
        )
    finally:
        await ipc.stop()
        if tg_channel:
            await tg_channel.stop()


# ── 客户端 ────────────────────────────────────────────────────────

async def connect_cli(config_path: str = "config.json") -> None:
    from channels.cli import CLIClient
    socket_path = Config.load(config_path).channels.socket
    await CLIClient(socket_path).run()


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
        asyncio.run(connect_cli(config_path))
    else:
        asyncio.run(serve(config_path))
