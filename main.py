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
from agent.memory import MemoryStore
from agent.tools.registry import ToolRegistry
from agent.scheduler import LatencyTracker, SchedulerService
from agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from agent.tools.message_push import MessagePushTool
from agent.tools.schedule import CancelScheduleTool, ListSchedulesTool, ScheduleTool
from agent.tools.shell import ShellTool
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from session.manager import SessionManager
from feeds.store import FeedStore
from feeds.registry import FeedRegistry
from feeds.rss import RSSFeedSource
from feeds.tools import FeedManageTool, FeedQueryTool
from proactive.loop import ProactiveLoop
from proactive.state import ProactiveStateStore
from proactive.presence import PresenceStore
from proactive.memory_optimizer import MemoryOptimizer, MemoryOptimizerLoop

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

def _build_agent(config: Config, workspace: Path) -> tuple[AgentLoop, MessageBus, ToolRegistry, MessagePushTool, SessionManager, SchedulerService, FeedRegistry, FeedStore, LLMProvider]:
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

    tracker = LatencyTracker()
    scheduler = SchedulerService(
        store_path=workspace / "schedules.json",
        push_tool=push_tool,
        agent_loop=None,   # set after loop is created
        tracker=tracker,
    )

    provider = LLMProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        system_prompt=config.system_prompt,
        extra_body=config.extra_body,
    )
    session_manager = SessionManager(workspace)
    presence = PresenceStore(workspace / "presence.json")
    loop = AgentLoop(
        bus=bus, provider=provider, tools=tools,
        session_manager=session_manager,
        workspace=workspace,
        model=config.model,
        max_iterations=config.max_iterations,
        max_tokens=config.max_tokens,
        presence=presence,
    )

    # Wire agent_loop back into scheduler (circular dependency resolved here)
    scheduler.agent_loop = loop

    # Register schedule tools
    tools.register(ScheduleTool(scheduler))
    tools.register(ListSchedulesTool(scheduler))
    tools.register(CancelScheduleTool(scheduler))

    # Feed store + registry
    feed_store = FeedStore(workspace / "feeds.json")
    feed_registry = FeedRegistry(feed_store)
    feed_registry.register_source_type("rss", lambda sub: RSSFeedSource(sub))

    # Register feed tools
    tools.register(FeedManageTool(feed_store))
    tools.register(FeedQueryTool(feed_store, feed_registry))

    return loop, bus, tools, push_tool, session_manager, scheduler, feed_registry, feed_store, provider


async def serve(config_path: str = "config.json") -> None:
    config = Config.load(config_path)
    workspace = Path.home() / ".akasic" / "workspace"
    agent_loop, bus, tools, push_tool, session_manager, scheduler, feed_registry, feed_store, provider = _build_agent(config, workspace)

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

    tasks = [
        agent_loop.run(),
        bus.dispatch_outbound(),
        scheduler.run(),
    ]

    memory_store = MemoryStore(workspace)
    presence = PresenceStore(workspace / "presence.json")

    if config.proactive.enabled:
        proactive_state = ProactiveStateStore(workspace / "proactive_state.json")
        proactive_loop = ProactiveLoop(
            feed_registry=feed_registry,
            session_manager=session_manager,
            provider=provider,
            push_tool=push_tool,
            config=config.proactive,
            model=config.model,
            max_tokens=config.max_tokens,
            state_store=proactive_state,
            memory_store=memory_store,
            presence=presence,
        )
        tasks.append(proactive_loop.run())

    # 每日 00:00 记忆质量优化 + 问题生成
    mem_optimizer = MemoryOptimizer(
        memory=memory_store,
        provider=provider,
        model=config.model,
    )
    tasks.append(MemoryOptimizerLoop(mem_optimizer).run())
    print("MemoryOptimizerLoop 已启动，每日 00:00 执行")

    try:
        await asyncio.gather(*tasks)
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
