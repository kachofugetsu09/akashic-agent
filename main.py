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
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from core.memory.port import MemoryPort
    from core.memory.runtime import MemoryRuntime

from bus.processing import ProcessingState
from bus.queue import MessageBus
from agent.config import Config
from agent.loop import AgentLoop
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from agent.scheduler import LatencyTracker, SchedulerService
from agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from agent.tools.message_push import MessagePushTool
from agent.tools.schedule import CancelScheduleTool, ListSchedulesTool, ScheduleTool
from agent.tools.shell import ShellTool
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from agent.tools.skill_action_tool import (
    SkillActionListTool,
    SkillActionRegisterTool,
    SkillActionResetTool,
    SkillActionRestartTool,
    SkillActionRewriteTool,
    SkillActionStatusTool,
    SkillActionUnregisterTool,
    SkillActionUpdateTool,
)
from agent.tools.fitbit import FitbitHealthSnapshotTool, FitbitSleepReportTool
from agent.tools.update_now import UpdateNowTool
from agent.mcp.registry import McpServerRegistry
from agent.mcp.manage_tools import McpAddTool, McpRemoveTool, McpListTool
from session.manager import SessionManager
from feeds.store import FeedStore
from feeds.registry import FeedRegistry
from feeds.rss import RSSFeedSource
from feeds.novel import NovelKBFeedSource
from feeds.tools import FeedManageTool, FeedQueryTool
from proactive.loop import ProactiveLoop
from proactive.feed_poller import FeedPoller
from proactive.state import ProactiveStateStore
from proactive.presence import PresenceStore
from proactive.schedule import ScheduleStore
from proactive.memory_optimizer import MemoryOptimizer, MemoryOptimizerLoop
from core.memory.runtime import MemoryRuntime
from memory2.post_response_worker import PostResponseMemoryWorker
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)

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

logger = logging.getLogger(__name__)


# ── 服务端 ────────────────────────────────────────────────────────


def build_providers(config: Config) -> tuple[LLMProvider, LLMProvider | None]:
    """
    构建主 LLM provider 与轻量 provider。

    # 1. 构建主 provider
    # 2. 构建轻量 provider（若已配置独立 key/url）
    # 3. 返回 (provider, light_provider)
    """
    # 1. 构建主 provider
    provider = LLMProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        system_prompt=config.system_prompt,
        extra_body=config.extra_body,
        request_timeout_s=180.0,
    )

    # 2. 构建轻量 provider（若已配置独立 key/url，否则 AgentLoop 内部降级到主 provider）
    light_provider: LLMProvider | None = None
    if config.light_model and (config.light_api_key or config.light_base_url):
        _light_url = config.light_base_url or config.base_url or ""
        # Gemini 不支持 enable_thinking，只对非 Google 端点传该字段
        _light_extra: dict = (
            {}
            if "googleapis.com" in _light_url or "generativelanguage" in _light_url
            else {"enable_thinking": False}
        )
        light_provider = LLMProvider(
            api_key=config.light_api_key or config.api_key,
            base_url=config.light_base_url or config.base_url,
            system_prompt=config.system_prompt,
            extra_body=_light_extra,
        )

    # 3. 返回
    return provider, light_provider


async def _run_cleanup_steps(
    *steps: tuple[str, Callable[[], Awaitable[None]]]
) -> None:
    """Run shutdown steps in order, continuing after failures."""
    first_error: Exception | None = None
    for name, step in steps:
        try:
            await step()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            logger.warning("shutdown step failed: %s: %s", name, exc)
    if first_error is not None:
        raise first_error


async def _noop_async() -> None:
    return None


def build_memory_runtime(
    config: Config,
    workspace: Path,
    tools: ToolRegistry,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    http_resources: SharedHttpResources,
) -> MemoryRuntime:
    """
    初始化 memory v2（向量检索体系），并注册相关工具。

    # 1. 检查 memory_v2 开关
    # 2. 初始化 store / embedder / memorizer / retriever
    # 3. 注册 MemorizeTool、WriteFileTool、EditFileTool（含 SopIndexer）
    # 4. 返回 MemoryRuntime
    """
    from agent.memory import MemoryStore
    from core.memory.port import DefaultMemoryPort

    store = MemoryStore(workspace)

    # 1. 检查开关
    if not config.memory_v2.enabled:
        tools.register(WriteFileTool())
        tools.register(EditFileTool())
        return MemoryRuntime(port=DefaultMemoryPort(store))

    # 2. 初始化 store / embedder / memorizer / retriever
    from memory2.store import MemoryStore2
    from memory2.embedder import Embedder
    from memory2.memorizer import Memorizer
    from memory2.retriever import Retriever
    from agent.tools.memorize import MemorizeTool

    db_path = (
        Path(config.memory_v2.db_path)
        if config.memory_v2.db_path
        else workspace / "memory" / "memory2.db"
    )
    mem2_store = MemoryStore2(db_path)
    embedder = Embedder(
        base_url=config.light_base_url or config.base_url or "",
        api_key=config.light_api_key or config.api_key,
        model=config.memory_v2.embed_model,
        requester=http_resources.external_default,
    )
    memorizer = Memorizer(mem2_store, embedder)
    retriever = Retriever(
        mem2_store,
        embedder,
        top_k=config.memory_v2.retrieve_top_k,
        score_threshold=config.memory_v2.score_threshold,
        score_thresholds={
            "procedure": config.memory_v2.score_threshold_procedure,
            "preference": config.memory_v2.score_threshold_preference,
            "event": config.memory_v2.score_threshold_event,
            "profile": config.memory_v2.score_threshold_profile,
        },
        relative_delta=config.memory_v2.relative_delta,
        inject_max_chars=config.memory_v2.inject_max_chars,
        inject_max_forced=config.memory_v2.inject_max_forced,
        inject_max_procedure_preference=config.memory_v2.inject_max_procedure_preference,
        inject_max_event_profile=config.memory_v2.inject_max_event_profile,
        sop_guard_enabled=config.memory_v2.sop_guard_enabled,
    )

    port = DefaultMemoryPort(store, memorizer=memorizer, retriever=retriever)
    post_mem_worker = PostResponseMemoryWorker(
        memorizer=memorizer,
        retriever=retriever,
        light_provider=light_provider or provider,
        light_model=config.light_model or config.model,
    )

    # 3. 注册工具（含 SopIndexer）
    tools.register(MemorizeTool(port))
    from memory2.sop_indexer import SopIndexer

    sop_indexer = SopIndexer(mem2_store, embedder, workspace / "sop")
    tools.register(WriteFileTool(sop_indexer=sop_indexer))
    tools.register(EditFileTool(sop_indexer=sop_indexer))

    # 4. 返回
    return MemoryRuntime(
        port=port,
        post_response_worker=post_mem_worker,
        sop_indexer=sop_indexer,
        closeables=[mem2_store, embedder],
    )


def build_core_runtime(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
) -> tuple[
    AgentLoop,
    MessageBus,
    ToolRegistry,
    MessagePushTool,
    SessionManager,
    SchedulerService,
    LLMProvider,
    LLMProvider | None,
    McpServerRegistry,
    "MemoryRuntime",
    PresenceStore,
]:
    """
    构建核心运行时：消息总线、工具注册表、AgentLoop、调度器、MCP。

    # 1. 初始化总线与基础工具集
    # 2. 构建 providers
    # 3. 初始化 memory runtime（v2 工具注册）
    # 4. 初始化调度器
    # 5. 构建 AgentLoop
    # 6. 注册调度工具与 MCP 工具
    # 7. 返回
    """
    # 1. 初始化总线与基础工具集
    bus = MessageBus()
    tools = ToolRegistry()
    tools.register(ShellTool())
    tools.register(WebSearchTool())
    tools.register(WebFetchTool(http_resources.external_default))
    tools.register(ReadFileTool())
    tools.register(ListDirTool())
    push_tool = MessagePushTool()
    tools.register(push_tool)

    skill_actions_path = workspace / "skill_actions.json"
    agent_tasks_dir = workspace / "agent-tasks"
    db_path = agent_tasks_dir / "task_notes.db"
    tools.register(
        SkillActionRegisterTool(skill_actions_path, agent_tasks_dir=agent_tasks_dir)
    )
    tools.register(SkillActionUnregisterTool(skill_actions_path))
    tools.register(
        SkillActionListTool(skill_actions_path, agent_tasks_dir=agent_tasks_dir)
    )
    tools.register(SkillActionStatusTool(agent_tasks_dir))
    tools.register(SkillActionUpdateTool(agent_tasks_dir))
    tools.register(SkillActionRestartTool(agent_tasks_dir, db_path=db_path))
    tools.register(SkillActionResetTool(agent_tasks_dir))
    tools.register(SkillActionRewriteTool(agent_tasks_dir, db_path=db_path))

    _fitbit_url = getattr(config.proactive, "fitbit_url", "http://127.0.0.1:18765")
    if getattr(config.proactive, "fitbit_enabled", False):
        tools.register(
            FitbitHealthSnapshotTool(
                _fitbit_url,
                requester=http_resources.local_service,
            )
        )
        tools.register(
            FitbitSleepReportTool(
                _fitbit_url,
                requester=http_resources.local_service,
            )
        )

    # 2. 构建 providers
    provider, light_provider = build_providers(config)

    # 3. 初始化 memory runtime
    memory_runtime = build_memory_runtime(
        config,
        workspace,
        tools,
        provider,
        light_provider,
        http_resources,
    )
    tools.register(UpdateNowTool(memory_runtime.port))

    # 4. 初始化调度器（agent_loop 暂为 None，后续回填）
    tracker = LatencyTracker()
    scheduler = SchedulerService(
        store_path=workspace / "schedules.json",
        push_tool=push_tool,
        agent_loop=None,
        tracker=tracker,
    )

    # 5. 构建 AgentLoop
    session_manager = SessionManager(workspace)
    presence = PresenceStore(workspace / "presence.json")
    processing_state = ProcessingState()

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        tools=tools,
        session_manager=session_manager,
        workspace=workspace,
        model=config.model,
        max_iterations=config.max_iterations,
        max_tokens=config.max_tokens,
        presence=presence,
        light_model=config.light_model,
        light_provider=light_provider,
        processing_state=processing_state,
        memory_top_k_procedure=config.memory_v2.top_k_procedure,
        memory_top_k_history=config.memory_v2.top_k_history,
        memory_route_intention_enabled=config.memory_v2.route_intention_enabled,
        memory_sop_guard_enabled=config.memory_v2.sop_guard_enabled,
        memory_gate_llm_timeout_ms=config.memory_v2.gate_llm_timeout_ms,
        memory_gate_max_tokens=config.memory_v2.gate_max_tokens,
        memory_runtime=memory_runtime,
    )

    # 6. 回填 agent_loop，注册调度与 MCP 工具
    scheduler.agent_loop = loop
    tools.register(ScheduleTool(scheduler))
    tools.register(ListSchedulesTool(scheduler))
    tools.register(CancelScheduleTool(scheduler))

    mcp_registry = McpServerRegistry(
        config_path=workspace / "mcp_servers.json",
        tool_registry=tools,
    )
    tools.register(McpAddTool(mcp_registry))
    tools.register(McpRemoveTool(mcp_registry))
    tools.register(McpListTool(mcp_registry))

    # 7. 返回
    return (
        loop,
        bus,
        tools,
        push_tool,
        session_manager,
        scheduler,
        provider,
        light_provider,
        mcp_registry,
        memory_runtime,
        presence,
    )


def build_feed_runtime(
    workspace: Path,
    tools: ToolRegistry,
    http_resources: SharedHttpResources,
) -> tuple[FeedRegistry, FeedStore, FeedManageTool]:
    """
    构建订阅系统：FeedStore、FeedRegistry 与相关工具。

    # 1. 初始化 FeedStore 与 FeedRegistry
    # 2. 注册内置 source 类型
    # 3. 注册 Feed 工具
    # 4. 返回
    """
    # 1. 初始化存储与注册表
    feed_store = FeedStore(workspace / "feeds.json")
    feed_registry = FeedRegistry(feed_store)

    # 2. 注册内置 source 类型
    feed_registry.register_source_type(
        "rss",
        lambda sub: RSSFeedSource(sub, requester=http_resources.feed_fetcher),
    )
    feed_registry.register_source_type("novel-kb", lambda sub: NovelKBFeedSource(sub))

    # 3. 注册 Feed 工具（scorer 由 proactive runtime 稍后注入）
    feed_manage_tool = FeedManageTool(feed_store)
    tools.register(feed_manage_tool)
    tools.register(FeedQueryTool(feed_store, feed_registry))

    # 4. 返回
    return feed_registry, feed_store, feed_manage_tool


def build_proactive_runtime(
    config: Config,
    workspace: Path,
    *,
    feed_registry: FeedRegistry,
    feed_store: FeedStore,
    feed_manage_tool: FeedManageTool,
    session_manager: SessionManager,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    push_tool: MessagePushTool,
    memory_store: "MemoryPort | None" = None,
    presence: PresenceStore,
    agent_loop: AgentLoop,
) -> tuple[list, "ProactiveLoop | None"]:
    """
    构建主动循环运行时，返回 (需要并发运行的 asyncio 任务列表, ProactiveLoop 实例)。
    若 proactive 未启用则返回 ([], None)。

    # 1. 检查 proactive 开关
    # 2. 初始化 ProactiveLoop
    # 3. 注入 SourceScorer 到 FeedManageTool
    # 4. 组装 FeedPoller（若启用）
    # 5. 添加可选的 fitbit monitor
    # 6. 返回任务列表
    """
    tasks: list = []

    # 1. 检查开关
    if not config.proactive.enabled:
        return tasks, None

    # 2. 初始化 ProactiveLoop
    proactive_state = ProactiveStateStore(workspace / "proactive_state.json")
    schedule_store = ScheduleStore(workspace / "schedule.json")
    proactive_cfg = config.proactive
    if proactive_cfg.skill_actions_enabled and not proactive_cfg.skill_actions_path:
        proactive_cfg.skill_actions_path = str(workspace / "skill_actions.json")

    proactive_loop = ProactiveLoop(
        feed_registry=feed_registry,
        session_manager=session_manager,
        provider=provider,
        push_tool=push_tool,
        config=proactive_cfg,
        model=config.model,
        max_tokens=config.max_tokens,
        state_store=proactive_state,
        memory_store=memory_store,
        presence=presence,
        schedule=schedule_store,
        light_provider=light_provider,
        light_model=config.light_model,
        feed_store=feed_store,
        passive_busy_fn=agent_loop.processing_state.is_busy
        if agent_loop.processing_state
        else None,
    )
    tasks.append(proactive_loop.run())

    # 3. 注入 SourceScorer 到 FeedManageTool
    if proactive_loop._source_scorer is not None:
        feed_manage_tool.set_scorer(proactive_loop._source_scorer)

    # 4. 启动 FeedPoller（若启用）
    if config.proactive.feed_poller_enabled and proactive_loop.feed_buffer is not None:
        feed_poller = FeedPoller(
            feed_registry,
            proactive_loop.feed_buffer,
            config.proactive,
            source_scorer=proactive_loop._source_scorer,
            feed_store=feed_store,
            memory_provider=memory_store,
        )
        tasks.append(feed_poller.run())
        print(
            f"FeedPoller 已启动  |  间隔={config.proactive.feed_poller_interval_seconds}s"
            + (f"  source_scorer=enabled" if proactive_loop._source_scorer else "")
        )

    # 5. 添加可选的 fitbit monitor
    _fitbit_path = getattr(config.proactive, "fitbit_monitor_path", "").strip()
    if config.proactive.fitbit_enabled and _fitbit_path:
        from proactive.fitbit_sleep import run_fitbit_monitor

        tasks.append(run_fitbit_monitor(_fitbit_path, config.proactive.fitbit_url))
        print(f"fitbit-monitor 已启动  |  路径={_fitbit_path}")

    # 6. 返回任务列表和 ProactiveLoop 实例
    return tasks, proactive_loop


async def serve(config_path: str = "config.json") -> None:
    # 1. 加载配置
    config = Config.load(config_path)
    print(f"[DEBUG] config_path={config_path} route_intention={config.memory_v2.route_intention_enabled}", flush=True)
    workspace = Path.home() / ".akasic" / "workspace"
    http_resources = SharedHttpResources()
    configure_default_shared_http_resources(http_resources)
    ipc = None
    tg_channel = None
    qq_channel = None
    memory_runtime = None
    try:
        # 2. 构建核心运行时
        (
            agent_loop,
            bus,
            tools,
            push_tool,
            session_manager,
            scheduler,
            provider,
            light_provider,
            mcp_registry,
            memory_runtime,
            presence,
        ) = build_core_runtime(config, workspace, http_resources)
        await mcp_registry.load_and_connect_all()

        # 3. 构建订阅系统
        feed_registry, feed_store, feed_manage_tool = build_feed_runtime(
            workspace,
            tools,
            http_resources,
        )

        # 4. 启动 IPC 服务
        from channels.ipc_server import IPCServerChannel

        ipc = IPCServerChannel(bus, config.channels.socket)
        await ipc.start()
        print(f"Agent 已启动  |  CLI 连接地址: {config.channels.socket}")

        # 5. 启动各渠道
        if config.channels.telegram:
            from channels.telegram_channel import TelegramChannel

            tg = config.channels.telegram
            tg_channel = TelegramChannel(
                token=tg.token,
                bus=bus,
                session_manager=session_manager,
                allow_from=tg.allow_from,
            )
            await tg_channel.start()
            push_tool.register_channel(
                "telegram",
                text=tg_channel.send,
                file=tg_channel.send_file,
                image=tg_channel.send_image,
            )
            print(f"Telegram Bot 已启动")

        if config.channels.qq:
            from channels.qq_channel import QQChannel

            qq = config.channels.qq
            qq_channel = QQChannel(
                bot_uin=qq.bot_uin,
                bus=bus,
                session_manager=session_manager,
                allow_from=qq.allow_from,
                groups=qq.groups,
                http_requester=http_resources.external_default,
            )
            await qq_channel.start()
            push_tool.register_channel(
                "qq",
                text=qq_channel.send,
                file=qq_channel.send_file,
                image=qq_channel.send_image,
            )
            print(f"QQ Bot 已启动  |  QQ 号: {qq.bot_uin}")

        # 6. 组装并发任务
        tasks = [
            agent_loop.run(),
            bus.dispatch_outbound(),
            scheduler.run(),
        ]

        proactive_tasks, proactive_loop = build_proactive_runtime(
            config,
            workspace,
            feed_registry=feed_registry,
            feed_store=feed_store,
            feed_manage_tool=feed_manage_tool,
            session_manager=session_manager,
            provider=provider,
            light_provider=light_provider,
            push_tool=push_tool,
            memory_store=memory_runtime.port,
            presence=presence,
            agent_loop=agent_loop,
        )
        tasks.extend(proactive_tasks)

        # 将 ProactiveLoop 注入 IPC server，以支持手动触发 skill action
        if proactive_loop is not None:
            ipc.set_proactive_loop(proactive_loop)

        # 7. 启动记忆优化器
        if config.memory_optimizer_enabled:
            mem_optimizer = MemoryOptimizer(
                memory=memory_runtime.port,
                provider=provider,
                model=config.model,
            )
            interval = config.memory_optimizer_interval_seconds
            tasks.append(
                MemoryOptimizerLoop(mem_optimizer, interval_seconds=interval).run()
            )
            print(f"MemoryOptimizerLoop 已启动，间隔={interval}s ({interval / 3600:.1f}h)")
        else:
            print("MemoryOptimizerLoop 已禁用（memory_optimizer_enabled=false）")

        # 8. 运行直到退出，清理资源
        await asyncio.gather(*tasks)
    finally:
        try:
            await _run_cleanup_steps(
                ("ipc.stop", ipc.stop if ipc else _noop_async),
                ("telegram.stop", tg_channel.stop if tg_channel else _noop_async),
                ("qq.stop", qq_channel.stop if qq_channel else _noop_async),
                (
                    "memory_runtime.aclose",
                    memory_runtime.aclose if memory_runtime else _noop_async,
                ),
                ("http_resources.aclose", http_resources.aclose),
            )
        finally:
            clear_default_shared_http_resources(http_resources)


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
        print(
            f"找不到配置文件 {config_path!r}，请先复制 config.example.json 为 config.json。"
        )
        sys.exit(1)

    if "cli" in args:
        connect_cli(config_path)
    else:
        asyncio.run(serve(config_path))
