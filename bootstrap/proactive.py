from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agent.config_models import Config
from agent.looping.core import AgentLoop
from agent.provider import LLMProvider
from agent.tools.message_push import MessagePushTool
from proactive_v2.loop import ProactiveLoop
from proactive_v2.memory_optimizer import MemoryOptimizer, MemoryOptimizerLoop
from proactive_v2.presence import PresenceStore
from proactive_v2.state import ProactiveStateStore
from session.manager import SessionManager

if TYPE_CHECKING:
    from bus.event_bus import EventBus
    from core.memory.profile import MemoryOptimizerStore, ProfileMaintenanceStore
    from core.memory.runtime_facade import MemoryRuntimeFacade


_FITBIT_DEFAULT_URL = "http://127.0.0.1:18765"
_FITBIT_DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "fitbit-monitor"
)


def _build_proactive_provider(config: Config, provider: LLMProvider) -> LLMProvider:
    api_key = str(getattr(config, "api_key", "") or "").strip()
    system_prompt = str(getattr(config, "system_prompt", "") or "")
    base_url = getattr(config, "base_url", None)
    if not api_key:
        return provider

    extra_body = dict(getattr(config, "extra_body", {}) or {})
    extra_body.pop("enable_thinking", None)
    return LLMProvider(
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        extra_body=extra_body,
        provider_name=str(getattr(config, "provider", "") or ""),
        force_disable_thinking=True,
    )


def build_proactive_runtime(
    config: Config,
    workspace: Path,
    *,
    session_manager: SessionManager,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    push_tool: MessagePushTool,
    memory_store: "MemoryRuntimeFacade | None",
    presence: PresenceStore,
    agent_loop: AgentLoop,
    observe_writer=None,
    event_bus: "EventBus | None" = None,
) -> tuple[list, ProactiveLoop | None]:
    tasks: list = []
    # 1. 总开关关闭时，主动链路完全不启动。
    if not config.proactive.enabled:
        return tasks, None

    # 2. 先准备 proactive 独立状态存储和配置快照。
    proactive_state = ProactiveStateStore(workspace / "proactive.db")
    proactive_cfg = config.proactive
    proactive_provider = _build_proactive_provider(config, provider)

    # 3. 构建 ProactiveLoop。
    #    这里把主动链路需要的外部依赖一次性注入进去：
    #    session / provider / push_tool / memory / presence / passive_busy_fn。
    proactive_loop = ProactiveLoop(
        session_manager=session_manager,
        provider=proactive_provider,
        push_tool=push_tool,
        config=proactive_cfg,
        model=config.model,
        max_tokens=config.max_tokens,
        state_store=proactive_state,
        memory_store=memory_store,
        presence=presence,
        light_provider=light_provider,
        light_model=config.light_model,
        passive_busy_fn=(
            agent_loop.processing_state.is_busy if agent_loop.processing_state else None
        ),
        observe_writer=observe_writer,
        event_bus=event_bus,
        shared_tools=getattr(agent_loop, "tools", None),
        fitbit_enabled=config.fitbit.enabled,
        fitbit_url=_FITBIT_DEFAULT_URL,
    )

    # 4. 主动链路本体以后台任务方式常驻运行。
    tasks.append(proactive_loop.run())

    if config.fitbit.enabled:
        from proactive_v2.fitbit_sleep import run_fitbit_monitor

        # 5. 可选挂载 fitbit 监控子任务，给主动链路提供睡眠上下文。
        tasks.append(run_fitbit_monitor(_FITBIT_DEFAULT_PATH, _FITBIT_DEFAULT_URL))
        print(f"fitbit-monitor 已启动  |  路径={_FITBIT_DEFAULT_PATH}")

    return tasks, proactive_loop


def build_memory_optimizer_task(
    config: Config,
    *,
    provider: LLMProvider,
    memory_store: "MemoryOptimizerStore",
) -> list:
    if not config.memory_optimizer_enabled:
        print("MemoryOptimizerLoop 已禁用（memory_optimizer_enabled=false）")
        return []

    mem_optimizer = MemoryOptimizer(
        memory=memory_store,
        provider=provider,
        model=config.model,
    )
    interval = config.memory_optimizer_interval_seconds
    print(f"MemoryOptimizerLoop 已启动，间隔={interval}s ({interval / 3600:.1f}h)")
    return [MemoryOptimizerLoop(mem_optimizer, interval_seconds=interval).run()]
