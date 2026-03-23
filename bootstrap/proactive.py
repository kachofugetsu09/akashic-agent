from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agent.config_models import Config
from agent.looping.core import AgentLoop
from agent.provider import LLMProvider
from agent.tools.message_push import MessagePushTool
from proactive.loop import ProactiveLoop
from proactive.memory_optimizer import MemoryOptimizer, MemoryOptimizerLoop
from proactive.presence import PresenceStore
from proactive.state import ProactiveStateStore
from session.manager import SessionManager

if TYPE_CHECKING:
    from core.memory.port import MemoryPort


def build_proactive_runtime(
    config: Config,
    workspace: Path,
    *,
    session_manager: SessionManager,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    push_tool: MessagePushTool,
    memory_store: "MemoryPort | None",
    presence: PresenceStore,
    agent_loop: AgentLoop,
    observe_writer=None,
) -> tuple[list, ProactiveLoop | None]:
    tasks: list = []
    if not config.proactive.enabled:
        return tasks, None

    proactive_state = ProactiveStateStore(workspace / "proactive_state.json")
    proactive_cfg = config.proactive
    if proactive_cfg.skill_actions_enabled and not proactive_cfg.skill_actions_path:
        proactive_cfg.skill_actions_path = str(workspace / "skill_actions.json")

    proactive_loop = ProactiveLoop(
        session_manager=session_manager,
        provider=provider,
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
        tool_registry=getattr(agent_loop, "tools", None),
    )
    tasks.append(proactive_loop.run())

    fitbit_path = getattr(config.proactive, "fitbit_monitor_path", "").strip()
    if config.proactive.fitbit_enabled and fitbit_path:
        from proactive.fitbit_sleep import run_fitbit_monitor

        tasks.append(run_fitbit_monitor(fitbit_path, config.proactive.fitbit_url))
        print(f"fitbit-monitor 已启动  |  路径={fitbit_path}")

    return tasks, proactive_loop


def build_memory_optimizer_task(
    config: Config,
    *,
    provider: LLMProvider,
    memory_store: "MemoryPort",
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
