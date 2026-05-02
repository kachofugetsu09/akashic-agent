from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agent.config_models import Config
from agent.looping.core import AgentLoop
from agent.provider import LLMProvider
from agent.tool_hooks import ToolHook
from agent.tools.message_push import MessagePushTool
from proactive_v2.loop import ProactiveLoop
from proactive_v2.memory_optimizer import MemoryOptimizer, MemoryOptimizerLoop
from proactive_v2.presence import PresenceStore
from proactive_v2.state import ProactiveStateStore
from session.manager import SessionManager

if TYPE_CHECKING:
    from core.memory.profile import MemoryOptimizerStore, ProfileMaintenanceStore
    from core.memory.runtime_facade import MemoryRuntimeFacade


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
    tool_hooks: list[ToolHook] | None = None,
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
        shared_tools=getattr(agent_loop, "tools", None),
        tool_hooks=tool_hooks,
    )

    # 4. 主动链路本体以后台任务方式常驻运行。
    tasks.append(proactive_loop.run())

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
