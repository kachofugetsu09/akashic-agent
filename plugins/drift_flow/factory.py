from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from plugins.drift_flow.runtime import DriftTurnPipeline, DriftTurnPipelineDeps
from plugins.drift_flow.state import DriftStateStore
from plugins.drift_flow.tools import DriftToolDeps
from proactive_v2.runtime_scope import ProactiveRuntimeScope


logger = logging.getLogger(__name__)
JsonObject = dict[str, Any]
LlmFn = Callable[
    [list[JsonObject], list[JsonObject], str | JsonObject, bool],
    Awaitable[JsonObject | None],
]
RecentChatFn = Callable[[int], Awaitable[list[JsonObject]]]


def build_drift_llm_fn(scope: ProactiveRuntimeScope) -> LlmFn:
    """构造完整 Drift 与主动链路共用的模型调用适配器。"""

    agent_model = scope.cfg.agent_tick_model or scope.model
    provider = scope.provider

    async def llm_fn(
        messages: list[JsonObject],
        schemas: list[JsonObject],
        tool_choice: str | JsonObject = "auto",
        disable_thinking: bool = False,
    ) -> JsonObject | None:
        _ = disable_thinking
        response = await provider.chat(
            messages=messages,
            tools=schemas,
            model=agent_model,
            max_tokens=scope.max_tokens,
            tool_choice=tool_choice,
            disable_thinking=True,
        )
        if not response.tool_calls:
            text = (response.content or "").strip()
            logger.warning(
                "[drift] llm_fn: no tool call returned (text=%r)",
                text[:300] if text else "(empty)",
            )
            return None
        call = response.tool_calls[0]
        return {
            "id": call.id,
            "name": call.name,
            "input": call.arguments,
            "_cache_prompt_tokens": response.cache_prompt_tokens,
            "_cache_hit_tokens": response.cache_hit_tokens,
        }

    return llm_fn


def build_drift_recent_chat_fn(scope: ProactiveRuntimeScope) -> RecentChatFn:
    """按 Default 的真实 Sensor 读取方式构造近期对话函数。"""

    async def recent_chat_fn(n: int = 20) -> list[JsonObject]:
        _ = n
        return await asyncio.get_running_loop().run_in_executor(
            None,
            scope.sense.collect_recent,
        )

    return recent_chat_fn


def build_drift_pipeline(
    scope: ProactiveRuntimeScope,
    recent_chat_fn: RecentChatFn,
) -> DriftTurnPipeline | None:
    """用统一依赖构造 Default 与 Wake 共用的完整 Drift pipeline。"""

    if not scope.cfg.drift_enabled:
        return None

    # 1. 收集当前插件 generation 提供的 Drift skill 根目录
    from agent.plugins.snapshot import get_current_runtime_snapshot

    snapshot = get_current_runtime_snapshot()
    plugin_skill_roots = (
        tuple(
            root
            for generation in snapshot.active_generations()
            for root in generation.contributions.drift_skill_roots
        )
        if snapshot is not None
        else ()
    )

    # 2. 绑定同一套工作区、工具、hooks 和事件总线
    workspace = Path(scope.state_store.workspace_dir)
    drift_dir = workspace / "drift"
    store = DriftStateStore(drift_dir, plugin_skill_roots=plugin_skill_roots)
    return DriftTurnPipeline(
        DriftTurnPipelineDeps(
            store=store,
            tool_deps=DriftToolDeps(
                drift_dir=drift_dir,
                store=store,
                workspace_dir=workspace,
                memory=scope.memory,
                recent_chat_fn=recent_chat_fn,
                shared_tools=scope.shared_tools,
                event_bus=scope.event_bus,
            ),
            max_steps=scope.cfg.drift_max_steps,
            tool_hooks=scope.tool_hooks,
        )
    )
