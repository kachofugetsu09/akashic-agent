"""
drift_runner — 向后兼容适配层。

DriftRunner 主逻辑已迁移至 agent.core.drift_turn.DriftTurnPipeline。
此模块提供旧 API 兼容：DriftRunner(store=..., tool_deps=..., ...) 和 .run(ctx, llm_fn)。
"""

from __future__ import annotations

from typing import Any

from agent.core.drift_turn import (
    DriftTurnPipeline,
    DriftTurnPipelineDeps,
    LlmFn,
    StepRecorder,
)
from proactive_v2.context import AgentTickContext
from proactive_v2.drift_state import DriftStateStore, SkillMeta
from proactive_v2.drift_tools import DriftToolDeps


class DriftRunner:
    """向后兼容包装器：旧 kwarg 构造 API → 新的 DriftTurnPipeline。"""

    def __init__(
        self,
        *,
        store: DriftStateStore,
        tool_deps: DriftToolDeps,
        max_steps: int = 20,
        step_recorder: StepRecorder | None = None,
        tool_hooks: list[Any] | None = None,
    ) -> None:
        self._pipeline = DriftTurnPipeline(
            DriftTurnPipelineDeps(
                store=store,
                tool_deps=tool_deps,
                max_steps=max_steps,
                step_recorder=step_recorder,
                tool_hooks=list(tool_hooks or []),
            )
        )

    async def run(self, ctx: AgentTickContext, llm_fn: LlmFn | None) -> bool:
        return await self._pipeline.run(ctx, llm_fn)

    @property
    def step_recorder(self) -> StepRecorder | None:
        return self._pipeline.step_recorder

    @step_recorder.setter
    def step_recorder(self, value: StepRecorder | None) -> None:
        self._pipeline.step_recorder = value

    def _build_system_prompt(self) -> str:
        return self._pipeline._build_system_prompt()

    def _build_runtime_context_message(
        self,
        skills: list[SkillMeta],
        connected_servers: set[str] | None = None,
    ) -> dict[str, str]:
        return self._pipeline._build_runtime_context_message(skills, connected_servers)
