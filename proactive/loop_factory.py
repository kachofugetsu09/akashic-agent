from __future__ import annotations

import logging
from pathlib import Path

from core.net.http import get_default_http_requester
from proactive.loop_helpers import _build_sandboxed_shell

logger = logging.getLogger("proactive.loop")


class ProactiveLoopFactoryMixin:
    def _build_skill_action_runner(self):
        if not self._cfg.skill_actions_enabled:
            return None
        skill_path_str = (self._cfg.skill_actions_path or "").strip()
        if not skill_path_str:
            return None
        skill_path = Path(skill_path_str).expanduser()
        state_path = (
            self._state.path.parent / "skill_action_state.json"
            if hasattr(self._state, "path")
            else None
        )
        from proactive.skill_action import SkillActionRegistry, SkillActionRunner

        registry = SkillActionRegistry(skill_path)
        workspace = getattr(self._sessions, "workspace", None)
        agent_tasks_dir = (Path(workspace) / "agent-tasks") if workspace else None
        if agent_tasks_dir:
            agent_tasks_dir.mkdir(parents=True, exist_ok=True)
        return SkillActionRunner(
            registry,
            rng=self._rng,
            state_path=state_path,
            subagent_factory=self._build_subagent_factory(),
            agent_tasks_dir=agent_tasks_dir,
            memory_retrieve_fn=(
                self._memory.retrieve_related if self._memory is not None else None
            ),
            memory_format_fn=(
                self._memory.format_injection_block
                if self._memory is not None
                else None
            ),
        )

    def _build_subagent_factory(self):
        from agent.background.subagent_profiles import (
            SubagentRuntime,
            build_skill_action_spec,
        )
        from prompts.background import SKILL_ACTION_AGENT_BASE_PROMPT

        workspace = getattr(self._sessions, "workspace", None)
        if workspace is None:
            logger.warning("[proactive] 无法获取 workspace，SubAgent 不可用")
            return None
        agent_tasks_dir = Path(workspace) / "agent-tasks"
        agent_tasks_dir.mkdir(parents=True, exist_ok=True)
        runtime = SubagentRuntime(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
            memory=self._memory,
        )
        db_path = agent_tasks_dir / "task_notes.db"
        shell_tool = _build_sandboxed_shell(agent_tasks_dir)
        fetch_requester = get_default_http_requester("external_default")

        def factory(
            action_id: str,
            system_prompt_override: str = SKILL_ACTION_AGENT_BASE_PROMPT,
        ):
            action_dir = agent_tasks_dir / action_id
            action_dir.mkdir(parents=True, exist_ok=True)
            spec = build_skill_action_spec(
                agent_tasks_dir=agent_tasks_dir,
                action_dir=action_dir,
                fetch_requester=fetch_requester,
                shell_tool=shell_tool,
                db_path=db_path,
                push_tool=self._push,
                channel=self._cfg.default_channel or "",
                chat_id=self._cfg.default_chat_id or "",
                system_prompt=system_prompt_override,
                max_iterations=40,
            )
            return spec.build(runtime)

        return factory


__all__ = [
    "ProactiveLoopFactoryMixin",
    "_build_sandboxed_shell",
    "get_default_http_requester",
]
