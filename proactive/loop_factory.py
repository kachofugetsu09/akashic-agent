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

        subagent_factory = self._build_subagent_factory()
        runner = SkillActionRunner(
            registry,
            rng=self._rng,
            state_path=state_path,
            subagent_factory=subagent_factory,
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
        enabled_count = len(registry.list_enabled())
        logger.info(
            "[proactive] skill_action_runner 已初始化 path=%s enabled_actions=%d subagent_factory=%s",
            skill_path,
            enabled_count,
            "yes" if subagent_factory else "no",
        )
        return runner

    def _build_subagent_factory(self):
        from agent.subagent import SubAgent
        from agent.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
        from agent.tools.notify_owner import NotifyOwnerTool
        from agent.tools.task_note import TaskDoneTool, TaskNoteTool, TaskRecallTool
        from agent.tools.web_fetch import WebFetchTool
        from agent.tools.web_search import WebSearchTool
        from proactive.skill_action import _AGENT_SYSTEM_PROMPT

        workspace = getattr(self._sessions, "workspace", None)
        if workspace is None:
            logger.warning("[proactive] 无法获取 workspace，SubAgent 不可用")
            return None
        agent_tasks_dir = Path(workspace) / "agent-tasks"
        agent_tasks_dir.mkdir(parents=True, exist_ok=True)
        shared_dir = agent_tasks_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)

        channel = self._cfg.default_channel or ""
        chat_id = self._cfg.default_chat_id or ""
        provider = self._provider
        model = self._model
        max_tokens = self._max_tokens
        push = self._push
        db_path = agent_tasks_dir / "task_notes.db"
        shell_tool = _build_sandboxed_shell(agent_tasks_dir)

        def factory(
            action_id: str,
            system_prompt_override: str = _AGENT_SYSTEM_PROMPT,
            shared_config_dir: str = str(shared_dir),
        ) -> SubAgent:
            assert system_prompt_override is not None
            action_dir = agent_tasks_dir / action_id
            action_dir.mkdir(parents=True, exist_ok=True)
            tools = [
                WebSearchTool(),
                WebFetchTool(get_default_http_requester("external_default")),
                ReadFileTool(allowed_dir=agent_tasks_dir),
                ListDirTool(allowed_dir=agent_tasks_dir),
                WriteFileTool(allowed_dir=agent_tasks_dir),
                shell_tool,
                TaskNoteTool(db_path),
                TaskRecallTool(db_path),
                TaskDoneTool(action_dir),
                NotifyOwnerTool(push, channel, chat_id),
            ]
            return SubAgent(
                provider=provider,
                model=model,
                tools=tools,
                system_prompt=system_prompt_override,
                max_iterations=40,
                max_tokens=max_tokens,
                mandatory_exit_tools=["task_note", "notify_owner"],
            )

        return factory
