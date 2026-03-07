from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ProactiveLoopTriggerMixin:
    async def trigger_skill_action(
        self, action_id: str | None = None
    ) -> tuple[bool, str]:
        """
        手动触发一次 skill action，与正常 idle 路径完全一致。

        - 阻塞 proactive 正常 tick（唤醒正在等待的 sleep，跳过本轮 tick）
        - 若 action_id 指定，则直接执行该 action；否则按权重随机抽取
        - 返回 (success, message)：success=True 时 message 为结果描述；False 时为错误原因

        注意：同一时刻只允许一个手动触发并发执行，额外的调用会立即返回 (False, "busy")。
        """
        if not self._cfg.skill_actions_enabled:
            return False, "skill_actions 未启用（skill_actions_enabled=false）"

        runner = self._engine._skill_action_runner
        if runner is None:
            return False, "SkillActionRunner 未初始化"

        if self._manual_trigger_lock.locked():
            return False, "已有手动触发正在执行，请稍后再试"

        async with self._manual_trigger_lock:
            self._manual_trigger_event.set()
            now_utc = datetime.now(timezone.utc)

            if action_id:
                registry = runner._registry
                action = registry.get(action_id)
                if action is None:
                    return False, f"找不到 action_id={action_id!r}"
                if not action.enabled:
                    return False, f"action_id={action_id!r} 已禁用"
            else:
                action = runner.pick()
                if action is None:
                    return False, "当前无可用 skill action（配额已满或间隔未到）"

            logger.info(
                "[proactive] 手动触发 skill_action id=%s name=%r",
                action.id,
                action.name,
            )

            if self._cfg.anyaction_enabled and self._engine._anyaction:
                self._engine._anyaction.record_action(now_utc=now_utc)

            success, stdout_str = await runner.run(action)
            logger.info(
                "[proactive] 手动触发 skill_action 完成 id=%s success=%s",
                action.id,
                success,
            )

            if success and stdout_str:
                await self._engine._try_send_proactive_text(action.id, stdout_str)

            if success:
                return True, f"skill_action {action.id!r} 已完成"
            return False, f"skill_action {action.id!r} 执行失败"
