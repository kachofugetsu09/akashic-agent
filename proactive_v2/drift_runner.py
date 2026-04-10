from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES
from agent.tool_hooks import ShellRmToRestoreHook, ToolExecutionRequest, ToolExecutor
from proactive_v2.context import AgentTickContext
from proactive_v2.drift_state import DriftStateStore, SkillMeta
from proactive_v2.drift_tools import (
    DRIFT_TOOL_SCHEMAS,
    FORCED_FINISH_PROMPT,
    FORCED_WRITE_PROMPT,
    DriftToolDeps,
    dispatch,
)


LlmFn = Callable[[list[dict], list[dict], str | dict], Awaitable[dict | None]]
logger = logging.getLogger(__name__)


@dataclass
class DriftRunner:
    store: DriftStateStore
    tool_deps: DriftToolDeps
    max_steps: int = 20

    def __post_init__(self) -> None:
        self._tool_executor = ToolExecutor([ShellRmToRestoreHook()])

    async def run(self, ctx: AgentTickContext, llm_fn: LlmFn | None) -> bool:
        if llm_fn is None:
            logger.info("[drift] skip: llm_fn is None")
            return False
        skills = self.store.scan_skills()
        if not skills:
            logger.info("[drift] skip: no available skills under %s", self.store.skills_dir)
            return False
        logger.info(
            "[drift] enter: skills=%d max_steps=%d drift_dir=%s",
            len(skills),
            self.max_steps,
            self.store.drift_dir,
        )

        ctx.drift_entered = True
        ctx.drift_finished = False
        ctx.drift_message_sent = False

        messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt(skills)}
        ]
        steps = 0
        warned = False

        while steps < self.max_steps and not ctx.drift_finished:
            tool_choice: str | dict = "required"
            schemas = list(DRIFT_TOOL_SCHEMAS)

            if steps == self.max_steps - 3 and not warned:
                warned = True
                logger.info("[drift] forced-landing warning at step=%d", steps)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "你还剩 3 步。请整理本次执行结果，"
                            "下一步必须写入 working files，最后一步调用 finish_drift。"
                        ),
                    }
                )
            elif steps == self.max_steps - 2:
                logger.info("[drift] forced write at step=%d", steps)
                messages[0] = {"role": "system", "content": FORCED_WRITE_PROMPT}
                tool_choice = {
                    "type": "function",
                    "function": {"name": "writefile"},
                }
            elif steps == self.max_steps - 1:
                logger.info("[drift] forced finish at step=%d", steps)
                messages[0] = {"role": "system", "content": FORCED_FINISH_PROMPT}
                tool_choice = {
                    "type": "function",
                    "function": {"name": "finish_drift"},
                }

            if ctx.drift_message_sent:
                allowed_after_send = {"writefile", "finish_drift"}
                schemas = [
                    s for s in schemas
                    if s["function"]["name"] in allowed_after_send
                ]
                logger.info(
                    "[drift] send_message already used, restricting schema to writefile/finish_drift"
                )

            tool_call = await llm_fn(messages, schemas, tool_choice)
            if tool_call is None:
                logger.warning("[drift] llm returned no tool call at step=%d", steps)
                break
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("input", {})
            logger.info(
                "[drift] step=%d tool=%s args=%s",
                steps,
                tool_name,
                json.dumps(tool_args, ensure_ascii=False)[:200],
            )
            steps += 1
            ctx.steps_taken += 1
            result = await self._tool_executor.execute(
                ToolExecutionRequest(
                    call_id=str(tool_call.get("id") or f"drift_{steps}"),
                    tool_name=tool_name,
                    arguments=tool_args,
                    source="proactive",
                    session_key=ctx.session_key,
                ),
                lambda name, args: dispatch(name, args, ctx, self.tool_deps),
            )
            if result.status == "error":
                logger.warning("[drift] tool executor error at step=%d: %s", steps, result.output)
                break
            logger.info(
                "[drift] step=%d tool=%s result=%s",
                steps,
                tool_name,
                str(result.output)[:300],
            )
            self._append_tool_messages(
                messages,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=str(tool_call.get("id") or f"drift_{steps}"),
                result=str(result.output),
            )
        logger.info(
            "[drift] exit: finished=%s message_sent=%s steps=%d",
            ctx.drift_finished,
            ctx.drift_message_sent,
            steps,
        )
        return True

    def _build_system_prompt(self, skills: list[SkillMeta]) -> str:
        memory_text = ""
        if self.tool_deps.memory is not None:
            try:
                raw = str(self.tool_deps.memory.read_long_term_context() or "").strip()
                if raw:
                    memory_text = raw
            except Exception:
                memory_text = ""

        lines = []
        for skill in skills[:8]:
            next_text = skill.next[:80] if skill.next else ""
            line = f"- {skill.name}/   {skill.run_count}次运行"
            if next_text:
                line += f'   next: "{next_text}"'
            lines.append(line)
        skill_block = "\n".join(lines) if lines else "- (none)"

        recent_rows = []
        for row in self.store.load_drift().get("recent_runs", [])[-5:][::-1]:
            run_at = str(row.get("run_at") or "")
            try:
                dt = datetime.fromisoformat(run_at).astimezone(timezone.utc)
                time_text = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                time_text = run_at[:16]
            recent_rows.append(
                f"- {time_text}  {row.get('skill', '')}   {str(row.get('one_line', ''))[:150]}"
            )
        recent_block = "\n".join(recent_rows) if recent_rows else "- (none)"

        drift_note = str(self.store.load_drift().get("note") or "")[:150]

        return (
            f"{AKASHIC_IDENTITY}\n\n"
            f"{PERSONALITY_RULES}\n\n"
            "你现在有一段空闲时间（Drift 模式）。没有外部内容需要推送，\n"
            "你可以自主决定做一件有意义的事。\n\n"
            f"【Drift 工作区绝对路径】\n{self.store.drift_dir}\n\n"
            f"【用户长期记忆】\n{memory_text}\n\n"
            f"【可用 Drift Skills】\n{skill_block}\n\n"
            f"【最近的 Drift 记录】\n{recent_block}\n\n"
            f"【全局备注】\n{drift_note}\n\n"
            "【执行规则】\n"
            "1. 自主选择一个 skill，readfile 读它的 SKILL.md 了解细节。\n"
            "   标准路径格式是 skills/<skill_name>/...，例如 skills/explore-curiosity/SKILL.md。\n"
            "2. readfile 读该 skill 的 working files 了解当前进度。\n"
            "   working file 也优先使用 skills/<skill_name>/... 或 drift 工作区下的绝对路径。\n"
            "3. 执行任务。\n"
            "4. 有价值的发现必须立即 writefile，不要积累到最后再写。\n"
            "5. 单次 run 最多只能 send_message 一次。\n"
            "6. send_message 成功后不要再调用 get_recent_chat / recall_memory / web_fetch，"
            "后续只允许 writefile 和 finish_drift 收尾。\n"
            "7. 执行结束前必须调用 finish_drift 保存状态。\n\n"
            "【可用工具】\n"
            "readfile, writefile, recall_memory, web_fetch, get_recent_chat, send_message, finish_drift"
        )

    @staticmethod
    def _append_tool_messages(
        messages: list[dict],
        *,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        result: str,
    ) -> None:
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
