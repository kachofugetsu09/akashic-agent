from __future__ import annotations

import json
import logging
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, cast

from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES
from agent.tool_hooks import ShellRmToRestoreHook, ToolExecutionRequest, ToolExecutor
from proactive_v2.context import AgentTickContext
from proactive_v2.drift_state import DriftStateStore, SkillMeta
from proactive_v2.drift_tools import (
    FORCED_FINISH_PROMPT,
    FORCED_WRITE_PROMPT,
    DriftToolDeps,
    build_drift_tool_registry,
)


LlmFn = Callable[[list[dict], list[dict], str | dict, bool], Awaitable[dict | None]]
StepRecorder = Callable[[AgentTickContext, str, str, str, dict[str, Any], str], None]
logger = logging.getLogger(__name__)


@dataclass
class DriftRunner:
    store: DriftStateStore
    tool_deps: DriftToolDeps
    max_steps: int = 20
    step_recorder: StepRecorder | None = None

    def __post_init__(self) -> None:
        self._tool_executor = ToolExecutor([ShellRmToRestoreHook()])

    async def run(self, ctx: AgentTickContext, llm_fn: LlmFn | None) -> bool:
        if llm_fn is None:
            logger.info("[drift] skip: llm_fn is None")
            return False
        skills = self.store.scan_skills()
        if not skills:
            logger.info("[drift] skip: no available drift skills")
            return False

        # 过滤掉 requires_mcp 未满足的 skill
        shared = self.tool_deps.shared_tools
        connected_servers = shared.get_mcp_server_names() if shared else set()
        skills = [
            s for s in skills
            if not s.requires_mcp or set(s.requires_mcp) <= connected_servers
        ]
        if not skills:
            logger.info("[drift] skip: all skills require unavailable MCP servers")
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

        mounted_tool_names: set[str] = set()
        tools = build_drift_tool_registry(
            ctx=ctx, deps=self.tool_deps, mounted_tool_names=mounted_tool_names,
        )
        base_schemas = tools.get_schemas()

        messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt(skills, connected_servers)}
        ]
        steps = 0
        warned = False

        while steps < self.max_steps and not ctx.drift_finished:
            tool_choice: str | dict = "required"
            schemas = list(base_schemas)
            # 拼接已挂载 MCP 工具的 schema
            if mounted_tool_names and shared:
                schemas += shared.get_schemas(names=mounted_tool_names)

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
                schemas = [
                    s for s in schemas
                    if s["function"]["name"] in {"write_file", "edit_file"}
                ]
                tool_choice = "required"
            elif steps == self.max_steps - 1:
                logger.info("[drift] forced finish at step=%d", steps)
                messages[0] = {"role": "system", "content": FORCED_FINISH_PROMPT}
                tool_choice = {
                    "type": "function",
                    "function": {"name": "finish_drift"},
                }

            if ctx.drift_message_sent:
                allowed_after_send = {"write_file", "edit_file", "finish_drift"}
                schemas = [
                    s for s in schemas
                    if s["function"]["name"] in allowed_after_send
                ]
                logger.info(
                    "[drift] message_push already used, restricting schema to write_file/edit_file/finish_drift"
                )

            if "disable_thinking" in inspect.signature(llm_fn).parameters:
                tool_call = await cast(Any, llm_fn)(
                    messages,
                    schemas,
                    tool_choice,
                    disable_thinking=True,
                )
            else:
                tool_call = await cast(Any, llm_fn)(messages, schemas, tool_choice)
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

            # 双路分发：本地 drift registry → shared registry (mounted MCP tools)
            if tools.has_tool(tool_name):
                exec_fn = tools.execute
            elif tool_name in mounted_tool_names and shared:
                exec_fn = shared.execute
            else:
                # 未知工具，走本地 registry 让它返回错误
                exec_fn = tools.execute

            result = await self._tool_executor.execute(
                ToolExecutionRequest(
                    call_id=str(tool_call.get("id") or f"drift_{steps}"),
                    tool_name=tool_name,
                    arguments=tool_args,
                    source="proactive",
                    session_key=ctx.session_key,
                ),
                exec_fn,
            )
            if result.status == "error":
                logger.warning("[drift] tool executor error at step=%d: %s", steps, result.output)
                if self.step_recorder is not None:
                    self.step_recorder(
                        ctx,
                        "drift:error",
                        tool_name,
                        str(tool_call.get("id") or f"drift_{steps}"),
                        tool_args,
                        str(result.output),
                    )
                break
            if self.step_recorder is not None:
                self.step_recorder(
                    ctx,
                    "drift",
                    tool_name,
                    str(tool_call.get("id") or f"drift_{steps}"),
                    tool_args,
                    str(result.output),
                )
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

    def _build_system_prompt(
        self, skills: list[SkillMeta], connected_servers: set[str] | None = None,
    ) -> str:
        memory_text = ""
        recent_context_text = ""
        if self.tool_deps.memory is not None:
            try:
                raw = str(self.tool_deps.memory.read_long_term_context() or "").strip()
                if raw:
                    memory_text = raw
            except Exception:
                memory_text = ""
            try:
                rc = str(self.tool_deps.memory.read_recent_context() or "").strip()
                if rc:
                    recent_context_text = rc
            except Exception:
                pass

        lines = []
        for skill in skills[:8]:
            next_text = skill.next[:80] if skill.next else ""
            line = f"- {skill.name}/   {skill.run_count}次运行"
            if skill.builtin:
                line += "   [builtin]"
            if next_text:
                line += f'   next: "{next_text}"'
            if skill.requires_mcp:
                line += f"   [需要: {', '.join(skill.requires_mcp)}]"
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

        # 动态生成可挂载 MCP server 目录（只列 server 名和工具数，不展开工具名）
        mcp_block = ""
        shared = self.tool_deps.shared_tools
        if connected_servers and shared:
            mcp_lines = []
            for srv in sorted(connected_servers):
                tool_count = len(shared.get_tool_names_by_source("mcp", srv))
                mcp_lines.append(f"- {srv}（{tool_count} 个工具）")
            mcp_block = (
                "【可挂载的外部能力】\n"
                + "\n".join(mcp_lines) + "\n"
                "使用 mount_server(server=\"名称\") 挂载后即可调用其中的工具。\n\n"
            )

        return (
            f"{AKASHIC_IDENTITY}\n\n"
            f"{PERSONALITY_RULES}\n\n"
            "你现在有一段空闲时间（Drift 模式）。没有外部内容需要推送，\n"
            "你可以自主决定做一件有意义的事。\n\n"
            f"【Drift 工作区绝对路径】\n{self.store.drift_dir}\n\n"
            f"【用户长期记忆】\n{memory_text}\n\n"
            f"【近期交互上下文】\n{recent_context_text or '（空）'}\n\n"
            f"【可用 Drift Skills】\n{skill_block}\n\n"
            f"【最近的 Drift 记录】\n{recent_block}\n\n"
            f"【全局备注】\n{drift_note}\n\n"
            "【执行规则】\n"
            "1. 每次进入 Drift 都先重新比较所有可用 skill，不要因为某个 skill 最近刚运行过，"
            "或它的 next 很明确，就默认继续它。\n"
            "   只有当它仍然是当前最值得做的事时，才继续它；如果别的 skill 更久没运行、"
            "更有价值、或更适合当前空档时间，优先选别的 skill。\n"
            "2. 自主选择一个 skill，read_file 读它的 SKILL.md 了解细节。\n"
            "   标准路径格式是 skills/<skill_name>/...，例如 skills/explore-curiosity/SKILL.md。\n"
            "   这个路径同时适用于 drift 工作区 skill 和内建 drift builtin skill。\n"
            "3. read_file 读该 skill 的 working files 了解当前进度。\n"
            "   working file 也优先使用 skills/<skill_name>/... 或 drift 工作区下的绝对路径。\n"
            "4. 读完 skill 和 working files 后，要执行这个 skill 当前最直接的下一步动作，"
            "不要只因为看到了 queue、next 或等待描述，就立刻 finish_drift。\n"
            "   如果这个 skill 当前明显处于“等待用户回复/等待外部条件”的状态，"
            "就不要选它，改选别的 skill。\n"
            "5. 只有在本轮已经完成了一个明确动作后，或确认该 skill 当前确实无事可做时，"
            "才允许 finish_drift。\n"
            "6. 有价值的发现必须立即 write_file 或 edit_file，不要积累到最后再写。\n"
            "7. 如果你决定 message_push，对用户的表达要像此刻自然想到的一句聊天，"
            "而不是像在执行队列、候选列表、记忆检索或内部流程。\n"
            "   先把内部依据转写成自然联想，再说出口：像突然想到、顺着刚才的感觉延伸、"
            "隐约记得用户会偏好什么、或此刻真的有点好奇。\n"
            "   目标是让用户感受到你是真想聊这个，而不是在汇报你为什么会想到它。\n"
            "8. 单次 run 最多只能 message_push 一次。\n"
            "9. message_push 成功后不要再调用 recall_memory / web_fetch / web_search / fetch_messages / search_messages / shell，"
            "后续只允许 write_file、edit_file 和 finish_drift 收尾。\n"
            "10. 执行结束前必须调用 finish_drift 保存状态。\n\n"
            f"{mcp_block}"
            "【可用工具】\n"
            "read_file, write_file, edit_file, recall_memory, web_fetch, web_search, "
            "fetch_messages, search_messages, shell, message_push, finish_drift"
            + (", mount_server" if mcp_block else "")
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
                "content": f"调用工具 {tool_name}",
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
