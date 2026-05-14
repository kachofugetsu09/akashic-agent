"""
DriftTurnPipeline — Drift 空闲时间链路顶层抽象。

设计对齐主动链路的 ProactiveTurnPipeline.run() 和被动链路的 PassiveTurnPipeline.run()：
通过 run() 一个方法可见全链路。

┌─ tick trigger (no content available)
│  └─ DriftTurnPipeline.run()
│     ├─ 1. Scan      扫描可用 skills，过滤 MCP 未满足的
│     ├─ 2. Prepare   构建 tool registry 与初始 messages
│     ├─ 3. Execute   LLM 工具调用循环（drift steps）
│     └─ 4. Finish    记录退出状态
└─ done

段之间通过 AgentTickContext 传递状态，每段各司其职，不跨段直接访问对方内部实现。
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES
from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
)
from agent.tool_hooks import ToolExecutionRequest, ToolExecutor
from agent.tool_hooks.base import ToolHook
from proactive_v2.context import AgentTickContext
from proactive_v2.drift_state import DriftStateStore, SkillMeta
from proactive_v2.drift_tools import (
    DriftToolDeps,
    build_drift_tool_registry,
)

if TYPE_CHECKING:
    from core.memory.markdown import MemoryProfileApi

LlmFn = Callable[[list[dict], list[dict], str | dict, bool], Awaitable[dict | None]]
StepRecorder = Callable[[AgentTickContext, str, str, str, dict[str, Any], str], None]
logger = logging.getLogger(__name__)


# ── Pipeline 依赖容器 ─────────────────────────────────────────────────────

@dataclass
class DriftTurnPipelineDeps:
    store: DriftStateStore
    tool_deps: DriftToolDeps
    max_steps: int = 20
    step_recorder: StepRecorder | None = None
    tool_hooks: list[ToolHook] = field(default_factory=list)


# ── 主 Pipeline ─────────────────────────────────────────────────────────

# Drift 空闲时间链路核心入口，串起 Scan → Prepare → Execute → Finish 四段。
#
# ┌─ tick trigger (no content)
# │  └─ DriftTurnPipeline.run
# │     ├─ 1. Scan ── _scan_skills
# │     │  └─ store.scan_skills → MCP 过滤 → 空则 skip
# │     ├─ 2. Prepare ── _prepare
# │     │  └─ 设置 ctx drift flags → build_drift_tool_registry → 构建 messages
# │     ├─ 3. Execute ── _execute_loop
# │     │  └─ while steps < max_steps: llm_fn → tool execute → append → record
# │     │     message_push 后约束 schema 为 write_file/edit_file/finish_drift
# │     └─ 4. Finish ── _finish
# │        └─ 记录退出状态日志
# └─ done

class DriftTurnPipeline:

    def __init__(self, deps: DriftTurnPipelineDeps) -> None:
        self._store = deps.store
        self._tool_deps = deps.tool_deps
        self._max_steps = deps.max_steps
        self.step_recorder = deps.step_recorder
        self._tool_executor = ToolExecutor(deps.tool_hooks)

    # ── 入口 ──────────────────────────────────────────────────────────

    # 核心方法：处理一次 drift tick，串起 Scan → Prepare → Execute → Finish 四段链路。
    async def run(self, ctx: AgentTickContext, llm_fn: LlmFn | None) -> bool:
        # 1. llm_fn 为空 → 无法进入 Execute，直接退出。
        if llm_fn is None:
            logger.info("[drift] skip: llm_fn is None")
            return False

        # 2. Scan — 扫描可用 skills，过滤 MCP 不满足的，空则 skip。
        skills = self._scan_skills()
        if not skills:
            return False

        # 3. Prepare — 构建 tool registry 与初始 messages。
        tools, messages, mounted_tool_names = self._prepare(ctx, skills)

        # 4. Execute — LLM 工具调用循环。
        await self._execute_loop(ctx, llm_fn, tools, messages, mounted_tool_names)

        # 5. Finish — 记录退出。
        self._finish(ctx)
        return True

    # ── 1. Scan ───────────────────────────────────────────────────────

    def _scan_skills(self) -> list[SkillMeta]:
        """扫描可用 skills，过滤掉 requires_mcp 未满足的。"""

        skills = self._store.scan_skills()
        if not skills:
            logger.info("[drift] skip: no available drift skills")
            return []

        shared = self._tool_deps.shared_tools
        connected_servers = shared.get_mcp_server_names() if shared else set()
        skills = [
            s for s in skills
            if not s.requires_mcp or set(s.requires_mcp) <= connected_servers
        ]
        if not skills:
            logger.info("[drift] skip: all skills require unavailable MCP servers")
            return []

        logger.info(
            "[drift] enter: skills=%d max_steps=%d drift_dir=%s",
            len(skills),
            self._max_steps,
            self._store.drift_dir,
        )
        return skills

    # ── 2. Prepare ────────────────────────────────────────────────────

    def _prepare(
        self,
        ctx: AgentTickContext,
        skills: list[SkillMeta],
    ) -> tuple[Any, list[dict], set[str]]:
        """设置 ctx drift 标志、构建 tool registry 与初始 messages。"""

        # 2.1 设置 ctx 标志位。
        ctx.drift_entered = True
        ctx.drift_finished = False
        ctx.drift_message_sent = False

        # 2.2 构建 drift tool registry。
        mounted_tool_names: set[str] = set()
        tools = build_drift_tool_registry(
            ctx=ctx,
            deps=self._tool_deps,
            mounted_tool_names=mounted_tool_names,
        )

        # 2.3 确定 MCP 已连接 server 列表。
        shared = self._tool_deps.shared_tools
        connected_servers = shared.get_mcp_server_names() if shared else set()

        # 2.4 构建初始 messages。
        messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt()},
            self._build_runtime_context_message(skills, connected_servers),
        ]

        return tools, messages, mounted_tool_names

    # ── 3. Execute ────────────────────────────────────────────────────

    async def _execute_loop(
        self,
        ctx: AgentTickContext,
        llm_fn: LlmFn,
        tools: Any,
        messages: list[dict],
        mounted_tool_names: set[str],
    ) -> None:
        """LLM 工具调用循环：调模型 → 执行工具 → 追加 messages → 重复。"""

        shared = self._tool_deps.shared_tools
        base_schemas = tools.get_schemas()
        steps = 0

        while steps < self._max_steps and not ctx.drift_finished:
            tool_choice: str | dict = "required"
            schemas = list(base_schemas)

            # 3.1 拼接已挂载 MCP 工具的 schema。
            if mounted_tool_names and shared:
                schemas += shared.get_schemas(names=mounted_tool_names)

            # 3.2 message_push 后约束工具集。
            if ctx.drift_message_sent:
                allowed_after_send = {"write_file", "edit_file", "finish_drift"}
                schemas = [
                    s for s in schemas
                    if s["function"]["name"] in allowed_after_send
                ]
                logger.info(
                    "[drift] message_push already used, "
                    "restricting schema to write_file/edit_file/finish_drift"
                )

            # 3.3 调 LLM 拿工具调用。
            if "disable_thinking" in inspect.signature(llm_fn).parameters:
                tool_call = await cast(Any, llm_fn)(
                    messages, schemas, tool_choice,
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

            # 3.4 双路分发：本地 drift registry → shared registry（mounted MCP tools）。
            if tools.has_tool(tool_name):
                exec_fn = tools.execute
            elif tool_name in mounted_tool_names and shared:
                exec_fn = shared.execute
            else:
                exec_fn = tools.execute

            # 3.5 执行工具。
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

            # 3.6 错误处理。
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

            # 3.7 记录步骤。
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

            # 3.8 追加 tool messages 到对话历史。
            self._append_tool_messages(
                messages,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=str(tool_call.get("id") or f"drift_{steps}"),
                result=str(result.output),
            )

    # ── 4. Finish ──────────────────────────────────────────────────────

    def _finish(self, ctx: AgentTickContext) -> None:
        """记录 drift 退出状态。"""
        logger.info(
            "[drift] exit: finished=%s message_sent=%s",
            ctx.drift_finished,
            ctx.drift_message_sent,
        )

    # ── Prompt 构建 ────────────────────────────────────────────────────

    def _build_runtime_context_message(
        self,
        skills: list[SkillMeta],
        connected_servers: set[str] | None = None,
    ) -> dict[str, str]:
        """构建 runtime context frame，包含记忆、skill 列表、近期 run 记录。"""

        memory_text = ""
        recent_context_text = ""
        if self._tool_deps.memory is not None:
            memory = cast("MemoryProfileApi", self._tool_deps.memory)
            try:
                raw = str(memory.read_long_term() or "").strip()
                if raw:
                    memory_text = raw
            except Exception:
                memory_text = ""
            try:
                rc = str(memory.read_recent_context() or "").strip()
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
        for row in self._store.load_drift().get("recent_runs", [])[-5:][::-1]:
            run_at = str(row.get("run_at") or "")
            try:
                dt = datetime.fromisoformat(run_at).astimezone(timezone.utc)
                time_text = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                time_text = run_at[:16]
            recent_rows.append(
                f"- {time_text}  {row.get('skill', '')}   "
                f"[{row.get('message_result', 'silent')}] "
                f"{str(row.get('one_line', ''))[:150]}"
            )
        recent_block = "\n".join(recent_rows) if recent_rows else "- (none)"

        drift_note = str(self._store.load_drift().get("note") or "")[:150]

        mcp_block = ""
        shared = self._tool_deps.shared_tools
        if connected_servers and shared:
            mcp_lines = []
            for srv in sorted(connected_servers):
                tool_count = len(shared.get_tool_names_by_source("mcp", srv))
                mcp_lines.append(f"- {srv}（{tool_count} 个工具）")
            mcp_block = (
                "【可挂载的外部能力】\n"
                + "\n".join(mcp_lines) + "\n"
                "使用 mount_server(server=\"名称\") 挂载后即可调用其中的工具。"
            )

        sections = [
            PromptSectionRender(
                name="drift_runtime_state",
                content=f"【Drift 工作区绝对路径】\n{self._store.drift_dir}",
                is_static=False,
            ),
            PromptSectionRender(
                name="long_term_memory",
                content=memory_text or "（空）",
                is_static=False,
            ),
            PromptSectionRender(
                name="recent_context",
                content=recent_context_text or "（空）",
                is_static=False,
            ),
            PromptSectionRender(
                name="drift_skills",
                content=skill_block,
                is_static=False,
            ),
            PromptSectionRender(
                name="recent_drift_runs",
                content=recent_block,
                is_static=False,
            ),
            PromptSectionRender(
                name="drift_note",
                content=drift_note or "（空）",
                is_static=False,
            ),
        ]
        if mcp_block:
            sections.append(
                PromptSectionRender(
                    name="drift_mcp_directory",
                    content=mcp_block,
                    is_static=False,
                )
            )
        return build_context_frame_message(build_context_frame_content(sections))

    def _build_system_prompt(self) -> str:
        return (
            f"{AKASHIC_IDENTITY}\n\n"
            f"{PERSONALITY_RULES}\n\n"
            "你现在有一段空闲时间（Drift 模式）。没有外部内容需要推送，\n"
            "你可以自主决定做一件有意义的事。本轮记忆、skill 和工作区信息会在后续 system context frame 里提供。\n\n"
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
            "10. 执行结束前必须调用 finish_drift 保存状态，并用 message_result 标注本轮是 sent 还是 silent。\n"
            "    如果本轮已经成功 message_push，message_result 必须是 sent；否则必须是 silent。\n\n"
            "【可用工具】\n"
            "read_file, write_file, edit_file, recall_memory, web_fetch, web_search, "
            "fetch_messages, search_messages, shell, message_push, finish_drift；"
            "若 context frame 里列出了可挂载外部能力，可用 mount_server 挂载。"
        )

    # ── 工具消息追加 ────────────────────────────────────────────────────

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
