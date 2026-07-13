"""
DriftTurnPipeline — Drift 空闲时间链路顶层抽象。

设计对齐主动链路的 ProactiveFlowRuntime.run() 和被动链路的 PassiveTurnPipeline.run()：
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
from bus.events_lifecycle import DriftFinished
from plugins.default_proactive.context import AgentTickContext
from plugins.drift_flow.state import DriftStateStore, SkillMeta
from plugins.drift_flow.tools import (
    DriftToolDeps,
    build_drift_tool_registry,
)

if TYPE_CHECKING:
    from core.memory.markdown import MemoryProfileApi

LlmFn = Callable[[list[dict], list[dict], str | dict, bool], Awaitable[dict | None]]
StepRecorder = Callable[[AgentTickContext, str, str, str, dict[str, Any], str], None]
logger = logging.getLogger(__name__)
_WRAP_UP_MAX_ATTEMPTS = 2
_BEFORE_SELECT_TOOLS = frozenset({"select_skill", "idle_drift"})
_AFTER_SEND_TOOLS = frozenset({"finish_drift"})
_TOOL_CONSTRAINT_RETRY_LIMIT = 2


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
# ┌─ tick 触发（无内容）
# │  └─ DriftTurnPipeline.run
# │     ├─ 1. Scan（扫描）── _scan_skills
# │     │  └─ store.scan_skills → MCP 过滤 → 空则 skip
# │     ├─ 2. Prepare（准备）── _prepare
# │     │  └─ 设置 ctx drift flags → build_drift_tool_registry → 构建 messages
# │     ├─ 3. Execute（执行）── _execute_loop
# │     │  └─ while steps < max_steps: llm_fn → 执行工具 → 追加消息 → 记录
# │     │     message_push 后约束 schema 为 finish_drift
# │     └─ 4. Finish（收尾）── _finish
# │        └─ 记录退出状态日志
# └─ 完成

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
        tools, messages = await self._prepare(ctx, skills)

        # 4. Execute — LLM 工具调用循环。
        await self._execute_loop(ctx, llm_fn, tools, messages)

        # 5. Finish — 记录退出。
        self._finish(ctx)
        return True

    # ── 1. Scan（扫描）───────────────────────────────────────────────

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

    # ── 2. Prepare（准备）───────────────────────────────────────────

    async def _prepare(
        self,
        ctx: AgentTickContext,
        skills: list[SkillMeta],
    ) -> tuple[Any, list[dict]]:
        """设置 ctx drift 标志、构建 tool registry 与初始 messages。"""

        # 2.1 设置 ctx 标志位。
        ctx.drift_entered = True
        ctx.drift_finished = False
        ctx.drift_message_staged = False
        ctx.drift_message_sent = False
        ctx.drift_selected_skill = ""
        ctx.drift_finish_status = ""
        ctx.drift_finish_briefing = ""

        # 2.2 构建 drift tool registry。
        tools = build_drift_tool_registry(
            ctx=ctx,
            deps=self._tool_deps,
        )

        # 2.3 确定 MCP 已连接 server 列表。
        shared = self._tool_deps.shared_tools
        connected_servers = shared.get_mcp_server_names() if shared else set()

        # 2.4 构建初始 messages。
        messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt()},
            await self._build_runtime_context_message(skills, connected_servers, ctx=ctx),
        ]

        return tools, messages

    # ── 3. Execute（执行）───────────────────────────────────────────

    async def _execute_loop(
        self,
        ctx: AgentTickContext,
        llm_fn: LlmFn,
        tools: Any,
        messages: list[dict],
    ) -> None:
        """LLM 工具调用循环：调模型 → 执行工具 → 追加 messages → 重复。"""

        steps = 0
        constraint_rejections = 0

        while steps < self._max_steps and not ctx.drift_finished:
            tool_choice: str | dict = "required"
            schemas = tools.get_schemas()
            allowed_tool_names: set[str] | None = None
            before_select = not str(ctx.drift_selected_skill or "").strip()

            # 3.1 必须先声明执行对象，或说明本轮为什么空闲。
            if before_select:
                allowed_tool_names = set(_BEFORE_SELECT_TOOLS)
                tool_choice = "required"
                schemas = [
                    s for s in schemas
                    if s["function"]["name"] in allowed_tool_names
                ]
                logger.info("[drift] selected_skill missing, forcing select_skill or idle_drift")
            elif ctx.drift_message_staged:
                allowed_tool_names = set(_AFTER_SEND_TOOLS)
                schemas = [
                    s for s in schemas
                    if s["function"]["name"] in allowed_tool_names
                ]
                logger.info(
                    "[drift] message_push already used, "
                    "restricting schema to finish_drift"
                )

            # 3.2 调 LLM 拿工具调用。
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

            ctx.record_llm_cache(
                cache_prompt_tokens=tool_call.get("_cache_prompt_tokens"),
                cache_hit_tokens=tool_call.get("_cache_hit_tokens"),
            )
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

            if allowed_tool_names is not None and tool_name not in allowed_tool_names:
                constraint_rejections += 1
                allowed_text = ", ".join(sorted(allowed_tool_names))
                output = (
                    f"错误：当前阶段不能调用 {tool_name}。"
                    f"当前只允许调用：{allowed_text}。"
                )
                if "finish_drift" in allowed_tool_names:
                    output += "请调用 finish_drift 保存 completed 或 paused 状态。"
                logger.warning("[drift] tool constraint rejected tool=%s", tool_name)
                self._store.append_step(
                    step_index=steps,
                    tool_name=tool_name,
                    input_preview=json.dumps(tool_args, ensure_ascii=False),
                    output_preview=output,
                    now_utc=ctx.now_utc,
                )
                if self.step_recorder is not None:
                    self.step_recorder(
                        ctx,
                        "drift:error",
                        tool_name,
                        str(tool_call.get("id") or f"drift_{steps}"),
                        tool_args,
                        output,
                    )
                self._append_tool_messages(
                    messages,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_call_id=str(tool_call.get("id") or f"drift_{steps}"),
                    result=output,
                )
                if constraint_rejections >= _TOOL_CONSTRAINT_RETRY_LIMIT:
                    if "finish_drift" in allowed_tool_names:
                        await self._wrap_up(ctx, llm_fn, tools, messages)
                    else:
                        logger.warning("[drift] selection rejected repeatedly, aborting drift")
                    return
                continue

            # 3.3 执行工具。
            result = await self._tool_executor.execute(
                ToolExecutionRequest(
                    call_id=str(tool_call.get("id") or f"drift_{steps}"),
                    tool_name=tool_name,
                    arguments=tool_args,
                    source="proactive",
                    session_key=ctx.session_key,
                ),
                tools.execute,
            )

            # 3.4 错误处理。
            if result.status == "error":
                logger.warning("[drift] tool executor error at step=%d: %s", steps, result.output)
                self._store.append_step(
                    step_index=steps,
                    tool_name=tool_name,
                    input_preview=json.dumps(tool_args, ensure_ascii=False),
                    output_preview=str(result.output),
                    now_utc=ctx.now_utc,
                )
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

            # 3.5 记录步骤。
            self._store.append_step(
                step_index=steps,
                tool_name=tool_name,
                input_preview=json.dumps(tool_args, ensure_ascii=False),
                output_preview=str(result.output),
                now_utc=ctx.now_utc,
            )
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
            constraint_rejections = 0

        if steps >= self._max_steps and not ctx.drift_finished:
            await self._wrap_up(ctx, llm_fn, tools, messages)

    async def _wrap_up(
        self,
        ctx: AgentTickContext,
        llm_fn: LlmFn,
        tools: Any,
        messages: list[dict],
    ) -> None:
        finish_schemas = [
            schema
            for schema in tools.get_schemas()
            if schema["function"]["name"] == "finish_drift"
        ]
        if not finish_schemas:
            logger.warning("[drift] wrap-up skipped: finish_drift schema missing")
            return

        messages.append(
            {
                "role": "system",
                "content": (
                    "【系统强制收尾】本轮 Drift 可用步数已耗尽。"
                    "不要继续推进任务，只根据上方已发生的工具结果调用 finish_drift。"
                    "如果本轮小闭环已完成，status 写 completed。"
                    "如果没做完，status 写 paused，并在 scratchpad_update 写清已经做到哪里、"
                    "当前卡在什么条件、下次从哪里继续。"
                    "不要编造额外下一步。"
                ),
            }
        )

        rejection = ""
        for attempt in range(1, _WRAP_UP_MAX_ATTEMPTS + 1):
            if rejection:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "【系统强制收尾重试】上一次收尾无效："
                            f"{rejection}。你已经可以看到本轮完整工具历史，"
                            "现在只能调用 finish_drift，不能调用任何其他工具。"
                        ),
                    }
                )

            tool_choice = {"type": "function", "function": {"name": "finish_drift"}}
            if "disable_thinking" in inspect.signature(llm_fn).parameters:
                tool_call = await cast(Any, llm_fn)(
                    messages, finish_schemas, tool_choice, disable_thinking=True
                )
            else:
                tool_call = await cast(Any, llm_fn)(messages, finish_schemas, tool_choice)
            if tool_call is None:
                rejection = "没有返回工具调用"
                logger.warning("[drift] wrap-up llm returned no tool call attempt=%d", attempt)
                continue

            ctx.record_llm_cache(
                cache_prompt_tokens=tool_call.get("_cache_prompt_tokens"),
                cache_hit_tokens=tool_call.get("_cache_hit_tokens"),
            )
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("input", {})
            if tool_name != "finish_drift":
                rejection = f"返回了非 finish_drift 工具 {tool_name}"
                logger.warning(
                    "[drift] wrap-up rejected non-finish tool=%s attempt=%d",
                    tool_name,
                    attempt,
                )
                continue

            result = await self._tool_executor.execute(
                ToolExecutionRequest(
                    call_id=str(tool_call.get("id") or "drift_wrap_up"),
                    tool_name=tool_name,
                    arguments=tool_args,
                    source="proactive",
                    session_key=ctx.session_key,
                ),
                tools.execute,
            )
            self._store.append_step(
                step_index=ctx.steps_taken + attempt,
                tool_name=tool_name,
                input_preview=json.dumps(tool_args, ensure_ascii=False),
                output_preview=str(result.output),
                now_utc=ctx.now_utc,
            )
            if self.step_recorder is not None:
                self.step_recorder(
                    ctx,
                    "drift",
                    tool_name,
                    str(tool_call.get("id") or "drift_wrap_up"),
                    tool_args,
                    str(result.output),
                )
            self._append_tool_messages(
                messages,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=str(tool_call.get("id") or "drift_wrap_up"),
                result=str(result.output),
            )
            if result.status == "error":
                rejection = f"finish_drift 执行失败：{result.output}"
                logger.warning("[drift] wrap-up finish error: %s", result.output)
                continue
            if ctx.drift_finished:
                return
            rejection = f"finish_drift 未完成：{result.output}"

        logger.warning("[drift] wrap-up exhausted, fallback pause: %s", rejection)
        self._fallback_pause(ctx)

    def _fallback_pause(self, ctx: AgentTickContext) -> None:
        skill_name = str(ctx.drift_selected_skill or "").strip() or "unknown"
        message_result = "staged" if ctx.drift_message_staged else "silent"
        self._store.save_finish(
            skill_used=skill_name,
            status="paused",
            briefing="达到步数上限后模型未按要求调用 finish_drift，runtime 自动保存为 paused。",
            message_result=message_result,
            scratchpad_update="下次先阅读 Drift Briefing，再根据上一轮已执行的工具结果继续或改选更合适的 skill。",
            global_note_update=None,
            now_utc=ctx.now_utc,
            self_update={
                "next_tendency": "下次根据停点和当时状态重新选择是否继续",
            },
        )
        ctx.drift_finished = True
        ctx.drift_finish_status = "paused"
        ctx.drift_finish_briefing = "达到步数上限后模型未按要求调用 finish_drift，runtime 自动保存为 paused。"

    # ── 4. Finish（收尾）─────────────────────────────────────────────

    def _finish(self, ctx: AgentTickContext) -> None:
        """记录 drift 退出状态。"""
        logger.info(
            "[drift] exit: finished=%s message_staged=%s selected_skill=%s",
            ctx.drift_finished,
            ctx.drift_message_staged,
            ctx.drift_selected_skill,
        )

    def record_commit_result(self, ctx: AgentTickContext, sent: bool) -> None:
        message_result = "sent" if sent else "silent"
        self._store.update_last_message_result(message_result)
        event_bus = self._tool_deps.event_bus
        if event_bus is not None:
            event_bus.enqueue(
                DriftFinished(
                    session_key=ctx.session_key,
                    skill_name=ctx.drift_selected_skill,
                    status=ctx.drift_finish_status,
                    briefing=ctx.drift_finish_briefing,
                    message_result=message_result,
                    timestamp=datetime.now(timezone.utc),
                )
            )

    # ── Prompt 构建 ────────────────────────────────────────────────────

    async def _build_runtime_context_message(
        self,
        skills: list[SkillMeta],
        connected_servers: set[str] | None = None,
        ctx: AgentTickContext | None = None,
    ) -> dict[str, str]:
        """构建 runtime context frame，包含记忆、skill 列表、近期 run 记录。"""

        memory_text = ""
        if self._tool_deps.memory is not None:
            memory = cast("MemoryProfileApi", self._tool_deps.memory)
            raw = str(memory.read_long_term() or "").strip()
            if raw:
                memory_text = raw
        recent_chat_text = await self._build_recent_raw_chat(limit=5)

        display_skills = sorted(skills[:8], key=lambda item: item.name)
        lines = []
        for skill in display_skills:
            line = (
                f"- {skill.name}/   {skill.run_count}次运行   "
                f"status: {skill.status}   {skill.description[:80]}"
            )
            if skill.builtin:
                line += "   [builtin]"
            if skill.requires_mcp:
                line += f"   [需要: {', '.join(skill.requires_mcp)}]"
            lines.append(line)
        skill_block = "\n".join(lines) if lines else "- (none)"
        selection_context = self._build_selection_context(display_skills)

        recent_rows = []
        for row in self._store.load_drift().get("recent_runs", [])[-5:][::-1]:
            run_at = str(row.get("run_at") or "")
            try:
                dt = datetime.fromisoformat(run_at).astimezone(timezone.utc)
                time_text = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                time_text = run_at[:16]
            recent_rows.append(
                f"- {time_text}  {row.get('skill', '')}   "
                f"[{row.get('message_result', 'silent')}] "
                f"{str(row.get('briefing', ''))[:150]}"
            )
        recent_block = "\n".join(recent_rows) if recent_rows else "- (none)"

        drift_note = str(self._store.load_drift().get("note") or "")[:150]
        drift_briefing = self._store.load_briefing(skills)
        self_state = self._build_self_state_context()
        self_observations = self._build_self_observations_context()
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
                name="drift_self_state",
                content=self_state,
                is_static=False,
            ),
            PromptSectionRender(
                name="drift_self_observations",
                content=self_observations,
                is_static=False,
            ),
            PromptSectionRender(
                name="drift_selection_context",
                content=selection_context,
                is_static=False,
            ),
            PromptSectionRender(
                name="drift_skills",
                content=skill_block,
                is_static=False,
            ),
            PromptSectionRender(
                name="long_term_memory",
                content=memory_text or "（空）",
                is_static=False,
            ),
            PromptSectionRender(
                name="recent_raw_chat",
                content=recent_chat_text or "（空）",
                is_static=False,
            ),
            PromptSectionRender(
                name="drift_briefing",
                content=drift_briefing,
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
        sections.append(
            PromptSectionRender(
                name="runtime_clock",
                content=self._build_runtime_clock(ctx),
                is_static=False,
            )
        )
        return build_context_frame_message(build_context_frame_content(sections))

    def _build_self_state_context(self) -> str:
        state = self._store.load_self_state()
        if not state:
            return "（还没有过去的 Drift 意图，可以自由探索。）"
        return (
            "这是上轮留下的自我连续性，不是必须执行的命令；可以延续，也可以改变主意。\n"
            f"- 当时选择：{state.get('last_decision') or '（空）'}\n"
            f"- 当时想做：{state.get('current_intention') or '（空）'}\n"
            f"- 选择原因：{state.get('decision_reason') or '（空）'}\n"
            f"- 下次倾向：{state.get('next_tendency') or '（尚未收尾）'}\n"
            f"- 关联 skill：{state.get('current_skill') or '（空）'}\n"
            f"- 更新时间：{state.get('updated_at') or '（空）'}"
        )

    def _build_self_observations_context(self) -> str:
        rows = self._store.load_recent_self_observations()
        if not rows:
            return "（还没有 Drift 自我观察。）"
        lines = [
            "这些只是过去多轮 Drift 对自身行为的暂定观察，不是长期记忆、人格结论或行动命令。",
            "结合具体情境寻找重复、矛盾和变化；单次观察不能定义自己，本轮也不必刻意证明它。",
        ]
        for row in rows:
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            lines.append(
                f"- {str(row.get('created_at') or '')[:16]} {row.get('skill_name') or 'unknown'} "
                f"[{payload.get('effect') or 'question'}] {str(payload.get('statement') or '')[:200]}；"
                f"依据：{str(payload.get('basis') or '')[:240]}"
            )
        return "\n".join(lines)

    def _build_runtime_clock(self, ctx: AgentTickContext | None) -> str:
        now_utc = ctx.now_utc if ctx is not None else datetime.now(timezone.utc)
        local_now = now_utc.astimezone()
        return (
            f"current_time_utc={now_utc.isoformat()}\n"
            f"current_time_local={local_now.isoformat()}"
        )

    def _build_selection_context(self, skills: list[SkillMeta]) -> str:
        if not skills:
            return "- （无）"

        lines = [
            "下面按 skill 名称排列，顺序不代表优先级，也不是强制首选。",
            "选择依据：runtime_clock、status、上次 finish 时间、上次摘要、scratchpad、cursor、recent_raw_chat 和最近 runs。",
            "completed 表示上次主动行为已闭环，包含已行动、检查后无事可做、或判断不合时宜后静默结束。",
            "paused 表示存在一个可以续接的停点，不代表本轮必须立刻继续，也不代表要从头重做。",
            "先判断本轮与已有停点的关系：从停点继续、暂时搁置、改做其他 skill，或在没有合适前情时自由探索。",
            "如果决定继续 paused skill，应把 scratchpad、cursor 和已有工作文件视为进度依据，找到尚未完成的下一步。",
            "SKILL.md 是能力说明书、约束和路径地图，不是每轮都要从第一条重新执行的清单。",
            "对 paused skill，local_context 记录的已完成进度高于 SKILL.md 中面向全新任务的完整流程和固定工具序列。",
            "local_context 只在 select_skill 后作为执行上下文参考，其中 scratchpad 是自然语言前情，cursor 是结构化游标。",
            "用户回应与否不是 skill 状态；回答出现后可作为新上下文使用，但未出现回答不是可观测事件，不要写成‘用户没回’。",
            "上次提问主题只作为短期去重信号：本轮可以换主题行动，也可以因时机不合适静默闭环。",
            "默认应选择一个合适 skill 做一个小的原子动作；idle_drift 是例外路径，只用于近期气氛、频率或风险明确不合适。",
            "遇到丧亲、疾病、强压力、明显情绪低落等近期语境时，优先选择 idle_drift 静默结束，除非 selected skill 明确是低打扰的支持性动作。",
            "判断“刚刚、今天、昨天、两天前”等相对时间时，必须以 runtime_clock 的完整日期和时间为准；只有时分没有日期时，不要断言它发生在今天。",
        ]
        for skill in skills:
            continuum = self._store.load_skill_continuum(skill.name)
            briefing = str(continuum.get("last_briefing") or "").strip()[:120]
            scratchpad = str(continuum.get("scratchpad") or "").strip()[:160]
            finished_at = str(continuum.get("updated_at") or continuum.get("last_run_at") or "").strip()
            cursor = continuum.get("cursor")
            cursor_text = ""
            if isinstance(cursor, dict) and cursor:
                cursor_text = (
                    " cursor="
                    + json.dumps(cursor, ensure_ascii=False, sort_keys=True)[:160]
                )
            local_context = (
                f"local_context=completed{cursor_text}"
                if skill.status == "completed"
                else f"scratchpad={scratchpad or '（空）'}{cursor_text}"
            )
            lines.append(
                f"- {skill.name}: status={skill.status} run_count={skill.run_count} "
                f"last_finish={finished_at or 'never'} briefing={briefing or '（空）'} "
                f"{local_context}"
            )
        return "\n".join(lines)

    async def _build_recent_raw_chat(self, *, limit: int) -> str:
        recent_chat_fn = self._tool_deps.recent_chat_fn
        if recent_chat_fn is None:
            return "（空）"
        try:
            rows = await recent_chat_fn(limit)
        except Exception as exc:
            logger.warning("[drift] recent_raw_chat read failed: %s", exc)
            return "（读取失败）"

        lines: list[str] = []
        for row in list(rows or [])[-limit:]:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "unknown").strip() or "unknown"
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            marker = " proactive=true" if row.get("proactive") else ""
            compact = " ".join(content.split())
            lines.append(f"- {role}{marker}: {compact[:500]}")
        return "\n".join(lines) if lines else "（空）"

    def _build_system_prompt(self) -> str:
        return (
            f"{AKASHIC_IDENTITY}\n\n"
            f"{PERSONALITY_RULES}\n\n"
            "你现在有一段空闲时间（Drift 模式）。没有外部内容需要推送，\n"
            "这段时间更像一个人没有被叫住时的自处：优先尝试做一点合适的小事，例如整理想法、延续小兴趣、准备以后可能用得上的素材，或发一个低打扰的轻量问题。"
            "Drift 不是服务用户当前请求，也不是补跑所有历史任务；但它默认应该行动一小步。"
            "它也不需要总是提问、产出或维护系统；现有 skill 能做，不等于此刻就想做。"
            "只有近期气氛、频率或风险明确不合适时，才安静待着。"
            "本轮记忆、skill 和工作区信息会在后续 system context frame 里提供。\n\n"
            "【状态语义】\n"
            "Drift 只有 completed 和 paused 两种收尾状态。"
            "completed 表示本轮主动行为已闭环，包含已行动、检查后无事可做、或判断当前不合时宜后静默结束。"
            "paused 只用于系统自己没完成的情况，例如工具失败、外部服务不可用、步数上限、或处理中间文件尚未写完；"
            "paused 必须在 scratchpad_update 写清已经做到哪里、卡住原因、下次从哪里继续。"
            "paused 保存的是可续接停点：下轮可以继续、延后或改做别的事；若选择继续，应从未完成处接上，而不是重新开始。"
            "paused 和 idle 只能描述系统自己的进度、时机或选择，不描述用户需要做什么。\n\n"
            "【异步消息语义】\n"
            "message_push 是 fire-and-forget：成功写入会话后，本轮动作已经完成，不进入等待回复状态。"
            "用户以后若感兴趣并回答，回答会作为新的会话上下文并由记忆系统自然提炼；Drift 不维护提问与回答的 pending 状态。"
            "回答真的出现时可以理解和使用它；回答尚未出现不是可靠事件，不能声称‘用户没回’，也不能据此安排下一轮。"
            "scratchpad、cursor、next_tendency 和 observation 都不得保存‘等待用户回复’或‘如果用户不回复’。\n\n"
            "【自我定位】\n"
            "调用首个工具前，先在内部确认：上一轮在做什么、哪些步骤和产物已经完成、为何停下、当前最自然的下一步是什么。"
            "Drift 会被反复触发，recent_drift_runs 是自己刚刚真实度过的空闲时间，不是互相独立的测试样本。"
            "drift_self_state 是过去的自己留下的意图和倾向，不是待办指令；本轮可以继续、延后、切换、自由探索或安静休息。"
            "留意自己是否只是逐字复制 next_tendency，或反复选择最容易执行的同一类动作；重复本身没有错，但要由此刻的意愿和价值重新成立。"
            "当最近多轮都落在同一类服务或维护动作时，停一下判断：这是正在形成的兴趣，还是只是候选里最省事的路径。"
            "drift_self_observations 是 Drift 多轮使用中留下的暂定自我观察；可以参考、质疑或修正，但不要把它当作稳定人格或强行复现。"
            "也要判断此刻想做的活动是否已被现有 skill 覆盖；如果没有覆盖，可以选择候选中的元能力，把可反复进行的小活动沉淀成新 skill。"
            "已有计划和工作文件是过去行动留下的进度，不要仅因为进入了新一轮 Drift 就重新创建。"
            "SKILL.md 是说明书，不是必须从头播放的脚本；只执行当前决定所需要的部分。\n\n"
            "选择 paused skill 后，把 SKILL.md 的完整工作流拆成“已经完成”和“尚未完成”两部分。"
            "local_context 明确已经完成的读文件、查重、规划、写计划等前置动作，不得仅为遵循固定工具序列而重复。"
            "已有工作文件不需要为了确认存在而先 list_dir；下一步工具本身能消费该文件时，也不必无目的地重新读取。"
            "只有 local_context 与实际结果矛盾、文件明确缺失、或下一步确实需要读取其内容时，才回查已完成步骤。\n\n"
            "【执行规则】\n"
            "1. 先根据 context frame 比较所有可用 skill 和最近聊天气氛。"
            "Drift 的含义是没有正在服务用户时，自己尝试做一点合适的小事；"
            "skill 上次 completed 不代表不能再做，只代表上次已闭环。"
            "默认调用 select_skill(skill_name, decision, intention, reason)，先保存本轮选择，再让被选 skill 完成一个原子动作。"
            "decision 表示本轮与既有意图的关系，只能是 continue、defer、switch、explore。"
            "选择 paused skill 表示接回原来的意图；select_skill 返回 local_context 后，先定位停点，再执行最小的未完成动作。"
            "此时第一次执行动作通常应是停点后的下一步，而不是 SKILL.md 完整流程的第一步。"
            "本轮也可以暂时不继续 paused skill，改选其他 skill；不要为了续接而续接。"
            "只有最近刚主动打扰过、当前气氛不适合、或所有 skill 都会产生明显低价值重复时，才调用 idle_drift(reason) 静默结束。"
            "idle_drift 的 reason 必须写具体的时机或风险原因，不能只写 completed、无用户交互、无新信号。\n"
            "2. 选中 skill 后执行一个原子动作；需要更多上下文时，只读取 SKILL.md 声明的 working files。"
            "路径由 drift mount resolver 解析，skills/<skill_name>/... 同时适用于工作区和内建 skill。\n"
            "3. 有用户价值且适合打扰时可调用 message_push，单次 run 最多一次；"
            "message_push 成功后只能调用 finish_drift。\n"
            "4. 结束前必须调用 finish_drift；skill_used 必须等于 selected_skill。\n"
            "5. finish_drift.status 只能为 completed 或 paused。"
            "completed 表示本轮主动行为已闭环；paused 必须写 scratchpad_update，说明做到哪里和下次从哪里继续。"
            "结构化接续写 cursor_update；已经完成的事实追加到 journal_append。"
            "收尾时必须通过 self_update.next_tendency 保存下次空闲时的自然倾向；如果原意图改变，再更新 current_intention。"
            "只有本轮或它与近期多轮的对照确实显露出关于自己如何选择或行动的新证据时，才写 self_update.observation；"
            "初次发现用 question，重复证据用 reinforce，反例或变化用 revise。没有新发现就省略，不要为了显得成长而编造。\n\n"
            "【可用工具】\n"
            "select_skill, idle_drift, read_file, list_dir, write_file, edit_file, recall_memory, web_fetch, web_search, "
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
