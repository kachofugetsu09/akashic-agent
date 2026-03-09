"""
SubAgent — 通用子 Agent

有固定工具集、独立的 LLM 循环，执行单个任务后返回结果。
可作为 skill_action agent 类型的执行引擎，也可用于未来其他子 Agent 场景。

用法示例：
    agent = SubAgent(
        provider=provider,
        model="deepseek-chat",
        tools=[WebSearchTool(), WebFetchTool(), NotifyOwnerTool(...)],
        system_prompt="你是后台研究助手...",
    )
    result = await agent.run("调研最新的 agent 相关论文，总结后发给我")
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from agent.provider import LLMProvider
from agent.tools.base import Tool

logger = logging.getLogger(__name__)

_REFLECT_PROMPT = (
    "根据上述工具结果，决定下一步操作。\n"
    "若任务已完成，直接输出最终结果；若需要继续，继续调用工具。\n"
    "禁止把工具调用失败的原因写进最终回复，遇到失败时换个方式或跳过该步骤。"
)
_REFLECT_PROMPT_WARN = (
    "根据上述工具结果，决定下一步操作。\n"
    "⚠️ 步骤预算剩余 {remaining} 步，请优先完成核心目标，跳过非必要步骤。\n"
    "尽快用 task_note 记录当前进度，以防步骤耗尽时能断点续接。\n"
    "若任务已完成，调用 notify_owner 汇报结果后输出最终结果；若需要继续，继续调用工具。\n"
    "禁止把工具调用失败的原因写进最终回复，遇到失败时换个方式或跳过该步骤。"
)
_REFLECT_PROMPT_LAST = (
    "⚠️ 步骤预算将在下一步耗尽。请立即优先完成核心目标，"
    "系统随后会强制调用 task_note / notify_owner 进行收尾。"
)
_CLEANUP_PROMPT = (
    "步骤预算已耗尽，进入强制收尾阶段。\n"
    "你必须调用 {tool_name}，如实汇报当前进度（已完成的步骤、产出路径、未完成的原因）。"
)
_WARN_THRESHOLD = 5  # 剩余步数 <= 此值时开始提示
_MAX_TOOL_RESULT_CHARS = 100_000  # 单条工具结果字符上限（约 ~25K tokens）
_RECENT_TOOL_ROUNDS = 3  # 保留完整 tool result 的最近轮次数
_CLEARED = "[已清除]"  # 旧 tool result 的占位符
_TOOL_LOOP_REPEAT_LIMIT = 3  # 连续同签名工具调用达到该次数时判定循环
_SUMMARY_MAX_TOKENS = 512
_INCOMPLETE_SUMMARY_PROMPT = (
    "当前任务未在步骤预算内完成，请直接输出中文进度总结，不要 JSON。\n"
    "必须覆盖：1) 已完成内容；2) 当前未完成点；3) 下一步计划。\n"
    "禁止输出模板句“已达到最大迭代次数”。"
)
_FORCED_FINAL_SUMMARY_PROMPT = (
    "你已用完任务执行预算，禁止再调用工具。\n"
    "现在必须直接输出中文最终总结，供主 agent 回传给用户。\n"
    "必须覆盖：1) 已完成内容；2) 当前未完成内容；3) 产出文件路径（如果有）；4) 下一步建议。\n"
    "禁止：继续规划工具调用；说“需要继续调用工具”；输出“已达到最大迭代次数”等模板句。"
)
_FORCED_FINAL_SUMMARY_FALLBACK = (
    "这次后台任务已先停在当前进度。我已经完成了一部分关键步骤，"
    "但还有剩余工作未收束；下一次可从当前检查点继续推进。"
)


def _tool_call_signature(tool_calls) -> str:
    parts: list[str] = []
    for tc in tool_calls:
        args = json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)
        parts.append(f"{tc.name}:{args}")
    return "|".join(parts)


def _trim_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将旧轮次的 tool result 替换为占位符，防止长对话累积撑爆上下文。

    保留最近 _RECENT_TOOL_ROUNDS 个 assistant(tool_calls) 轮次的完整结果；
    更早轮次的 tool result 替换为 _CLEARED，保留因果结构供 LLM 理解调用链。
    """
    # 找到所有含 tool_calls 的 assistant 消息索引（即每轮工具调用的起点）
    tool_round_indices = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    if len(tool_round_indices) <= _RECENT_TOOL_ROUNDS:
        return messages  # 轮次不多，无需清理

    # 需要清理的轮次边界：第 (len - _RECENT_TOOL_ROUNDS) 个之前的所有轮次
    cutoff = tool_round_indices[-_RECENT_TOOL_ROUNDS]

    out = []
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and i < cutoff:
            # 找到这条 tool 消息对应的 assistant 轮次起点（往前找最近的含 tool_calls 的 assistant）
            out.append({**m, "content": _CLEARED})
        else:
            out.append(m)
    return out


class SubAgent:
    """有界子 Agent：固定工具集 + 单任务执行。

    与主 AgentLoop 的区别：
    - 不维护对话历史，每次 run() 是独立的一次性任务
    - 工具集在构造时固定，不可在运行时扩展
    - 没有 session/memory 写入能力（由调用方决定是否保存结果）
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        tools: list[Tool],
        *,
        system_prompt: str = "",
        max_iterations: int = 30,
        max_tokens: int = 8192,
        mandatory_exit_tools: Sequence[str] = (),
    ) -> None:
        self._provider = provider
        self._model = model
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._mandatory_exit_tools = list(mandatory_exit_tools)
        self.last_exit_reason: str = "idle"
        self._tool_map: dict[str, Tool] = {t.name: t for t in tools}
        self._tool_schemas: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    async def run(self, task: str) -> str:
        """执行任务并返回文本结果。

        - 任务正常完成：返回最终结果文本
        - 命中循环保护或达到最大迭代：返回进度收尾总结
        - LLM 调用等硬错误：返回空字符串
        """
        messages: list[dict[str, Any]] = []
        self.last_exit_reason = "running"
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": task})
        last_tool_signature = ""
        repeat_count = 0

        for iteration in range(self._max_iterations):
            try:
                response = await self._provider.chat(
                    messages=_trim_tool_results(messages),
                    tools=self._tool_schemas,
                    model=self._model,
                    max_tokens=self._max_tokens,
                    tool_choice="auto",
                )
            except Exception as e:
                logger.error("[subagent] LLM 调用失败 iteration=%d: %s", iteration, e)
                self.last_exit_reason = "error"
                return ""

            if not response.tool_calls:
                logger.info("[subagent] 任务完成 iterations=%d", iteration + 1)
                self.last_exit_reason = "completed"
                return (response.content or "").strip()

            signature = _tool_call_signature(response.tool_calls)
            if signature and signature == last_tool_signature:
                repeat_count += 1
            else:
                repeat_count = 1
                last_tool_signature = signature

            if repeat_count >= _TOOL_LOOP_REPEAT_LIMIT:
                logger.warning(
                    "[subagent] 检测到工具调用循环 signature=%s repeat=%d，提前收尾",
                    signature[:160],
                    repeat_count,
                )
                self.last_exit_reason = "tool_loop"
                if self._mandatory_exit_tools:
                    await self._run_mandatory_exit(messages)
                return await self._summarize_incomplete_progress(
                    messages,
                    reason="tool_call_loop",
                    iteration=iteration + 1,
                )

            # 追加 assistant 消息（含 tool_calls）
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(
                                    tc.arguments, ensure_ascii=False
                                ),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            # 执行工具
            for tc in response.tool_calls:
                tool = self._tool_map.get(tc.name)
                if tool:
                    logger.info(
                        "[subagent] 调用工具 %s args=%s",
                        tc.name,
                        str(tc.arguments)[:120],
                    )
                    try:
                        result = await tool.execute(**tc.arguments)
                    except Exception as e:
                        result = f"工具执行出错: {e}"
                    logger.info("[subagent] 工具结果 %s: %s", tc.name, result[:120])
                else:
                    result = f"未知工具: {tc.name}"
                # 兜底截断：防止超长结果撑爆 LLM 上下文
                if len(result) > _MAX_TOOL_RESULT_CHARS:
                    original_len = len(result)
                    result = (
                        result[:_MAX_TOOL_RESULT_CHARS]
                        + f"\n...[结果已截断，原始长度 {original_len} 字符，超出上限 {_MAX_TOOL_RESULT_CHARS}]"
                    )
                    logger.warning(
                        "[subagent] 工具结果 %s 过长已截断 original=%d",
                        tc.name,
                        original_len,
                    )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

            remaining = self._max_iterations - iteration - 1
            if remaining == 0:
                reflect = _REFLECT_PROMPT_LAST
            elif remaining <= _WARN_THRESHOLD:
                reflect = _REFLECT_PROMPT_WARN.format(remaining=remaining)
            else:
                reflect = _REFLECT_PROMPT
            messages.append({"role": "user", "content": reflect})

        logger.warning("[subagent] 已达到最大迭代次数 %d", self._max_iterations)
        if self._mandatory_exit_tools:
            await self._run_mandatory_exit(messages)
        return await self._force_final_summary(
            messages,
            reason="max_iterations",
            iteration=self._max_iterations,
        )

    async def _summarize_incomplete_progress(
        self,
        messages: list[dict[str, Any]],
        *,
        reason: str,
        iteration: int,
    ) -> str:
        prompt = (
            f"[收尾原因] {reason}\n"
            f"[已执行轮次] {iteration}\n\n" + _INCOMPLETE_SUMMARY_PROMPT
        )
        try:
            resp = await self._provider.chat(
                messages=messages + [{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=min(_SUMMARY_MAX_TOKENS, self._max_tokens),
            )
            text = (resp.content or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning("[subagent] 生成收尾总结失败: %s", e)
        return "本轮步骤预算已用完：已完成部分关键步骤，但仍有未完成项，下一轮将从当前检查点继续推进。"

    async def _force_final_summary(
        self,
        messages: list[dict[str, Any]],
        *,
        reason: str,
        iteration: int,
    ) -> str:
        prompt = (
            f"[结束原因] {reason}\n"
            f"[已执行任务轮次] {iteration}\n\n"
            + _FORCED_FINAL_SUMMARY_PROMPT
        )
        try:
            resp = await self._provider.chat(
                messages=messages + [{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=min(_SUMMARY_MAX_TOKENS, self._max_tokens),
            )
            text = (resp.content or "").strip()
            if text:
                self.last_exit_reason = "forced_summary"
                return text
        except Exception as e:
            logger.warning("[subagent] 强制最终总结失败: %s", e)
        self.last_exit_reason = "forced_summary_fallback"
        return _FORCED_FINAL_SUMMARY_FALLBACK

    async def _run_mandatory_exit(self, messages: list[dict[str, Any]]) -> None:
        """强制收尾：逐个调用 mandatory_exit_tools 中的工具。"""
        for tool_name in self._mandatory_exit_tools:
            if tool_name not in self._tool_map:
                continue
            prompt = _CLEANUP_PROMPT.format(tool_name=tool_name)
            messages.append({"role": "user", "content": prompt})
            try:
                response = await self._provider.chat(
                    messages=messages,
                    tools=self._tool_schemas,
                    model=self._model,
                    max_tokens=self._max_tokens,
                    tool_choice={"type": "function", "function": {"name": tool_name}},
                )
            except Exception as e:
                logger.error("[subagent] mandatory_exit %s 调用失败: %s", tool_name, e)
                continue

            if not response.tool_calls:
                continue

            tc = response.tool_calls[0]
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(
                                    tc.arguments, ensure_ascii=False
                                ),
                            },
                        }
                    ],
                }
            )
            tool = self._tool_map.get(tc.name)
            if tool:
                try:
                    result = await tool.execute(**tc.arguments)
                except Exception as e:
                    result = f"工具执行出错: {e}"
                logger.info(
                    "[subagent] mandatory_exit %s 结果: %s", tc.name, str(result)[:120]
                )
            else:
                result = f"未知工具: {tc.name}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
