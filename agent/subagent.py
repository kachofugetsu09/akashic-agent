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
from typing import Any

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
    "这是最后一步，步骤预算已耗尽。按顺序完成以下两件事：\n"
    "1. 调用 task_note 记录当前进度（已完成的步骤、结果路径、待续事项）\n"
    "2. 调用 notify_owner 发送消息（说明已完成的步骤和当前结果）\n"
    "哪怕任务未完全完成，也要如实汇报当前进度，不得声称已完成。"
)
_WARN_THRESHOLD = 5  # 剩余步数 <= 此值时开始提示


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
        max_iterations: int = 15,
        max_tokens: int = 4096,
    ) -> None:
        self._provider = provider
        self._model = model
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
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
        """执行任务，返回最终文本结果。失败时返回空字符串。"""
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": task})

        for iteration in range(self._max_iterations):
            try:
                response = await self._provider.chat(
                    messages=messages,
                    tools=self._tool_schemas,
                    model=self._model,
                    max_tokens=self._max_tokens,
                    tool_choice="auto",
                )
            except Exception as e:
                logger.error("[subagent] LLM 调用失败 iteration=%d: %s", iteration, e)
                return ""

            if not response.tool_calls:
                logger.info("[subagent] 任务完成 iterations=%d", iteration + 1)
                return (response.content or "").strip()

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
        return ""
