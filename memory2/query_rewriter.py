from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GateDecision:
    needs_episodic: bool
    episodic_query: str
    latency_ms: int


class QueryRewriter:
    def __init__(
        self,
        llm_client: Any,
        *,
        model: str = "",
        max_tokens: int = 220,
        timeout_ms: int = 800,
    ) -> None:
        self._llm_client = llm_client
        self._model = model
        self._max_tokens = max(64, int(max_tokens))
        self._timeout_s = max(0.1, float(timeout_ms) / 1000.0)

    async def decide(self, user_msg: str, recent_history: str) -> GateDecision:
        # 1. 先准备 prompt 和 fail-open 默认值。
        started = time.perf_counter()
        fallback = self._build_decision(
            started=started,
            user_msg=user_msg,
            needs_episodic=True,
            episodic_query=user_msg,
        )
        prompt = self._build_prompt(user_msg=user_msg, recent_history=recent_history)

        # 2. 再调用 LLM；异常或超时都直接 fail-open。
        try:
            raw_output = await asyncio.wait_for(
                self._call_llm(prompt),
                timeout=self._timeout_s,
            )
        except Exception:
            return fallback

        # 3. 最后解析 XML；结构无效则继续回退原始消息。
        decision = self._parse_output(raw_output)
        if decision is None:
            return fallback
        return self._build_decision(started=started, user_msg=user_msg, **decision)

    async def _call_llm(self, prompt: str) -> str:
        response = await self._llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            model=self._model,
            max_tokens=self._max_tokens,
        )
        content = getattr(response, "content", response)
        return str(content or "")

    def _parse_output(self, raw_output: str) -> dict[str, Any] | None:
        decision_text = self._extract_tag(raw_output, "decision").upper()
        if decision_text not in {"RETRIEVE", "NO_RETRIEVE"}:
            return None
        return {
            "needs_episodic": decision_text == "RETRIEVE",
            "episodic_query": self._extract_tag(raw_output, "history_query"),
        }

    def _build_decision(
        self,
        *,
        started: float,
        user_msg: str,
        needs_episodic: bool,
        episodic_query: str,
    ) -> GateDecision:
        fallback_query = user_msg.strip()
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        return GateDecision(
            needs_episodic=needs_episodic,
            episodic_query=episodic_query.strip() or fallback_query,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_tag(raw_output: str, tag: str) -> str:
        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            raw_output or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _build_prompt(*, user_msg: str, recent_history: str) -> str:
        history_block = recent_history.strip() or "（无）"
        return f"""你是记忆检索决策器。根据近期对话和当前用户消息，判断是否需要检索 episodic memory，并输出一个查询。

近期对话：
{history_block}

当前用户消息：
{user_msg}

规则：
- NO_RETRIEVE：打招呼、闲聊、确认当前轮内容、通用知识问答、简单回应”好/嗯/继续”
- RETRIEVE：询问过去发生的事、用户偏好、个人信息，或要求执行某类操作时需要查 memory

聚合类问题处理（包含”都有哪些/列举/所有/一共/总共/历史上”等词）：
- 判断为 RETRIEVE
- history_query 改写为宽泛的语义 query，覆盖该主题下所有可能的记录
  例：用户问”我买过哪些键盘” → history_query: “购买 键盘 外设”
  例：用户问”我们讨论过哪些游戏” → history_query: “游戏 推荐 讨论”

代词消解（优先执行，再做其他推断）：
- 消息中出现”他 / 她 / 它 / 这个 / 那个 / 这玩意 / 这东西 / 这 / 那”等指示词时，必须根据近期对话将其替换为实际指代的实体名称
- 例：近期讨论了 “recursive language model”，用户说”他会经常返回奇怪的 repl 字符” → history_query: “recursive language model repl 字符 输出异常”
- 例：近期讨论了 MCP 协议，用户说”这玩意为啥突然又火了” → history_query: “MCP 协议 突然流行 原因”
- 若实在无法确定指代，保留原词并追加近期对话中最相关的实体词

隐式意图推断（先想再决策）：
- 在输出 XML 之前，先用 <thinking>...</thinking> 推断用户消息的隐含背景
- 提到快递 / 物流 / 单号 / 包裹 / 到货：隐含意图通常是查用户最近的购买行为
- 提到身体症状 / 药 / 复查：隐含意图通常是查用户健康档案
- 提到那个任务 / 项目 / 上次说的”：隐含意图通常是查用户正在进行的事项
- 如果隐含意图指向历史记录，则应 RETRIEVE，history_query 应面向隐含意图，而不是表面词
- <thinking> 只用于内部推理，不要把它混入最终 XML 字段

输出要求：
- history_query：面向 event/profile 的完整语义 query，可以包含上下文

只输出 XML：
<decision>RETRIEVE|NO_RETRIEVE</decision>
<history_query>...</history_query>
"""
