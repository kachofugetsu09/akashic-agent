from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RewriteDecision:
    needs_retrieval: bool
    procedure_query: str
    history_query: str
    memory_types_hint: list[str]
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

    async def decide(self, user_msg: str, recent_history: str) -> RewriteDecision:
        # 1. 先准备 prompt 和 fail-open 默认值。
        started = time.perf_counter()
        fallback = self._build_decision(
            started=started,
            user_msg=user_msg,
            needs_retrieval=True,
            procedure_query=user_msg,
            history_query=user_msg,
            memory_types_hint=[],
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
            "needs_retrieval": decision_text == "RETRIEVE",
            "procedure_query": self._extract_tag(raw_output, "procedure_query"),
            "history_query": self._extract_tag(raw_output, "history_query"),
            "memory_types_hint": self._parse_memory_types(
                self._extract_tag(raw_output, "memory_types")
            ),
        }

    def _build_decision(
        self,
        *,
        started: float,
        user_msg: str,
        needs_retrieval: bool,
        procedure_query: str,
        history_query: str,
        memory_types_hint: list[str],
    ) -> RewriteDecision:
        fallback_query = user_msg.strip()
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        return RewriteDecision(
            needs_retrieval=needs_retrieval,
            procedure_query=procedure_query.strip() or fallback_query,
            history_query=history_query.strip() or fallback_query,
            memory_types_hint=memory_types_hint,
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
    def _parse_memory_types(raw_value: str) -> list[str]:
        allowed = {"procedure", "preference", "event", "profile"}
        result: list[str] = []
        seen: set[str] = set()
        for part in str(raw_value or "").split(","):
            item = part.strip().lower()
            if not item or item not in allowed or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @staticmethod
    def _build_prompt(*, user_msg: str, recent_history: str) -> str:
        history_block = recent_history.strip() or "（无）"
        return f"""你是记忆检索决策器。根据近期对话和当前用户消息，判断是否需要检索记忆，并输出两个查询。

近期对话：
{history_block}

当前用户消息：
{user_msg}

规则：
- NO_RETRIEVE：打招呼、闲聊、确认当前轮内容、通用知识问答、简单回应“好/嗯/继续”
- RETRIEVE：询问过去发生的事、用户偏好、个人信息，或要求执行某类操作时需要查 memory

隐式意图推断（先想再决策）：
- 在输出 XML 之前，先用 <thinking>...</thinking> 推断用户消息的隐含背景
- 提到快递 / 物流 / 单号 / 包裹 / 到货：隐含意图通常是查用户最近的购买行为
- 提到身体症状 / 药 / 复查：隐含意图通常是查用户健康档案
- 提到“那个任务 / 项目 / 上次说的”：隐含意图通常是查用户正在进行的事项
- 如果隐含意图指向历史记录，则应 RETRIEVE，history_query 应面向隐含意图，而不是表面词
- <thinking> 只用于内部推理，不要把它混入最终 XML 字段

输出要求：
- procedure_query：面向 procedure/preference 的精简动作意图，不要写成问句，不要用“用户/我”做主语
- history_query：面向 event/profile 的完整语义 query，可以包含上下文
- memory_types：从 procedure,preference,event,profile 中选择一个或多个，逗号分隔；不确定可留空

只输出 XML：
<decision>RETRIEVE|NO_RETRIEVE</decision>
<procedure_query>...</procedure_query>
<history_query>...</history_query>
<memory_types>从上述四个类型中选择，逗号分隔，不确定可留空</memory_types>"""
