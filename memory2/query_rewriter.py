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
    procedure_query: str = ""


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

        # 2. 分开改写 history 和 procedure，避免任一路失败吞掉另一路结果。
        raw_output = ""
        procedure_query = ""
        main_task = asyncio.create_task(
            self._call_llm(
                self._build_prompt(
                    user_msg=user_msg,
                    recent_history=recent_history,
                )
            )
        )
        procedure_task = asyncio.create_task(self._rewrite_procedure_query(user_msg))
        done, pending = await asyncio.wait(
            {main_task, procedure_task},
            timeout=self._timeout_s,
        )
        for task in pending:
            _ = task.cancel()
        if not done:
            return fallback
        if main_task in done:
            try:
                raw_output = main_task.result()
            except Exception:
                raw_output = ""
        if procedure_task in done:
            try:
                procedure_query = procedure_task.result()
            except Exception:
                procedure_query = ""

        # 3. 最后解析；history 结构无效则回退原始消息，但保留 procedure 改写。
        decision = self._parse_output(raw_output)
        if decision is None:
            return self._build_decision(
                started=started,
                user_msg=user_msg,
                needs_episodic=True,
                episodic_query=user_msg,
                procedure_query=procedure_query,
            )
        decision["procedure_query"] = procedure_query
        return self._build_decision(started=started, user_msg=user_msg, **decision)

    async def _call_llm(self, prompt: str) -> str:
        response = await self._llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            model=self._model,
            max_tokens=self._max_tokens,
            disable_thinking=True,
        )
        content = getattr(response, "content", response)
        return str(content or "")

    async def _rewrite_procedure_query(self, user_msg: str) -> str:
        prompt = self._build_procedure_prompt(user_msg)
        try:
            raw_output = await self._call_llm(prompt)
        except Exception:
            return ""
        return self._clean_procedure_query(raw_output)

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
        procedure_query: str = "",
    ) -> GateDecision:
        fallback_query = user_msg.strip()
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        return GateDecision(
            needs_episodic=needs_episodic,
            episodic_query=episodic_query.strip() or fallback_query,
            latency_ms=latency_ms,
            procedure_query=procedure_query.strip(),
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
        return f"""你是记忆检索决策器。根据近期对话和当前用户消息，判断是否需要检索用户个人事实或历史事件，并输出查询。

近期对话：
{history_block}

当前用户消息：
{user_msg}

规则：
- NO_RETRIEVE：打招呼、闲聊、确认当前轮内容、通用知识问答、简单回应“好/嗯/继续”、用户提出新的服务偏好或执行规则
- RETRIEVE：询问过去发生的事、用户个人信息、用户是否告诉过某事
- 用户提出新的偏好或规则时，decision 仍是 NO_RETRIEVE；这类内容只交给 procedure/preference 检索 query 处理
- 出现“都有哪些/列举/所有/一共/总共/历史上”这类聚合问题 → RETRIEVE，并改写成覆盖主题的宽泛 query
- 出现“他/她/它/这个/那个/这东西/这玩意”这类指示词时，优先用近期对话消解为实际实体
- “你还记得吗/你知道我的/你记不记得/我跟你说过”等元问题是在查事实本身，history_query 要贴近记忆 summary
- 提到快递、物流、包裹、到货时，若语境指向用户最近购买行为，应查购买历史；纯工具查询可以不查
- 提到身体症状、药、复查时，若语境指向用户健康状态，应查健康档案或历史记录

<example id="new_service_rule_not_history">
用户消息：以后讲复杂问题先给我一个能贯穿始终的例子
输出：
<decision>NO_RETRIEVE</decision>
<history_query></history_query>
</example>

<example id="external_resource_not_history">
用户消息：【视频标题-示例站点】 https://short.example/item
输出：
<decision>NO_RETRIEVE</decision>
<history_query></history_query>
</example>

<example id="memory_question_history">
用户消息：你还记得我用的是哪个 Fitbit 吗
输出：
<decision>RETRIEVE</decision>
<history_query>用户使用的 Fitbit 设备型号</history_query>
</example>

只输出 XML，不要解释：
<decision>RETRIEVE|NO_RETRIEVE</decision>
<history_query>...</history_query>
"""


    @staticmethod
    def _build_procedure_prompt(user_msg: str) -> str:
        return f"""只输出一行检索 query，不要解释。

把用户消息改写成 preference/procedure 库能命中的 summary 风格查询：
- 用户希望 agent 怎样服务、讲解、推荐
- agent 在某类请求下必须怎么做、用什么工具
- 用户发来某类外部资源、文件、图片、链接时 agent 应如何处理
不要抽一次性标题词，要写可复用场景。
丢弃一次性标题、情绪词、短链路径；保留可复用类别词，如平台、资源类型、输入形态、用户动作、agent 应执行的对象。
如果用户消息里出现了具体平台名或资源类型，要保留这些可复用类别词；不要输出“某平台”“某资源”这类占位词。

<example id="procedure_explicit_command">
用户消息：把这个资源下载下来
输出：用户要求 agent 下载外部资源
</example>

<example id="procedure_direct_action">
用户消息：帮我把这个内容整理成表格
输出：用户要求 agent 整理内容
</example>

<example id="procedure_resource_share">
用户消息：【视频标题-哔哩哔哩】 https://short.example/item
输出：用户发送哔哩哔哩视频链接时 agent 应如何处理
</example>

<example id="procedure_document_link">
用户消息：这个文档链接你看一下 https://docs.example.com/item
输出：用户发送文档链接时 agent 应如何处理
</example>

<example id="procedure_attachment">
用户消息：这是文件，帮我处理一下
输出：用户发送文件并要求 agent 处理
</example>

<example id="procedure_media">
用户消息：帮我看看这张图
输出：用户发送图片并要求 agent 分析
</example>

<example id="preference_service_style">
用户消息：以后讲复杂问题先给我一个能贯穿始终的例子
输出：用户希望 agent 讲解复杂问题时先给贯穿始终的例子
</example>

<example id="preference_future_rule">
用户消息：以后遇到这种问题先给结论再解释
输出：用户希望 agent 回答时先给结论再解释
</example>

<example id="memory_answer_rule">
用户消息：你还记得我之前告诉你的设备型号吗
输出：用户询问记忆内容时 agent 应如何查找依据
</example>

<example id="answer_style_rule">
用户消息：解释一下这两个协议有什么区别
输出：用户询问知识问题时 agent 应如何组织回答
</example>

用户消息：{user_msg}
输出：
"""

    @staticmethod
    def _clean_procedure_query(raw_output: str) -> str:
        text = re.sub(r"\s+", " ", str(raw_output or "")).strip("。 .")
        if text.lower() in {"", "空", "无", "none", "null", "n/a", "not applicable", "(empty)"}:
            return ""
        return text
