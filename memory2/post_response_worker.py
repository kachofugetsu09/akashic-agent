"""回复后异步记忆提取与 supersede 处理。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import json_repair

from agent.provider import LLMProvider
from agent.model_runtime.registry import model_execution_scope
from core.memory.events import MemoryWritten, TurnIngested
from memory2.memorizer import Memorizer
from memory2.retriever import Retriever
from memory2.store import MemoryHit, memory_hit_score

if TYPE_CHECKING:
    from bus.publisher import EventPublisher

logger = logging.getLogger(__name__)

_LEGACY_MEMORY_ID = re.compile(
    r"(?:new|reinforced|merged):([A-Za-z0-9_-]{1,128})"
)
_EXPLICIT_MEMORY_ID = re.compile(r"item_id=([A-Za-z0-9:_-]{1,128})")


@dataclass(frozen=True)
class _RunContext:
    session_key: str
    channel: str
    chat_id: str
    source_ref: str


class PostResponseMemoryWorker:
    """在回复后检测并退休用户明确否定的旧记忆。"""

    SUPERSEDE_THRESHOLD = 0.82
    SUPERSEDE_CANDIDATE_K = 5
    TOKEN_BUDGET_PER_RUN = 1000
    TOKENS_EXTRACT_INVALIDATION = 96
    TOKENS_CHECK_INVALIDATE = 96

    def __init__(
        self,
        memorizer: Memorizer,
        retriever: Retriever,
        light_provider: LLMProvider,
        light_model: str,
        event_publisher: "EventPublisher | None" = None,
    ) -> None:
        self._memorizer = memorizer
        self._retriever = retriever
        self._provider = light_provider
        self._model = light_model
        self._event_publisher = event_publisher

    async def handle(self, event: TurnIngested) -> None:
        await self.run(
            user_msg=event.user_message,
            agent_response=event.assistant_response,
            tool_chain=list(event.tool_chain),
            source_ref=event.source_ref,
            session_key=event.session_key,
            channel=event.channel,
            chat_id=event.chat_id,
        )

    async def run(
        self,
        user_msg: str,
        agent_response: str,
        tool_chain: list[dict],
        source_ref: str,
        session_key: str = "",
        channel: str = "",
        chat_id: str = "",
    ) -> None:
        """在独立转次上下文中识别并废弃失效记忆。"""

        async with model_execution_scope(self._provider):
            await self._run_bound(
                user_msg=user_msg,
                agent_response=agent_response,
                tool_chain=tool_chain,
                source_ref=source_ref,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
            )

    async def _run_bound(
        self,
        *,
        user_msg: str,
        agent_response: str,
        tool_chain: list[dict],
        source_ref: str,
        session_key: str,
        channel: str,
        chat_id: str,
    ) -> None:
        """Process one post-response memory job inside its model snapshot."""

        # 1. 初始化本轮异步提炼的上下文和 token 预算。
        context = _RunContext(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            source_ref=source_ref,
        )
        token_budget = self.TOKEN_BUDGET_PER_RUN
        logger.debug(
            "post_response_memorize start session=%s source_ref=%s user_len=%d resp_len=%d tool_steps=%d",
            context.session_key or "-",
            context.source_ref or "-",
            len(user_msg.strip()),
            len(agent_response.strip()),
            len(tool_chain),
        )

        # 2. 保护本轮刚写入的记忆，避免紧接着被退休。
        protected_ids = self._collect_protected_memory_ids(tool_chain)
        logger.debug(
            "post_response_memorize explicit_memories session=%s protected_ids=%d",
            context.session_key or "-",
            len(protected_ids),
        )

        # 3. 隐式提炼已由 consolidation 负责，此处只处理显式废弃信号。
        token_budget = await self._handle_invalidations(
            user_msg,
            context,
            protected_ids,
            token_budget,
        )
        logger.debug(
            "post_response_memorize done session=%s source_ref=%s remain_budget=%d",
            context.session_key or "-",
            context.source_ref or "-",
            token_budget,
        )

    @staticmethod
    def _consume_budget(remain: int, cost: int) -> tuple[bool, int]:
        if remain < cost:
            return False, remain
        return True, remain - cost

    @staticmethod
    def _preview_text(text: str, limit: int = 80) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[:limit] + "..."

    def _collect_protected_memory_ids(self, tool_chain: list[dict]) -> set[str]:
        """收集本轮 memorize 工具真实写入的记忆 ID。"""

        protected_ids: set[str] = set()
        # 1. 遍历本轮工具调用，只解析 memorize 结果。
        for step in tool_chain:
            calls = step.get("calls", [])
            if not isinstance(calls, list):
                raise TypeError("tool_chain[].calls 必须是数组")
            for call in calls:
                if not isinstance(call, dict):
                    raise TypeError("tool_chain[].calls[] 必须是对象")
                if call.get("name") != "memorize":
                    continue

                # 2. 从结果中解析真实 ID，不重复校验 memorize 入参。
                result = call["result"]
                if not isinstance(result, str):
                    raise TypeError("memorize call result 必须是字符串")
                match = _LEGACY_MEMORY_ID.search(result) or _EXPLICIT_MEMORY_ID.search(
                    result
                )
                if match:
                    protected_ids.add(match.group(1))
        return protected_ids

    async def _handle_invalidations(
        self,
        user_msg: str,
        context: _RunContext,
        protected_ids: set[str],
        token_budget: int,
    ) -> int:
        """检测用户明确指出 agent 旧行为有误的情况，无需替代规则即直接 supersede 旧条目。"""
        # 1. 先从当前用户消息里提取"要废弃什么旧行为"的主题。
        topics, token_budget = await self._extract_invalidation_topics(
            user_msg,
            token_budget,
        )
        logger.debug(
            "post_response invalidation_topics session=%s count=%d remain_budget=%d topics=%s",
            context.session_key or "-",
            len(topics),
            token_budget,
            [self._preview_text(topic, 40) for topic in topics[:3]],
        )
        if not topics:
            return token_budget
        for topic in topics:
            # 2. 再到现有 procedure/preference 里召回和该主题最相关的旧条目。
            candidates = await self._retriever.retrieve(
                topic,
                memory_types=["procedure", "preference"],
            )
            high_sim = [
                c
                for c in candidates
                if memory_hit_score(c) >= self.SUPERSEDE_THRESHOLD
                and c["id"] not in protected_ids
            ][: self.SUPERSEDE_CANDIDATE_K]
            if not high_sim:
                continue

            # 3. 最后让 light model 判断这些旧条目里哪些该真正 supersede。
            supersede_ids, token_budget = await self._check_invalidate(
                topic,
                high_sim,
                token_budget,
            )
            if supersede_ids:
                self._memorizer.supersede_batch(supersede_ids)
                logger.info(
                    "post_response invalidation: superseded %s for topic '%s'",
                    supersede_ids,
                    topic,
                )
                if self._event_publisher is not None and context.session_key:
                    await self._event_publisher.fanout(
                        MemoryWritten(
                            session_key=context.session_key,
                            channel=context.channel,
                            chat_id=context.chat_id,
                            source_ref=context.source_ref,
                            action="supersede",
                            superseded_ids=supersede_ids,
                        )
                    )
        return token_budget

    @staticmethod
    def _parse_json_string_array(text: str, response_name: str) -> list[str]:
        """解析并校验模型返回的 JSON 字符串数组。"""
        # 1. 去掉模型常见的 Markdown 代码围栏。
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) < 2 or lines[-1].strip() != "```":
                raise ValueError(f"{response_name} 返回了未闭合的代码围栏")
            text = "\n".join(lines[1:-1]).strip()
        if not text:
            raise ValueError(f"{response_name} 返回了空内容")

        # 2. 解析 JSON；json_repair 只负责既有的轻微格式修复，不负责吞掉错误。
        result = json_repair.loads(text)
        if not isinstance(result, list):
            raise ValueError(f"{response_name} 必须返回 JSON 数组")

        # 3. 严格校验数组元素，区分合法空数组和损坏的模型响应。
        if any(not isinstance(item, str) or not item.strip() for item in result):
            raise ValueError(f"{response_name} 只能包含非空字符串")
        return result

    async def _extract_invalidation_topics(
        self,
        user_msg: str,
        token_budget: int,
    ) -> tuple[list[str], int]:
        """从用户消息中提取被明确声明为有误/需废弃的 agent 行为主题。"""
        # 1. 这里只负责抽取"被否定的行为主题"，不直接做 supersede 决策。
        prompt = f"""判断用户消息是否在明确声明 agent 某个现有行为/流程有误，且希望废弃它。

用户消息：{user_msg}

【必须同时满足才触发】
1. 用户表达了明确的否定/纠错/废弃意图——句子里有"错了/不对/不要再/忘掉/废弃/过时/改掉"等否定词
2. 否定的对象是 agent 的某个操作行为（不是用户自己的事，不是第三方信息）

【以下情况绝对不触发，返回 []】
✗ 用户在询问/确认 agent 的流程（"你的流程是什么""你怎么做的""你是按什么步骤"）
✗ 用户在描述/回顾自己的操作
✗ 用户提问句、疑问句（即使涉及 agent 行为）
✗ 含"也许/可能/猜测"等不确定措辞且无明确废弃指令

若触发，提取受影响的行为主题（简短描述，如"steam查询流程"）。
返回 JSON 数组，大多数消息应返回 []。"""
        ok, token_budget = self._consume_budget(
            token_budget,
            self.TOKENS_EXTRACT_INVALIDATION,
        )
        if not ok:
            logger.debug("post_response invalidation skipped: token budget exhausted")
            return [], token_budget

        resp = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            model=self._model,
            max_tokens=self.TOKENS_EXTRACT_INVALIDATION,
        )
        text = resp.content
        if not isinstance(text, str):
            raise ValueError("extract_invalidation_topics 未返回文本")
        topics = self._parse_json_string_array(
            text.strip(),
            "extract_invalidation_topics",
        )
        return topics, token_budget

    async def _check_invalidate(
        self,
        topic: str,
        candidates: list[MemoryHit],
        token_budget: int,
    ) -> tuple[list[str], int]:
        """用户声明旧行为有误时，判断哪些旧条目应被 supersede（无需新规则替代）。"""
        old_block = "\n".join(f"- id={c['id']} | {c['summary']}" for c in candidates)
        prompt = f"""用户明确表示 agent 关于"{topic}"的现有行为/流程有误，需要废弃。
以下是数据库中与该主题相关的现有规则，判断哪些应被标记为废弃：

{old_block}

规则：
- 若条目确实描述了"{topic}"相关的 agent 操作流程/行为，输出其 id
- 若条目与该主题无关，不输出
- 若无关联条目，返回 []

只返回 JSON 数组，如 ["abc123"] 或 []"""
        ok, token_budget = self._consume_budget(
            token_budget,
            self.TOKENS_CHECK_INVALIDATE,
        )
        if not ok:
            logger.debug(
                "post_response check_invalidate skipped: token budget exhausted"
            )
            return [], token_budget
        resp = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            model=self._model,
            max_tokens=self.TOKENS_CHECK_INVALIDATE,
        )
        text = resp.content
        if not isinstance(text, str):
            raise ValueError("check_invalidate 未返回文本")
        selected_ids = self._parse_json_string_array(
            text.strip(),
            "check_invalidate",
        )
        valid_ids = {c["id"] for c in candidates}
        unknown_ids = [item_id for item_id in selected_ids if item_id not in valid_ids]
        if unknown_ids:
            raise ValueError(
                f"check_invalidate 返回了未知候选 ID: {unknown_ids}"
            )
        return selected_ids, token_budget
