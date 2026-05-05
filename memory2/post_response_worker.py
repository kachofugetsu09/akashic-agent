"""回复后异步记忆提取与 supersede 处理。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import json_repair

from agent.provider import LLMProvider
from core.memory.events import MemoryWritten, TurnIngested
from memory2.memorizer import Memorizer
from memory2.retriever import Retriever

if TYPE_CHECKING:
    from bus.publisher import EventPublisher

logger = logging.getLogger(__name__)


class PostResponseMemoryWorker:
    """
    回复后异步执行：
    1. 检测并退休用户明确否定的旧行为（invalidation）。

    隐式 procedure/preference/profile 提炼已移至 consolidation 窗口期，
    与 event 提取并行、用主模型处理，不再在每轮 post-response 里跑。
    """

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
        self._current_run_session_key = ""
        self._current_run_channel = ""
        self._current_run_chat_id = ""

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
        # 1. 初始化本轮异步提炼的上下文和 token 预算。
        self._current_run_session_key = session_key
        self._current_run_channel = channel
        self._current_run_chat_id = chat_id
        token_budget = self.TOKEN_BUDGET_PER_RUN
        logger.debug(
            "post_response_memorize start session=%s source_ref=%s user_len=%d resp_len=%d tool_steps=%d",
            session_key or "-",
            source_ref or "-",
            len((user_msg or "").strip()),
            len((agent_response or "").strip()),
            len(tool_chain or []),
        )
        try:
            # 2. 先从本轮 tool_chain 里找显式 memorize 结果，后续 supersede 都要用。
            already_memorized, protected_ids = self._collect_explicit_memorized(
                tool_chain
            )
            logger.debug(
                "post_response_memorize explicit_memories session=%s summaries=%d protected_ids=%d",
                session_key or "-",
                len(already_memorized),
                len(protected_ids),
            )

            # 3. 处理"旧的有误/需要遗忘"的显式废弃信号，优先退休旧记忆。
            # 隐式 procedure/preference/profile 提炼已移至 consolidation 窗口期，
            # 与 event 提取并行、用主模型处理，不再在每轮 post-response 里跑。
            token_budget = await self._handle_invalidations(
                user_msg,
                source_ref,
                protected_ids,
                token_budget,
            )

            logger.debug(
                "post_response_memorize done session=%s source_ref=%s remain_budget=%d",
                session_key or "-",
                source_ref or "-",
                token_budget,
            )
        except Exception as e:
            logger.warning(f"post_response_memorize run failed: {e}")

    @staticmethod
    def _consume_budget(remain: int, cost: int) -> tuple[bool, int]:
        if remain < cost:
            return False, remain
        return True, remain - cost

    @staticmethod
    def _preview_text(text: str, limit: int = 80) -> str:
        import re
        compact = re.sub(r"\s+", " ", str(text or "").strip())
        if len(compact) <= limit:
            return compact
        return compact[:limit] + "..."

    def _collect_explicit_memorized(
        self, tool_chain: list[dict]
    ) -> tuple[list[str], set[str]]:
        """从 tool_chain 收集本轮 memorize tool 显式写入的 summary 和 DB id。

        返回 (summaries, protected_ids)：
        - summaries：传给 light model 的排除列表
        - protected_ids：memorize tool 本轮写入的条目 id，不允许被 worker supersede
        """
        import re as _re

        _legacy_pattern = _re.compile(
            r"(?:new|reinforced|merged):([A-Za-z0-9_-]{1,128})"
        )
        _explicit_pattern = _re.compile(r"item_id=([A-Za-z0-9:_-]{1,128})")

        summaries: list[str] = []
        protected_ids: set[str] = set()
        # 1. 遍历本轮工具调用，只关心 memorize 工具。
        for step in tool_chain:
            if not isinstance(step, dict):
                continue
            for call in step.get("calls", []):
                if not isinstance(call, dict) or call.get("name") != "memorize":
                    continue
                # 2. 从参数里拿 summary，后面给隐式提取做排重。
                args = call.get("arguments")
                if isinstance(args, dict):
                    summary = (args.get("summary") or "").strip()
                    if summary:
                        summaries.append(summary)
                # 3. 再从工具结果文本里解析真实写入的 DB id，避免后续误删本轮新记忆。
                result = call.get("result") or ""
                m = _legacy_pattern.search(result)
                if m:
                    protected_ids.add(m.group(1))
                    continue
                m = _explicit_pattern.search(result)
                if m:
                    protected_ids.add(m.group(1))
        return summaries, protected_ids

    async def _handle_invalidations(
        self,
        user_msg: str,
        source_ref: str,
        protected_ids: set[str] | None = None,
        token_budget: int = TOKEN_BUDGET_PER_RUN,
    ) -> int:
        """检测用户明确指出 agent 旧行为有误的情况，无需替代规则即直接 supersede 旧条目。"""
        # 1. 先从当前用户消息里提取"要废弃什么旧行为"的主题。
        topics, token_budget = await self._extract_invalidation_topics(
            user_msg,
            token_budget,
        )
        logger.debug(
            "post_response invalidation_topics session=%s count=%d remain_budget=%d topics=%s",
            self._current_run_session_key or "-",
            len(topics),
            token_budget,
            [self._preview_text(topic, 40) for topic in topics[:3]],
        )
        if not topics:
            return token_budget
        _protected = protected_ids or set()
        for topic in topics:
            # 2. 再到现有 procedure/preference 里召回和该主题最相关的旧条目。
            candidates = await self._retriever.retrieve(
                topic,
                memory_types=["procedure", "preference"],
            )
            high_sim = [
                c
                for c in candidates
                if isinstance(c, dict)
                and c.get("score", 0) >= self.SUPERSEDE_THRESHOLD
                and c.get("id") not in _protected
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
                if self._event_publisher is not None and self._current_run_session_key:
                    await self._event_publisher.fanout(
                        MemoryWritten(
                            session_key=self._current_run_session_key,
                            channel=self._current_run_channel,
                            chat_id=self._current_run_chat_id,
                            source_ref=source_ref,
                            action="supersede",
                            superseded_ids=supersede_ids,
                        )
                    )
        return token_budget

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

        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=self.TOKENS_EXTRACT_INVALIDATION,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if isinstance(result, list):
                return [
                    t for t in result if isinstance(t, str) and t.strip()
                ], token_budget
        except Exception as e:
            logger.warning(f"extract_invalidation_topics failed: {e}")
        return [], token_budget

    async def _check_invalidate(
        self,
        topic: str,
        candidates: list[dict],
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
        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=self.TOKENS_CHECK_INVALIDATE,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if isinstance(result, list):
                valid_ids = {c["id"] for c in candidates}
                return [
                    i for i in result if isinstance(i, str) and i in valid_ids
                ], token_budget
        except Exception as e:
            logger.warning(f"check_invalidate failed: {e}")
        return [], token_budget
