"""回复后异步记忆提取与 supersede 处理。"""

from __future__ import annotations

import logging
import re

import json_repair

from agent.provider import LLMProvider
from memory2.dedup_decider import DedupDecider, DedupDecision, MemoryAction
from memory2.memorizer import Memorizer
from memory2.profile_extractor import ProfileFact
from memory2.retriever import Retriever
from memory2.rule_schema import (
    build_procedure_rule_schema,
    procedure_rules_conflict,
    resolve_procedure_rule_schema,
)

logger = logging.getLogger(__name__)


class PostResponseMemoryWorker:
    """
    回复后异步执行：
    1. light model 提取隐式偏好
    2. 检测并退休矛盾旧条目
    3. 写入新条目
    """

    SUPERSEDE_THRESHOLD = 0.82
    SUPERSEDE_THRESHOLD_PROFILE = 0.78
    SUPERSEDE_CANDIDATE_K = 5
    TOKEN_BUDGET_PER_RUN = 1000
    TOKENS_EXTRACT_IMPLICIT = 384
    TOKENS_EXTRACT_INVALIDATION = 96
    TOKENS_CHECK_INVALIDATE = 96
    TOKENS_CHECK_SUPERSEDE = 96
    TOKENS_EXTRACT_PROFILE = 200
    TOKENS_SAVE_PROFILE = 96

    DEDUP_TYPES = frozenset({"procedure", "preference"})

    def __init__(
        self,
        memorizer: Memorizer,
        retriever: Retriever,
        light_provider: LLMProvider,
        light_model: str,
        tagger=None,  # ProcedureTagger | None
        profile_extractor=None,  # ProfileFactExtractor | None
        profile_supersede_enabled: bool = True,
        observe_writer=None,
        dedup_decider: DedupDecider | None = None,
    ) -> None:
        self._memorizer = memorizer
        self._retriever = retriever
        self._provider = light_provider
        self._model = light_model
        self._tagger = tagger
        self._profile_extractor = profile_extractor
        self._profile_supersede_enabled = profile_supersede_enabled
        self._observe_writer = observe_writer
        self._dedup_decider = dedup_decider
        self._current_run_session_key = ""

    async def run(
        self,
        user_msg: str,
        agent_response: str,
        tool_chain: list[dict],
        source_ref: str,
        session_key: str = "",
    ) -> None:
        self._current_run_session_key = session_key
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
            already_memorized, protected_ids = self._collect_explicit_memorized(
                tool_chain
            )
            logger.debug(
                "post_response_memorize explicit_memories session=%s summaries=%d protected_ids=%d",
                session_key or "-",
                len(already_memorized),
                len(protected_ids),
            )

            # 先处理"旧的有误/需要遗忘"的显式废弃信号，无需新规则即可 supersede
            token_budget = await self._handle_invalidations(
                user_msg,
                source_ref,
                protected_ids,
                token_budget,
            )

            new_items, token_budget = await self._extract_implicit(
                user_msg,
                agent_response,
                already_memorized,
                token_budget,
            )
            logger.debug(
                "post_response_memorize implicit_extracted session=%s count=%d remain_budget=%d",
                session_key or "-",
                len(new_items),
                token_budget,
            )
            new_items = await self._dedupe_against_explicit(
                new_items,
                already_memorized,
                protected_ids,
            )
            logger.debug(
                "post_response_memorize implicit_after_dedupe session=%s count=%d remain_budget=%d",
                session_key or "-",
                len(new_items),
                token_budget,
            )
            batch_vecs: list[tuple[list[float], dict]] = []
            for item in new_items:
                token_budget = await self._save_with_dedup(
                    item,
                    source_ref,
                    protected_ids,
                    token_budget,
                    batch_vecs=batch_vecs,
                )
            if self._profile_extractor is not None:
                token_budget = await self._run_profile_extraction(
                    user_msg,
                    agent_response,
                    source_ref,
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
    def _normalize_text(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[，。！？、；：,.!?;:\-—_`'\"()\[\]{}<>《》]", "", text)
        return text

    @staticmethod
    def _preview_text(text: str, limit: int = 80) -> str:
        compact = re.sub(r"\s+", " ", str(text or "").strip())
        if len(compact) <= limit:
            return compact
        return compact[:limit] + "..."

    @staticmethod
    def _coerce_memory_type(
        memory_type: str,
        tool_requirement: str | None,
        steps: list[str] | None,
    ) -> str:
        if memory_type not in ("procedure", "preference", "event", "profile"):
            return "procedure"
        return memory_type

    @staticmethod
    def _extract_obvious_preferences(user_msg: str) -> list[dict]:
        text = (user_msg or "").strip()
        if not text:
            return []
        if "中文" in text and any(k in text for k in ("跟我说话", "回复", "回答")):
            if any(k in text for k in ("只用中文", "不要夹杂英文", "别夹英文", "不用英文", "尽量翻译")):
                return [
                    {
                        "summary": text,
                        "memory_type": "preference",
                        "tool_requirement": None,
                        "steps": [],
                        "_force_create": True,
                    }
                ]
        return []

    async def _dedupe_against_explicit(
        self,
        items: list[dict],
        explicit_summaries: list[str],
        protected_ids: set[str],
    ) -> list[dict]:
        """过滤掉与本轮显式 memorize 同义的隐式条目，避免同轮重复写入。"""
        if not items:
            return []
        if not explicit_summaries:
            return items

        explicit_norms = [self._normalize_text(s) for s in explicit_summaries if s]
        filtered: list[dict] = []

        for item in items:
            if not isinstance(item, dict):
                continue
            summary = (item.get("summary") or "").strip()
            if not summary:
                continue

            cur = self._normalize_text(summary)
            # 1) 先做轻量文本去重
            text_dup = any(n and cur and (n in cur or cur in n) for n in explicit_norms)
            if text_dup:
                logger.debug(
                    "post_response_memorize skip implicit (explicit text-dup): %s",
                    summary,
                )
                continue

            # 2) 再做向量同义校验：命中本轮显式写入 id 且高相似时跳过
            mtype = item.get("memory_type", "procedure")
            if protected_ids and mtype in ("procedure", "preference"):
                try:
                    candidates = await self._retriever.retrieve(
                        summary,
                        memory_types=["procedure", "preference"],
                    )
                    if any(
                        c.get("id") in protected_ids and c.get("score", 0) >= 0.78
                        for c in candidates
                        if isinstance(c, dict)
                    ):
                        logger.debug(
                            "post_response_memorize skip implicit (explicit sem-dup): %s",
                            summary,
                        )
                        continue
                except Exception as e:
                    logger.warning(f"implicit-explicit dedupe retrieve failed: {e}")

            filtered.append(item)

        return filtered

    async def _handle_invalidations(
        self,
        user_msg: str,
        source_ref: str,
        protected_ids: set[str] | None = None,
        token_budget: int = TOKEN_BUDGET_PER_RUN,
    ) -> int:
        """检测用户明确指出 agent 旧行为有误的情况，无需替代规则即直接 supersede 旧条目。"""
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
                if self._observe_writer is not None and self._current_run_session_key:
                    try:
                        from core.observe.events import MemoryWriteTrace
                        self._observe_writer.emit(MemoryWriteTrace(
                            session_key=self._current_run_session_key,
                            source_ref=source_ref,
                            action="supersede",
                            superseded_ids=supersede_ids,
                        ))
                    except Exception:
                        pass
        return token_budget

    async def _extract_invalidation_topics(
        self,
        user_msg: str,
        token_budget: int,
    ) -> tuple[list[str], int]:
        """从用户消息中提取被明确声明为有误/需废弃的 agent 行为主题。"""
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
        fallback_items = [
            self._normalize_extracted_item(item)
            for item in self._extract_obvious_preferences(user_msg)
        ]

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

    def _collect_explicit_memorized(
        self, tool_chain: list[dict]
    ) -> tuple[list[str], set[str]]:
        """从 tool_chain 收集本轮 memorize tool 显式写入的 summary 和 DB id。

        返回 (summaries, protected_ids)：
        - summaries：传给 light model 的排除列表
        - protected_ids：memorize tool 本轮写入的条目 id，不允许被 worker supersede
        """
        import re as _re

        _id_pattern = _re.compile(r"(?:new|reinforced):([A-Za-z0-9_-]{8,64})")

        summaries: list[str] = []
        protected_ids: set[str] = set()
        for step in tool_chain:
            if not isinstance(step, dict):
                continue
            for call in step.get("calls", []):
                if not isinstance(call, dict) or call.get("name") != "memorize":
                    continue
                args = call.get("arguments")
                if isinstance(args, dict):
                    summary = (args.get("summary") or "").strip()
                    if summary:
                        summaries.append(summary)
                # 从 tool result 中解析写入的 DB id
                # 格式："已记住（new:0e750b742fa4）：..."
                result = call.get("result") or ""
                m = _id_pattern.search(result)
                if m:
                    protected_ids.add(m.group(1))
        return summaries, protected_ids

    async def _extract_implicit(
        self,
        user_msg: str,
        agent_response: str,
        already_memorized: list[str],
        token_budget: int,
    ) -> tuple[list[dict], int]:
        """light model 提取隐式偏好，返回 behavior_updates 格式列表。"""
        exclusion_block = ""
        if already_memorized:
            lines = "\n".join(f"- {s}" for s in already_memorized if s)
            exclusion_block = f"\n\n【本轮已显式记录，不要重复提取】\n{lines}"

        prompt = f"""你是记忆提取专家。从对话中提取对 agent 有长期价值的记忆条目。

默认答案是 []。只有通过以下三步检查，才写入一条条目。

【步骤一：引用 USER 原话】
提取任何条目前，先在脑中写出引用：USER 说了哪句话支撑这条记忆？
- 无法在 USER 原话中找到直接支撑句 → 不提取，返回 []
- ASSISTANT 的回复只是背景，不能作为证据；即使 ASSISTANT 描述了正确的流程或规则，只要 USER 没有说出对应要求，就不能提取

【步骤二：判断主体与类型】
此步骤只输出 procedure 或 preference，不输出 event（event 由其他模块负责）。

类型判断（二选一，不要混用）：

问题 A：这条规则是否在说"agent 做某类事情时应该怎么做"？
  （讲解方式、调查方式、回复风格、工具使用、操作流程）
  是 → procedure
  例："查资料时别只看 GitHub" / "讲流程时不能省略前置步骤" / "简单问题直接回答"

问题 B：这条规则是否在说"用户喜欢或厌恶某类内容本身"？
  （作品、题材、游戏、音乐、技术框架、信息来源的个人喜好）
  是 → preference（必须有"以后""下次""不要再""别再"等未来约束词或稳定喜好信号）
  例："用户讨厌恐怖游戏" / "用户不喜欢某作者" / "以后别推荐 X 框架"

两个问题都不是，或只是一次性感受/感悟 → 不提取
✗ 主体是外部系统、技术协议、算法 → 不提取

【步骤三：泛化检查】
summary 中不得出现任何具体实体名称（项目名、产品名、平台名、人名）：
- 将 summary 中所有具体实体名替换成"某X"，若规则失去意义 → 无法泛化，不提取
- 若规则仍然成立 → 按泛化后的无实体名形式写入

【输出要求】
- 返回 [] 是正常且正确的结果，绝大多数对话应返回 []
- 每条条目独立通过三步检查；不要为了凑数写第二条
- summary 以客观规则形式书写（如"遇到简单概念类问题时应直接回答""调查时应参考多方来源"），
  不得以 agent 第一人称表达（"以后我会...""以后简单问题就直接答"这类均不合格）
- summary 语气不得强于 USER 原话；不要把"单次评价"升级成"长期禁令"
- 已显式记录的内容不要重复提取{exclusion_block}

【对话内容】
USER: {user_msg}
ASSISTANT: {agent_response}

只返回合法 JSON 数组，无内容时返回 []。
每项格式：{{"summary": "...", "memory_type": "procedure|preference", "tool_requirement": null或"工具名", "steps": [], "rule_schema": {{"required_tools": [], "forbidden_tools": [], "mentioned_tools": []}}}}

【tool_requirement 填写规则】
- 若该条目要求 agent 在某类请求下必须调用特定工具/skill，填写工具关键名称；否则填 null

【rule_schema 填写规则】
- 仅对 procedure 填写；preference 可省略或填空对象
- required_tools：用户明确要求必须使用的工具
- forbidden_tools：用户明确禁止的工具
- mentioned_tools：规则涉及的工具别名
- 无法确认的约束留空，不要猜"""

        ok, token_budget = self._consume_budget(
            token_budget,
            self.TOKENS_EXTRACT_IMPLICIT,
        )
        if not ok:
            logger.info(
                "post_response extract_implicit skipped: token budget exhausted"
            )
            return [], token_budget

        fallback_items = [
            self._normalize_extracted_item(item)
            for item in self._extract_obvious_preferences(user_msg)
        ]

        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=self.TOKENS_EXTRACT_IMPLICIT,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if isinstance(result, list):
                llm_items = [
                    self._normalize_extracted_item(r)
                    for r in result
                    if isinstance(r, dict) and r.get("summary")
                ]
                items = self._merge_extracted_items(fallback_items, llm_items)
                log_fn = logger.info if items else logger.debug
                log_fn(
                    "post_response_memorize extract_result session=%s count=%d items=%s",
                    self._current_run_session_key or "-",
                    len(items),
                    [
                        f"{str(item.get('memory_type', 'procedure'))}:{self._preview_text(item.get('summary', ''), 50)}"
                        for item in items[:3]
                    ],
                )
                return items, token_budget
        except Exception as e:
            logger.warning(f"post_response_memorize extract failed: {e}")
        return fallback_items, token_budget

    @staticmethod
    def _merge_extracted_items(
        first: list[dict],
        second: list[dict],
    ) -> list[dict]:
        merged: list[dict] = []
        seen: set[str] = set()
        for bucket in (first, second):
            for item in bucket:
                if not isinstance(item, dict):
                    continue
                summary = str(item.get("summary", "") or "").strip()
                if not summary:
                    continue
                key = PostResponseMemoryWorker._normalize_text(summary)
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    @staticmethod
    def _normalize_extracted_item(item: dict) -> dict:
        normalized = dict(item)
        normalized["memory_type"] = PostResponseMemoryWorker._coerce_memory_type(
            str(normalized.get("memory_type", "procedure") or "procedure"),
            normalized.get("tool_requirement"),
            normalized.get("steps"),
        )
        if normalized.get("memory_type") == "procedure":
            normalized["rule_schema"] = build_procedure_rule_schema(
                summary=str(normalized.get("summary", "") or ""),
                tool_requirement=normalized.get("tool_requirement"),
                steps=normalized.get("steps") or [],
                rule_schema=normalized.get("rule_schema"),
            )
        return normalized

    async def _save_item_direct(
        self,
        item: dict,
        source_ref: str,
        token_budget: int,
    ) -> tuple[int, str | None]:
        """仅执行写入，不做 supersede 检查。返回 (remaining_budget, saved_id|None)。
        保留 rule_schema 构建、tagger 标注、observe emit 全部逻辑。
        """
        summary = (item.get("summary") or "").strip()
        if not summary:
            return token_budget, None

        mtype = item.get("memory_type", "procedure")
        if mtype not in ("procedure", "preference", "event", "profile"):
            mtype = "procedure"

        extra: dict = {
            "tool_requirement": item.get("tool_requirement"),
            "steps": item.get("steps") or [],
        }
        if mtype == "procedure":
            extra["rule_schema"] = build_procedure_rule_schema(
                summary=summary,
                tool_requirement=item.get("tool_requirement"),
                steps=item.get("steps") or [],
                rule_schema=item.get("rule_schema"),
            )
        if mtype == "procedure" and self._tagger is not None:
            try:
                trigger_tags = await self._tagger.tag(summary)
                if trigger_tags is not None:
                    extra["trigger_tags"] = trigger_tags
                    logger.debug(
                        "post_response_memorize: trigger_tags generated scope=%s",
                        trigger_tags.get("scope"),
                    )
            except Exception as e:
                logger.warning(
                    "post_response_memorize: trigger_tags generation failed: %s", e
                )
        try:
            result = await self._memorizer.save_item(
                summary=summary,
                memory_type=mtype,
                extra=extra,
                source_ref=source_ref,
            )
            logger.info(
                "post_response_memorize saved session=%s source_ref=%s type=%s result=%s summary=%s",
                self._current_run_session_key or "-",
                source_ref or "-",
                mtype,
                result,
                self._preview_text(summary, 80),
            )
            if self._observe_writer is not None and self._current_run_session_key:
                try:
                    from core.observe.events import MemoryWriteTrace
                    self._observe_writer.emit(MemoryWriteTrace(
                        session_key=self._current_run_session_key,
                        source_ref=source_ref,
                        action="write",
                        memory_type=mtype,
                        item_id=result,
                        summary=summary,
                    ))
                except Exception:
                    pass
            saved_id = result.split(":", 1)[1] if ":" in result else None
            return token_budget, saved_id
        except Exception as e:
            logger.warning("post_response_memorize save failed: %s", e)
            return token_budget, None

    async def _save_with_supersede(
        self,
        item: dict,
        source_ref: str,
        protected_ids: set[str] | None = None,
        token_budget: int = TOKEN_BUDGET_PER_RUN,
    ) -> int:
        """写入新条目，同时检测并退休矛盾的旧条目。save 段委托给 _save_item_direct。"""
        summary = (item.get("summary") or "").strip()
        if not summary:
            return token_budget

        mtype = item.get("memory_type", "procedure")
        if mtype not in ("procedure", "preference", "event", "profile"):
            mtype = "procedure"

        _protected = protected_ids or set()
        if mtype in ("procedure", "preference"):
            candidates = await self._retriever.retrieve(
                summary,
                memory_types=["procedure", "preference"],
            )
            high_sim = [
                c
                for c in candidates
                if isinstance(c, dict)
                and c.get("score", 0) >= self.SUPERSEDE_THRESHOLD
                and c.get("id") not in _protected
            ][: self.SUPERSEDE_CANDIDATE_K]

            if high_sim:
                supersede_ids: list[str] = []
                if mtype == "procedure":
                    rule_schema = build_procedure_rule_schema(
                        summary=summary,
                        tool_requirement=item.get("tool_requirement"),
                        steps=item.get("steps") or [],
                        rule_schema=item.get("rule_schema"),
                    )
                    supersede_ids = [
                        str(candidate.get("id", ""))
                        for candidate in high_sim
                        if procedure_rules_conflict(
                            rule_schema,
                            resolve_procedure_rule_schema(
                                str(candidate.get("summary", "") or ""),
                                candidate.get("extra_json") or {},
                            ),
                        )
                    ]
                remaining_candidates = [
                    candidate
                    for candidate in high_sim
                    if str(candidate.get("id", "")) not in set(supersede_ids)
                ]
                if remaining_candidates:
                    llm_supersede_ids, token_budget = await self._check_supersede(
                        summary,
                        remaining_candidates,
                        token_budget,
                    )
                    for item_id in llm_supersede_ids:
                        if item_id not in supersede_ids:
                            supersede_ids.append(item_id)
                if supersede_ids:
                    self._memorizer.supersede_batch(supersede_ids)
                    if self._observe_writer is not None and self._current_run_session_key:
                        try:
                            from core.observe.events import MemoryWriteTrace
                            self._observe_writer.emit(MemoryWriteTrace(
                                session_key=self._current_run_session_key,
                                source_ref=source_ref,
                                action="supersede",
                                superseded_ids=supersede_ids,
                            ))
                        except Exception:
                            pass

        token_budget, _ = await self._save_item_direct(item, source_ref, token_budget)
        return token_budget

    async def _save_with_dedup(
        self,
        item: dict,
        source_ref: str,
        protected_ids: set[str] | None,
        token_budget: int,
        batch_vecs: list[tuple[list[float], dict]] | None = None,
    ) -> int:
        """去重路由层：procedure/preference 走 DedupDecider，其他走原路径。"""
        summary = (item.get("summary") or "").strip()
        if not summary:
            return token_budget

        mtype = item.get("memory_type", "procedure")

        if item.get("_force_create"):
            remain, saved_id = await self._save_item_direct(item, source_ref, token_budget)
            if saved_id and batch_vecs is not None and self._dedup_decider is not None:
                query_vec = await self._dedup_decider._embedder.embed(summary)
                batch_vecs.append(
                    (
                        query_vec,
                        {
                            "id": saved_id,
                            "summary": summary,
                            "memory_type": mtype,
                            "_batch_internal": True,
                        },
                    )
                )
            return remain

        # 类型守卫：非 procedure/preference，或未启用 dedup，走原路径
        if mtype not in self.DEDUP_TYPES or self._dedup_decider is None:
            return await self._save_with_supersede(item, source_ref, protected_ids, token_budget)

        result = await self._dedup_decider.decide(candidate=item, batch_vecs=batch_vecs)
        logger.debug("dedup decision=%s reason=%s", result.decision, result.reason)

        _protected = protected_ids or set()

        merge_actions = [
            a for a in (result.actions or [])
            if a.action == MemoryAction.MERGE and a.item_id not in _protected
        ]
        delete_ids = [
            a.item_id for a in (result.actions or [])
            if a.action == MemoryAction.DELETE and a.item_id not in _protected
        ]

        # SKIP → 不写不改
        if result.decision == DedupDecision.SKIP:
            logger.debug("dedup skip: %s", summary[:60])
            return token_budget

        # NONE → 不写候选，处理旧条目
        if result.decision == DedupDecision.NONE:
            if merge_actions:
                # 有 merge：先尝试 merge，成功后才执行 delete
                # 顺序重要：避免"先删后并失败"导致信息丢失
                target = merge_actions[0]  # MVP 守卫保证最多 1 个
                merged_summary = await self._build_merge_summary(
                    candidate_summary=summary,
                    existing_summary=target.summary,
                    token_budget=token_budget,
                )
                if merged_summary is not None:
                    await self._memorizer.merge_item(target.item_id, merged_summary)
                    logger.info("dedup merge into id=%s", target.item_id)
                    if delete_ids:
                        self._memorizer.supersede_batch(delete_ids)
                        logger.info("dedup delete(supersede) ids=%s", delete_ids)
                else:
                    # merge 失败：不删、不写候选，保持现状（信息无损）
                    logger.info(
                        "dedup merge skipped (budget/llm failure), old item preserved, "
                        "candidate dropped: %s", summary[:60],
                    )
            elif delete_ids:
                # 无 merge，只有 delete：直接执行（无信息丢失风险）
                self._memorizer.supersede_batch(delete_ids)
                logger.info("dedup delete(supersede) ids=%s", delete_ids)
            return token_budget

        # CREATE → 先执行 delete，再写入新条目
        if delete_ids:
            self._memorizer.supersede_batch(delete_ids)
            logger.info("dedup delete(supersede) ids=%s", delete_ids)

        token_budget, saved_id = await self._save_item_direct(item, source_ref, token_budget)
        if result.query_vector is not None and saved_id and batch_vecs is not None:
            batch_vecs.append((result.query_vector, {"id": saved_id, "summary": summary}))

        return token_budget

    async def _build_merge_summary(
        self,
        candidate_summary: str,
        existing_summary: str,
        token_budget: int,
    ) -> str | None:
        """生成合并摘要。失败时返回 None，调用方应跳过 merge 而非覆写旧条目。"""
        COST = 128
        ok, _ = self._consume_budget(token_budget, COST)
        if not ok:
            logger.debug("_build_merge_summary skipped: budget exhausted")
            return None

        prompt = (
            f"将以下两条记忆合并为一条简洁的记忆摘要，保留所有有效信息：\n\n"
            f"现有记忆：{existing_summary}\n\n"
            f"新增信息：{candidate_summary}\n\n"
            f"只返回合并后的摘要文本，不加说明。"
        )
        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=COST,
            )
            merged = (resp.content or "").strip()
            if not merged:
                logger.warning("_build_merge_summary: LLM returned empty")
                return None
            return merged
        except Exception as e:
            logger.warning("_build_merge_summary failed: %s", e)
            return None

    async def _run_profile_extraction(
        self,
        user_msg: str,
        agent_response: str,
        source_ref: str,
        token_budget: int,
    ) -> int:
        # 1. 先检查本轮是否还有 profile 提取预算。
        ok, token_budget = self._consume_budget(
            token_budget,
            self.TOKENS_EXTRACT_PROFILE,
        )
        if not ok:
            logger.debug("post_response profile extraction skipped: token budget exhausted")
            return token_budget
        try:
            # 2. 再加载少量近期 profile 摘要，供 extractor 做轻量查重提示。
            existing_profile = await self._load_existing_profile_context(user_msg)
            logger.debug(
                "post_response profile_context session=%s existing_count=%d",
                self._current_run_session_key or "-",
                len([line for line in existing_profile.splitlines() if line.strip()]),
            )

            # 3. 最后提取并逐条写入 profile facts。
            facts = await self._profile_extractor.extract_from_exchange(
                user_msg,
                agent_response,
                existing_profile=existing_profile,
            )
            log_fn = logger.info if facts else logger.debug
            log_fn(
                "post_response profile_extracted session=%s count=%d facts=%s remain_budget=%d",
                self._current_run_session_key or "-",
                len(facts),
                [
                    f"{fact.category}:{self._preview_text(fact.summary, 50)}"
                    for fact in facts[:3]
                ],
                token_budget,
            )
            for fact in facts:
                token_budget = await self._save_profile_with_supersede(
                    fact,
                    source_ref,
                    token_budget,
                )
        except Exception as e:
            logger.warning("per-turn profile extraction failed: %s", e)
        return token_budget

    async def _load_existing_profile_context(self, query: str) -> str:
        try:
            items = await self._retriever.retrieve(
                query,
                memory_types=["profile"],
                top_k=3,
            )
        except Exception as e:
            logger.warning("post_response existing_profile retrieve failed: %s", e)
            return ""

        lines: list[str] = []
        seen: set[str] = set()
        for item in items:
            summary = str(item.get("summary", "") or "").strip()
            key = self._normalize_text(summary)
            if not summary or not key or key in seen:
                continue
            seen.add(key)
            lines.append(summary)
        return "\n".join(lines)

    async def _save_profile_with_supersede(
        self,
        fact: ProfileFact,
        source_ref: str,
        token_budget: int,
    ) -> int:
        # 1. 先写入新条目，保证 supersede 检查失败时新事实仍保留。
        try:
            saved_result = await self._memorizer.save_item(
                summary=fact.summary,
                memory_type="profile",
                extra={"category": fact.category},
                source_ref=source_ref,
                happened_at=fact.happened_at,
            )
            logger.info(
                "post_response profile_saved session=%s source_ref=%s category=%s result=%s summary=%s happened_at=%s",
                self._current_run_session_key or "-",
                source_ref or "-",
                fact.category,
                saved_result,
                self._preview_text(fact.summary, 80),
                fact.happened_at or "-",
            )
        except Exception as e:
            logger.warning("post_response profile save failed: %s", e)
            return token_budget

        if fact.category not in {"status", "purchase"} or not self._profile_supersede_enabled:
            return token_budget

        ok, token_budget = self._consume_budget(token_budget, self.TOKENS_SAVE_PROFILE)
        if not ok:
            logger.debug("post_response profile supersede skipped: token budget exhausted")
            return token_budget

        try:
            candidates = await self._retriever.retrieve(
                fact.summary,
                memory_types=["profile"],
                top_k=self.SUPERSEDE_CANDIDATE_K,
            )
        except Exception as e:
            logger.warning("post_response profile retrieve failed: %s", e)
            return token_budget

        saved_id = self._parse_saved_item_id(saved_result)
        high_sim = [
            item
            for item in candidates
            if isinstance(item, dict)
            and item.get("score", 0) >= self.SUPERSEDE_THRESHOLD_PROFILE
            and (item.get("extra_json") or {}).get("category") == fact.category
            and str(item.get("id", "")) != saved_id
        ][: self.SUPERSEDE_CANDIDATE_K]
        if not high_sim:
            return token_budget

        try:
            supersede_ids, token_budget = await self._check_supersede(
                fact.summary,
                high_sim,
                token_budget,
                consume_budget=False,
            )
        except Exception as e:
            logger.warning("profile supersede check failed: %s", e)
            return token_budget
        if supersede_ids:
            logger.info(
                "post_response profile_supersede session=%s saved_id=%s supersede_ids=%s summary=%s",
                self._current_run_session_key or "-",
                saved_id or "-",
                supersede_ids,
                self._preview_text(fact.summary, 80),
            )
            self._memorizer.supersede_batch(supersede_ids)
        return token_budget

    @staticmethod
    def _parse_saved_item_id(result: str) -> str:
        text = str(result or "")
        if ":" not in text:
            return ""
        return text.split(":", 1)[1].strip()

    async def _check_supersede(
        self,
        new_summary: str,
        candidates: list[dict],
        token_budget: int,
        consume_budget: bool = True,
    ) -> tuple[list[str], int]:
        """让 light model 判断新条目覆盖了哪些旧条目。"""
        old_block = "\n".join(f"- id={c['id']} | {c['summary']}" for c in candidates)
        prompt = f"""判断新规则是否覆盖/取代了以下旧规则。

新规则：{new_summary}

旧规则列表：
{old_block}

规则：
- 若新规则与旧规则语义矛盾（新的做法取代旧做法），输出旧规则的 id
- 若新规则是旧规则的细化/补充，不输出
- 若无矛盾，返回空数组

只返回 JSON 数组，如 ["abc123", "def456"] 或 []"""
        if consume_budget:
            ok, token_budget = self._consume_budget(
                token_budget,
                self.TOKENS_CHECK_SUPERSEDE,
            )
            if not ok:
                logger.debug(
                    "post_response check_supersede skipped: token budget exhausted"
                )
                return [], token_budget

        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=self.TOKENS_CHECK_SUPERSEDE,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if isinstance(result, list):
                valid_ids = {c["id"] for c in candidates}
                return [
                    item_id
                    for item_id in result
                    if isinstance(item_id, str) and item_id in valid_ids
                ], token_budget
        except Exception as e:
            logger.warning(f"supersede check failed: {e}")
        return [], token_budget
