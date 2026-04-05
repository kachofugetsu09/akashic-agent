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
    TOKENS_FINALIZE_IMPLICIT = 220
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
        # 1. 初始化本轮异步提炼的上下文和 token 预算。
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
            # 2. 先从本轮 tool_chain 里找显式 memorize 结果，后续去重和 supersede 都要用。
            already_memorized, protected_ids = self._collect_explicit_memorized(
                tool_chain
            )
            logger.debug(
                "post_response_memorize explicit_memories session=%s summaries=%d protected_ids=%d",
                session_key or "-",
                len(already_memorized),
                len(protected_ids),
            )

            # 3. 先处理"旧的有误/需要遗忘"的显式废弃信号，优先退休旧记忆。
            token_budget = await self._handle_invalidations(
                user_msg,
                source_ref,
                protected_ids,
                token_budget,
            )

            # 4. 再从 user_msg + agent_response 里提取隐式长期记忆候选。
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

            # 5. 用本轮显式 memorize 结果过滤隐式候选，避免同轮重复写入。
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

            # 6. 逐条做去重 / supersede / 保存；batch_vecs 用来复用本轮向量结果。
            batch_vecs: list[tuple[list[float], dict]] = []
            for item in new_items:
                token_budget = await self._save_with_dedup(
                    item,
                    source_ref,
                    protected_ids,
                    token_budget,
                    batch_vecs=batch_vecs,
                )

            # 7. 最后按需补做 profile 提取，这条链和普通 preference/procedure 分开。
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

    async def _extract_implicit(
        self,
        user_msg: str,
        agent_response: str,
        already_memorized: list[str],
        token_budget: int,
    ) -> tuple[list[dict], int]:
        """light model 提取隐式偏好，返回 behavior_updates 格式列表。"""
        # 1. 先把本轮显式 memorize 的内容做成排除块，避免重复抽取。
        exclusion_block = ""
        if already_memorized:
            lines = "\n".join(f"- {s}" for s in already_memorized if s)
            exclusion_block = f"\n\n【本轮已显式记录，不要重复提取】\n{lines}"

        prompt = self._build_implicit_prompt(
            user_msg=user_msg,
            agent_response=agent_response,
            exclusion_block=exclusion_block,
        )

        ok, token_budget = self._consume_budget(
            token_budget,
            self.TOKENS_EXTRACT_IMPLICIT,
        )
        if not ok:
            logger.info(
                "post_response extract_implicit skipped: token budget exhausted"
            )
            return [], token_budget

        # 2. 再准备一个确定性 fallback，兜底少量高置信偏好规则。
        fallback_items = [
            self._normalize_extracted_item(item)
            for item in self._extract_obvious_preferences(user_msg)
        ]

        try:
            # 3. 主路径由 light model 产出候选，再经过 normalize + heuristic 过滤。
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
                llm_items = [
                    item for item in llm_items
                    if not PostResponseMemoryWorker._should_drop_by_heuristic(item)
                ]
                # 4. 二阶段收口：不是所有候选都值得进长期记忆。
                llm_items, token_budget = await self._finalize_implicit_candidates(
                    user_msg=user_msg,
                    agent_response=agent_response,
                    candidates=llm_items,
                    token_budget=token_budget,
                )
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

    async def _finalize_implicit_candidates(
        self,
        *,
        user_msg: str,
        agent_response: str,
        candidates: list[dict],
        token_budget: int,
    ) -> tuple[list[dict], int]:
        """二阶段收口：判断候选是否真的值得进入长期 procedure/preference。"""
        # 1. 第一阶段只是"提候选"；这里才决定是否真的入长期库。
        if not candidates:
            return [], token_budget

        ok, token_budget = self._consume_budget(
            token_budget,
            self.TOKENS_FINALIZE_IMPLICIT,
        )
        if not ok:
            logger.info(
                "post_response finalize_implicit skipped: token budget exhausted"
            )
            return [], token_budget

        prompt = self._build_finalize_prompt(
            user_msg=user_msg,
            agent_response=agent_response,
            candidates=candidates,
        )
        try:
            # 2. 让 light model 做最后一轮筛选和 type 修正。
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=self.TOKENS_FINALIZE_IMPLICIT,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if isinstance(result, list):
                finalized = [
                    self._normalize_extracted_item(item)
                    for item in result
                    if isinstance(item, dict) and item.get("summary")
                ]
                finalized = [
                    item for item in finalized
                    if not PostResponseMemoryWorker._should_drop_by_heuristic(item)
                ]
                return finalized, token_budget
        except Exception as e:
            logger.warning(f"post_response finalize_implicit failed: {e}")
        return [], token_budget

    @staticmethod
    def _build_implicit_prompt(
        *,
        user_msg: str,
        agent_response: str,
        exclusion_block: str = "",
    ) -> str:
        return f"""你是记忆提取专家，负责从一轮对话中识别用户的长期偏好和 agent 应长期遵守的规则。

默认答案是 []。提取门槛要高，宁可不提取，也不要把当前对话的局部信息误写成长期记忆。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【核心判断标准】
一条信息是否值得提取，只看一件事：
把这条信息放进 6 个月后的一次全新对话，它还有用吗？
→ 是 → 可能是长期记忆，继续检查
→ 否 → 不是长期记忆，返回 []
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【提取前必须完成三项检查，顺序执行，任一不通过即返回 []】

▸ 检查 A — USER 原话锚点
在 USER 消息里找到支撑这条记忆的直接原句（逐字存在，不是推断）。
  - 找不到 USER 的直接原句 → 返回 []
  - ASSISTANT 的解释、建议、工具返回的数据，不算 USER 原句
  - USER 没有反驳 ASSISTANT ≠ USER 认同且希望长期记忆
  - USER 消息是纯状态汇报（"复习中"/"在看书"/"工作中"等，没有任何偏好或规则表达）→ 没有可提取内容，返回 []

▸ 检查 B — 时效性
这条信息是否只对当前这次对话成立？
  - 涉及当前任务、当前时间段、当前情境（本次/今天/这个项目/明天的事）→ 返回 []
  - 只有明确跨 session 稳定成立，才继续

▸ 检查 C — 来源方向
这条信息的核心内容来自谁？
  - ASSISTANT 解释了某知识点、给出了建议、工具返回了数据 → 返回 []
  - 即使内容是对的、有意义的，只要来源是 ASSISTANT，就不提取
  - ASSISTANT 主动给出建议，USER 没有明确说"以后都这样"/"记住这个"/"你要这么做" → 返回 []
  - "USER 没有反驳"不等于"USER 授权 AGENT 长期执行这条规则"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【示例：三项检查的应用】

<example id="drop_situational">
场景: 用户临时问了今晚聚餐去哪好
USER: 今晚几个同学来，想找个气氛好的日料店
ASSISTANT: 推荐几家适合聚餐的日料…

检查A: USER说的是"今晚聚餐"，没有表达长期偏好
检查B: "今晚"明确是当前情境，不跨session
→ []
×不能提取: "用户喜欢日料" / "用户聚餐时喜欢气氛好的餐厅"（均为推断）
</example>

<example id="drop_knowledge">
场景: 用户问了技术知识，ASSISTANT 做了解释
USER: TCP 和 UDP 的区别是什么
ASSISTANT: TCP 是可靠传输协议，有拥塞控制和重传机制；UDP 是无连接的……

检查A: USER在提问，没有表达任何偏好或行为要求
检查C: 知识内容全部来自ASSISTANT
→ []
×不能提取: "TCP 是可靠传输协议" / "用户了解TCP原理"（知识不是用户规则）
</example>

<example id="drop_assistant_advice">
场景: 用户说了状态，ASSISTANT 顺势给出建议
USER: 最近玩游戏太久了，眼睛有点疲
ASSISTANT: 建议每隔一小时休息10分钟，可以做眼保健操……

检查A: USER表达了"眼睛疲"，没有要求agent做任何事
检查C: "每隔一小时休息"是ASSISTANT的建议，不是用户规则
→ []
×不能提取: "用户玩游戏时需要定期休息提醒" / "用户偏好护眼习惯"（推断放大）
</example>

<example id="drop_assistant_proactive_advice_no_user_endorsement">
场景: 用户说了一句当前状态，ASSISTANT 主动给出具体行动建议，用户没有授权或要求记住
USER: 在看书
ASSISTANT: 记得每隔45分钟起来活动一下，喝点水，对颈椎好……

检查A: USER原话只有"在看书"，没有表达任何对agent的要求
检查C: "每隔45分钟活动"完全来自ASSISTANT主动建议，USER没有说"以后都这样提醒我"或"记住这个"
→ []
×不能提取: "每隔45分钟应起身活动并补水"（来自ASSISTANT，不是用户规则）
关键判断: ASSISTANT建议得再具体、再合理，只要USER没有明确授权，就不是长期记忆
</example>

<example id="drop_situational_advice_specific_timing">
场景: 用户问了当前情境下的建议，ASSISTANT 给出具体操作指导
USER: 我现在有点困，要不要小睡一下
ASSISTANT: 可以睡，但控制在20分钟以内，设好闹钟，不然进入深睡会更难受……

检查A: USER在问当前这一次要不要小睡，没有表达长期规则要求
检查B: "我现在有点困"是当前状态，不跨session
检查C: 时长限制和闹钟建议来自ASSISTANT，不是USER的要求
→ []
×不能提取: "小睡时应控制在20分钟以内并设闹钟"（把当前一次建议升格为长期procedure）
</example>

<example id="drop_workaround">
场景: 当前任务遇到障碍，临时换了方案
USER: 那就直接写个脚本绕过去吧
ASSISTANT: 好，我来写一个 Python 脚本，用 requests 库来处理这个请求……

检查A: USER说的是"绕过去"，是对当前任务的临时决策
检查B: 这是当前任务的局部策略，不跨session
→ []
×不能提取: "遇到此类问题应优先用Python脚本绕过"（临时方案不是长期规则）
</example>

<example id="keep_explicit_rule">
场景: 用户明确要求 agent 以后的行为方式
USER: 以后帮我查菜谱只给 20 分钟以内能做完的，我没时间搞复杂的
ASSISTANT: 明白，以后只推荐快手菜……

检查A: USER原句="以后帮我查菜谱只给20分钟以内能做完的" ✓
检查B: "以后"明确跨session ✓
检查C: 来自USER主动要求 ✓
→ [{{"summary": "查询菜谱时只推荐 20 分钟内可完成的菜式", "memory_type": "procedure"}}]
</example>

<example id="keep_multi_source_research">
场景: 用户要求以后调研时使用多来源交叉验证
USER: 以后帮我查耳机先看 B 站评测和 Reddit 讨论，别只看官网参数
ASSISTANT: 好的，以后推荐耳机时会先参考 B 站评测和 Reddit 的用户反馈……

检查A: USER原句="以后帮我查耳机先看B站评测和Reddit讨论" ✓
检查B: "以后"明确跨session，不绑定当前任务 ✓
检查C: 调研方法论由USER主动提出，不是ASSISTANT建议 ✓
注意: 提到具体平台（B站/Reddit）不等于"当前任务特定"，只要USER用"以后"明确泛化，就是长期rule
→ [{{"summary": "查询耳机时先看 B 站评测和 Reddit 讨论，不只依赖官网参数", "memory_type": "procedure"}}]
</example>

<example id="keep_preference_trimmed">
场景: 用户表达了喜好，但 ASSISTANT 的回复有过度延伸
USER: 我不喜欢这种悬疑风格的游戏，太压抑了
ASSISTANT: 明白！你是偏好轻松明快风格的玩家，喜欢治愈系或休闲类游戏，追求积极愉快的游戏体验……

检查A: USER原句="不喜欢悬疑风格，太压抑" ✓
检查B: 游戏喜好跨session成立 ✓
检查C: 来自USER ✓
summary只能写USER说的，不得包含ASSISTANT延伸的"治愈系""休闲类""积极愉快"：
→ [{{"summary": "不喜欢悬疑压抑风格的游戏", "memory_type": "preference"}}]
×不能写: "偏好治愈系或休闲类游戏"（USER没说过）
</example>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【允许输出的两类记忆】

procedure — agent 在未来类似场景下应遵守的长期执行规则
  特征：面向 agent 行为、跨任务可复用、用户明确要求
  句式：客观规则句，"查询 X 时应先 Y"

preference — 用户跨 session 稳定成立的长期偏好或倾向
  特征：用户的偏好/厌恶/倾向，而不是硬约束
  句式："用户偏好/不喜欢/倾向于……"

【绝对不输出】
event（有时间性的具体事件）、profile（用户身份背景）
纯知识点、当前任务局部策略、一次性纠错、一次性建议

【procedure 和 preference 的边界】
有明确执行步骤或工具要求 → procedure
只是方向性的偏好倾向 → preference（优先选 preference）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【summary 写法约束】
- 只能包含 USER 原话中直接出现的内容，不能加入推断或延伸
- summary 语气不得强于 USER 原话（"不太喜欢" ≠ "强烈反感且要求永久避免"）
- summary 脱离原对话也能独立成立，不能含"这次""今天""当前"等时间锚
- 不能只是原话碎片，必须是完整句{exclusion_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

    @staticmethod
    def _build_finalize_prompt(
        *,
        user_msg: str,
        agent_response: str,
        candidates: list[dict],
    ) -> str:
        candidate_lines = []
        for idx, item in enumerate(candidates, start=1):
            candidate_lines.append(
                f"{idx}. type={item.get('memory_type', 'procedure')} | summary={item.get('summary', '')}"
            )
        candidate_block = "\n".join(candidate_lines) if candidate_lines else "(none)"
        return f"""你在做"长期记忆入库决策"，不是重新抽取。

目标：对候选列表做两件事——
1. 丢弃不该入库的候选
2. 把留下来的候选的 summary 修剪到与 USER 原话对齐

默认答案是 []。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【对每条候选，依次做两步判断】

第一步 — 入库资格审查（不通过 → 丢弃）
  □ 这条信息在 6 个月后的全新对话里还有用吗？
    - 只对当前任务/事件/时间段成立 → 丢弃
    - ASSISTANT 给的建议被当成用户偏好 → 丢弃
    - 核心内容来自 ASSISTANT（解释/工具返回/顺势建议）而非 USER → 丢弃

第二步 — Summary 忠实度核查（通过资格审查后执行）
  □ 逐句检查 summary 里的每个断言，在 USER 消息里能找到对应的原话吗？
    - 找不到的断言 → 从 summary 里删掉
    - 删减后仍有实质内容 → 用删减后的内容重写 summary
    - 删减后没有实质内容 → 丢弃整条

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【示例：入库资格审查】

<example id="finalize_drop_event">
USER: 我今天把那个 bug 修好了，总算搞定了
候选: preference | 用户在解决长期困扰的 bug 后会感到强烈成就感，倾向于分享修复过程
资格审查: "今天"是当前事件，候选描述的是本次情绪，不跨session → 丢弃
→ []
</example>

<example id="finalize_drop_assistant_backfill">
USER: 嗯
ASSISTANT: 看来你平时比较注重效率，喜欢直接拿到结论，不需要过多解释……
候选: preference | 用户偏好高效简洁的沟通方式，不需要冗长解释
资格审查: USER只说了"嗯"，核心内容来自ASSISTANT的推断 → 丢弃
→ []
</example>

<example id="finalize_drop_proactive_health_rule">
USER: 在赶代码
ASSISTANT: 别忘了每隔一段时间起来活动下，喝点水，久坐对颈椎不好……
候选: procedure | 每隔45分钟应起身活动并补水

资格审查: USER只说了当前状态"在赶代码"，没有要求或授权任何规则。
"每隔45分钟活动并补水"是ASSISTANT主动给出的健康提醒，不是USER提出的要求。
USER未反驳 ≠ USER希望永久记录这条规则 → 丢弃
→ []
</example>

【示例：Summary 忠实度核查】

<example id="finalize_trim_summary">
USER: 我不太喜欢那种剧情太拖的动漫
ASSISTANT: 明白！你是追求节奏感的观众，偏好情节紧凑、快节奏的作品，对冗长的铺垫和反复的情感渲染容忍度较低……
候选: preference | 用户偏好节奏紧凑的动漫，对冗长铺垫和反复情感渲染容忍度低，追求高效叙事体验

忠实度核查:
  - "不太喜欢剧情太拖" → USER说了 ✓
  - "对冗长铺垫和反复情感渲染容忍度低" → USER没说，来自ASSISTANT延伸 ✗ 删去
  - "追求高效叙事体验" → USER没说 ✗ 删去

重写后: preference | 不太喜欢剧情拖沓的动漫
→ [{{"summary": "不太喜欢剧情拖沓的动漫", "memory_type": "preference"}}]
</example>

<example id="finalize_keep_intact">
USER: 以后查快递帮我直接给结论，不用说运输过程，我只关心几号到
候选: procedure | 查询快递时直接告知预计到达时间，不展示中间运输节点

忠实度核查:
  - "直接给结论" → ✓
  - "不用说运输过程" → ✓
  - "只关心几号到" → ✓
  全部来自USER → summary 忠实，无需修改
→ [{{"summary": "查询快递时直接告知预计到达时间，不展示中间运输节点", "memory_type": "procedure"}}]
</example>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【原对话】
USER: {user_msg}
ASSISTANT: {agent_response}

【候选 memory】
{candidate_block}

请对每条候选完成"资格审查 → 忠实度核查 → 重写或丢弃"，只输出最终保留的条目。
只返回 JSON 数组，无内容时返回 []。
每项格式：{{"summary": "...", "memory_type": "procedure|preference", "tool_requirement": null, "steps": [], "rule_schema": {{"required_tools": [], "forbidden_tools": [], "mentioned_tools": []}}}}"""

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
    def _should_drop_by_heuristic(item: dict) -> bool:
        """确定性后置过滤：兜底拦截 LLM 可能漏掉的知识点句式和碎片。

        作用域：只针对高置信度信号，不应误伤正常 procedure/preference。
        R3/R4（技术知识点）的主要防线；R1/R5/R6/R7 的防线在 prompt 层。
        """
        summary = str(item.get("summary", "") or "").strip()
        if len(summary) <= 2:
            return True
        if item.get("memory_type") == "procedure":
            _KNOWLEDGE_SIGNALS = (
                "是指", "即为", "原理是", "的概念是",
                "协议规定", "定义为", "实现原理",
            )
            if any(s in summary for s in _KNOWLEDGE_SIGNALS):
                return True
            # 架构/设计讨论：同时出现 2+ 个信号时判定为项目设计讨论而非 agent 执行规则
            _ARCH_SIGNALS = ("在设计", "应采用", "架构时", "设计模式", "架构上应")
            if sum(1 for s in _ARCH_SIGNALS if s in summary) >= 2:
                return True
        return False

    @staticmethod
    def _normalize_extracted_item(item: dict) -> dict:
        normalized = dict(item)
        normalized["memory_type"] = PostResponseMemoryWorker._coerce_memory_type(
            str(normalized.get("memory_type", "procedure") or "procedure"),
            normalized.get("tool_requirement"),
            normalized.get("steps"),
        )
        summary = str(normalized.get("summary", "") or "")
        if (
            normalized.get("memory_type") == "procedure"
            and PostResponseMemoryWorker._extract_obvious_preferences(summary)
        ):
            normalized["memory_type"] = "preference"
        if normalized.get("memory_type") == "procedure":
            normalized["rule_schema"] = build_procedure_rule_schema(
                summary=summary,
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
                    old_items = self._get_store_items(supersede_ids)
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
                else:
                    old_items = []
            else:
                old_items = []
        else:
            old_items = []

        token_budget, saved_id = await self._save_item_direct(item, source_ref, token_budget)
        if old_items and saved_id:
            self._record_replacements(
                old_items=old_items,
                new_item=self._get_store_item(saved_id),
                source_ref=source_ref,
            )
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

        # 1. 强制创建项直接写入，不参与本轮 dedup 决策。
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

        # 2. 只有 procedure/preference 且启用了 dedup，才走新决策器。
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

        # 3. SKIP：候选和旧记忆等价，直接什么都不做。
        if result.decision == DedupDecision.SKIP:
            logger.debug("dedup skip: %s", summary[:60])
            return token_budget

        # 4. NONE：不写新候选，只对已有条目做 merge / delete。
        if result.decision == DedupDecision.NONE:
            if merge_actions:
                # 4.1 有 merge 时先 merge 再 delete，避免先删后并失败。
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
                        old_items = self._get_store_items(delete_ids)
                        self._memorizer.supersede_batch(delete_ids)
                        logger.info("dedup delete(supersede) ids=%s", delete_ids)
                        self._record_replacements(
                            old_items=old_items,
                            new_item=self._get_store_item(target.item_id),
                            source_ref=source_ref,
                            relation_type="merge",
                        )
                else:
                    # merge 失败：不删、不写候选，保持现状（信息无损）
                    logger.info(
                        "dedup merge skipped (budget/llm failure), old item preserved, "
                        "candidate dropped: %s", summary[:60],
                    )
            elif delete_ids:
                # 4.2 无 merge 只有 delete 时，直接退休旧条目。
                self._memorizer.supersede_batch(delete_ids)
                logger.info("dedup delete(supersede) ids=%s", delete_ids)
            return token_budget

        # 5. CREATE：先清理需要淘汰的旧条目，再写入新条目。
        old_items = self._get_store_items(delete_ids) if delete_ids else []
        if delete_ids:
            self._memorizer.supersede_batch(delete_ids)
            logger.info("dedup delete(supersede) ids=%s", delete_ids)

        token_budget, saved_id = await self._save_item_direct(item, source_ref, token_budget)
        if old_items and saved_id:
            self._record_replacements(
                old_items=old_items,
                new_item=self._get_store_item(saved_id),
                source_ref=source_ref,
            )
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
                # 4. profile 逐条单独写入并按需 supersede 旧 profile。
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
            old_items = self._get_store_items(supersede_ids)
            logger.info(
                "post_response profile_supersede session=%s saved_id=%s supersede_ids=%s summary=%s",
                self._current_run_session_key or "-",
                saved_id or "-",
                supersede_ids,
                self._preview_text(fact.summary, 80),
            )
            self._memorizer.supersede_batch(supersede_ids)
            if saved_id:
                self._record_replacements(
                    old_items=old_items,
                    new_item=self._get_store_item(saved_id),
                    source_ref=source_ref,
                )
        return token_budget

    @staticmethod
    def _parse_saved_item_id(result: str) -> str:
        text = str(result or "")
        if ":" not in text:
            return ""
        return text.split(":", 1)[1].strip()

    def _get_store_item(self, item_id: str) -> dict | None:
        store = getattr(self._memorizer, "_store", None)
        if store is None or not item_id or not hasattr(store, "get_items_by_ids"):
            return None
        try:
            rows = store.get_items_by_ids([item_id])
        except Exception as e:
            logger.warning("post_response get_store_item failed: %s", e)
            return None
        return rows[0] if rows else None

    def _get_store_items(self, ids: list[str]) -> list[dict]:
        store = getattr(self._memorizer, "_store", None)
        if store is None or not ids or not hasattr(store, "get_items_by_ids"):
            return []
        try:
            return store.get_items_by_ids(ids)
        except Exception as e:
            logger.warning("post_response get_store_items failed: %s", e)
            return []

    def _record_replacements(
        self,
        *,
        old_items: list[dict],
        new_item: dict | None,
        source_ref: str,
        relation_type: str = "supersede",
    ) -> None:
        if not old_items or not new_item:
            return
        store = getattr(self._memorizer, "_store", None)
        if store is None or not hasattr(store, "record_replacements"):
            return
        try:
            store.record_replacements(
                old_items=old_items,
                new_item=new_item,
                source_ref=source_ref,
                relation_type=relation_type,
            )
        except Exception as e:
            logger.warning("post_response record_replacements failed: %s", e)

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
