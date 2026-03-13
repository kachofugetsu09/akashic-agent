"""回复后异步记忆提取与 supersede 处理。"""

from __future__ import annotations

import logging
import re

import json_repair

from agent.provider import LLMProvider
from memory2.memorizer import Memorizer
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
    SUPERSEDE_CANDIDATE_K = 5
    TOKEN_BUDGET_PER_RUN = 768
    TOKENS_EXTRACT_IMPLICIT = 384
    TOKENS_EXTRACT_INVALIDATION = 96
    TOKENS_CHECK_INVALIDATE = 96
    TOKENS_CHECK_SUPERSEDE = 96

    def __init__(
        self,
        memorizer: Memorizer,
        retriever: Retriever,
        light_provider: LLMProvider,
        light_model: str,
        tagger=None,  # ProcedureTagger | None
        observe_writer=None,
    ) -> None:
        self._memorizer = memorizer
        self._retriever = retriever
        self._provider = light_provider
        self._model = light_model
        self._tagger = tagger
        self._observe_writer = observe_writer

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
        try:
            already_memorized, protected_ids = self._collect_explicit_memorized(
                tool_chain
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
            new_items = await self._dedupe_against_explicit(
                new_items,
                already_memorized,
                protected_ids,
            )
            if not new_items:
                return

            for item in new_items:
                token_budget = await self._save_with_supersede(
                    item,
                    source_ref,
                    protected_ids,
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

        prompt = f"""你是记忆提取专家。从以下对话中提取两类长期有效的信息：
1. 用户对 agent 行为的隐式偏好或操作规范
2. 用户对内容题材/作品/游戏/作者/来源的稳定喜欢或厌恶，尤其是会影响未来主动推送的偏好

【唯一有效依据来源：USER 说的话】
ASSISTANT 的回复只是对话背景，不能作为提取依据。
即使 ASSISTANT 描述了某个流程/规则，若 USER 没有明确表达纠正/要求/不满，也不能提取。

【提取标准——必须同时满足以下全部条件】
1. 能从 USER 的原话中直接引用出明确的纠正/不满/要求/偏好信号
   - 行为规范信号示例："你应该""你不能""你之前错了""下次要"
   - 内容偏好信号示例："我就讨厌《X》""以后别给我推 X""我很喜欢《Y》"
   无法引用 USER 原话 → 直接返回 []
2. 该信号必须具有跨对话持久意义
   - 新对话开始时仍然适用，才写入
3. 若是内容偏好，必须足够明确，能够影响未来推荐/主动推送/举例方式

【内容偏好提取重点】
✓ 用户明确表达喜欢/偏好某作品、题材、作者、来源，可提取为 preference
✓ 用户明确表达讨厌/厌恶/拒绝某作品、题材、作者、来源，可提取为 preference
✓ 用户明确要求“以后别再推/别再发/别再拿 X 打比方”，必须优先提取为 preference
✓ 这类偏好应直接写成对未来行为有指导意义的总结，如：
  - 用户明确厌恶《鬼灭之刃》相关内容；主动消息不要再推送、引用或拿它打比方
  - 用户明确喜欢《仁王》相关内容；主动消息可优先推送相关内容

【明确不写】
✗ ASSISTANT 回复中描述的任何流程、规则、步骤（即使正确）
✗ USER 只是在查询/确认 agent 的行为（"你的流程是什么""你是怎么做的"）
✗ 无法在 USER 原话中找到纠正/不满/强调句的条目
✗ agent 自己选择的回复格式/风格（用户没有要求）
✗ 用户对自身情况的感慨或困惑
✗ 一次性操作记录
✗ 带明确时间锚点的短期计划、近期打算、当前想做的事（如“这周末最想重玩 X”“最近想看 Y”）不能写成 preference；若确有记忆价值，应写成 event
✗ USER 只是随口评价某作品好坏，但没有表现出稳定喜欢/厌恶或未来规避意图
✗ 显式 memorize 内容（见排除列表）{exclusion_block}

【重要】大多数对话不包含可提取的偏好，返回 [] 是正常且正确的结果。
【数量限制】最多 2 条，只写有 USER 原文依据的。

【对话内容】
USER: {user_msg}
ASSISTANT: {agent_response}

只返回合法 JSON 数组，无内容时返回 []。
每项格式：{{"summary": "...", "memory_type": "procedure|preference|event", "tool_requirement": null或"工具名", "steps": [], "rule_schema": {{"required_tools": [], "forbidden_tools": [], "mentioned_tools": []}}}}

【tool_requirement 填写规则】
- 若该条目要求 agent 在某类请求下必须调用特定工具/skill（如"查询天气时必须用 weather skill"、"查 Steam 必须用 MCP 工具"），则填写触发该工具的关键名称（如 "weather_skill"、"steam_mcp"）
- 这样系统才能在相关请求时强制注入该规则，不受相似度分数影响
- 若无强制工具要求，填 null

【rule_schema 填写规则】
- 仅对 procedure 填写；preference 可省略或填空对象
- `required_tools` 只放用户明确要求必须使用的工具
- `forbidden_tools` 只放用户明确禁止或要求不要直接使用的工具
- `mentioned_tools` 放该规则涉及到的工具别名
- 若无法确认某个约束，就留空，不要猜"""

        ok, token_budget = self._consume_budget(
            token_budget,
            self.TOKENS_EXTRACT_IMPLICIT,
        )
        if not ok:
            logger.info(
                "post_response extract_implicit skipped: token budget exhausted"
            )
            return [], token_budget

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
                return [
                    self._normalize_extracted_item(r)
                    for r in result
                    if isinstance(r, dict) and r.get("summary")
                ], token_budget
        except Exception as e:
            logger.warning(f"post_response_memorize extract failed: {e}")
        return [], token_budget

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
        if normalized.get("memory_type") == "procedure":
            normalized["rule_schema"] = build_procedure_rule_schema(
                summary=str(normalized.get("summary", "") or ""),
                tool_requirement=normalized.get("tool_requirement"),
                steps=normalized.get("steps") or [],
                rule_schema=normalized.get("rule_schema"),
            )
        return normalized

    async def _save_with_supersede(
        self,
        item: dict,
        source_ref: str,
        protected_ids: set[str] | None = None,
        token_budget: int = TOKEN_BUDGET_PER_RUN,
    ) -> int:
        """写入新条目，同时检测并退休矛盾的旧条目。"""
        summary = (item.get("summary") or "").strip()
        if not summary:
            return token_budget

        mtype = item.get("memory_type", "procedure")
        if mtype not in ("procedure", "preference", "event", "profile"):
            mtype = "procedure"

        _protected = protected_ids or set()
        rule_schema = None
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

        extra: dict = {
            "tool_requirement": item.get("tool_requirement"),
            "steps": item.get("steps") or [],
        }
        if mtype == "procedure":
            extra["rule_schema"] = rule_schema or build_procedure_rule_schema(
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
            logger.debug(f"post_response_memorize saved ({mtype}): {result}")
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
        except Exception as e:
            logger.warning(f"post_response_memorize save failed: {e}")
        return token_budget

    async def _check_supersede(
        self,
        new_summary: str,
        candidates: list[dict],
        token_budget: int,
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
