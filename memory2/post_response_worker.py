"""回复后异步记忆提取与 supersede 处理。"""

from __future__ import annotations

import logging

import json_repair

from agent.provider import LLMProvider
from memory2.memorizer import Memorizer
from memory2.retriever import Retriever

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

    def __init__(
        self,
        memorizer: Memorizer,
        retriever: Retriever,
        light_provider: LLMProvider,
        light_model: str,
    ) -> None:
        self._memorizer = memorizer
        self._retriever = retriever
        self._provider = light_provider
        self._model = light_model

    async def run(
        self,
        user_msg: str,
        agent_response: str,
        tool_chain: list[dict],
        source_ref: str,
    ) -> None:
        try:
            already_memorized, protected_ids = self._collect_explicit_memorized(tool_chain)

            # 先处理"旧的有误/需要遗忘"的显式废弃信号，无需新规则即可 supersede
            await self._handle_invalidations(user_msg, source_ref, protected_ids)

            new_items = await self._extract_implicit(
                user_msg,
                agent_response,
                already_memorized,
            )
            if not new_items:
                return

            for item in new_items:
                await self._save_with_supersede(item, source_ref, protected_ids)
        except Exception as e:
            logger.warning(f"post_response_memorize run failed: {e}")

    async def _handle_invalidations(
        self, user_msg: str, source_ref: str, protected_ids: set[str] | None = None
    ) -> None:
        """检测用户明确指出 agent 旧行为有误的情况，无需替代规则即直接 supersede 旧条目。"""
        topics = await self._extract_invalidation_topics(user_msg)
        if not topics:
            return
        _protected = protected_ids or set()
        for topic in topics:
            candidates = await self._retriever.retrieve(
                topic,
                memory_types=["procedure", "preference"],
            )
            high_sim = [
                c for c in candidates
                if isinstance(c, dict)
                and c.get("score", 0) >= self.SUPERSEDE_THRESHOLD
                and c.get("id") not in _protected
            ][: self.SUPERSEDE_CANDIDATE_K]
            if not high_sim:
                continue
            supersede_ids = await self._check_invalidate(topic, high_sim)
            if supersede_ids:
                self._memorizer.supersede_batch(supersede_ids)
                logger.info(
                    "post_response invalidation: superseded %s for topic '%s'",
                    supersede_ids,
                    topic,
                )

    async def _extract_invalidation_topics(self, user_msg: str) -> list[str]:
        """从用户消息中提取被明确声明为有误/需废弃的 agent 行为主题。"""
        prompt = f"""判断用户消息是否在指出 agent 某个现有行为/流程有误，且希望废弃它（即使没给出替代方案）。

用户消息：{user_msg}

触发条件（需同时满足）：
1. 用户明确指出 agent 某个行为/流程是错的、过时的、需要忘记/废弃
2. 主语是 agent 的行为（不是用户自己的事，不是第三方的事）

若触发，提取受影响的行为主题（简短描述，如"steam查询流程"、"健康数据获取方式"）。
"也许/可能"等不确定措辞仍算触发——宁可多检查，不可漏掉。
若不触发，返回 []。

只返回 JSON 数组，如 ["steam查询流程"] 或 []"""
        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=128,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if isinstance(result, list):
                return [t for t in result if isinstance(t, str) and t.strip()]
        except Exception as e:
            logger.warning(f"extract_invalidation_topics failed: {e}")
        return []

    async def _check_invalidate(
        self, topic: str, candidates: list[dict]
    ) -> list[str]:
        """用户声明旧行为有误时，判断哪些旧条目应被 supersede（无需新规则替代）。"""
        old_block = "\n".join(
            f'- id={c["id"]} | {c["summary"]}' for c in candidates
        )
        prompt = f"""用户明确表示 agent 关于"{topic}"的现有行为/流程有误，需要废弃。
以下是数据库中与该主题相关的现有规则，判断哪些应被标记为废弃：

{old_block}

规则：
- 若条目确实描述了"{topic}"相关的 agent 操作流程/行为，输出其 id
- 若条目与该主题无关，不输出
- 若无关联条目，返回 []

只返回 JSON 数组，如 ["abc123"] 或 []"""
        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=128,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if isinstance(result, list):
                valid_ids = {c["id"] for c in candidates}
                return [
                    i for i in result
                    if isinstance(i, str) and i in valid_ids
                ]
        except Exception as e:
            logger.warning(f"check_invalidate failed: {e}")
        return []

    def _collect_explicit_memorized(
        self, tool_chain: list[dict]
    ) -> tuple[list[str], set[str]]:
        """从 tool_chain 收集本轮 memorize tool 显式写入的 summary 和 DB id。

        返回 (summaries, protected_ids)：
        - summaries：传给 light model 的排除列表
        - protected_ids：memorize tool 本轮写入的条目 id，不允许被 worker supersede
        """
        import re as _re
        _id_pattern = _re.compile(r'(?:new|reinforced):([a-f0-9]{10,16})')

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
    ) -> list[dict]:
        """light model 提取隐式偏好，返回 behavior_updates 格式列表。"""
        exclusion_block = ""
        if already_memorized:
            lines = "\n".join(f"- {s}" for s in already_memorized if s)
            exclusion_block = f"\n\n【本轮已显式记录，不要重复提取】\n{lines}"

        prompt = f"""你是记忆提取专家。从以下对话中提取用户对 agent 行为的隐式偏好或操作规范。

【提取标准——必须同时满足以下全部条件】
1. 对话原文中存在明确的用户纠正/不满/强调信号——你必须能直接引用用户说的具体句子作为依据。
   无法引用原文 → 直接返回 []，不得靠推断补全。
2. 该信号针对的是 agent 的行为（不是用户对自身情况的感慨、困惑或自我评价）。
3. 跨对话有持久意义——新对话开始时这条规则还适用吗？若否，不写。

【明确不写】
✗ 无法从原文中找到用户纠正/不满句子的任何条目
✗ agent 自己选择的回复格式/风格（用户没有要求）
✗ 从单次操作推断的一般规律
✗ 用户在表达对自身情况的困惑或感慨（"感觉不太对劲/我怎么了"等）
✗ 一次性操作记录
✗ 显式 memorize 内容（见排除列表）{exclusion_block}

【重要】大多数对话不包含可提取的偏好，返回 [] 是正常且正确的结果。
【数量限制】最多 2 条，只写有原文依据的。

【对话内容】
USER: {user_msg}
ASSISTANT: {agent_response}

只返回合法 JSON 数组，无内容时返回 []。
每项格式：{{"summary": "...", "memory_type": "procedure|preference", "tool_requirement": null或"工具名", "steps": []}}"""

        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=512,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if isinstance(result, list):
                return [r for r in result if isinstance(r, dict) and r.get("summary")]
        except Exception as e:
            logger.warning(f"post_response_memorize extract failed: {e}")
        return []

    async def _save_with_supersede(
        self,
        item: dict,
        source_ref: str,
        protected_ids: set[str] | None = None,
    ) -> None:
        """写入新条目，同时检测并退休矛盾的旧条目。"""
        summary = (item.get("summary") or "").strip()
        if not summary:
            return

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
                supersede_ids = await self._check_supersede(summary, high_sim)
                if supersede_ids:
                    self._memorizer.supersede_batch(supersede_ids)

        extra = {
            "tool_requirement": item.get("tool_requirement"),
            "steps": item.get("steps") or [],
        }
        try:
            result = await self._memorizer.save_item(
                summary=summary,
                memory_type=mtype,
                extra=extra,
                source_ref=source_ref,
            )
            logger.info(f"post_response_memorize saved ({mtype}): {result}")
        except Exception as e:
            logger.warning(f"post_response_memorize save failed: {e}")

    async def _check_supersede(
        self, new_summary: str, candidates: list[dict]
    ) -> list[str]:
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

        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=128,
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
                ]
        except Exception as e:
            logger.warning(f"supersede check failed: {e}")
        return []
