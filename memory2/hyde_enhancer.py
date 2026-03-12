"""
HyDE（Hypothetical Document Embeddings）检索增强。

工作流：
  1. 并行：raw 检索 + light LLM 生成假想记忆条目
  2. 等 hypothesis 就绪后，发起第二次检索
  3. union dedup：保留 raw 全部结果，追加 hyde 中 raw 没有的条目
  4. 任何步骤失败/超时 → 降级返回 raw 结果，used_hyde=False
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from agent.provider import LLMProvider

logger = logging.getLogger(__name__)


class HyDEEnhancer:
    HYPOTHESIS_MAX_TOKENS = 80
    DEFAULT_TIMEOUT_S = 2.0

    def __init__(
        self,
        light_provider: "LLMProvider",
        light_model: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._provider = light_provider
        self._model = light_model
        self._timeout_s = max(0.5, float(timeout_s))

    async def generate_hypothesis(self, query: str, context: str) -> str | None:
        """
        生成假想记忆条目。失败/超时返回 None，调用方降级为原始检索。

        关键 prompt 约束：
        - 保持原问题的语义极性（否定问题生成否定式条目）
        - 只改写语态为第三人称书面陈述，不添加原问题没有的信息
        """
        context_section = f"\n近期对话背景：\n{context}\n" if context else ""
        prompt = (
            "你是个人助手的记忆系统。根据用户提问，生成一条"
            "**如果该信息存在于记忆数据库中会长什么样**的假想条目。\n"
            f"{context_section}"
            "规则：\n"
            "- 始终生成肯定式条目，描述**如果该记忆存在会记录什么事实**，不要否定该事件的存在\n"
            '- 第三人称（"用户..."），与数据库条目语体一致（简洁的事实陈述）\n'
            "- 只输出那一条文本，不要解释，不要回答问题本身\n\n"
            f"用户提问：{query}\n"
            "假想记忆条目："
        )
        try:
            resp = await asyncio.wait_for(
                self._provider.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self._model,
                    max_tokens=self.HYPOTHESIS_MAX_TOKENS,
                ),
                timeout=self._timeout_s,
            )
            text = (resp.content or "").strip()
            return text if text else None
        except Exception as e:
            logger.debug("hyde hypothesis generation failed: %s", e)
            return None

    async def augment(
        self,
        *,
        raw_query: str,
        context: str,
        retrieve_fn: Callable[..., Awaitable[list[dict]]],
        top_k: int,
        **retrieve_kwargs,
    ) -> tuple[list[dict], bool]:
        """
        双路检索 + union dedup。
        返回 (results, used_hyde)，used_hyde=True 表示 HyDE 实际追加了新条目。
        raw 结果完整保留，hyde 只追加 raw 中不存在的新条目。
        """
        # 并行：raw 检索 + hypothesis 生成
        raw_task = asyncio.create_task(
            retrieve_fn(raw_query, top_k=top_k, **retrieve_kwargs)
        )
        hyp_task = asyncio.create_task(self.generate_hypothesis(raw_query, context))
        raw_hits, hypothesis = await asyncio.gather(raw_task, hyp_task)

        if not hypothesis:
            logger.debug("hyde: no hypothesis, using raw results only")
            return raw_hits, False

        # hypothesis 就绪后，串行发起第二次检索
        try:
            hyde_hits = await retrieve_fn(hypothesis, top_k=top_k, **retrieve_kwargs)
        except Exception as e:
            logger.debug("hyde retrieve failed: %s", e)
            return raw_hits, False

        result = _union_dedup(raw_hits, hyde_hits)
        used_hyde = len(result) > len(raw_hits)
        logger.info(
            "hyde: raw=%d hyde=%d merged=%d used_hyde=%s hypothesis=%r",
            len(raw_hits),
            len(hyde_hits),
            len(result),
            used_hyde,
            hypothesis[:60],
        )
        return result, used_hyde


def _union_dedup(raw: list[dict], hyde: list[dict]) -> list[dict]:
    """
    保留 raw 全部结果（含原始分数），追加 hyde 中 raw 没有的条目。
    不修改任何条目的 score，避免影响下游 select_for_injection 的 type_best 计算。
    """
    seen_ids: set[str] = set()
    result = []
    for item in raw:
        item_id = str(item.get("id", ""))
        if item_id:
            seen_ids.add(item_id)
        result.append(item)
    for item in hyde:
        item_id = str(item.get("id", ""))
        if item_id and item_id in seen_ids:
            continue
        result.append(item)
        if item_id:
            seen_ids.add(item_id)
    return result
