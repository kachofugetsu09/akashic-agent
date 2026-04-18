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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from agent.provider import LLMProvider

logger = logging.getLogger(__name__)


async def _retrieve_hits(
    retrieve_fn: Callable[..., Awaitable[list[dict]]],
    query: str,
    top_k: int,
    retrieve_kwargs: dict[str, object],
) -> list[dict]:
    return await retrieve_fn(query, top_k=top_k, **retrieve_kwargs)


@dataclass
class HyDEAugmentResult:
    """augment() 的返回值，包含 raw/hyde 两路的原始数据，供 trace 使用。"""

    items: list[dict]               # 合并后的最终结果（raw ∪ hyde_added）
    used_hyde: bool                 # HyDE 是否实际追加了新条目
    hypothesis: str | None          # LLM 生成的假设文本
    raw_hits: list[dict] = field(default_factory=list)  # raw query 检索到的条目

    def __iter__(self):
        """兼容旧调用方的二元解包：results, used_hyde = augment(...)。"""
        yield self.items
        yield self.used_hyde


class HyDEEnhancer:
    HYPOTHESIS_MAX_TOKENS = 80
    DEFAULT_TIMEOUT_S = 2.0

    def __init__(
        self,
        light_provider: "LLMProvider",
        light_model: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        prompt_builder: Callable[[str, str], str] | None = None,
    ) -> None:
        self._provider = light_provider
        self._model = light_model
        self._timeout_s = max(0.5, float(timeout_s))
        self._prompt_builder = prompt_builder or self._build_default_prompt

    async def generate_hypothesis(self, query: str, context: str) -> str | None:
        """
        生成假想记忆条目。失败/超时返回 None，调用方降级为原始检索。

        关键 prompt 约束：
        - 保持原问题的语义极性（否定问题生成否定式条目）
        - 只改写语态为第三人称书面陈述，不添加原问题没有的信息
        """
        prompt = self._prompt_builder(query, context)
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

    @staticmethod
    def _build_default_prompt(query: str, context: str) -> str:
        context_section = f"\n近期对话背景：\n{context}\n" if context else ""
        return (
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

    async def augment(
        self,
        *,
        raw_query: str,
        context: str,
        retrieve_fn: Callable[..., Awaitable[list[dict]]],
        top_k: int,
        **retrieve_kwargs,
    ) -> HyDEAugmentResult:
        """
        双路检索 + union dedup。
        返回 HyDEAugmentResult，包含合并结果、是否追加了新条目、假设文本和 raw hits。
        raw 结果完整保留，hyde 只追加 raw 中不存在的新条目。
        """
        # 并行：raw 检索 + hypothesis 生成
        raw_task = asyncio.create_task(
            _retrieve_hits(retrieve_fn, raw_query, top_k, retrieve_kwargs)
        )
        hyp_task = asyncio.create_task(self.generate_hypothesis(raw_query, context))
        raw_hits, hypothesis = await asyncio.gather(raw_task, hyp_task)

        if not hypothesis:
            logger.debug("hyde: no hypothesis, using raw results only")
            return HyDEAugmentResult(
                items=raw_hits, used_hyde=False, hypothesis=None, raw_hits=raw_hits
            )

        # hypothesis 就绪后，串行发起第二次检索
        try:
            hyde_hits = await retrieve_fn(hypothesis, top_k=top_k, **retrieve_kwargs)
        except Exception as e:
            logger.debug("hyde retrieve failed: %s", e)
            return HyDEAugmentResult(
                items=raw_hits, used_hyde=False, hypothesis=hypothesis, raw_hits=raw_hits
            )

        merged = _union_dedup(raw_hits, hyde_hits)
        used_hyde = len(merged) > len(raw_hits)
        logger.info(
            "hyde: raw=%d hyde=%d merged=%d used_hyde=%s hypothesis=%r",
            len(raw_hits),
            len(hyde_hits),
            len(merged),
            used_hyde,
            hypothesis[:60],
        )
        return HyDEAugmentResult(
            items=merged, used_hyde=used_hyde, hypothesis=hypothesis, raw_hits=raw_hits
        )


def _union_dedup(raw: list[dict], hyde: list[dict]) -> list[dict]:
    """
    保留 raw 全部结果（含原始分数），追加 hyde 中 raw 没有的条目。
    不修改任何条目的 score，避免影响下游注入筛选的 type_best 计算。
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
