"""
SourceScorer — 基于用户 memory 对 RSS 信息源打分，按 softmax 分配拉取配额。

设计原则：
- 全量初始化：所有源一次塞进一个 prompt，LLM 横向比较打分（0-10分）
- 新增源：把已有分数作参照，只对新源单独打分（保证标准一致性）
- 删除源：直接从缓存 pop，不调 LLM
- 缓存文件：~/.akasic/workspace/source_scores.json
  key = sorted(source_ids) 的 sha1 hash，hash 变化时重算
- 分配算法：softmax(temperature=2.0) + min_per_source 保底 + max_per_source 封顶
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.provider import LLMProvider

from feeds.base import FeedSubscription

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1


class SourceScorer:
    """为每个订阅 feed 源打分，并按比例分配拉取配额。"""

    def __init__(
        self,
        light_provider: LLMProvider,
        light_model: str,
        cache_path: Path,
    ) -> None:
        self._provider = light_provider
        self._model = light_model
        self._cache_path = cache_path
        self._cache: dict = {}  # 内存中的缓存，启动时懒加载

    # ── public API ──────────────────────────────────────────────────

    async def get_limits(
        self,
        subscriptions: list[FeedSubscription],
        memory_text: str,
        total_budget: int,
        min_per_source: int,
        max_per_source: int,
    ) -> dict[str, int]:
        """返回 source_id → limit 的分配字典。"""
        if not subscriptions:
            return {}

        enabled = [s for s in subscriptions if s.enabled]
        if not enabled:
            return {}

        scores = await self._get_scores(enabled, memory_text)
        limits = _allocate_limits(
            scores=scores,
            sub_ids=[s.id for s in enabled],
            total_budget=total_budget,
            min_per_source=min_per_source,
            max_per_source=max_per_source,
        )
        logger.debug(
            "[source_scorer] 配额分配完成 total_budget=%d sources=%d limits=%s",
            total_budget,
            len(enabled),
            {k[:8]: v for k, v in limits.items()},
        )
        return limits

    async def score_new_source(
        self,
        new_sub: FeedSubscription,
        memory_text: str,
    ) -> float:
        """
        新增源时：把已有分数作参照，只打新源。
        结果追加写入缓存（不改变现有分数，不改变 sources_hash，
        因为新源加入后 hash 已由调用方通过 get_limits 重算）。
        返回 0-10 分。
        """
        self._load_cache()
        existing_scores: dict[str, float] = self._cache.get("scores", {})

        score = await self._score_single_source(new_sub, existing_scores, memory_text)
        # 不在这里写缓存——调用方会通过 get_limits 触发全量/增量，由那边统一写
        logger.info(
            "[source_scorer] 新源打分完成 name=%r score=%.1f",
            new_sub.name,
            score,
        )
        return score

    def invalidate_source(self, source_id: str) -> None:
        """删除源时：从缓存 pop，更新文件。不调 LLM。"""
        self._load_cache()
        scores: dict[str, float] = self._cache.get("scores", {})
        if source_id in scores:
            scores.pop(source_id)
            self._cache["scores"] = scores
            # 重新计算 hash（基于剩余 source_id）
            self._cache["sources_hash"] = _hash_ids(list(scores.keys()))
            self._save_cache()
            logger.info("[source_scorer] 已从缓存移除 source_id=%s", source_id[:8])

    # ── internal ────────────────────────────────────────────────────

    async def _get_scores(
        self,
        subs: list[FeedSubscription],
        memory_text: str,
    ) -> dict[str, float]:
        """获取所有源的分数，优先命中缓存，否则重新打分。"""
        self._load_cache()
        current_hash = _hash_ids([s.id for s in subs])
        cached_hash = self._cache.get("sources_hash", "")
        cached_scores: dict[str, float] = self._cache.get("scores", {})

        if current_hash == cached_hash and cached_scores:
            logger.debug(
                "[source_scorer] 命中缓存 hash=%s sources=%d",
                current_hash[:8],
                len(subs),
            )
            return cached_scores

        # 判断是否属于"新增源"情况（旧 hash 存在，只是多了几个源）
        cached_ids = set(cached_scores.keys())
        current_ids = {s.id for s in subs}
        removed_ids = cached_ids - current_ids
        new_ids = current_ids - cached_ids

        if cached_scores and not removed_ids and new_ids:
            # 增量：只对新源打分，复用已有分数
            logger.info(
                "[source_scorer] 增量打分 new_sources=%d",
                len(new_ids),
            )
            new_subs = [s for s in subs if s.id in new_ids]
            for new_sub in new_subs:
                score = await self._score_single_source(
                    new_sub, cached_scores, memory_text
                )
                cached_scores[new_sub.id] = score
            self._update_cache(current_hash, cached_scores)
            return cached_scores

        # 全量打分
        logger.info("[source_scorer] 全量打分 sources=%d", len(subs))
        scores = await self._score_all_sources(subs, memory_text)
        self._update_cache(current_hash, scores)
        return scores

    async def _score_all_sources(
        self,
        subs: list[FeedSubscription],
        memory_text: str,
    ) -> dict[str, float]:
        """全量打分：把所有源一次塞进 prompt，LLM 横向比较。"""
        source_lines = []
        for sub in subs:
            note_part = f"，备注: {sub.note}" if sub.note else ""
            source_lines.append(f'- id: "{sub.id}", 名称: "{sub.name}"{note_part}')
        sources_text = "\n".join(source_lines)

        prompt = f"""你是一个信息筛选助手。根据以下用户兴趣画像，对每个 RSS 信息源的相关度打分。

用户兴趣画像：
{memory_text}

打分规则：
- 分值范围 0-10（整数或一位小数）
- 10 = 用户极度感兴趣，几乎每条都会看
- 5  = 有一定相关性，偶尔感兴趣
- 0  = 完全不相关或用户明确不喜欢
- 横向比较各源，保证评分标准一致

信息源列表：
{sources_text}

只返回 JSON，格式如下（不要 markdown 代码块）：
{{"scores": {{"<id>": <score>, ...}}}}"""

        try:
            response = await self._provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是信息筛选助手，只返回合法 JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self._model,
                max_tokens=1024,
            )
            text = (response.content or "").strip()
            scores = _parse_scores_json(text, subs)
            logger.info(
                "[source_scorer] 全量打分完成 scores=%s",
                {k[:8]: v for k, v in scores.items()},
            )
            return scores
        except Exception as e:
            logger.warning("[source_scorer] 全量打分失败，回退均等分: %s", e)
            return {s.id: 5.0 for s in subs}

    async def _score_single_source(
        self,
        new_sub: FeedSubscription,
        existing_scores: dict[str, float],
        memory_text: str,
    ) -> float:
        """对单个新源打分，把已有分数作为参照系。"""
        # 构造参照信息（取已有源的名称+分数）
        ref_lines = []
        for sid, score in list(existing_scores.items())[:15]:
            ref_lines.append(f"  - id: {sid[:8]}...  分数: {score:.1f}")
        ref_text = "\n".join(ref_lines) if ref_lines else "  （无参照）"

        note_part = f"，备注: {new_sub.note}" if new_sub.note else ""
        prompt = f"""你是一个信息筛选助手。根据用户兴趣画像，对新订阅的信息源打分。

用户兴趣画像：
{memory_text}

以下是已有信息源的打分参照（用于保证标准一致）：
{ref_text}

需要打分的新信息源：
- 名称: "{new_sub.name}"{note_part}

打分规则：
- 分值范围 0-10（整数或一位小数），与参照系标准一致
- 10 = 用户极度感兴趣，0 = 完全不相关

只返回 JSON，格式如下（不要 markdown 代码块）：
{{"score": <number>}}"""

        try:
            response = await self._provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是信息筛选助手，只返回合法 JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self._model,
                max_tokens=64,
            )
            text = (response.content or "").strip()
            score = _parse_single_score(text)
            return score
        except Exception as e:
            logger.warning("[source_scorer] 单源打分失败，默认 5.0: %s", e)
            return 5.0

    def _load_cache(self) -> None:
        """懒加载缓存文件到内存。"""
        if self._cache:
            return
        if not self._cache_path.exists():
            self._cache = {"version": _CACHE_VERSION, "sources_hash": "", "scores": {}}
            return
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("version") == _CACHE_VERSION:
                self._cache = data
            else:
                logger.warning("[source_scorer] 缓存版本不匹配，忽略旧缓存")
                self._cache = {
                    "version": _CACHE_VERSION,
                    "sources_hash": "",
                    "scores": {},
                }
        except Exception as e:
            logger.warning("[source_scorer] 缓存读取失败: %s", e)
            self._cache = {"version": _CACHE_VERSION, "sources_hash": "", "scores": {}}

    def _update_cache(self, sources_hash: str, scores: dict[str, float]) -> None:
        """更新内存缓存并写文件。"""
        self._cache = {
            "version": _CACHE_VERSION,
            "sources_hash": sources_hash,
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "scores": scores,
        }
        self._save_cache()

    def _save_cache(self) -> None:
        """将内存缓存写入文件。"""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug("[source_scorer] 缓存已写入 path=%s", self._cache_path)
        except Exception as e:
            logger.warning("[source_scorer] 缓存写入失败: %s", e)


# ── helpers ──────────────────────────────────────────────────────────


def _hash_ids(ids: list[str]) -> str:
    """对 source_id 列表排序后取 sha1，用于缓存失效判断。"""
    raw = "|".join(sorted(ids))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _parse_scores_json(text: str, subs: list[FeedSubscription]) -> dict[str, float]:
    """从 LLM 输出中解析 {"scores": {id: score}} JSON。解析失败时回退均等分。"""
    # 尝试去掉 markdown 代码块
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        text = m.group(1) if m else text
    # 找 { ... }
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        logger.warning("[source_scorer] 无法提取 JSON: %r", text[:200])
        return {s.id: 5.0 for s in subs}
    try:
        data = json.loads(m.group())
        raw_scores = data.get("scores", {})
        result: dict[str, float] = {}
        for sub in subs:
            if sub.id in raw_scores:
                result[sub.id] = max(0.0, min(10.0, float(raw_scores[sub.id])))
            else:
                logger.warning(
                    "[source_scorer] LLM 未返回 source %r 的分数，默认 5.0", sub.name
                )
                result[sub.id] = 5.0
        return result
    except Exception as e:
        logger.warning("[source_scorer] JSON 解析失败: %s  raw=%r", e, text[:200])
        return {s.id: 5.0 for s in subs}


def _parse_single_score(text: str) -> float:
    """从 LLM 输出中解析 {"score": N} JSON。"""
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        text = m.group(1) if m else text
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        # 尝试直接找数字
        nm = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if nm:
            return max(0.0, min(10.0, float(nm.group(1))))
        return 5.0
    try:
        data = json.loads(m.group())
        return max(0.0, min(10.0, float(data.get("score", 5.0))))
    except Exception:
        return 5.0


def _softmax(values: list[float], temperature: float = 2.0) -> list[float]:
    """Softmax with temperature，temperature 越高越平均分配。"""
    t = max(temperature, 1e-6)
    scaled = [v / t for v in values]
    max_v = max(scaled) if scaled else 0.0
    exps = [math.exp(x - max_v) for x in scaled]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _allocate_limits(
    scores: dict[str, float],
    sub_ids: list[str],
    total_budget: int,
    min_per_source: int,
    max_per_source: int,
) -> dict[str, int]:
    """
    按 softmax 权重分配 total_budget 条配额。
    保证每源 >= min_per_source（若预算允许），上限 max_per_source。
    """
    if not sub_ids:
        return {}

    n = len(sub_ids)
    min_total = n * min_per_source
    effective_budget = max(total_budget, min_total)

    # 先给每源分配保底
    result = {sid: min_per_source for sid in sub_ids}
    remaining = effective_budget - min_total

    if remaining <= 0:
        # 预算不足以超过保底，直接返回保底值
        return result

    # softmax 迭代分配剩余配额，处理 max_per_source 截断后余量重新分配
    score_values = [scores.get(sid, 5.0) for sid in sub_ids]
    extra = {sid: 0 for sid in sub_ids}  # 保底之上的额外配额

    budget_left = remaining
    free_ids = list(sub_ids)  # 还未到上限的源

    while budget_left > 0 and free_ids:
        free_scores = [scores.get(sid, 5.0) for sid in free_ids]
        weights = _softmax(free_scores, temperature=2.0)

        float_alloc = [w * budget_left for w in weights]
        int_alloc = [int(f) for f in float_alloc]
        leftover = budget_left - sum(int_alloc)
        fracs = sorted(
            ((float_alloc[i] - int_alloc[i], i) for i in range(len(free_ids))),
            reverse=True,
        )
        for _, i in fracs[:leftover]:
            int_alloc[i] += 1

        overflow = 0
        next_free = []
        for i, sid in enumerate(free_ids):
            cap = max_per_source - min_per_source - extra[sid]
            give = min(int_alloc[i], cap)
            extra[sid] += give
            overflow += int_alloc[i] - give
            if extra[sid] < cap:
                next_free.append(sid)

        budget_left = overflow
        free_ids = next_free

    for sid in sub_ids:
        result[sid] = min_per_source + extra[sid]

    return result
