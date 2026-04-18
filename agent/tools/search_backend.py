"""SearchBackend：工具目录搜索后端接口与默认实现。

评分字段（KeywordSearchBackend）：
  name parts（精确/部分/兜底）→ 10/5/3（MCP: 12/6/3）
  search_hint（substring）    → +4
  description（substring）    → +2

中文查询通过字符级 substring + CJK bigram 处理，不依赖外部分词库。
why_matched 由 _explain() 独立生成，与 _score() 算法解耦。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.tools.registry import ToolDocument


class SearchBackend(ABC):
    """工具目录搜索后端接口。"""

    @abstractmethod
    def rebuild(self, documents: list["ToolDocument"]) -> None:
        """用给定文档列表重建全量索引。"""

    @abstractmethod
    def add(self, document: "ToolDocument") -> None:
        """增量添加一个工具文档到索引（动态注册用）。"""

    @abstractmethod
    def remove(self, name: str) -> None:
        """从索引中移除指定工具（动态注销用）。"""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
        excluded_names: AbstractSet[str] | None = None,
    ) -> list[dict]:
        """搜索工具目录，返回匹配的工具信息列表。

        返回格式：
          [{"name": ..., "summary": ..., "why_matched": [...],
            "risk": ..., "always_on": ...}, ...]
        """


# ── KeywordSearchBackend ──────────────────────────────────────────────────────


class KeywordSearchBackend(SearchBackend):
    """基于字段加权 + 字符级 substring 的关键词搜索后端。"""

    def __init__(self) -> None:
        self._docs: dict[str, "ToolDocument"] = {}

    def rebuild(self, documents: list["ToolDocument"]) -> None:
        self._docs = {doc.name: doc for doc in documents}

    def add(self, document: "ToolDocument") -> None:
        self._docs[document.name] = document

    def remove(self, name: str) -> None:
        self._docs.pop(name, None)

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
        excluded_names: AbstractSet[str] | None = None,
    ) -> list[dict]:
        risk_filter = set(allowed_risk) if allowed_risk else None
        skip = excluded_names or set()

        # ── Fast path：精确名称匹配，跳过评分循环 ──────────────────────────
        query_stripped = query.strip()
        if query_stripped in self._docs:
            doc = self._docs[query_stripped]
            if query_stripped not in skip:
                if not risk_filter or doc.risk in risk_filter:
                    return [_doc_to_result(doc, ["名称:精确匹配"])]

        # ── 关键词评分路径 ────────────────────────────────────────────────
        keywords = _default_normalize(query)
        if not keywords:
            # 空/纯空白 query 归一化后无有效词，直接返回空，不做全量扫描
            return []

        results = []
        for name, doc in self._docs.items():
            if name in skip:
                continue
            if risk_filter and doc.risk not in risk_filter:
                continue

            score = _score(doc, keywords)
            if score > 0:
                results.append(
                    (score, name, _doc_to_result(doc, _explain(doc, keywords)))
                )

        # 主键：score 降序；次键：name 字母序（稳定 tie-break）
        results.sort(key=lambda x: (-x[0], x[1]))
        return [r for _, _, r in results[:top_k]]


# ── 内部工具函数 ───────────────────────────────────────────────────────────────


def _doc_to_result(doc: "ToolDocument", why_matched: list[str]) -> dict:
    """将 ToolDocument 转为 search() 标准返回格式。"""
    return {
        "name": doc.name,
        "summary": doc.description[:120],
        "why_matched": why_matched,
        "risk": doc.risk,
        "always_on": doc.always_on,
    }


def _default_normalize(query: str) -> set[str]:
    """将 query 归一化为搜索词集合，无需手写同义词表或外部分词库。

    策略：
    1. 整体 lowercase，保留原始串
    2. 空格切词（对英文 / 带空格查询）
    3. CJK / 非 CJK 边界切词："RSS订阅" → ["rss", "订阅"]
    4. CJK bigram：让"定时提醒"能命中"定时任务"中的"定时"；
       单字覆盖"文"能命中"文件"

    完全基于工具自身 name/description 做召回，无需同义词表或 jieba。
    """
    import re

    query_lower = query.lower().strip()
    tokens: set[str] = set()
    tokens.add(query_lower)

    # 空格切词
    for part in query_lower.split():
        tokens.add(part)

    # CJK / 非 CJK 边界切词（处理 "RSS订阅"、"fitbit健康" 类混合串）
    for segment in re.split(r"([\u4e00-\u9fff]+)", query_lower):
        segment = segment.strip()
        if segment:
            tokens.add(segment)

    # CJK bigram + unigram
    cjk = [c for c in query_lower if "\u4e00" <= c <= "\u9fff"]
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i] + cjk[i + 1])
    tokens.update(cjk)

    # 过滤空串：防止 "" in any_string 永真
    tokens.discard("")
    return tokens


def _score(doc: "ToolDocument", keywords: set[str]) -> int:
    """给单个工具文档打分（纯 int score，不生成解释）。

    名称 parts 匹配（最高权重，MCP 工具额外 +2）：
      精确命中 part → 10（MCP: 12）
      部分命中 part → 5  (MCP: 6)
      全名兜底      → 3

    其他字段独立叠加（不与名称互斥）：
      search_hint → +4
      description → +2
    """
    name_parts = [p for p in doc.name.lower().split("_") if p]
    name_lower = doc.name.lower()
    hint_lower = (doc.search_hint or "").lower()
    desc_lower = doc.description.lower()
    is_mcp = doc.source_type == "mcp"

    score = 0
    for kw in keywords:
        # ── 名称 parts 匹配（精确 > 部分 > 全名兜底）────────────────────
        if kw in name_parts:
            score += 12 if is_mcp else 10
        elif any(kw in part or part in kw for part in name_parts):
            score += 6 if is_mcp else 5
        elif kw in name_lower:
            score += 3

        # ── 其他字段独立叠加 ─────────────────────────────────────────────
        if hint_lower and kw in hint_lower:
            score += 4
        if kw in desc_lower:
            score += 2

    return score


def _explain(doc: "ToolDocument", keywords: set[str]) -> list[str]:
    """生成 why_matched 解释（与 _score 解耦）。"""
    name_parts = [p for p in doc.name.lower().split("_") if p]
    name_lower = doc.name.lower()
    hint_lower = (doc.search_hint or "").lower()
    desc_lower = doc.description.lower()

    reasons: list[str] = []
    seen: set[str] = set()

    def _add(r: str) -> None:
        if r not in seen:
            seen.add(r)
            reasons.append(r)

    for kw in keywords:
        if kw in name_parts:
            _add(f"名称精确:{kw}")
        elif any(kw in part or part in kw for part in name_parts):
            _add(f"名称部分:{kw}")
        elif kw in name_lower:
            _add(f"名称:{kw}")

        if hint_lower and kw in hint_lower:
            _add(f"提示:{kw}")
        if kw in desc_lower:
            _add(f"描述:{kw}")

    return reasons
