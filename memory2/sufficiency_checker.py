from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from agent.provider import LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class SufficiencyResult:
    is_sufficient: bool
    reason: str
    refined_query: str | None
    latency_ms: int


class _ParsedSufficiency(TypedDict):
    is_sufficient: bool
    reason: str
    refined_query: str | None


def should_check_sufficiency(items: list[dict]) -> bool:
    """仅在检索没有结果时触发质检。"""
    return not items


class SufficiencyChecker:
    def __init__(
        self,
        llm_client: Any,
        *,
        max_tokens: int = 120,
        timeout_ms: int = 600,
        model: str = "",
    ) -> None:
        self._llm_client = llm_client
        self._max_tokens = max(64, int(max_tokens))
        self._timeout_s = max(0.1, float(timeout_ms) / 1000.0)
        self._model = model

    async def check(
        self,
        query: str,
        items: list[dict],
        context: str = "",
    ) -> SufficiencyResult:
        # 1. 构造 prompt。
        started = time.perf_counter()
        prompt = self._build_prompt(query=query, items=items, context=context)

        # 2. 模型边界失败时保留现有结果，并留下明确原因。
        try:
            response = await asyncio.wait_for(
                self._llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self._model,
                    max_tokens=self._max_tokens,
                ),
                timeout=self._timeout_s,
            )
            content = self._response_content(response)
        except Exception as exc:
            logger.warning("memory2 sufficiency check failed: %s", exc)
            return self._result(
                started=started,
                is_sufficient=True,
                reason="checker_error",
                refined_query=None,
            )

        # 3. 解析模型输出；结构无效时不触发重查。
        parsed = self._parse_output(content)
        if parsed is None:
            return self._result(
                started=started,
                is_sufficient=True,
                reason="parse_error",
                refined_query=None,
            )
        return self._result(started=started, **parsed)

    @staticmethod
    def _response_content(response: object) -> str:
        if isinstance(response, str):
            return response
        if not isinstance(response, LLMResponse):
            raise TypeError("LLM response 必须是字符串或 LLMResponse")
        return response.content or ""

    def _build_prompt(self, *, query: str, items: list[dict], context: str) -> str:
        context_block = f"\n补充上下文：\n{context.strip()}\n" if context.strip() else ""
        items_block = self._format_items(items)
        return f"""你是检索结果质检器。请判断当前 query 与已检索到的记忆条目是否相关且足够支持回答。

当前 query：
{query}
{context_block}

已检索到的记忆条目：
{items_block}

判断规则：
- yes：结果相关且足够，直接使用现有结果
- partial：部分相关但不完整，仍然保留现有结果，不要触发重查
- no：结果为空或明显无关，需要给出更精确的 refined_query

只输出 XML：
<sufficient>yes|no|partial</sufficient>
<refined_query>...</refined_query>"""

    def _format_items(self, items: list[dict]) -> str:
        if not items:
            return "（无结果）"
        lines: list[str] = []
        for index, item in enumerate(items[:8], start=1):
            summary = str(item.get("summary", "") or "")[:120]
            lines.append(
                f"{index}. [{str(item.get('memory_type', ''))}] "
                f"score={float(item.get('score', 0.0) or 0.0):.3f} {summary}"
            )
        return "\n".join(lines)

    def _parse_output(self, raw_output: str) -> _ParsedSufficiency | None:
        decision = self._extract_tag(raw_output, "sufficient").lower()
        refined = self._extract_tag(raw_output, "refined_query") or None
        if decision == "yes":
            return cast(
                _ParsedSufficiency,
                {
                    "is_sufficient": True,
                    "reason": "sufficient",
                    "refined_query": None,
                },
            )
        if decision == "partial":
            return cast(
                _ParsedSufficiency,
                {
                    "is_sufficient": True,
                    "reason": "partial",
                    "refined_query": None,
                },
            )
        if decision == "no":
            return cast(
                _ParsedSufficiency,
                {
                    "is_sufficient": False,
                    "reason": "irrelevant",
                    "refined_query": refined,
                },
            )
        return None

    def _result(
        self,
        *,
        started: float,
        is_sufficient: bool,
        reason: str,
        refined_query: str | None,
    ) -> SufficiencyResult:
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        return SufficiencyResult(
            is_sufficient=is_sufficient,
            reason=reason,
            refined_query=refined_query,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_tag(raw_output: str, tag: str) -> str:
        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            raw_output or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""
