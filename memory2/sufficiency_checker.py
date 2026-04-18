from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, TypedDict, cast


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
    """触发条件：RETRIEVE 路径返回空结果，且没有 forced procedure。

    有结果时不重查——分数过滤已在注入筛选阶段完成，
    低分条目不会进入 selected_items，无需再做一次 LLM 质量判断。
    """
    if not items:
        return True
    if _has_forced_procedure(items):
        return False
    return False


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
        # 1. 先构造 prompt 与 fail-open 默认结果。
        started = time.perf_counter()
        prompt = self._build_prompt(query=query, items=items, context=context)

        # 2. 再调用 LLM；异常时保留现有结果，不阻断主路径。
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
        except Exception:
            return self._result(
                started=started,
                is_sufficient=True,
                reason="checker_error",
                refined_query=None,
            )

        # 3. 最后解析输出；乱码时同样 fail-open。
        parsed = self._parse_output(str(getattr(response, "content", response) or ""))
        if parsed is None:
            return self._result(
                started=started,
                is_sufficient=True,
                reason="parse_error",
                refined_query=None,
            )
        return self._result(started=started, **parsed)

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


def _has_forced_procedure(items: list[dict]) -> bool:
    for item in items:
        extra = item.get("extra_json")
        if item.get("memory_type") == "procedure" and isinstance(extra, dict):
            if str(extra.get("tool_requirement", "") or "").strip():
                return True
    return False
