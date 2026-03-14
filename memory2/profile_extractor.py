from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ProfileFact:
    summary: str
    category: str
    happened_at: str | None


class ProfileFactExtractor:
    def __init__(
        self,
        llm_client: Any,
        *,
        model: str = "",
        max_tokens: int = 400,
        timeout_ms: int = 1200,
    ) -> None:
        self._llm_client = llm_client
        self._model = model
        self._max_tokens = max(128, int(max_tokens))
        self._timeout_s = max(0.1, float(timeout_ms) / 1000.0)

    async def extract(
        self,
        conversation: str,
        *,
        existing_profile: str = "",
    ) -> list[ProfileFact]:
        # 1. 先构造 prompt；空对话直接返回空列表。
        if not str(conversation or "").strip():
            return []
        prompt = self._build_prompt(
            conversation=conversation,
            existing_profile=existing_profile,
        )

        # 2. 再调用 LLM；异常时 fail-open 返回空列表。
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
            return []

        # 3. 最后解析 XML 并做去重；乱码时同样返回空列表。
        content = str(getattr(response, "content", response) or "")
        return self._parse_facts(content)

    @staticmethod
    def _build_prompt(*, conversation: str, existing_profile: str) -> str:
        return f"""你是 profile 事实提取器。请只从对话里提取用户长期可检索的 profile 事实，并输出 XML。

仅允许以下 5 类：
- purchase：用户购买 / 下单了什么
- decision：用户明确拍板了什么方案 / 计划
- preference：用户明确表达的稳定偏好
- status：用户某件事的状态变化（等待 / 完成 / 放弃）
- personal_fact：用户关于自身的事实性披露

必须遵守：
- 纯技术讨论、闲聊、打招呼，不输出
- 若 existing_profile 已有相同事实，不重复输出
- summary 要简洁、可独立检索

当前已有 profile（用于查重）：
{existing_profile or "（空）"}

待处理对话：
{conversation}

只输出 XML：
<facts>
<fact>
  <summary>...</summary>
  <category>purchase|decision|preference|status|personal_fact</category>
  <happened_at>YYYY-MM-DD</happened_at>
</fact>
</facts>"""

    def _parse_facts(self, raw_output: str) -> list[ProfileFact]:
        allowed = {"purchase", "decision", "preference", "status", "personal_fact"}
        matches = re.findall(r"<fact>\s*(.*?)\s*</fact>", raw_output or "", re.DOTALL)
        facts: list[ProfileFact] = []
        seen: set[tuple[str, str]] = set()
        for block in matches:
            summary = self._extract_tag(block, "summary")
            category = self._extract_tag(block, "category").lower()
            happened_at = self._extract_tag(block, "happened_at") or None
            if not summary or category not in allowed:
                continue
            key = (summary, category)
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                ProfileFact(
                    summary=summary,
                    category=category,
                    happened_at=happened_at,
                )
            )
        return facts

    @staticmethod
    def _extract_tag(raw_output: str, tag: str) -> str:
        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            raw_output or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""
