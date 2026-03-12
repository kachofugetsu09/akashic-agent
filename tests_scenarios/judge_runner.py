from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent.provider import LLMProvider

from tests_scenarios.fixtures import ScenarioJudgeSpec, ScenarioSpec


@dataclass
class ScenarioJudgeVerdict:
    passed: bool
    score: float
    reasons: list[str]
    raw_text: str


class ScenarioJudgeRunner:
    def __init__(self, provider: LLMProvider, model: str) -> None:
        self._provider = provider
        self._model = model

    async def run(
        self,
        spec: ScenarioSpec,
        judge: ScenarioJudgeSpec,
        *,
        final_content: str,
        memory_trace: dict,
        tool_calls: list[dict],
    ) -> ScenarioJudgeVerdict:
        prompt = self._build_prompt(spec, judge, final_content, memory_trace, tool_calls)
        response = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            model=self._model,
            max_tokens=400,
        )
        return self._parse_verdict(response.content or "")

    def _build_prompt(
        self,
        spec: ScenarioSpec,
        judge: ScenarioJudgeSpec,
        final_content: str,
        memory_trace: dict,
        tool_calls: list[dict],
    ) -> str:
        rubric_text = "\n".join(f"- {line}" for line in judge.rubric)
        return (
            "你是 AgentLoop 场景测试的 judge，只能输出 JSON。\n"
            "请根据场景目标与最终回答判断是否通过。\n\n"
            f"[场景目标]\n{judge.goal}\n\n"
            f"[用户消息]\n{spec.message}\n\n"
            f"[评分标准]\n{rubric_text}\n\n"
            f"[记忆轨迹]\n{json.dumps(memory_trace, ensure_ascii=False)}\n\n"
            f"[工具轨迹]\n{json.dumps(tool_calls, ensure_ascii=False)}\n\n"
            f"[最终回答]\n{final_content}\n\n"
            '输出格式：{"passed": true, "score": 0.95, "reasons": ["..."]}'
        )

    def _parse_verdict(self, text: str) -> ScenarioJudgeVerdict:
        payload = self._extract_json(text)
        return ScenarioJudgeVerdict(
            passed=bool(payload.get("passed", False)),
            score=float(payload.get("score", 0.0) or 0.0),
            reasons=[str(item) for item in payload.get("reasons", [])],
            raw_text=text,
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
