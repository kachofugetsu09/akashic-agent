from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from agent.provider import LLMProvider
from prompts.proactive import build_post_judge_prompt_messages
from proactive_v2.json_utils import extract_json_object

logger = logging.getLogger(__name__)


def _format_recent_proactive_entries(recent_proactive: list[object]) -> str:
    lines: list[str] = []
    for index, message in enumerate(recent_proactive, 1):
        content = _field(message, "content")
        if not content:
            continue
        meta = _recent_meta(message)
        suffix = f" ({'; '.join(meta)})" if meta else ""
        lines.append(f"[{index}]{suffix} {content}")
    return "\n---\n".join(lines)


def _field(raw: object, name: str, default: str = "") -> str:
    if isinstance(raw, dict):
        return str(raw.get(name, default) or default).strip()
    return str(getattr(raw, name, default) or default).strip()


def _recent_meta(message: object) -> list[str]:
    meta: list[str] = []
    timestamp = getattr(message, "timestamp", None)
    if timestamp is not None:
        try:
            meta.append(f"time={timestamp.isoformat()}")
        except Exception:
            meta.append(f"time={timestamp}")
    tag = _field(message, "state_summary_tag", "none")
    if tag and tag != "none":
        meta.append(f"state_tag={tag}")
    return meta


@dataclass(frozen=True)
class ProactiveJudgeResult:
    final_score: float
    should_send: bool
    vetoed_by: str | None
    dims_deterministic: dict[str, float]
    dims_llm: dict[str, float]
    dims_llm_raw: dict[str, int]


class Judge:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        max_tokens: int,
        format_recent: Callable[[list[dict]], str],
        cfg: Any,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._format_recent = format_recent
        self._cfg = cfg

    def pre_compose_veto(
        self,
        *,
        age_hours: float,
        sent_24h: int,
        interrupt_factor: float,
    ) -> str | None:
        deterministic = self._build_deterministic_dims(
            age_hours=age_hours,
            sent_24h=sent_24h,
            interrupt_factor=interrupt_factor,
        )
        vetoed = self._deterministic_veto(deterministic)
        if vetoed:
            logger.info(
                "[judge] pre-compose 确定性否决 vetoed_by=%s urgency=%.3f balance=%.3f",
                vetoed,
                deterministic["urgency"],
                deterministic["balance"],
            )
        return vetoed

    async def judge_message(
        self,
        *,
        message: str,
        recent: list[dict],
        recent_proactive_text: str,
        preference_block: str = "",
        age_hours: float,
        sent_24h: int,
        interrupt_factor: float,
    ) -> ProactiveJudgeResult:
        deterministic = self._build_deterministic_dims(
            age_hours=age_hours,
            sent_24h=sent_24h,
            interrupt_factor=interrupt_factor,
        )
        vetoed = self._deterministic_veto(deterministic)
        if vetoed:
            return ProactiveJudgeResult(0.0, False, vetoed, deterministic, {}, {})
        llm_dims, llm_dims_raw = await self._score_llm_dims(
            message=message,
            recent=recent,
            recent_proactive_text=recent_proactive_text,
            preference_block=preference_block,
        )
        low_dim_reason = self._llm_dim_reject_reason(llm_dims)
        if low_dim_reason is not None:
            logger.info(
                "[judge] LLM 维度否决 vetoed_by=%s llm_dims=%s",
                low_dim_reason,
                {k: f"{v:.2f}" for k, v in llm_dims.items()},
            )
            return ProactiveJudgeResult(
                0.0,
                False,
                low_dim_reason,
                deterministic,
                llm_dims,
                llm_dims_raw,
            )
        final_score = self._compute_final_score(deterministic, llm_dims)
        threshold = float(getattr(self._cfg, "judge_send_threshold", 0.60))
        should_send = final_score >= threshold
        logger.info(
            "[judge] 最终决策 final_score=%.3f threshold=%.3f should_send=%s deterministic=%s llm=%s",
            final_score,
            threshold,
            should_send,
            {k: f"{v:.2f}" for k, v in deterministic.items()},
            {k: f"{v:.2f}" for k, v in llm_dims.items()},
        )
        return ProactiveJudgeResult(
            final_score,
            should_send,
            None,
            deterministic,
            llm_dims,
            llm_dims_raw,
        )

    def _build_deterministic_dims(
        self,
        *,
        age_hours: float,
        sent_24h: int,
        interrupt_factor: float,
    ) -> dict[str, float]:
        daily_max = max(1, int(getattr(self._cfg, "judge_balance_daily_max", 8)))
        urgency_horizon = max(
            1.0, float(getattr(self._cfg, "judge_urgency_horizon_hours", 12.0))
        )
        urgency = max(0.0, 1.0 - (max(age_hours, 0.0) / urgency_horizon))
        balance = max(0.0, 1.0 - (max(sent_24h, 0) / float(daily_max)))
        dynamics = 0.6 + 0.4 * max(0.0, min(1.0, float(interrupt_factor)))
        return {"urgency": urgency, "balance": balance, "dynamics": dynamics}

    def _deterministic_veto(self, deterministic: dict[str, float]) -> str | None:
        if deterministic["balance"] < float(
            getattr(self._cfg, "judge_veto_balance_min", 0.1)
        ):
            return "balance"
        return None

    async def _score_llm_dims(
        self,
        *,
        message: str,
        recent: list[dict],
        recent_proactive_text: str,
        preference_block: str,
    ) -> tuple[dict[str, float], dict[str, int]]:
        system_msg, user_msg = build_post_judge_prompt_messages(
            recent_summary=self._format_recent(recent) or "（无近期对话）",
            last_proactive=recent_proactive_text or "（无近期主动消息）",
            composed_message=message,
            preference_block=preference_block,
        )
        try:
            response = await self._provider.chat(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                tools=[],
                model=self._model,
                max_tokens=min(256, self._max_tokens),
            )
            raw = extract_json_object(response.content or "")
        except Exception:
            raw = {}
        raw_dims = {
            "information_gap": self._clamp_dim(raw.get("information_gap")),
            "relevance": self._clamp_dim(raw.get("relevance")),
            "expected_impact": self._clamp_dim(raw.get("expected_impact")),
        }
        normalized = {key: (value - 1) / 4.0 for key, value in raw_dims.items()}
        return normalized, raw_dims

    def _llm_dim_reject_reason(self, llm_dims: dict[str, float]) -> str | None:
        llm_veto_min = (int(getattr(self._cfg, "judge_veto_llm_dim_min", 2)) - 1) / 4.0
        if any(value < llm_veto_min for value in llm_dims.values()):
            return "llm_dim"
        return None

    @staticmethod
    def _clamp_dim(raw: object) -> int:
        try:
            value = int(raw) if isinstance(raw, int | float | str) else 2
        except Exception:
            value = 2
        return max(1, min(5, value))

    def _compute_final_score(
        self,
        deterministic: dict[str, float],
        llm_dims: dict[str, float],
    ) -> float:
        weights = {
            "urgency": float(getattr(self._cfg, "judge_weight_urgency", 0.15)),
            "balance": float(getattr(self._cfg, "judge_weight_balance", 0.10)),
            "dynamics": float(getattr(self._cfg, "judge_weight_dynamics", 0.10)),
            "information_gap": float(
                getattr(self._cfg, "judge_weight_information_gap", 0.25)
            ),
            "relevance": float(getattr(self._cfg, "judge_weight_relevance", 0.20)),
            "expected_impact": float(
                getattr(self._cfg, "judge_weight_expected_impact", 0.20)
            ),
        }
        dims = dict(deterministic)
        dims.update(llm_dims)
        weight_sum = sum(weights.get(key, 0.0) for key in dims)
        if weight_sum <= 0:
            return 0.0
        return sum(weights.get(key, 0.0) * dims[key] for key in dims) / weight_sum


class MessageDeduper:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        max_tokens: int,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens

    async def is_duplicate(
        self,
        new_message: str,
        recent_proactive: list[object],
        new_state_summary_tag: str = "none",
    ) -> tuple[bool, str]:
        if not recent_proactive:
            return False, "无近期主动消息，放行"
        try:
            response = await self._provider.chat(
                messages=self._build_messages(
                    new_message,
                    recent_proactive,
                    new_state_summary_tag,
                ),
                tools=[],
                model=self._model,
                max_tokens=min(128, self._max_tokens),
            )
            payload = extract_json_object((response.content or "").strip())
        except Exception as exc:
            logger.warning("[proactive.deduper] 检测失败，放行: %s", exc)
            return False, str(exc)
        is_duplicate = bool(payload.get("is_duplicate", False))
        reason = str(payload.get("reason", ""))
        logger.info(
            "[proactive.deduper] is_duplicate=%s reason=%r",
            is_duplicate,
            reason[:80],
        )
        return is_duplicate, reason

    def _build_messages(
        self,
        new_message: str,
        recent_proactive: list[object],
        new_state_summary_tag: str,
    ) -> list[dict[str, str]]:
        system_msg = (
            "你是消息重复检测器。判断【新消息】是否与【近期已发消息】在实质信息上重复。\n"
            "重复包括：同一事件重复，或同一用户状态总结/安慰框架重复。\n"
            "不重复包括：同话题但有真正新进展或明显不同角度。\n"
            "只输出 JSON。"
        )
        user_msg = (
            f"近期已发消息：\n{_format_recent_proactive_entries(recent_proactive)}\n\n"
            f"---\n新消息：{new_message}\n"
            f"新消息 state_summary_tag：{new_state_summary_tag}\n\n"
            "---\n只输出 JSON：\n"
            '{"is_duplicate": false, "reason": "简短说明"}'
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
