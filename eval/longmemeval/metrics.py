"""Scoring functions for LongMemEval results.

Primary metric: token-level F1 (same as SQuAD / LoCoMo papers).
Secondary: exact match (normalised).
Tertiary: LLM-as-judge (semantic correctness, handles paraphrase/translation).
Both are computed per question_type so you can see where the system
is strong vs weak (single-session vs multi-session vs temporal etc.).
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are a strict judge for a long-term memory benchmark.

The gold answer describes what the user's preferences or facts are.
The predicted answer is what the agent actually said.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Judge strictly: the predicted answer is correct only if it reflects the specific preferences or facts stated in the gold answer. If the predicted answer asks the user for information that should already be in memory, or gives generic responses that ignore the user's specific preferences, answer no.

Reply with exactly one word: yes or no."""


# ── text normalisation ────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenise(text: str) -> list[str]:
    return _normalise(text).split()


# ── per-pair metrics ──────────────────────────────────────────────────────────

def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _tokenise(pred)
    gold_tokens = _tokenise(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return _normalise(pred) == _normalise(gold)


# ── llm judge ────────────────────────────────────────────────────────────────

async def judge_answer(
    provider,
    model: str,
    *,
    question: str,
    gold: str,
    predicted: str,
) -> bool:
    """Single LLM call: returns True if predicted is semantically correct."""
    if not predicted or not predicted.strip():
        return False
    prompt = _JUDGE_PROMPT.format(
        question=question.strip(),
        gold=gold.strip(),
        predicted=predicted.strip(),
    )
    try:
        resp = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            model=model,
            max_tokens=4,
        )
        content = getattr(resp, "content", None)
        if content is None:
            content = ""
        verdict = str(content).strip().lower()
        return verdict.startswith("yes")
    except Exception as e:
        logger.warning("judge_answer failed: %s", e)
        return False


# ── dataset-level scoring ─────────────────────────────────────────────────────

def score_results(results: list[dict]) -> dict:
    """Compute aggregate and per-type scores.

    Args:
        results: List of dicts from qa_runner.run_qa_instance.

    Returns:
        {
            "overall": {"f1": float, "em": float, "n": int, "errors": int},
            "by_type": {question_type: {"f1": float, "em": float, "n": int}},
        }
    """
    by_type: dict[str, list[dict]] = {}
    for r in results:
        qt = r.get("question_type") or "unknown"
        by_type.setdefault(qt, []).append(r)

    def _agg(items: list[dict]) -> dict:
        errors = sum(1 for r in items if r.get("error"))
        f1s = [
            0.0 if r.get("error") else token_f1(r["predicted_answer"], r["gold_answer"])
            for r in items
        ]
        ems = [
            0.0 if r.get("error") else (1.0 if exact_match(r["predicted_answer"], r["gold_answer"]) else 0.0)
            for r in items
        ]
        judged = [r for r in items if r.get("judge_correct") is not None and not r.get("error")]
        judge_acc = round(sum(1 for r in judged if r["judge_correct"]) / len(judged), 4) if judged else None
        n = len(items)
        if n == 0:
            return {"f1": 0.0, "em": 0.0, "judge_acc": None, "n": 0, "errors": 0}
        result = {
            "f1": round(sum(f1s) / n, 4),
            "em": round(sum(ems) / n, 4),
            "n": n,
            "errors": errors,
        }
        if judge_acc is not None:
            result["judge_acc"] = judge_acc
        return result

    return {
        "overall": _agg(results),
        "by_type": {qt: _agg(items) for qt, items in sorted(by_type.items())},
    }
