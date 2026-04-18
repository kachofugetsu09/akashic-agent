from __future__ import annotations

import re

from eval.longmemeval.metrics import exact_match, token_f1

_PAREN_LABEL_RE = re.compile(r"\(([a-z])\)", re.IGNORECASE)
_WORD_LABEL_RE = re.compile(r"\b([a-z])\b", re.IGNORECASE)


def extract_option_label(predicted: str, options: list[str]) -> str | None:
    text = str(predicted or "").strip()
    if not text:
        return None

    match = _PAREN_LABEL_RE.search(text)
    if match:
        value = (match.group(1) or "").lower()
        if value:
            index = ord(value) - ord("a")
            if 0 <= index < len(options):
                return f"({value})"

    word_matches = _WORD_LABEL_RE.findall(text)
    for raw_value in reversed(word_matches):
        value = str(raw_value or "").lower()
        index = ord(value) - ord("a")
        if 0 <= index < len(options):
            return f"({value})"

    best_label: str | None = None
    best_score = 0.0
    for idx, option in enumerate(options):
        if exact_match(text, option):
            return f"({chr(ord('a') + idx)})"
        score = token_f1(text, option)
        if score > best_score:
            best_score = score
            best_label = f"({chr(ord('a') + idx)})"
    if best_score >= 0.55:
        return best_label
    return None


def score_results(results: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = {}
    for result in results:
        question_type = str(result.get("question_type") or "unknown")
        by_type.setdefault(question_type, []).append(result)

    def _agg(items: list[dict]) -> dict:
        n = len(items)
        if n == 0:
            return {"accuracy": 0.0, "parsed_rate": 0.0, "n": 0, "errors": 0}
        errors = sum(1 for item in items if item.get("error"))
        parsed = sum(1 for item in items if item.get("predicted_label"))
        correct = sum(1 for item in items if item.get("is_correct"))
        return {
            "accuracy": round(correct / n, 4),
            "parsed_rate": round(parsed / n, 4),
            "n": n,
            "errors": errors,
        }

    return {
        "overall": _agg(results),
        "by_type": {qt: _agg(items) for qt, items in sorted(by_type.items())},
    }
