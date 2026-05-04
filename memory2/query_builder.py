from __future__ import annotations


def build_procedure_queries(user_msg: str, rewritten_query: str = "") -> list[str]:
    """为 procedure/preference 检索生成原始 query 和改写 query。"""
    msg = _normalize_text(user_msg)
    rewritten = _normalize_text(rewritten_query)
    queries = [item for item in (msg, rewritten) if item]
    if not queries:
        return []
    seen: set[str] = set()
    deduped: list[str] = []
    for item in queries:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())
