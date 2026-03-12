from __future__ import annotations

import json
import json_repair
import logging
import math
import random as _random_module
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from feeds.base import FeedItem
from proactive.item_id import compute_item_id, compute_source_key, normalize_url

logger = logging.getLogger("proactive.loop")


@dataclass
class _Decision:
    score: float
    should_send: bool
    message: str
    reasoning: str
    evidence_item_ids: list[str] = field(default_factory=list)


def _decision_with_randomized_score(
    decision: _Decision,
    *,
    strength: float,
    rng: _random_module.Random | None = None,
) -> tuple[_Decision, float]:
    s = max(0.0, min(1.0, strength))
    if s <= 0:
        return decision, 0.0
    delta = (rng or _random_module).uniform(-s, s)
    score = max(0.0, min(1.0, decision.score + delta))
    return (
        _Decision(
            score=score,
            should_send=decision.should_send,
            message=decision.message,
            reasoning=decision.reasoning,
            evidence_item_ids=decision.evidence_item_ids,
        ),
        delta,
    )


def _format_items(items: list[FeedItem]) -> str:
    out = []
    for i, item in enumerate(items, 1):
        title = item.title or "（无标题）"
        content = (item.content or "").strip().replace("\n", " ")
        if len(content) > 300:
            content = content[:300] + "..."
        meta = []
        if item.source_name:
            meta.append(item.source_name)
        if item.author:
            meta.append(item.author)
        if item.published_at:
            meta.append(str(item.published_at))
        meta_str = f" [{' / '.join(meta)}]" if meta else ""
        url_line = f"\n原文链接: {item.url}" if item.url else ""
        out.append(f"{i}. {title}{meta_str}\n{content}{url_line}")
    return "\n\n".join(out)


def _format_recent(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        role = m.get("role", "user")
        name = "用户" if role == "user" else "助手"
        content = (m.get("content") or "").strip()
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"{name}: {content}")
    return "\n".join(lines)


def _parse_decision(text: str) -> _Decision:
    try:
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json_repair.loads(text)
        if not isinstance(data, dict):
            raise ValueError("decision is not object")
        score = float(data.get("score", 0.0))
        should_send = _strict_bool(data.get("should_send", False))
        message = str(data.get("message", "") or "")
        reasoning = str(data.get("reasoning", "") or "")
        evidence_ids = data.get("evidence_item_ids") or []
        evidence_item_ids = [str(x) for x in evidence_ids if str(x).strip()]
        return _Decision(
            score=score,
            should_send=should_send,
            message=message,
            reasoning=reasoning,
            evidence_item_ids=evidence_item_ids,
        )
    except Exception as e:
        return _Decision(score=0.0, should_send=False, message="", reasoning=str(e))


def _strict_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _source_key(item: FeedItem) -> str:
    return compute_source_key(item)


def _item_id(item: FeedItem) -> str:
    return compute_item_id(item)


def _normalize_url(url: str | None) -> str:
    return normalize_url(url)


def _resolve_evidence_item_ids(decision: _Decision, items: list[FeedItem]) -> list[str]:
    if decision.evidence_item_ids:
        return decision.evidence_item_ids[:1]
    if not items:
        return []
    return [_item_id(item) for item in items[:1]]


def _build_delivery_key(item_ids: list[str], message: str) -> str:
    ids = ",".join(sorted(i for i in item_ids if i))
    return f"{ids}|{message.strip()}"


def _semantic_text(item: FeedItem, max_chars: int) -> str:
    title = (item.title or "").strip()
    content = re.sub(r"\s+", " ", (item.content or "").strip())
    text = f"{title}\n{content}".strip()
    return text[:max_chars]


def _semantic_entries(items: list[FeedItem], max_chars: int) -> list[dict[str, str]]:
    return [
        {
            "item_id": _item_id(item),
            "source_key": _source_key(item),
            "text": _semantic_text(item, max_chars),
        }
        for item in items
    ]


def _char_ngrams(text: str, n: int) -> list[str]:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i : i + n] for i in range(0, len(compact) - n + 1)]


def _build_tfidf_vectors(texts: list[str], n: int) -> list[dict[str, float]]:
    doc_tokens = [_char_ngrams(text, n) for text in texts]
    df: Counter[str] = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    doc_count = max(1, len(doc_tokens))
    vectors = []
    for toks in doc_tokens:
        tf = Counter(toks)
        total = max(1, len(toks))
        vec: dict[str, float] = {}
        for tok, cnt in tf.items():
            idf = math.log((1 + doc_count) / (1 + df[tok])) + 1.0
            vec[tok] = (cnt / total) * idf
        vectors.append(vec)
    return vectors


def _build_sandboxed_shell(workspace_dir: Path):
    import shutil

    from agent.tools.shell import ShellTool

    if not shutil.which("firejail"):
        logger.warning("[proactive] firejail 未找到，SubAgent 使用未沙箱化的 ShellTool")
        return ShellTool()

    class _FirejailShellTool(ShellTool):
        async def execute(self, **kwargs):
            import shlex as _shlex

            command = kwargs.get("command", "").strip()
            if not command:
                return json.dumps({"error": "命令不能为空"}, ensure_ascii=False)
            sandboxed = (
                "firejail --quiet "
                "--blacklist=~/.ssh "
                "--blacklist=~/.gnupg "
                f"-- bash -c {_shlex.quote(command)}"
            )
            kwargs["command"] = sandboxed
            return await super().execute(**kwargs)

    logger.info(
        "[proactive] SubAgent ShellTool 已启用 firejail 沙箱 dir=%s", workspace_dir
    )
    return _FirejailShellTool()


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = 0.0
    for k, v in small.items():
        dot += v * large.get(k, 0.0)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
