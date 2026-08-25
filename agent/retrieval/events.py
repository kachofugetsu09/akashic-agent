from __future__ import annotations

from agent.retrieval.protocol import RetrievalRequest
from core.memory.engine import MemoryQueryResult, MemoryRecord
from core.memory.events import RetrievalCompleted, RetrievalHitSummary


def build_retrieval_completed(
    request: RetrievalRequest,
    result: MemoryQueryResult | None,
    *,
    error: BaseException | None = None,
) -> RetrievalCompleted:
    """Freeze a plugin retrieval result into the shared observation event."""

    records = [] if result is None else list(result.records)
    raw = {} if result is None else result.raw
    trace = {} if result is None else result.trace
    query = _first_nonempty_string(raw.get("rewritten_query"), request.message)
    aux_queries = _string_list(raw.get("aux_queries")) or _string_list(
        trace.get("hyde_hypotheses")
    )
    hits = [_build_hit_summary(record) for record in records]
    return RetrievalCompleted(
        session_key=request.session_key,
        channel=request.channel,
        chat_id=request.chat_id,
        query=query,
        orig_query=request.message if query != request.message else None,
        hits=hits,
        injected_count=sum(1 for hit in hits if hit.injected),
        route_decision=_optional_string(trace.get("route_decision")),
        aux_queries=aux_queries,
        error=None if error is None else str(error) or type(error).__name__,
    )


def _build_hit_summary(record: MemoryRecord) -> RetrievalHitSummary:
    signals = dict(record.signals)
    confidence_label = signals.get("confidence_label")
    return RetrievalHitSummary(
        item_id=record.id,
        memory_type=record.kind,
        score=float(record.score),
        summary=record.summary[:120],
        injected=bool(record.injected),
        confidence_label=confidence_label if isinstance(confidence_label, str) else "",
        forced=bool(signals.get("forced", False)),
        metadata=signals,
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _first_nonempty_string(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
