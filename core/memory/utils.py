from __future__ import annotations

from core.memory.engine import EvidenceRef, MemoryQuery, MemoryScope


def evidence_from_source_ref(source_ref: str) -> list[EvidenceRef]:
    value = (source_ref or "").strip()
    if not value:
        return []
    return [EvidenceRef(refs=[value], source_ref=value)]


def source_ref_from_evidence(
    evidence: list[EvidenceRef],
    *,
    fallback: str = "",
) -> str:
    for item in evidence:
        if item.source_ref.strip():
            return item.source_ref.strip()
        if item.refs:
            return item.refs[0]
    return fallback


def resolve_memory_scope(scope: MemoryScope) -> MemoryScope:
    if scope.channel and scope.chat_id:
        return scope
    if not scope.session_key or ":" not in scope.session_key:
        return scope
    channel, chat_id = scope.session_key.split(":", 1)
    return MemoryScope(
        session_key=scope.session_key,
        channel=scope.channel or channel,
        chat_id=scope.chat_id or chat_id,
    )


def should_require_scope_match(request: MemoryQuery, scope: MemoryScope) -> bool:
    if request.intent in {"answer", "interest"}:
        return bool(scope.channel and scope.chat_id)
    return bool(request.filters.hints.get("require_scope_match", False))
