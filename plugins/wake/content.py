from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import cast
from urllib.parse import urlsplit

from .pool import PoolResult
from .selection import DutyProposal

def _content_candidates(
    proposal: DutyProposal,
) -> tuple[Mapping[str, object], ...]:
    return proposal.candidates or ({"ref": proposal.ref, "payload": proposal.payload},)


def _candidate_payloads(proposal: DutyProposal) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for candidate in _content_candidates(proposal):
        ref = _mapping(candidate.get("ref"), "Content candidate ref")
        payload = _mapping(candidate.get("payload"), "Content candidate payload")
        candidates.append({"candidate_id": _candidate_id(ref), **dict(payload)})
    return candidates


def _selected_content_refs(
    receipt: Mapping[str, object],
    item_ids: Sequence[str],
    *,
    allow_legacy_single: bool = False,
) -> tuple[Mapping[str, object], ...]:
    """Resolve the model's candidate ids only against the frozen Content batch."""

    raw_items = receipt.get("items")
    items = _sequence(raw_items, "Content selection items")
    if not item_ids and allow_legacy_single:
        if len(items) != 1:
            raise RuntimeError("legacy Content selection 必须恰好包含一个 member")
        return (_mapping(items[0].get("ref"), "Content selection ref"),)
    if not item_ids:
        raise ValueError("Content share_content 必须引用至少一个 candidate_id")
    candidates: dict[str, Mapping[str, object]] = {}
    for item in items:
        ref = _mapping(item.get("ref"), "Content selection ref")
        candidates[_candidate_id(ref)] = ref
    unknown = set(item_ids) - set(candidates)
    if unknown:
        raise ValueError(
            f"Content share_content 引用了批次外 candidate_id: {sorted(unknown)}"
        )
    return tuple(candidates[item_id] for item_id in item_ids)


def _candidate_id(ref: Mapping[str, object]) -> str:
    fields = (
        _string(ref.get("source_id"), "source_id"),
        _string(ref.get("item_id"), "item_id"),
        _string(ref.get("revision"), "revision"),
    )
    payload = "\x00".join(fields).encode("utf-8")
    return "candidate_" + hashlib.sha256(payload).hexdigest()[:16]


def _delivery_metadata(receipt: Mapping[str, object]) -> dict[str, object]:
    raw = receipt.get("message_metadata")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("Wake delivery message_metadata 必须是 Mapping")
    return dict(cast(Mapping[str, object], raw))


def _message_with_source_links(message: str, metadata: Mapping[str, object]) -> str:
    """Append selected source links so the user and later Turns retain provenance."""

    raw_refs = metadata.get("source_refs")
    if not isinstance(raw_refs, (list, tuple)):
        return message
    links: list[str] = []
    seen: set[str] = set()
    for raw_ref in cast(Sequence[object], raw_refs):
        if not isinstance(raw_ref, Mapping):
            continue
        source_ref = cast(Mapping[str, object], raw_ref)
        raw_url = source_ref.get("url")
        if not isinstance(raw_url, str):
            continue
        url = raw_url.strip()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if url in seen or url in message:
            continue
        seen.add(url)
        raw_title = source_ref.get("title")
        title = (
            " ".join(raw_title.split())
            if isinstance(raw_title, str) and raw_title.strip()
            else f"来源 {len(links) + 1}"
        )
        links.append(f"- {title}：<{url}>")
    if not links:
        return message
    return f"{message.rstrip()}\n\n来源：\n" + "\n".join(links)


def _pool_detail(
    pool: str,
    new_count: int,
    result: PoolResult,
) -> str:
    """Persist the values needed to reconstruct one fixed-score pool check."""

    return (
        f"{pool}, new={new_count}, new_mass={result.new_mass:.6f}, "
        f"pool_mass={result.pool_mass:.6f}, threshold={result.threshold:.6f}, "
        f"below_floor={result.below_floor}, "
        f"driver={result.driver_item_id or '-'}"
    )


def _proposal_next_due(
    proposals: Sequence[Mapping[str, object]], proposal: DutyProposal
) -> bool:
    return any(
        item.get("ref") == proposal.ref and item.get("next_due") is not None
        for item in proposals
    )


def _sequence(value: object, field: str) -> Sequence[Mapping[str, object]]:
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise ValueError(f"{field} 必须是 Mapping sequence")
    return cast(Sequence[Mapping[str, object]], value)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是 Mapping")
    return cast(Mapping[str, object], value)


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} 必须是整数")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} 必须是非空字符串")
    return value


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value)
    else:
        raise ValueError("Wake deadline 必须是 datetime 或 ISO 字符串")
    if result.tzinfo is None:
        raise ValueError("Wake deadline 必须带时区")
    return result.astimezone(UTC)


def _content_text(item: Mapping[str, object]) -> str:
    payload = _mapping(item.get("payload"), "Content item payload")
    text = "\n".join(
        part
        for part in (
            str(payload.get("title") or "").strip(),
            str(payload.get("content") or payload.get("body") or "").strip(),
        )
        if part
    )
    return text


def _preprocess_interest(payload: Mapping[str, object]) -> float:
    features = payload.get("preprocess_features")
    raw = (
        features.get("interest")
        if isinstance(features, Mapping)
        else payload.get("preprocess_score")
    )
    if not isinstance(raw, (int, float, str)):
        return 0.0
    try:
        return min(0.999, max(0.0, float(raw or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _semantic_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("semantic interest 必须是数字")
    score = float(value)
    if not 0.0 <= score <= 0.999:
        raise RuntimeError("semantic interest 必须在 0..0.999")
    return score
