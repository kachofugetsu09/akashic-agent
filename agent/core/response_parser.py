from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import cast


def _empty_cited_memory_ids() -> list[str]:
    return []


@dataclass
class ResponseMetadata:
    raw_text: str
    cited_memory_ids: list[str] = field(default_factory=_empty_cited_memory_ids)
    meme_tag: str | None = None


@dataclass
class ParsedResponse:
    clean_text: str
    metadata: ResponseMetadata


_CITED_RE = re.compile(r"(?:\n|\r\n)?§cited:\[([A-Za-z0-9_,\-\s]+)\]§\s*$")
_MEME_RE = re.compile(r"<meme:([a-zA-Z0-9_-]+)>", re.IGNORECASE)


def parse_response(
    raw_text: str,
    *,
    tool_chain: list[dict[str, object]],
) -> ParsedResponse:
    without_meme, meme_tag = _extract_meme_tag(raw_text)
    clean_text, cited_memory_ids = _extract_cited_ids(without_meme)
    if not cited_memory_ids:
        cited_memory_ids = extract_cited_ids_from_tool_chain(tool_chain)
    return ParsedResponse(
        clean_text=clean_text,
        metadata=ResponseMetadata(
            raw_text=raw_text,
            cited_memory_ids=cited_memory_ids,
            meme_tag=meme_tag,
        ),
    )


def _extract_meme_tag(response: str) -> tuple[str, str | None]:
    first = _MEME_RE.search(response)
    cleaned = _MEME_RE.sub("", response).strip()
    if first is None:
        return cleaned, None
    return cleaned, first.group(1).lower()


def _extract_cited_ids(response: str) -> tuple[str, list[str]]:
    match = _CITED_RE.search(response)
    if not match:
        return response, []
    raw = match.group(1)
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    clean = response[: match.start()].rstrip()
    return clean, ids


def extract_cited_ids_from_tool_chain(
    tool_chain: list[dict[str, object]],
) -> list[str]:
    cited: list[str] = []
    seen: set[str] = set()
    for group in tool_chain:
        calls_value = group.get("calls")
        if not isinstance(calls_value, list):
            continue
        calls = cast(list[object], calls_value)
        for raw_call in calls:
            if not isinstance(raw_call, dict):
                continue
            call = cast(dict[str, object], raw_call)
            if str(call.get("name", "") or "") != "recall_memory":
                continue
            raw_result = str(call.get("result", "") or "").strip()
            if not raw_result:
                continue
            try:
                decoded = json.loads(raw_result)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(decoded, dict):
                continue
            data = cast(dict[str, object], decoded)
            raw_ids: list[object] = []
            cited_ids = data.get("cited_item_ids")
            if isinstance(cited_ids, list):
                raw_ids.extend(cast(list[object], cited_ids))
            else:
                items_value = data.get("items")
                if isinstance(items_value, list):
                    items = cast(list[object], items_value)
                    for raw_item in items:
                        if isinstance(raw_item, dict):
                            item = cast(dict[str, object], raw_item)
                            raw_ids.append(item.get("id"))
            for raw_id in raw_ids:
                item_id = str(raw_id or "").strip()
                if item_id and item_id not in seen:
                    seen.add(item_id)
                    cited.append(item_id)
    return cited
