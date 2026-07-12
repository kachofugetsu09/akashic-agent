from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plugins.wake_proactive.context import event_item_id


@dataclass(slots=True)
class RenderedShare:
    message: str
    evidence: list[str]
    display_event_map: dict[int, str]
    source_refs: list[dict[str, Any]]


def render_share(
    *,
    opening: str,
    items: list[dict[str, str]],
    closing: str,
    events: list[dict[str, Any]],
) -> RenderedShare:
    event_map = {event_item_id(event): event for event in events}
    blocks: list[str] = []
    opening = opening.strip()
    closing = closing.strip()
    if opening:
        blocks.append(opening)

    evidence: list[str] = []
    display_event_map: dict[int, str] = {}
    source_refs: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        item_id = str(item["item_id"]).strip()
        event = event_map[item_id]
        title = str(event.get("title") or "这条内容").strip()
        summary = str(item.get("summary") or "").strip()
        why = str(item.get("why_it_matters") or "").strip()
        url = str(event.get("url") or "").strip()
        source = str(event.get("source") or event.get("source_name") or "").strip()

        heading = title if len(items) == 1 else f"{index}. {title}"
        lines = [heading, summary]
        if why:
            lines.append(f"和你有关的是：{why}")
        if url:
            lines.append(f"原始来源：{url}")
        blocks.append("\n".join(line for line in lines if line))

        evidence.append(item_id)
        display_event_map[index] = item_id
        source_refs.append(
            {
                "display_index": index,
                "event_id": item_id,
                "source_name": source,
                "title": title,
                "url": url,
            }
        )

    if closing:
        blocks.append(closing)
    return RenderedShare(
        message="\n\n".join(blocks),
        evidence=evidence,
        display_event_map=display_event_map,
        source_refs=source_refs,
    )
