from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plugins.wake_proactive.context import event_item_aliases


@dataclass(slots=True)
class RenderedShare:
    message: str
    evidence: list[str]
    display_event_map: dict[int, str]
    source_refs: list[dict[str, Any]]


def render_share(
    *,
    message: str = "",
    opening: str,
    items: list[dict[str, str]],
    closing: str,
    events: list[dict[str, Any]],
) -> RenderedShare:
    event_map = {
        alias: event
        for event in events
        for alias in event_item_aliases(event)
    }
    blocks: list[str] = []
    opening = opening.strip()
    closing = closing.strip()
    message = message.strip()
    if message:
        blocks.append(message)
    elif opening:
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

        if not message:
            heading = summary if len(items) == 1 else f"{index}. {summary}"
            lines = [heading]
            if why:
                lines.append(why)
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

    urls = [
        f"{index}. {str(event_map[str(item['item_id']).strip()].get('url') or '').strip()}"
        for index, item in enumerate(items, 1)
        if str(event_map[str(item['item_id']).strip()].get("url") or "").strip()
    ]
    if urls:
        blocks.append(
            f"来源：{urls[0].removeprefix('1. ')}"
            if len(urls) == 1
            else "来源：\n" + "\n".join(urls)
        )
    if closing and not message:
        blocks.append(closing)
    return RenderedShare(
        message="\n\n".join(blocks),
        evidence=evidence,
        display_event_map=display_event_map,
        source_refs=source_refs,
    )
