from __future__ import annotations

from typing import Any

from plugins.wake_proactive.context import WakeContext


def build_messages(
    *,
    ctx: WakeContext,
    memory_text: str,
    proactive_context: str,
    recent_session: str,
) -> list[dict[str, str]]:
    lines: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in ctx.content_events:
        source_id = str(
            event.get("_reservoir_original_source_id")
            or event.get("source_id")
            or event.get("source")
            or "unknown"
        )
        grouped.setdefault(source_id, []).append(event)
    for source_id, events in grouped.items():
        source_name = str(
            events[0].get("source_name")
            or events[0].get("source")
            or source_id
        )
        lines.append(f"来源：{source_name}")
        for event in sorted(
            events,
            key=lambda item: str(
                item.get("published_at") or item.get("first_seen_at") or ""
            ),
            reverse=True,
        ):
            lines.append(
                " | ".join(
                    (
                        f"id={event['id']}",
                        f"published_at={event.get('published_at') or event.get('first_seen_at') or ''}",
                        f"title={event.get('title') or ''}",
                        f"source_name={event.get('source_name') or event.get('source') or ''}",
                    )
                )
            )

    system = (
        "这是一次由连续 hazard 激活后，拟递给内容判断模型的只读观察输入。"
        "候选已经按来源分组，每个来源内部严格按 published_at 倒序；"
        "预处理和唤醒分数只决定激活与候选顺序，不会提供给模型。"
        "当前阶段不调用模型、不使用工具、不抓正文，也不发送消息；"
        "这里只记录未来模型会看到的 MEMORY、PROACTIVE_CONTEXT、截至当时的最近对话和标题页。"
    )
    user = (
        f"【固定 MEMORY.md】\n{memory_text}\n\n"
        f"【固定 PROACTIVE_CONTEXT.md】\n{proactive_context}\n\n"
        f"【截至当前时间的最近对话】\n{recent_session}\n\n"
        f"【本次标题页：{len(ctx.content_events)} 条，窗口内未展示 {getattr(ctx, 'content_backlog_count', 0)} 条】\n"
        + "\n".join(lines)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
