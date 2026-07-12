from __future__ import annotations

from typing import Any

from plugins.wake_proactive.context import WakeContext


def build_messages(
    *,
    ctx: WakeContext,
    memory_text: str,
    proactive_context: str,
    recent_session: str,
    current_context: str = "unknown（没有可靠 ContextEvent）",
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
        "你正在处理一次主动内容窗口。候选已经按来源分组，每个来源内部严格按 "
        "published_at 倒序；预处理和唤醒分数不会提供给你。先结合 MEMORY、"
        "PROACTIVE_CONTEXT 和最近对话快速阅读全部标题，再调用一次 scratchpad，"
        "只记录最多八条确实值得查正文或需要确认用户兴趣的候选。"
        "likely_interesting 用于已有明确兴趣依据的内容；uncertain 用于需要 RecallMemory "
        "确认的内容。宁可少选，不要为了覆盖资讯而选择。不要把预测当成用户反馈，"
        "不要在输出中提及记忆、画像、分数或筛选流程。长期记忆不只影响是否分享，"
        "也可以让表达带有自然的理解和共情：可以顺着用户稳定的喜好、期待和经历说话，"
        "但不要列档案、逐句复述旧对话，或用‘你之前说过’来证明自己记得。"
        "涉及焦虑、健康、财务或私密关系时，只在与当前事实直接相关且能带来帮助时轻柔提及，"
        "不得放大情绪、替用户定义感受或把脆弱经历当作吸引注意的钩子。"
        "关于用户此刻是否睡眠、忙碌、离线或在游戏，只允许依据当前 ContextEvent；"
        "ContextEvent 为 unknown 时不得根据时间、历史习惯或语气猜测当前状态。"
    )
    user = (
        f"【固定 MEMORY.md】\n{memory_text}\n\n"
        f"【固定 PROACTIVE_CONTEXT.md】\n{proactive_context}\n\n"
        f"【截至当前时间的最近对话】\n{recent_session}\n\n"
        f"【当前 ContextEvent】\n{current_context}\n\n"
        f"【本次标题页：{len(ctx.content_events)} 条，窗口内未展示 {getattr(ctx, 'content_backlog_count', 0)} 条】\n"
        + "\n".join(lines)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
