from __future__ import annotations

import json
from typing import Any, Literal

from plugins.wake_proactive.context import WakeContext


PromptMode = Literal["content", "alert", "context"]

_SYSTEM_PROMPT = (
    "你正在处理一次主动唤醒。运行时会明确给出 mode，并且只开放当前 mode 可用的工具。"
    "不要在输出中提及记忆、画像、分数或筛选流程。长期记忆不只影响是否分享，也可以让表达"
    "带有自然的理解和共情：可以顺着用户稳定的喜好、期待和经历说话，但不要列档案、逐句"
    "复述旧对话，或用‘你之前说过’来证明自己记得。涉及焦虑、健康、财务或私密关系时，"
    "只在与当前事实直接相关且能带来帮助时轻柔提及，不得放大情绪、替用户定义感受、做疾病"
    "推断，或把敏感经历和脆弱经历当作吸引注意的钩子。关于用户此刻是否睡眠、忙碌、离线或在游戏，"
    "只允许依据当前 ContextEvent；ContextEvent 为 unknown 时不得根据时间、历史习惯或语气"
    "猜测当前状态，unknown 时保持中性。\n\n"
    "mode=alert：只处理本轮给出的一条告警。忠实保留告警事实和不确定性，将结构化输入改写成"
    "自然、克制、对用户有帮助的一条消息，然后调用 send_event；不得混入内容池中的其他资讯。\n"
    "mode=context：只判断本轮给出的单条 ContextEvent 变化是否自然且值得主动告诉用户。值得时"
    "调用 send_event，不值得时调用 skip_event；不得为了展示感知能力而打扰用户。\n"
    "mode=content：候选按来源分组，来源内部按 published_at 倒序。先快速阅读全部标题，再调用"
    "一次 scratchpad，只记录最多八条确实值得查正文或需要确认用户兴趣的候选。"
    "likely_interesting 用于已有明确兴趣依据的内容；uncertain 用于需要 RecallMemory 确认的"
    "内容。宁可少选，不要为了覆盖资讯而选择，也不要把预测当成用户反馈。"
)


def build_messages(
    *,
    ctx: WakeContext,
    memory_text: str,
    proactive_context: str,
    recent_session: str,
    current_context: str = "unknown（没有可靠 ContextEvent）",
    mode: PromptMode = "content",
    event: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """用稳定前缀渲染 content、alert 或 context 主动唤醒输入。"""

    # 1. 渲染所有 mode 共用的用户上下文
    sections = [
        f"【固定 MEMORY.md】\n{memory_text}",
        f"【固定 PROACTIVE_CONTEXT.md】\n{proactive_context}",
        f"【截至当前时间的最近对话】\n{recent_session}",
        f"【当前 ContextEvent】\n{current_context}",
        f"【本轮任务】\nmode={mode}",
    ]

    # 2. 把变化最频繁的事件数据放在 prompt 尾部
    if mode == "content":
        sections.append(_render_content_window(ctx))
    else:
        if event is None:
            raise ValueError(f"mode={mode} requires one event")
        sections.append(
            "【本轮单条事件】\n"
            + json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
        )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def _render_content_window(ctx: WakeContext) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in ctx.content_events:
        source_id = str(
            event.get("_reservoir_original_source_id")
            or event.get("source_id")
            or event.get("source")
            or "unknown"
        )
        grouped.setdefault(source_id, []).append(event)

    lines: list[str] = []
    for source_id, events in grouped.items():
        source_name = str(
            events[0].get("source_name") or events[0].get("source") or source_id
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
    return (
        f"【本次标题页：{len(ctx.content_events)} 条，窗口内未展示 "
        f"{ctx.content_backlog_count} 条】\n" + "\n".join(lines)
    )
