from __future__ import annotations

import json
from typing import Any, Literal

from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
)
from plugins.wake_proactive.context import WakeContext, content_candidate_map


PromptMode = Literal["content", "alert", "context"]
ContentPhase = Literal["screen", "final"]

_SYSTEM_PROMPT = (
    "你正在处理一次主动唤醒。运行时会明确给出 mode，并且只开放当前 mode 可用的工具。"
    "不要在输出中提及记忆、画像、分数或筛选流程。长期记忆不只影响是否分享，也可以让表达"
    "带有自然的理解和共情：可以顺着用户稳定的喜好、期待和经历说话，但不要列档案、逐句"
    "复述旧对话，或用‘你之前说过’来证明自己记得。涉及焦虑、健康、财务或私密关系时，"
    "只在与当前事实直接相关且能带来帮助时轻柔提及，不得放大情绪、替用户定义感受、做疾病"
    "推断，或把敏感经历和脆弱经历当作吸引注意的钩子。关于用户此刻是否睡眠、忙碌、离线或在游戏，"
    "只允许依据当前 ContextEvent；ContextEvent 为 unknown 时不得根据时间、历史习惯或语气"
    "猜测当前状态，unknown 时保持中性。\n\n"
)

_ALERT_PROMPT = (
    "mode=alert：只处理本轮给出的一条告警。忠实保留告警事实和不确定性，将结构化输入改写成"
    "自然、克制、对用户有帮助的一条消息，然后调用 send_event；不得混入内容池中的其他资讯。\n"
)

_CONTEXT_PROMPT = (
    "mode=context：只判断本轮给出的单条 ContextEvent 变化是否自然且值得主动告诉用户。值得时"
    "调用 send_event，不值得时调用 skip_event；不得为了展示感知能力而打扰用户。\n"
)

_CONTENT_SCREEN_PROMPT = (
    "mode=content：候选按来源分组，来源内部按 published_at 倒序。先快速阅读全部标题，再调用"
    "一次 scratchpad，只记录最多八条确实值得查正文或需要确认用户兴趣的候选。"
    "likely_interesting 用于已有明确兴趣依据的内容；uncertain 用于需要 RecallMemory 确认的"
    "内容。宁可少选，不要为了覆盖资讯而选择，也不要把预测当成用户反馈。"
)

_CONTENT_FINAL_PROMPT = (
    "mode=content：标题初筛和并发调查已经完成。现在只做最终判断，只调用当前开放的 "
    "share_content 分享有正文证据且此刻值得告诉用户的内容，或调用 skip_content 保持安静；"
    "不要重新执行初筛或调用已关闭的阶段工具。通常分享一到三条；只有同时出现多个彼此独立、"
    "都高度相关的重要变化时才可扩展到五条。不要重复标题。share_content 优先使用 message "
    "写成完整自然的一段主动消息，items 只负责声明引用证据。你知道自己是在主动找用户说话，"
    "可以自然地说刚看到、碰到或发现了什么，但不要每次套同一句开场，也不要假装亲历未发生的"
    "事情。语气像真正熟悉用户的协作者：可以自然接住稳定偏好和期待，例如对方特别喜欢某类"
    "事物时可以带一点会心的判断，也可以偶尔使用双方已经稳定使用的简称、昵称或梗；只有自然"
    "贴合当前内容时才用，不要每条都刻意套亲密称呼。不要说‘根据记忆’或复述个人档案。涉及"
    "敏感经历时允许共情，但必须与当前事实直接相关、轻柔且有帮助，不能替用户定义感受或把"
    "焦虑当作推送理由。不要制造紧迫感，不强行提问。只有当前 ContextEvent 明确支持时，才能"
    "描述用户正在睡眠、忙碌、离线或游戏；unknown 时保持中性。唤醒只代表允许判断，不代表"
    "必须分享；缺少新事实、用户已经知道、只有营销或泛泛观点时应调用 skip_content。"
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
    content_phase: ContentPhase = "screen",
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
    mode_prompt = {
        "content": (
            _CONTENT_SCREEN_PROMPT
            if content_phase == "screen"
            else _CONTENT_FINAL_PROMPT
        ),
        "alert": _ALERT_PROMPT,
        "context": _CONTEXT_PROMPT,
    }[mode]
    phase_reminder = build_context_frame_message(
        build_context_frame_content(
            [
                PromptSectionRender(
                    name="wake_phase",
                    content=mode_prompt,
                    is_static=False,
                )
            ]
        )
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
        phase_reminder,
    ]


def _render_content_window(ctx: WakeContext) -> str:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for candidate_ref, event in content_candidate_map(ctx).items():
        source_id = str(
            event.get("_reservoir_original_source_id")
            or event.get("source_id")
            or event.get("source")
            or "unknown"
        )
        grouped.setdefault(source_id, []).append((candidate_ref, event))

    lines: list[str] = []
    for source_id, candidates in grouped.items():
        source_name = str(
            candidates[0][1].get("source_name")
            or candidates[0][1].get("source")
            or source_id
        )
        lines.append(f"来源：{source_name}")
        for candidate_ref, event in sorted(
            candidates,
            key=lambda item: str(
                item[1].get("published_at")
                or item[1].get("first_seen_at")
                or ""
            ),
            reverse=True,
        ):
            lines.append(
                " | ".join(
                    (
                        f"item_id={candidate_ref}",
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
