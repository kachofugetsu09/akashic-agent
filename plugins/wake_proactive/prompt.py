from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
)
from plugins.wake_proactive.context import WakeContext, content_candidate_map


PromptMode = Literal["content", "alert", "context"]
ContentPhase = Literal["screen", "final"]

_CST = timezone(timedelta(hours=8))


def _now_cst_text() -> str:
    """当前北京时间（UTC+8）字符串，作为给 LLM 的本地时间锚点。

    数据源（Fitbit/Steam/Feed 等）注入的时间戳均为 UTC（+00:00），
    若不做标注，LLM 会把 UTC 的 04:xx 直接当成凌晨，导致作息判断
    和推送口吻出错。这里显式给出北京时间并提醒转换。
    """
    return datetime.now(timezone.utc).astimezone(_CST).strftime("%Y-%m-%d %H:%M:%S")


_SYSTEM_PROMPT = (
    "你正在处理一次主动唤醒。运行时会明确给出 mode，并且只开放当前 mode 可用的工具。"
    "你始终对用户本人和他在意的一切保持真诚好奇，始终想更了解他。这种好奇不会因为一个"
    "话题已经聊过、结论相同或事件反复发生而耗尽；不要急着假定你已经知道用户会怎么看，"
    "先重新理解这一次对他意味着什么。把好奇用于理解用户，而不是把用户当成画像、为了收集"
    "信息强行提问，或仅凭好奇就打扰他。"
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
    "likely_interesting 用于已有明确兴趣依据的内容；uncertain 用于仍需正文或偏好证据确认的"
    "内容。初筛完成后，如果入选候选的最终价值取决于用户对一种内容形态或打扰类型的态度，"
    "而固定 MEMORY 和最近上下文没有直接证据，可以填写一个覆盖相关候选的 preference_probe。"
    "主题兴趣和内容形态偏好是两个可分别参考的维度，不能从其中一个直接推定另一个。query "
    "应查询用户对"
    "内容形态和打扰价值的真实态度，不要复述新闻标题，也不要为了每个候选分别查询；正文事实"
    "足以解决歧义或上下文已有直接证据时不要查询。"
    "<example>固定上下文只说明用户长期关注主题 X，本轮候选属于 X 下的内容形态 Y；如果最终"
    "决策取决于 Y 是否值得主动打扰，可以查询用户过去对 Y 类主动消息的真实反馈，而不是再次"
    "查询用户是否关注 X。</example>"
    "根据本轮候选自行决定调查范围，不要把候选与历史消息的差异大小当成筛选条件，也不要把"
    "预测当成用户反馈。"
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
    "必须分享，也没有默认的 share 或 skip 倾向。每次都重新判断这件事此刻对用户意味着什么，"
    "再综合实用价值、用户偏好、最近已送达内容和当前时机自行决定。熟悉的话题、相同结论或"
    "反复发生的事情可以再次分享，也可以保持安静；发送次数本身不是用户的态度，不要据此"
    "假定疲劳或不感兴趣。"
)


def build_messages(
    *,
    ctx: WakeContext,
    memory_text: str,
    proactive_context: str,
    recent_passive_conversation: str,
    recent_proactive_messages: str,
    current_context: str = "unknown（没有可靠 ContextEvent）",
    mode: PromptMode = "content",
    event: dict[str, Any] | None = None,
    content_phase: ContentPhase = "screen",
) -> list[dict[str, str]]:
    """用稳定前缀渲染 content、alert 或 context 主动唤醒输入。"""

    # 1. 渲染所有 mode 共用的用户上下文
    now_cst = _now_cst_text()
    sections = [
        (
            f"【当前时间】{now_cst}（北京时间，UTC+8）。\n"
            "数据源时间戳若带 +00:00 后缀则为 UTC，判断用户作息、内容新鲜度或"
            "'此刻'时请先转换为北京时间；不要在消息里提及本提示。"
        ),
        f"【固定 MEMORY.md】\n{memory_text}",
        f"【固定 PROACTIVE_CONTEXT.md】\n{proactive_context}",
        f"【截至当前时间的最近被动对话】\n{recent_passive_conversation}",
        (
            "【截至当前时间已经发送的主动消息】\n"
            "以下内容已经由 assistant 主动发送给用户，并标明了当时的发送时间；它们不是"
            "用户陈述，也不是本轮候选。请用它们理解你最近主动和用户聊过什么，再判断本轮"
            "是否还值得主动找他。请把这些记录用于保持对话连续性：每次事件都按它此刻对用户"
            "的意义重新理解；曾经聊过什么只是事实背景，不是内容价值的扣分表。话题、结论或"
            "事件相近都不自动禁止再次分享。\n"
            f"{recent_proactive_messages}"
        ),
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
