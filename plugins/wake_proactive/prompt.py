from __future__ import annotations

from itertools import groupby

from plugins.wake_proactive.context import WakeContext


def build_messages(
    *,
    ctx: WakeContext,
    memory_text: str,
    proactive_context: str,
    recent_session: str,
) -> list[dict[str, str]]:
    lines: list[str] = []
    for source_id, events in groupby(
        ctx.content_events,
        key=lambda event: str(event.get("_reservoir_source_id") or event.get("source") or "unknown"),
    ):
        lines.append(f"来源：{source_id}")
        for event in events:
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
        "你正在处理一次由概率 hazard 触发的主动内容窗口。"
        "候选已经按来源分组，每个来源内部严格按 published_at 倒序；"
        "预处理分数只负责决定何时唤醒，不参与这里的排序，也不会提供给你。\n"
        "必须严格按三步工具流程执行：\n"
        "1. scratchpad 一次覆盖全部标题，记录初步兴趣与需要查正文/兴趣记忆的项目。\n"
        "   likely_interesting 必须调查 content 或 both；uncertain 必须调查 recall 或 both；"
        "只有 not_interesting 可以使用 none。\n"
        "2. investigate_candidates 一次并发执行计划，不得逐条调用抓取或召回。\n"
        "3. share_content 只分享有事实证据的内容，或 skip_content 保持安静。\n"
        "scratchpad 的预测不是用户反馈；抓取失败或 recall 无命中也不是负反馈。\n"
        "最终最多产生一条消息。分享时解释发生了什么、为什么现在值得关注，"
        "不要使用生硬的资讯汇总口吻，也不要强行追问。"
    )
    user = (
        f"【固定 MEMORY.md】\n{memory_text}\n\n"
        f"【固定 PROACTIVE_CONTEXT.md】\n{proactive_context}\n\n"
        f"【截至当前时间的最近对话】\n{recent_session}\n\n"
        "【本次全部未读标题】\n"
        + "\n".join(lines)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
