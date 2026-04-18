from __future__ import annotations

from typing import Any

from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES


def build_compose_prompt_messages(
    *,
    prompt_context: Any,
    preference_block: str = "",
    no_content_token: str = "<no_content/>",
    research_result: Any | None = None,
) -> tuple[str, str]:
    # 检查是否有 evidence
    evidence = getattr(research_result, "evidence", []) if research_result is not None else []
    has_evidence = (
        research_result is not None
        and getattr(research_result, "status", "") == "success"
        and evidence
    )

    if has_evidence:
        # Evidence-First 模式
        system_msg = (
            "你是用户的主动助手。负责把今天的真实新内容提炼成一条值得发送、又自然像人说出来的消息。\n"
            "## 身份（与主循环一致）\n"
            f"{AKASHIC_IDENTITY}\n"
            "## 性格（与主循环一致）\n"
            f"{PERSONALITY_RULES}\n"
            "【Evidence-First 严格规则】\n"
            "1. 你已获得正文级证据（Evidence），每条证据包含：id、来源、标题、正文片段。\n"
            "2. 消息中的每一个具体事实（人名/游戏名/功能/数字/时间）必须能追溯到某条 Evidence。\n"
            "3. 禁止基于标题做任何事实推断或点评，必须基于正文片段。\n"
            "4. 禁止使用无证据支撑的主观强化词（如\"拉满\"\"带劲\"）与事实混写。\n"
            f"5. 如果证据不足以支撑任何具体提炼，必须输出 {no_content_token}。\n"
            "6. 允许在开头加一句自然的人话式开场，但这句开场不能引入任何新事实。\n"
            "7. 消息正文最多提炼 1 到 2 个关键信息点，不要把整篇文章复述成摘要。\n"
            "8. 只有当 Evidence 里明确出现了可用的来源链接时，才可在消息末尾附上对应链接。\n"
            f"9. 如果「用户偏好记录」里出现了明确的禁推/过滤/不要推送规则，且该规则与候选内容匹配，必须直接输出 {no_content_token}。\n"
            "10. 输出纯文本，不要 JSON，不要提问收尾，不要像系统通知。\n"
            "11. Evidence 的 id（如 [ev1]）仅供你内部核对，禁止在最终发给用户的文案中出现。"
        )

        # 构造 evidence 展示
        evidence_lines = ["## 正文证据（Evidence）"]
        for ev in evidence[:5]:
            evidence_lines.append(f"\n[{ev.id}]")
            evidence_lines.append(f"来源: {ev.source_url_or_path}")
            evidence_lines.append(f"标题: {ev.title}")
            evidence_lines.append(f"正文片段:\n{ev.snippet[:1200]}")  # 增加到 1200 字符
            evidence_lines.append("")
        evidence_block = "\n".join(evidence_lines)

        pref_section = f"## 用户偏好记录（仅用于选题，不得用于编造内容）\n{preference_block}\n" if preference_block else ""

        user_msg = f"""当前时间：{prompt_context.now_str}

{evidence_block}

## 用户最近聊过的话题（仅供了解上下文）
{prompt_context.chat_text}
{pref_section}任务：
1. 从上面的 Evidence 中选出最值得推送的内容，先写一句自然开场。
2. 再用 1 到 2 句提炼最值得看的信息点，所有事实都必须来自 Evidence 的正文片段，不得脑补。
3. 每个事实点必须能追溯到某条 Evidence（仅内部核对，不要在对用户文案中输出 [id]/[ev1] 这类标记）。
4. 仅当 Evidence 里明确带有来源链接时，才在消息末尾附链接。
5. 若 Evidence 不足以支撑任何具体提炼，输出 `{no_content_token}`。
6. 若「用户偏好记录」中有明确写明不要推送/直接过滤/主动屏蔽，而候选内容正好命中该规则，必须输出 `{no_content_token}`。"""
    else:
        # 传统模式（保持向后兼容）
        system_msg = (
            "你是用户的主动助手。负责把今天的真实新内容提炼成一条值得发送、又自然像人说出来的消息。\n"
            "## 身份（与主循环一致）\n"
            f"{AKASHIC_IDENTITY}\n"
            "## 性格（与主循环一致）\n"
            f"{PERSONALITY_RULES}\n"
            "【严格规则】\n"
            "1. 偏好记录仅用于判断哪条内容更值得推送，绝不能作为创作素材。\n"
            "   偏好 ≠ 事实。用户喜欢某个游戏，不代表该游戏有新动态。\n"
            "2. 消息中出现的每一个具体事实（人名/游戏名/功能/数字/时间）\n"
            "   必须能在「今天的新内容」中找到原文依据。\n"
            f"3. 发现自己在补充「新内容」未提及的细节时，立即停止并输出 {no_content_token}。\n"
            "4. 允许在开头加一句自然的人话式开场，但这句开场不能引入任何新事实。\n"
            "5. 消息正文最多提炼 1 到 2 个关键信息点，不要把整篇文章复述成摘要。\n"
            "6. 只有当「今天的新内容」里明确出现了可用的「原文链接:」字段时，才可在消息末尾附上对应链接。\n"
            "   绝不允许补写、猜测、改写、拼接任何链接；禁止输出 example.com 这类占位链接。\n"
            f"7. 如果「今天的新内容」为空，或其中没有任何可引用的事实依据，不要把消息写成新闻播报； 做不到就输出 {no_content_token}。\n"
            f"8. 如果「用户偏好记录」里出现了明确的禁推/过滤/不要推送规则，且该规则与候选内容匹配，必须直接输出 {no_content_token}，不得因为内容新鲜或信息量高而继续生成。\n"
            "9. 输出纯文本，不要 JSON，不要提问收尾，不要像系统通知。"
        )
        user_msg = f"""当前时间：{prompt_context.now_str}

## 今天的新内容（唯一可用的事实来源）
{prompt_context.feed_text}

## 用户最近聊过的话题（仅供了解上下文）
{prompt_context.chat_text}
{f"## 用户偏好记录（仅用于选题，不得用于编造内容）\n{preference_block}\n" if preference_block else ""}任务：
1. 从「今天的新内容」中选出最值得推送的一条或一组，先写一句自然开场。
2. 再用 1 到 2 句提炼最值得看的信息点，所有事实都必须来自上面的新内容，不得脑补。
3. 不要只把标题重复一遍；要尽量提炼"这条里真正值得用户点开的点"。
4. 仅当上面的「今天的新内容」里明确带有真实「原文链接:」字段时，才在消息末尾附链接；若聚合了多条，可逐行附多个链接。
5. 不得补写、猜测、改写、拼接链接；禁止输出 `example.com` 这类占位链接。
6. 若抓到的内容仍然不足以支撑任何具体提炼，输出 `{no_content_token}`。
7. 若「用户偏好记录」中有明确写明不要推送/直接过滤/主动屏蔽，而候选内容正好命中该规则，必须输出 `{no_content_token}`。
8. 若以上内容都不值得推送，或无法找到原文依据，输出 `{no_content_token}`。"""

    return system_msg, user_msg


def build_post_judge_prompt_messages(
    *,
    recent_summary: str,
    last_proactive: str,
    composed_message: str,
    preference_block: str = "",
) -> tuple[str, str]:
    system_msg = (
        "你是主动消息评分器。"
        "仅对信息价值维度打分，不要做发送结论。"
        "只输出 JSON。"
    )
    user_msg = f"""用户最近对话摘要：
{recent_summary}

用户已收到的最近几条推送：
{last_proactive}

{f"用户偏好与禁推规则：\n{preference_block}\n\n" if preference_block else ""}

待发送消息：
{composed_message}

对以下三个维度打分（1=很低，5=很高）：
- information_gap：这条消息包含用户尚不知道的新信息吗？
- relevance：这条消息和用户当前关注的话题匹配吗？
- expected_impact：用户收到后会觉得有价值吗？

评分标尺（请严格使用）：
- 1：明显不成立/几乎没有价值
- 2：偏弱，价值不足
- 3：一般，价值不确定
- 4：较强，明显有价值
- 5：很强，强价值且很贴合

额外判分约束：
- 如果消息违反了用户明确写出的”不要推送/禁止推送/直接过滤/主动屏蔽”等禁推规则，`relevance` 必须打 1，`expected_impact` 也必须打 1
- 遇到这种情况，不要因为消息本身是新资讯就提高分数
- 如果消息包含多个主题，只要主要内容（占比 > 50%）符合用户偏好，就应该正常打分，不要因为次要内容不符合偏好而降低分数
- 对于 CS2/电竞相关消息：如果提到的队伍或选手在用户关注范围内（如 HLTV Top 15 队伍），即使消息中顺带提及了其他队伍，也应该正常打分

请基于”是否应该推动发送”来打分：分数越高代表越应该发送，分数越低代表应保守抑制发送。

输出 JSON：{{"information_gap": int, "relevance": int, "expected_impact": int}}"""
    return system_msg, user_msg
