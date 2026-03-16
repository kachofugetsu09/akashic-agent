from __future__ import annotations

import json
from typing import Any


def build_feature_scoring_prompt_messages(
    *,
    prompt_context: Any,
    decision_signals: dict[str, object],
    retrieved_memory_block: str,
) -> tuple[str, str]:
    system_msg = (
        "你是主动触达特征评估器。只输出固定JSON字段。"
        "每个分数字段必须是0到1的小数；同时给每个字段一句简短理由。"
        "若决策信号含 alert_events，告警类触达优先级高于普通资讯触达。"
        "message_readiness_reason 应基于用户整体状态（时间、活跃度、对话节奏等）综合判断，无需引用具体健康数值。"
        "若决策信号不含健康来源告警，不得用健康状况作为触达理由。"
        "topic_continuity 代表与近期对话的连续性，它是加分项而不是硬门槛。"
        "如果订阅内容与用户长期兴趣明显匹配，即使与近期对话无关，也可以给出高 interest_match 和合理的 reconnect_value。"
        "不要给最终决策，不要输出额外文本。"
    )
    user_msg = f"""当前时间：{prompt_context.now_str}

## 决策信号（系统计算）
```json
{json.dumps(decision_signals, ensure_ascii=False, indent=2)}
```

## 订阅信息流
{prompt_context.feed_text}

## 长期记忆
{prompt_context.memory_text}
{f"## 相关记忆（本次触达召回）\n{retrieved_memory_block}\n" if retrieved_memory_block else ""}

## 近期对话
{prompt_context.chat_text}

补充原则：
- 不要求消息必须承接近期对话
- 若某条信息流与用户长期兴趣明显匹配，可把它视为一个自然的新切入点
- 只有当完全无关、打扰风险高、或内容不够新鲜时，才应压低分数

只输出 JSON，且仅包含以下键：
{{
  "topic_continuity": 0.0,
  "topic_continuity_reason": "",
  "interest_match": 0.0,
  "interest_match_reason": "",
  "content_novelty": 0.0,
  "content_novelty_reason": "",
  "reconnect_value": 0.0,
  "reconnect_value_reason": "",
  "disturb_risk": 0.0,
  "disturb_risk_reason": "",
  "message_readiness": 0.0,
  "message_readiness_reason": "",
  "confidence": 0.0,
  "confidence_reason": ""
}}
"""
    return system_msg, user_msg


def build_compose_prompt_messages(
    *,
    prompt_context: Any,
    preference_block: str = "",
    no_content_token: str = "<no_content/>",
) -> tuple[str, str]:
    system_msg = (
        "你是用户的主动助手。负责把今天的真实新内容提炼成一条值得发送的消息。\n"
        "【严格规则】\n"
        "1. 偏好记录仅用于判断哪条内容更值得推送，绝不能作为创作素材。\n"
        "   偏好 ≠ 事实。用户喜欢某个游戏，不代表该游戏有新动态。\n"
        "2. 消息中出现的每一个具体事实（人名/游戏名/功能/数字/时间）\n"
        "   必须能在「今天的新内容」中找到原文依据。\n"
        "3. 发现自己在补充「新内容」未提及的细节时，立即停止并输出"
        f" {no_content_token}。\n"
        "4. 消息末尾必须附上对应来源的原文链接（取自新内容中的「原文链接:」字段）。\n"
        "5. 输出纯文本，不要 JSON，不要提问收尾。"
    )
    user_msg = f"""当前时间：{prompt_context.now_str}

## 今天的新内容（唯一可用的事实来源）
{prompt_context.feed_text}

## 用户最近聊过的话题（仅供了解上下文）
{prompt_context.chat_text}
{f"## 用户偏好记录（仅用于选题，不得用于编造内容）\n{preference_block}\n" if preference_block else ""}
任务：
1. 从「今天的新内容」中选出最值得推送的一条，写一句简短说明（基于原文，不扩写）。
2. 在消息末尾另起一行附上该条内容的原文链接。
3. 若以上内容都不值得推送，或无法找到原文依据，输出 `{no_content_token}`。"""
    return system_msg, user_msg


def build_post_judge_prompt_messages(
    *,
    recent_summary: str,
    last_proactive: str,
    composed_message: str,
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

请基于“是否应该推动发送”来打分：分数越高代表越应该发送，分数越低代表应保守抑制发送。

输出 JSON：{{"information_gap": int, "relevance": int, "expected_impact": int}}"""
    return system_msg, user_msg
