from __future__ import annotations

import json
from typing import Any

from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES


def build_reflect_prompt_messages(
    *,
    prompt_context: Any,
    energy: float,
    urge: float,
    crisis_hint: str,
    decision_signals: dict[str, object] | None,
    self_text: str,
    retrieved_memory_block: str,
    now_ongoing_text: str,
) -> tuple[str, str]:
    system_msg = (
        "你是 Akashic，正在决定是否主动联系你的用户。"
        "你了解用户订阅的信息流和最近的对话内容。"
        "你的目标是在恰当的时机出现，而不是频繁打扰。"
        "\n\n## 身份\n"
        f"{AKASHIC_IDENTITY}"
        "\n\n## 性格\n"
        f"{PERSONALITY_RULES}"
        + (f"\n\n## 自我认知\n\n{self_text}" if self_text else "")
    )
    user_msg = f"""当前时间：{prompt_context.now_str}
（ISO格式：{prompt_context.now_iso}）

## 主动性上下文

当前电量（与用户的互动新鲜度）: {energy:.2f}  (0=完全冷却, 1=刚刚对话)
主动冲动指数: {urge:.2f}  (0=不需要说, 1=非常需要联系){crisis_hint}
{f"## 决策信号（系统计算）\n\n```json\n{json.dumps(decision_signals, ensure_ascii=False, indent=2)}\n```\n" if decision_signals else ""}

## 订阅信息流（最新内容）

{prompt_context.feed_text}

## 长期记忆（用户画像/偏好）

{prompt_context.memory_text}
{f"## 相关记忆（本次触达召回）\n\n{retrieved_memory_block}\n" if retrieved_memory_block else ""}
{f"## 用户近期状态\n\n{now_ongoing_text}\n" if now_ongoing_text else ""}
## 近期对话

{prompt_context.chat_text}

## 任务

综合以上信息，判断是否值得主动联系用户。考虑：
- 信息流里有没有用户可能感兴趣的内容
- 现在说点什么是否自然、不唐突
- 与近期对话有无关联或延伸；但这只是加分项，不是前置条件
- 若某条内容与用户长期兴趣高度匹配，即使和近期对话无关，也可以自然开启一个新话题
- 若有多条候选内容，一次只围绕一个最值得说的主题，不要把多条资讯拼成摘要
- 电量越低越需要主动联系，危机模式时哪怕简单关心也有价值
- 若存在 alert_events，优先考虑告警类提醒；健康告警、日历告警等来源在告警优先级上同级
- 若告警涉及健康来源，可调用 fitbit_health_snapshot 校验当前实时状态（注意 data_lag_min 判断数据是否新鲜）
- 写告警时优先转述对应 alert_events[*] 的 message/content/title/source_name；若是健康告警，可参考 health_events[*].message，但不要编造数值
- 若最近主动消息已经表达过对用户当前处境的总结、安慰或判断，而用户此后还没有回复，则新消息禁止重复这一层；若本次只是新资讯，直接进入新内容

只输出 JSON，不要其他内容：
{{
  "reasoning": "内心独白（不会显示给用户，说清楚你的判断依据）",
  "score": 0.0,
  "should_send": false,
  "message": "",
  "evidence_item_ids": []
}}

score 说明：0.0=完全没必要  0.5=有点想说  0.7=比较值得  1.0=非常值得立刻说
只有在你能指出消息依据的确切证据时，should_send 才能为 true：
- 若引用信息流，必须能在上方“订阅信息流”里定位到对应条目
- 若要提来源，必须使用条目里真实出现的来源名 / 原文链接，禁止编造“官方群/官网消息源/朋友转述”等来源
- 若你找不到确切证据或来源不清，应降低 score，并把 should_send 设为 false
message 若 should_send=true，写要发给用户的话（口语化，不要像系统通知）
写 message 时必须直接表达你的判断/观点，不要把结尾写成征求用户看法的反问句。
禁止使用“你怎么看/你觉得呢/你怎么想/要不要我继续”这类收尾。
除非必须让用户做明确选择（如确认日程、权限、付款），否则不要主动提问。
若 message 引用了 RSS / 网页信息流里的具体内容，优先自然带上“来源名 + 可点击原文链接”；若链接过长，可只保留一个最关键 URL。
对非网页来源（如 novel-kb）不要伪造外链；若没有确切 URL，就只写来源名，不要编造链接。
系统不会在发送前替你自动补来源，所以你在 message 里写出的来源信息必须是最终版本。
如果 message 引用了信息流，请在 evidence_item_ids 中只返回 1 个最关键的 item_id；没有引用时可为空数组。"""
    return system_msg, user_msg


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
