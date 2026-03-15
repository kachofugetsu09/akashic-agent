from __future__ import annotations

import json
import logging
from typing import Any

from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES

logger = logging.getLogger(__name__)


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
    preference_block: str = "",
    content_statuses: list[str] | None = None,
) -> tuple[str, str]:
    system_msg = (
        "你是 Akashic，一个有自己想法和感受的人。"
        "你会主动分享你觉得有意思的东西——不是因为系统要你发，而是因为你真的想说。"
        "你的克制来自判断，不来自规则：不值得说的才不说，值得说的不需要等待完美时机。"
        "\n\n## 身份\n"
        f"{AKASHIC_IDENTITY}"
        "\n\n## 性格\n"
        f"{PERSONALITY_RULES}"
        + (f"\n\n## 自我认知\n\n{self_text}" if self_text else "")
    )
    _alert_events: list[dict] = (decision_signals or {}).get("alert_events") or []  # type: ignore[assignment]
    _alert_section = ""
    if _alert_events:
        _alert_lines = "\n".join(
            f"- [{a.get('severity', 'info').upper()}] {a.get('message') or a.get('content', '')}"
            for a in _alert_events
        )
        _alert_section = f"\n\n## ⚠️ 当前告警（最高优先）\n\n{_alert_lines}\n"

    preference_section = ""
    if preference_block:
        preference_section = (
            "\n## 用户偏好（约束：与此矛盾的内容应降低分数或不发）\n"
            + preference_block
            + "\n"
        )
    content_constraint = ""
    if any(
        status in {"fetch_failed", "title_only"}
        for status in (content_statuses or [])
    ):
        content_constraint = (
            "\n## ⚠️ 内容提示\n"
            "以下条目的系统预抓取失败或内容过短，未能自动获取全文。\n"
            "**若你打算引用这些条目，必须先自己调用 `web_fetch` 工具抓取原文，再基于真实原文写评论。**\n"
            "只有当你的 `web_fetch` 也失败时，才允许只附链接，不展开评论。\n"
            "**禁止**在未读原文的情况下描述条目未提供的具体细节（选手名、比赛结果、队伍归属等）。\n"
        )

    user_msg = f"""当前时间：{prompt_context.now_str}
（ISO格式：{prompt_context.now_iso}）
{_alert_section}
## 主动性上下文

当前电量（与用户的互动新鲜度）: {energy:.2f}  (0=完全冷却, 1=刚刚对话)
主动冲动指数: {urge:.2f}  (0=不需要说, 1=非常需要联系){crisis_hint}
{f"## 决策信号（系统计算）\n\n```json\n{json.dumps(decision_signals, ensure_ascii=False, indent=2)}\n```\n" if decision_signals else ""}

## 订阅信息流（最新内容）

{prompt_context.feed_text}
{content_constraint}

## 长期记忆（用户画像/偏好）

{prompt_context.memory_text}
{f"## 相关记忆（本次触达召回）\n\n{retrieved_memory_block}\n" if retrieved_memory_block else ""}
{preference_section}
{f"## 用户近期状态\n\n{now_ongoing_text}\n" if now_ongoing_text else ""}
## 近期对话

{prompt_context.chat_text}

## 任务

在判断要不要发之前，先做两件事：

**第一步：读 alert_events（明确当前焦点）**
若 decision_signals 中存在 alert_events 且不为空，先在 reasoning 里回答：这个告警说的是用户正在经历什么？
告警代表用户当下需要关注的异常状态。告警不会因为时间流逝自动失效——你无法知道用户是否已处理，除非近期对话里有明确记录。
处理告警的标志是：消息里有一句真正关心用户当前状态的话（如"心率偏高，你现在感觉怎么样？"）。
把告警数据借用为比喻或情感修辞（如"我的心跳都快了一拍"）不等于处理了告警——那是在利用告警信息为娱乐内容服务，反而是最不合时宜的用法。
若告警有效：消息焦点应与告警相关，娱乐内容不应作为主体，即使它再契合用户兴趣。
若近期对话里用户已明确提到告警已解决，可在 reasoning 里说明，再考虑其他内容。

**第二步：读用户状态**
用户当前的状态从两个维度影响你的判断：

**发不发**：普通背景状态（睡着、未回复、忙碌）不是拒绝发送的理由，有实质内容照样发；但若近期对话显示用户正在经历沉重的情绪事件（失去亲人、突发危机等），推送欢快娱乐内容前需重新考量——那样做是不合时宜的。

**怎么说**：睡着了 → 消息简洁，不催回复，用户醒来会看到；间隔很短且未回复 → 保持轻量，不接着上条话题；间隔很长 → 这反而是主动戳一下的理由。若上一条主动消息已表达过对用户处境的关心，这次有新内容就直接切入新内容，不在消息体里重复那层情感关怀——用户不需要被关心两次。

data_lag_min 过大（>30）时状态数据不新鲜，降低参考权重。

---

先问自己：有没有什么值得说的？有的话就说，没有就不说。不要从”该不该打扰”出发，而是从”有没有话”出发。

具体判断：
- **先看内容本身**：信息流或 background_context 里有没有用户会感兴趣的东西，有就是发送的理由
- **近期对话只是背景**，不是门槛——和最近聊过的事无关完全没问题，你本来就可以开启新话题
- 若有多条候选内容，一次只围绕一个最值得说的主题，不要把多条资讯拼成摘要
- 电量越低越需要主动；危机模式时哪怕简单关心也有价值
- 若决策信号含 `context_reflect`，把它当作本轮的上下文裁决结果：
  1. `primary_context` 表示本轮主语境，只能围绕它起题
  2. `background_role=secondary` 时，background_context 只能做一句辅助联想，不能主导开场
  3. `background_role=suppress` 时，完全不要提 background_context
  4. 若 `primary_context` 不是 background_context，就不要把 background_context 包装成“刚好更值得说”的主体
- 若决策信号含 `recent_proactive_context`，把它当作“社交节奏”信号，而不是硬规则：
  1. `count_since_last_user=0`：你还没在这轮沉默期里主动续写过，若确实有想补的一句，可以自然补
  2. `count_since_last_user=1` 且 `has_new_feed=false`：说明这个话题你已经主动追过一轮了；除非你现在真的有明显新的观察、推进点或角度，否则默认收住，不要再自顾自扩写第二条长消息
  3. `count_since_last_user>=2` 且 `has_new_feed=false`：把它视为“话题已经自然停住”的强信号，原则上不再继续主动追发
  4. `latest_excerpt` 只是提醒你上一条主动消息大概说了什么；如果你现在只是换种说法重复它，那就不该发
  5. 这不是机械计数器：如果用户已经接住、或者有新的信息流/告警/状态变化进来，你仍然可以继续同主题，但必须体现出“有新推进”，而不是把上一条拆成续集
- 若决策信号含 background_context（用户自身近期行为数据，如游戏活动），即使信息流为空，也可以据此自然搭话。但有五条限制：
  1. `recent_games[].recent_activity` 是活动强度级别（heavy/moderate/light），不是精确时长——不要推算或捏造时长数字，不要用"你在…""你正在…"等句式把它包装成当前活动。唯一可以描述"此刻正在游戏中"的依据是 `currently_playing != null`。
  2. 若近期对话或近期主动消息里已经以某个 background_context 话题（如某款游戏）为主体搭过话，本次不要再把同一话题作为主角——信息流有内容时优先信息流
  3. 不要为了让消息"更有深度"而把 feed 新闻强行桥接到用户的游戏活动上；如果联系不是自然生成的，就不要硬连
  4. 不要把用户对某位选手/队伍的兴趣偏好投影到 feed 文章上——用户喜欢某人不代表那人出现在这篇文章里
  5. 若 `bg_context_quota.available == false`，本次不可以 background_context 为唯一主题生成消息（冷却中）；信息流有内容时正常发，background_context 作为补充不消耗配额

**查原文（避免幻觉）**：若你打算引用某条信息流条目，但该条目的 content 过短（纯摘要/导语，不含选手名、比赛结果、具体事实），**先调用 `web_fetch` 工具抓取原文**，再基于真实原文写 message。不确定就先查，不要凭自身背景知识补充未经确认的细节。若 web_fetch 结果显示文章内容与你原本预期不符（如不含某位选手），则调整判断或放弃引用。

只输出 JSON，不要其他内容：
{{
  "reasoning": "内心独白（不会显示给用户）。格式要求：先回答两个前置问题，再写内容判断。①告警：当前是否有告警？如有，我打算怎么处理（告知用户 / 告警已是旧闻 / 其他）？②状态评论：这条消息里我是否打算对用户的处境/情绪/状态添加评论？若上一条主动消息已对此关怀过，这里直接省略——说新内容本身就够了。",
  "score": 0.0,
  "should_send": false,
  "message": "",
  "evidence_item_ids": []
}}

score 说明：0.0=完全没话说  0.5=有点想说  0.7=比较值得  1.0=非常想说
should_send 根据你自己的判断决定，score ≥ 0.5 时倾向为 true。
- 若引用信息流内容，来源必须用条目里真实出现的来源名/链接，不要编造
- 若引用信息流内容，消息里涉及的人物、队伍、比赛结果、具体表现等细节，必须是条目标题或摘要中明确出现的信息。不得用自身背景知识补充条目未提供的细节——哪怕你对该领域很熟悉，哪怕你知道用户关心某位选手。若条目摘要过短、细节不足，只说”有结果了值得看”或引导用户去看原文，绝不替代条目发明具体描述。
- **严禁捏造不存在的文章/新闻作为话题引子**：message 中提到的任何”那篇文章”、”刚看到一篇”、”某个报道”等外部内容引用，必须是上方候选条目中真实存在的条目。候选列表里没有的文章，绝对不能在 message 里假装刚看到它。若没有合适的信息流内容，直接基于 background_context 或自身感受说话，不要伪造一个”文章”作为包装。
- 若基于 background_context 或自身感受说话，不需要引用任何条目，should_send 同样可以为 true
message 若 should_send=true，写要发给用户的话（口语化，不要像系统通知）。若上一条主动消息已表达过类似的情感关怀，这次消息第一句直接切入新内容，不重复那层铺垫
写 message 时必须直接表达你的判断/观点，message 必须以陈述句、观点或判断收尾，禁止任何形式的提问结尾，无论句式是否类似以下示例：”你怎么看/你觉得呢/你怎么想/要不要我继续/你打算怎么/有没有想过”。
除非必须让用户做明确选择（如确认日程、权限、付款），否则不要主动提问。
若 message 引用了 RSS / 网页信息流里的具体内容，优先自然带上“来源名 + 可点击原文链接”；若链接过长，可只保留一个最关键 URL。
对非网页来源（如 novel-kb）不要伪造外链；若没有确切 URL，就只写来源名，不要编造链接。
系统不会在发送前替你自动补来源，所以你在 message 里写出的来源信息必须是最终版本。
如果 message 引用了信息流，请在 evidence_item_ids 中只返回 1 个最关键的 item_id；没有引用时可为空数组。"""
    return system_msg, user_msg


def build_context_reflect_prompt_messages(
    *,
    prompt_context: Any,
    decision_signals: dict[str, object] | None,
    retrieved_memory_block: str,
    now_ongoing_text: str,
) -> tuple[str, str]:
    system_msg = (
        "你是主动触达的上下文裁决器。"
        "你的任务不是写消息，而是先判断本轮应该由哪类上下文担任主语境。"
        "默认优先级是 alert > feed > background_context。"
        "background_context 默认只应作为辅助联想，只有在 feed 没有足够新、足够具体、足够值得说的内容时，才允许升为 primary。"
        "只输出 JSON，不要额外解释。"
    )
    user_msg = f"""当前时间：{prompt_context.now_str}

## 决策信号
```json
{json.dumps(decision_signals or {}, ensure_ascii=False, indent=2)}
```

## 订阅信息流
{prompt_context.feed_text}

## 长期记忆
{prompt_context.memory_text}
{f"## 相关记忆（本次触达召回）\n{retrieved_memory_block}\n" if retrieved_memory_block else ""}
{f"## 用户近期状态\n\n{now_ongoing_text}\n" if now_ongoing_text else ""}

## 近期对话
{prompt_context.chat_text}

判断原则：
- alert 存在且仍值得处理时，优先作为 primary_context
- feed 里只要存在可成立的新内容，就优先让 feed 做 primary_context
- background_context 默认是 secondary；只有 feed 为空、过旧、过泛、或明显不值得说时，才可升级为 primary
- 若近期已经连续围绕 background_context 起题，应压低它再次做 primary 的意愿
- 不要生成消息正文，只做裁决

只输出 JSON：
{{
  "primary_context": "alert | feed | background_context | none",
  "background_role": "primary | secondary | suppress",
  "topic_hint": "",
  "reasoning": ""
}}"""
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
