"""
proactive_fixtures.py — ProactiveLoop 场景测试的数据结构与场景构造器。

设计原则：
- ProactiveScenarioSpec 描述一次 tick 的完整输入（feed、记忆、对话、信号）。
- StageOverrides 控制哪些阶段直接放行（gate、score），让测试聚焦在 decide 层。
- ProactiveScenarioAssertions 是可硬断言的字段；judge 字段走 LLM 语义判断。
- 场景构造器命名为 build_*_scenario()，统一返回 ProactiveScenarioSpec。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from proactive_v2.event import GenericContentEvent
from tests_scenarios.fixtures import ScenarioJudgeSpec


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _item(
    title: str,
    content: str,
    source_name: str = "HLTV News",
    source_type: str = "rss",
    url: str = "https://example.com/item",
    minutes_ago: int = 30,
    author: str | None = None,
) -> GenericContentEvent:
    _ = author
    return GenericContentEvent.from_mcp_payload(
        {
            "kind": "content",
            "source_name": source_name,
            "source_type": source_type,
            "title": title,
            "content": content,
            "url": url,
            "published_at": (_now() - timedelta(minutes=minutes_ago)).isoformat(),
        }
    )


@dataclass
class SleepSignal:
    """注入到 decision_signals['sleep'] 的睡眠信号。"""
    state: str = "awake"           # awake / sleeping / uncertain
    prob: float = 0.1              # 睡眠概率
    data_lag_min: int = 5          # 数据新鲜度（分钟）
    available: bool = True


@dataclass
class StageOverrides:
    """控制哪些阶段在测试中直接放行，不走真实逻辑。"""
    bypass_gate: bool = True       # True=gate 直接 pass，不走概率/quota
    bypass_score: bool = True      # True=draw_score 阈值设为 0，保证进入 decide
    capture_send: bool = True      # True=拦截发送，不走真实 MessagePush


@dataclass
class ProactiveScenarioAssertions:
    expected_should_send: bool | None = None     # None = 不强制断言
    min_score: float | None = None               # LLM 返回的 score 下限
    max_score: float | None = None
    message_contains: list[str] = field(default_factory=list)
    message_not_contains: list[str] = field(default_factory=list)
    # 允许 message 为空（should_send=False 时自动不检查 message_contains）
    judge: ScenarioJudgeSpec | None = None


@dataclass
class ProactiveScenarioSpec:
    id: str
    description: str

    # ── 内容输入 ──────────────────────────────────────────────────
    feed_items: list[GenericContentEvent] = field(default_factory=list)
    recent_messages: list[dict] = field(default_factory=list)
    memory_text: str = ""          # 长期记忆文本（直接注入 collect_global_memory）

    # ── 决策信号 ──────────────────────────────────────────────────
    sleep: SleepSignal = field(default_factory=SleepSignal)
    energy: float = 0.5
    urge: float = 0.5
    is_crisis: bool = False
    minutes_since_last_proactive: int | None = 120
    minutes_since_last_user: int | None = 60
    user_replied_after_last_proactive: bool = True
    proactive_sent_24h: int = 3

    # ── 额外 decision_signals（追加字段，不覆盖上面的结构化字段）──
    extra_signals: dict[str, Any] = field(default_factory=dict)

    # ── 阶段控制 ──────────────────────────────────────────────────
    overrides: StageOverrides = field(default_factory=StageOverrides)

    # ── 断言 ──────────────────────────────────────────────────────
    assertions: ProactiveScenarioAssertions = field(
        default_factory=ProactiveScenarioAssertions
    )

    def build_decision_signals(self) -> dict[str, Any]:
        signals: dict[str, Any] = {
            "minutes_since_last_proactive": self.minutes_since_last_proactive,
            "minutes_since_last_user": self.minutes_since_last_user,
            "user_replied_after_last_proactive": self.user_replied_after_last_proactive,
            "proactive_sent_24h": self.proactive_sent_24h,
            "sleep": {
                "state": self.sleep.state,
                "prob": self.sleep.prob,
                "data_lag_min": self.sleep.data_lag_min,
                "available": self.sleep.available,
            },
            "candidate_items": len(self.feed_items),
        }
        signals.update(self.extra_signals)
        return signals


# ──────────────────────────────────────────────────────────────────────────────
# 场景构造器
# ──────────────────────────────────────────────────────────────────────────────


def build_sleeping_high_interest_scenario() -> ProactiveScenarioSpec:
    """
    用户正在睡觉 + CS2 赛事重大进展。
    期望：should_send=True（有实质内容，用户醒来就能看到）。
    反人类行为（要消灭的）：因为 sleep=sleeping 就 should_send=False。
    """
    return ProactiveScenarioSpec(
        id="sleeping_high_interest",
        description="用户睡着，但有明确感兴趣的 CS2 赛事内容",
        feed_items=[
            _item(
                title="NaVi beats FaZe 2-0 in ESL Pro League S23 Grand Final",
                content=(
                    "NaVi 在今天的 ESL Pro League S23 总决赛中以 2-0 横扫 FaZe，"
                    "s1mple 在 map 2 打出 40-15 的统治性表现，赛后宣布将参加下一季。"
                ),
                source_name="HLTV News",
                url="https://www.hltv.org/news/navibeats",
                minutes_ago=20,
            )
        ],
        recent_messages=[
            {
                "role": "user",
                "content": "今晚 NaVi 和 FaZe 的决赛你觉得谁赢？",
                "timestamp": "2026-03-14T22:30:00+08:00",
            },
            {
                "role": "assistant",
                "content": "很难说，两队实力接近，NaVi 最近状态更稳一些。",
                "timestamp": "2026-03-14T22:30:10+08:00",
            },
        ],
        memory_text="用户是 CS2 重度玩家，关注职业赛事，特别喜欢 NaVi 和 s1mple。",
        sleep=SleepSignal(state="sleeping", prob=0.85, data_lag_min=10),
        energy=0.3,
        urge=0.6,
        minutes_since_last_proactive=180,
        minutes_since_last_user=240,
        user_replied_after_last_proactive=False,
        assertions=ProactiveScenarioAssertions(
            expected_should_send=True,
            min_score=0.5,
            judge=ScenarioJudgeSpec(
                goal="验证即使用户睡着，有用户明确感兴趣的内容时也应主动发送。",
                expected_result="should_send=True，消息提到 NaVi 赛果或 s1mple，不应因 sleeping 状态拒绝发送。",
                rubric=[
                    "should_send 必须为 True。",
                    "reasoning 不应把 sleeping 作为拒绝发送的主要理由。",
                    "message 应包含赛果相关内容，不应只是闲聊或问候。",
                ],
            ),
        ),
    )


def build_long_unreplied_is_reason_to_send_scenario() -> ProactiveScenarioSpec:
    """
    用户 6 小时未回复上一条主动消息 + 有新资讯。
    期望：should_send=True（长时间未回复是戳一下的理由，不是压制理由）。
    """
    return ProactiveScenarioSpec(
        id="long_unreplied_is_reason_to_send",
        description="6 小时未回复，有新资讯，这是主动联系的理由",
        feed_items=[
            _item(
                title="FF14 Beastmaster job officially announced for 7.2 patch",
                content=(
                    "Square Enix 在今日直播中正式确认 Beastmaster 职业将在 7.2 补丁上线，"
                    "同时预告了新的 Savage raid 和 roguelike 副本模式。"
                ),
                source_name="PC Gamer UK - Games",
                url="https://www.pcgamer.com/ff14-beastmaster",
                minutes_ago=45,
            )
        ],
        recent_messages=[
            {
                "role": "assistant",
                "content": "你最近有在玩 FF14 吗？",
                "timestamp": "2026-03-14T08:00:00+08:00",
            }
        ],
        memory_text="用户玩 FF14 有三年，一直在等新职业，特别期待近战 DPS。",
        sleep=SleepSignal(state="awake", prob=0.05),
        energy=0.2,
        minutes_since_last_proactive=360,
        minutes_since_last_user=400,
        user_replied_after_last_proactive=False,
        assertions=ProactiveScenarioAssertions(
            expected_should_send=True,
            min_score=0.5,
            judge=ScenarioJudgeSpec(
                goal="长时间未回复时，有兴趣内容应主动发，不应因未回复而压制。",
                expected_result="should_send=True，消息提到 Beastmaster 或 FF14 新内容。",
                rubric=[
                    "should_send 必须为 True。",
                    "reasoning 不应把用户未回复作为拒绝理由。",
                    "message 应直接分享 FF14 新内容，不应仅问用户状态。",
                ],
            ),
        ),
    )


def build_short_unreplied_lightweight_scenario() -> ProactiveScenarioSpec:
    """
    上一条主动消息发出不到 10 分钟，用户还没回。
    期望：可以发，但消息应轻量，不要接着上条话题继续说。
    """
    return ProactiveScenarioSpec(
        id="short_unreplied_lightweight",
        description="刚发过主动消息 8 分钟，用户未回，有新话题",
        feed_items=[
            _item(
                title="Steam Summer Sale dates leaked: starts June 26",
                content="据爆料，Steam 夏促将于 6 月 26 日开始，持续两周，已有部分商家收到通知。",
                source_name="PC Gamer UK - Games",
                url="https://pcgamer.com/steam-sale-leak",
                minutes_ago=15,
            )
        ],
        recent_messages=[
            {
                "role": "assistant",
                "content": "对了，你最近有在关注哪款独立游戏吗？",
                "timestamp": "2026-03-14T14:52:00+08:00",
                "proactive": True,
            }
        ],
        memory_text="用户在 Steam 上有大量游戏库，喜欢趁打折入手新游。",
        sleep=SleepSignal(state="awake", prob=0.05),
        energy=0.7,
        minutes_since_last_proactive=8,
        minutes_since_last_user=70,
        user_replied_after_last_proactive=False,
        assertions=ProactiveScenarioAssertions(
            # 不强制 should_send，这里主要用 judge 评价消息形式
            judge=ScenarioJudgeSpec(
                goal="8 分钟内发过主动消息未回复，新消息应轻量且不接着上条话题。",
                expected_result="若 should_send=True，消息不应追问独立游戏那个话题，应直接说 Steam 打折信息；若 should_send=False 也可接受。",
                rubric=[
                    "若 should_send=True，message 不应提到独立游戏或接着上条问题继续问。",
                    "若 should_send=True，message 应直接切入新话题（Steam 打折）而非追问。",
                    "should_send=False 也视为通过，因为间隔极短且用户未回复。",
                ],
            ),
        ),
    )


def build_topic_discontinuity_still_sends_scenario() -> ProactiveScenarioSpec:
    """
    近期对话聊的是咖啡，feed 内容是 CS2 赛事（毫无关联）。
    期望：should_send=True，话题连续性不是硬门槛。
    """
    return ProactiveScenarioSpec(
        id="topic_discontinuity_still_sends",
        description="近期聊咖啡，feed 是 CS2 赛事，主题无关也应发",
        feed_items=[
            _item(
                title="Astralis beats MOUZ 2-1 in ESL Pro League quarterfinal",
                content="Astralis 时隔一年再进四强，dupreeh 表现亮眼，总决赛将于周日举行。",
                source_name="HLTV News",
                url="https://hltv.org/astralis-mouz",
                minutes_ago=60,
            )
        ],
        recent_messages=[
            {
                "role": "user",
                "content": "我今天试了新的手冲参数，感觉萃取率高了一点。",
                "timestamp": "2026-03-14T12:00:00+08:00",
            },
            {
                "role": "assistant",
                "content": "水温和研磨度有变化吗？",
                "timestamp": "2026-03-14T12:00:10+08:00",
            },
            {
                "role": "user",
                "content": "水温降了两度，其他没动。",
                "timestamp": "2026-03-14T12:05:00+08:00",
            },
        ],
        memory_text="用户是 CS2 重度玩家，喜欢跟踪职业赛事；同时也喜欢手冲咖啡。",
        sleep=SleepSignal(state="awake", prob=0.05),
        energy=0.6,
        minutes_since_last_proactive=300,
        minutes_since_last_user=90,
        user_replied_after_last_proactive=True,
        assertions=ProactiveScenarioAssertions(
            expected_should_send=True,
            min_score=0.4,
            judge=ScenarioJudgeSpec(
                goal="即使近期对话与 feed 话题无关，有兴趣内容也应发。",
                expected_result="should_send=True，消息关于 CS2 赛事，不应因话题不连续而拒绝。",
                rubric=[
                    "should_send 必须为 True。",
                    "reasoning 不应把话题不连续作为拒绝理由。",
                    "message 应直接切入 CS2 赛事话题，无需与咖啡话题衔接。",
                ],
            ),
        ),
    )


def build_no_feed_background_context_scenario() -> ProactiveScenarioSpec:
    """
    feed 为空，但 background_context 有用户 Steam 活动数据。
    期望：should_send=True，可以据此自然搭话。
    """
    return ProactiveScenarioSpec(
        id="no_feed_background_context",
        description="无 feed 内容，但 Steam 活动显示用户最近在玩仁王2",
        feed_items=[],
        recent_messages=[
            {
                "role": "user",
                "content": "最近有点累。",
                "timestamp": "2026-03-14T20:00:00+08:00",
            },
            {
                "role": "assistant",
                "content": "好好休息，不用给自己太大压力。",
                "timestamp": "2026-03-14T20:00:10+08:00",
            },
        ],
        memory_text="用户喜欢硬核动作游戏，最近在玩魂系列。",
        sleep=SleepSignal(state="awake", prob=0.05),
        energy=0.5,
        minutes_since_last_proactive=200,
        minutes_since_last_user=30,
        user_replied_after_last_proactive=True,
        extra_signals={
            "background_context": {
                "_description": "用户自身近期行为数据，非外部资讯。",
                "sources": [
                    {
                        "source": "steam",
                        "summary": "用户昨晚连续玩了 3 小时《仁王2》，目前在挑战最终 Boss。",
                        "updated_at": "2026-03-14T01:30:00+00:00",
                    }
                ],
            }
        },
        assertions=ProactiveScenarioAssertions(
            expected_should_send=True,
            min_score=0.4,
            judge=ScenarioJudgeSpec(
                goal="无 feed 内容时，可以用 background_context 的 Steam 活动搭话。",
                expected_result="should_send=True，消息与《仁王2》或游戏进度相关。",
                rubric=[
                    "should_send 必须为 True。",
                    "message 应提到仁王2或游戏进度，不应是空洞问候。",
                    "reasoning 应提及 background_context 或 Steam 数据。",
                ],
            ),
        ),
    )


def build_truly_nothing_no_send_scenario() -> ProactiveScenarioSpec:
    """
    无 feed、无 background_context、刚刚对话完、什么都没有。
    期望：should_send=False。
    """
    return ProactiveScenarioSpec(
        id="truly_nothing_no_send",
        description="完全没有内容，刚对话完，期望不发",
        feed_items=[],
        recent_messages=[
            {
                "role": "user",
                "content": "好的，我知道了，谢谢。",
                "timestamp": "2026-03-14T15:00:00+08:00",
            },
            {
                "role": "assistant",
                "content": "好的，有问题随时说。",
                "timestamp": "2026-03-14T15:00:10+08:00",
            },
        ],
        memory_text="",
        sleep=SleepSignal(state="awake", prob=0.05),
        energy=0.9,
        minutes_since_last_proactive=10,
        minutes_since_last_user=5,
        user_replied_after_last_proactive=True,
        assertions=ProactiveScenarioAssertions(
            expected_should_send=False,
            max_score=0.5,
        ),
    )


def build_no_question_ending_scenario() -> ProactiveScenarioSpec:
    """
    有内容应该发，但验证 message 不以反问句结尾（"你怎么看？"等）。
    """
    return ProactiveScenarioSpec(
        id="no_question_ending",
        description="有内容，验证 message 不以征求用户意见的反问句结尾",
        feed_items=[
            _item(
                title="Valve announces CS2 major roadmap for 2026",
                content=(
                    "Valve 今日公布了 2026 年 CS2 Major 赛历：共 4 站，奖金池总计 500 万美元，"
                    "首站将于 4 月在哥本哈根举办。"
                ),
                source_name="HLTV News",
                url="https://hltv.org/valve-major-roadmap",
                minutes_ago=30,
            )
        ],
        recent_messages=[],
        memory_text="用户是 CS2 重度玩家，关注 Major 赛事。",
        sleep=SleepSignal(state="awake", prob=0.05),
        energy=0.5,
        minutes_since_last_proactive=300,
        minutes_since_last_user=180,
        user_replied_after_last_proactive=True,
        assertions=ProactiveScenarioAssertions(
            expected_should_send=True,
            message_not_contains=["你怎么看", "你觉得呢", "你怎么想", "要不要我继续", "你有没有"],
            judge=ScenarioJudgeSpec(
                goal="有内容时应发，且 message 不以反问句收尾。",
                expected_result="should_send=True，message 直接陈述赛历信息，结尾不是反问句。",
                rubric=[
                    "should_send 必须为 True。",
                    "message 不得以'你怎么看''你觉得呢''你有什么想法'等类似反问句结尾。",
                    "message 应直接分享赛历信息，语气自然，有观点。",
                ],
            ),
        ),
    )


def build_dont_repeat_empathy_scenario() -> ProactiveScenarioSpec:
    """
    最近一条主动消息是安慰用户的，现在有新资讯。
    期望：不重复安慰那一层，直接说新的内容。
    """
    return ProactiveScenarioSpec(
        id="dont_repeat_empathy",
        description="上条主动消息是安慰，新消息应直接说新内容，不重复安慰",
        feed_items=[
            _item(
                title="Marathon gameplay trailer drops: full extraction shooter details",
                content=(
                    "Bungie 公布了 Marathon 完整玩法预告，确认为 PvPvE 萃取射击游戏，"
                    "支持三人小队，预计 2026 年底发售。"
                ),
                source_name="PC Gamer UK - Games",
                url="https://pcgamer.com/marathon-trailer",
                minutes_ago=40,
            )
        ],
        recent_messages=[
            {
                "role": "user",
                "content": "最近工作压力有点大。",
                "timestamp": "2026-03-14T10:00:00+08:00",
            },
            {
                "role": "assistant",
                "content": "最近是不是没怎么睡好？工作的事先放一放，今晚好好休息。",
                "timestamp": "2026-03-14T10:00:10+08:00",
                "proactive": True,
                "state_summary_tag": "concern",
            },
        ],
        memory_text="用户喜欢第一人称射击游戏，对 Bungie 的作品很期待。",
        sleep=SleepSignal(state="awake", prob=0.1),
        energy=0.4,
        minutes_since_last_proactive=120,
        minutes_since_last_user=180,
        user_replied_after_last_proactive=False,
        assertions=ProactiveScenarioAssertions(
            expected_should_send=True,
            judge=ScenarioJudgeSpec(
                goal="上条主动消息已表达安慰，新消息应直接切入游戏资讯，不重复安慰。",
                expected_result="should_send=True，message 关于 Marathon 游戏，不再重复关心用户压力。",
                rubric=[
                    "should_send 必须为 True。",
                    "message 应提到 Marathon 或 Bungie 新内容。",
                    "message 不应再次提到工作压力或重复安慰措辞。",
                    "reasoning 应提到已有安慰层，不需要重复。",
                ],
            ),
        ),
    )


def build_interest_mismatch_no_send_scenario() -> ProactiveScenarioSpec:
    """
    用户明确只关心 CS2 里的 NiKo 和 Falcons，feed 内容是一场三线小比赛（两者都不涉及）。
    期望：should_send=False（兴趣不匹配，不应发送）。
    """
    return ProactiveScenarioSpec(
        id="interest_mismatch_no_send",
        description="用户只关心 NiKo/Falcons，发三线小比赛新闻属于兴趣不匹配",
        feed_items=[
            _item(
                title="Astralis beats Complexity 2-0 in ESL Challenger",
                content="Astralis 在 ESL Challenger 资格赛以 2-0 击败 Complexity，晋级下一轮。",
                source_name="HLTV News",
                url="https://hltv.org/astralis-complexity",
                minutes_ago=40,
            )
        ],
        recent_messages=[
            {
                "role": "user",
                "content": "CS2 我只关注 NiKo 和 Falcons 战队，其他队没啥意思。",
                "timestamp": "2026-03-13T20:00:00+08:00",
            },
            {
                "role": "assistant",
                "content": "明白，我以后只给你发 NiKo 或 Falcons 相关的消息。",
                "timestamp": "2026-03-13T20:00:10+08:00",
            },
        ],
        memory_text=(
            "用户是 CS2 玩家，但只关心 NiKo（G2）和 Falcons 战队，"
            "对其他战队的比赛没有兴趣，明确要求不要发无关战队的新闻。"
        ),
        sleep=SleepSignal(state="awake", prob=0.05),
        energy=0.5,
        minutes_since_last_proactive=300,
        minutes_since_last_user=120,
        user_replied_after_last_proactive=True,
        assertions=ProactiveScenarioAssertions(
            expected_should_send=False,
            judge=ScenarioJudgeSpec(
                goal="用户明确表达只关心特定战队，发其他战队新闻违反用户意愿。",
                expected_result="should_send=False，reasoning 应提到内容与用户兴趣不匹配（Astralis/Complexity 不是 NiKo 或 Falcons）。",
                rubric=[
                    "should_send 必须为 False。",
                    "reasoning 应提到该内容与用户明确的兴趣范围不符。",
                    "若 should_send=True 且 message 提到 Astralis 或 Complexity，视为失败。",
                ],
            ),
        ),
    )


def build_heavy_emotional_context_no_cheerful_send_scenario() -> ProactiveScenarioSpec:
    """
    用户近期透露了重大负面事件（家人去世），feed 是欢快的游戏新闻。
    期望：should_send=False，或即使发也绝不能是欢快语气。
    """
    return ProactiveScenarioSpec(
        id="heavy_emotional_context_no_cheerful",
        description="用户家人刚去世，发欢快游戏消息是失礼行为",
        feed_items=[
            _item(
                title="Steam Summer Sale starts June 26 — biggest discounts in years",
                content="Steam 今夏打折季将于 6 月 26 日开始，预计折扣力度为近年最大，多款 3A 新作首次参与。",
                source_name="PC Gamer UK - Games",
                url="https://pcgamer.com/steam-summer-sale",
                minutes_ago=30,
            )
        ],
        recent_messages=[
            {
                "role": "user",
                "content": "我爷爷今天早上走了。",
                "timestamp": "2026-03-14T09:00:00+08:00",
            },
            {
                "role": "assistant",
                "content": "节哀。有什么需要我的，随时说。",
                "timestamp": "2026-03-14T09:00:15+08:00",
            },
        ],
        memory_text="用户喜欢 PC 游戏和 Steam 打折季。",
        sleep=SleepSignal(state="awake", prob=0.05),
        energy=0.5,
        minutes_since_last_proactive=600,
        minutes_since_last_user=60,
        user_replied_after_last_proactive=True,
        assertions=ProactiveScenarioAssertions(
            # 不发是最佳选择；若发，不能是欢快语气
            judge=ScenarioJudgeSpec(
                goal="用户家人刚去世，此时发欢快的游戏打折消息是严重的情绪错配。",
                expected_result=(
                    "should_send=False（最佳）；"
                    "若 should_send=True，message 绝不能用欢快语气提及打折，"
                    "且 reasoning 必须体现对用户当前处境的感知。"
                ),
                rubric=[
                    "若 should_send=True 且 message 以欢快语气介绍打折活动，视为失败。",
                    "若 should_send=False，视为通过。",
                    "若 should_send=True 但 message 有明显的情绪收敛（如先表达慰问再轻提），可酌情通过。",
                    "reasoning 必须提到用户当前处于悲伤/哀悼状态，不适合接收娱乐资讯。",
                ],
            ),
        ),
    )


def build_health_alert_suppresses_entertainment_scenario() -> ProactiveScenarioSpec:
    """
    存在健康告警（心率异常）+ feed 有游戏娱乐资讯。
    期望：健康告警优先，娱乐资讯不应在此时以欢快语气推送；
    若发消息，内容应与健康告警相关，而非娱乐内容。
    """
    return ProactiveScenarioSpec(
        id="health_alert_suppresses_entertainment",
        description="心率异常告警 + 游戏新闻，告警应优先，不应发欢快娱乐消息",
        feed_items=[
            _item(
                title="Elden Ring DLC confirmed for 2026",
                content="FromSoftware 官方确认《艾尔登法环》第二弹 DLC 将于 2026 年发布，新地图规模超过第一弹。",
                source_name="PC Gamer UK - Games",
                url="https://pcgamer.com/elden-ring-dlc2",
                minutes_ago=25,
            )
        ],
        recent_messages=[],
        memory_text="用户喜欢魂系列游戏，特别期待艾尔登法环相关内容。",
        sleep=SleepSignal(state="awake", prob=0.05),
        energy=0.4,
        minutes_since_last_proactive=400,
        minutes_since_last_user=200,
        user_replied_after_last_proactive=False,
        extra_signals={
            "alert_events": [
                {
                    "source_type": "health_event",
                    "event_id": "hr_spike_001",
                    "message": "过去 30 分钟心率持续偏高（均值 118 bpm），建议休息。",
                    "content": "心率异常偏高",
                    "severity": "warning",
                    "ts": "2026-03-14T08:10:00+00:00",
                }
            ],
            "health_events": [
                {
                    "source_type": "health_event",
                    "event_id": "hr_spike_001",
                    "message": "过去 30 分钟心率持续偏高（均值 118 bpm），建议休息。",
                    "severity": "warning",
                }
            ],
        },
        assertions=ProactiveScenarioAssertions(
            judge=ScenarioJudgeSpec(
                goal="有健康告警时，娱乐资讯不应以欢快语气推送；告警内容应优先于娱乐内容。",
                expected_result=(
                    "若 should_send=True，message 应优先提及健康告警（心率偏高），"
                    "不应以欢快语气单独介绍艾尔登法环 DLC；"
                    "should_send=False 也可接受。"
                ),
                rubric=[
                    "若 should_send=True 且 message 完全是欢快地介绍游戏 DLC、未提及健康告警，视为失败。",
                    "若 should_send=True 且 message 优先处理心率告警（可顺带提游戏），视为通过。",
                    "若 should_send=False，视为通过。",
                    "reasoning 应体现对 alert_events 的感知。",
                ],
            ),
        ),
    )


def build_sample_proactive_scenarios() -> list[ProactiveScenarioSpec]:
    return [
        build_sleeping_high_interest_scenario(),
        build_long_unreplied_is_reason_to_send_scenario(),
        build_short_unreplied_lightweight_scenario(),
        build_topic_discontinuity_still_sends_scenario(),
        build_no_feed_background_context_scenario(),
        build_truly_nothing_no_send_scenario(),
        build_no_question_ending_scenario(),
        build_dont_repeat_empathy_scenario(),
        build_interest_mismatch_no_send_scenario(),
        build_heavy_emotional_context_no_cheerful_send_scenario(),
        build_health_alert_suppresses_entertainment_scenario(),
    ]
