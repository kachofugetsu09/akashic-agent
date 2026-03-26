"""
Post-response memory 提取质量基线场景 fixtures。

每个场景对应 docs/memory_extraction_baseline_20260326.md 中归纳的一类 Bug：

  B1  技术知识被误判为 procedure（八股文场景）
  B2  从 ASSISTANT 话语提取 memory
  B3  session-specific 内容写成长期 preference
  B4  同轮同质内容 dedup 未拦截
  B5  内容绑定具体项目，泛化不足

正向场景（期望提取）：
  P1  纠正 agent 行为 → 正确提取 preference
  P2  调查方法论 → 正确泛化为 preference
"""
from __future__ import annotations

from datetime import datetime

from tests_scenarios.fixtures import (
    ScenarioAssertions,
    ScenarioJudgeSpec,
    ScenarioMemoryItem,
    ScenarioMemoryRowAssertion,
    ScenarioSpec,
)

# ---------------------------------------------------------------------------
# 公共噪音条目
# ---------------------------------------------------------------------------

_NOISE_EVENTS = [
    ScenarioMemoryItem(
        summary="用户上周买了新的机械键盘，最近在测试不同轴体的手感。",
        memory_type="event",
        extra={"scope_channel": "cli", "scope_chat_id": "scenario-post-extract"},
        source_ref="noise-keyboard",
        happened_at="2026-03-20T10:00:00+08:00",
    ),
    ScenarioMemoryItem(
        summary="用户计划端午节去成都旅游，想去宽窄巷子和锦里。",
        memory_type="event",
        extra={"scope_channel": "cli", "scope_chat_id": "scenario-post-extract"},
        source_ref="noise-travel",
        happened_at="2026-03-18T11:00:00+08:00",
    ),
    ScenarioMemoryItem(
        summary="用户最近在看《三体》，已经到第二部了。",
        memory_type="event",
        extra={"scope_channel": "cli", "scope_chat_id": "scenario-post-extract"},
        source_ref="noise-book",
        happened_at="2026-03-22T22:00:00+08:00",
    ),
]

_NOISE_PREFS = [
    ScenarioMemoryItem(
        summary="用户偏好简洁直接的回复，不需要过多铺垫。",
        memory_type="preference",
        extra={"scope_channel": "cli", "scope_chat_id": "scenario-post-extract"},
        source_ref="noise-brevity-pref",
        happened_at="2026-03-10T09:00:00+08:00",
    ),
]


# ---------------------------------------------------------------------------
# B1：八股文场景 — 技术知识不应被提取为 procedure
#
# 场景：用户在做面试复习，向 agent 背诵/讲解分布式系统的工作原理。
# 对话内容充满技术"流程"描述，但这些都是用户在背书，不是对 agent 的行为要求。
# 期望：post-response worker 不提取任何 procedure 或 preference。
# ---------------------------------------------------------------------------

def build_b1_interview_prep_no_procedure() -> ScenarioSpec:
    """
    B1：用户背诵分布式协调相关原理，不应提取为 agent procedure。

    用户行为：复习 Raft 选主流程（面试八股），逐条陈述技术细节。
    期望：memory 中不出现关于 Raft、选主、quorum 的 procedure 或 preference 条目。
    """
    return ScenarioSpec(
        id="post_extract_b1_interview_prep_no_procedure",
        message=(
            "我来背一下 Raft 选主：leader 崩溃后，follower 等待 election timeout，"
            "然后 term+1，给自己投票，向其他节点发 RequestVote RPC。"
            "获得超过半数节点（quorum）投票后成为新 leader，"
            "并开始向所有节点发送心跳来抑制新一轮选举。"
            "你帮我纠错一下这段描述，看有没有讲错的地方。"
        ),
        channel="cli",
        chat_id="scenario-post-extract",
        request_time=datetime.fromisoformat("2026-03-26T10:00:00+08:00"),
        memory2_items=[
            *_NOISE_EVENTS,
            *_NOISE_PREFS,
        ],
        assertions=ScenarioAssertions(
            async_wait_timeout_s=15.0,
            # 噪音条目应保持不变（确认 worker 正常运行）
            async_memory_rows=[
                ScenarioMemoryRowAssertion(
                    status="active",
                    memory_type="preference",
                    summary_keywords=["简洁"],
                    source_ref_contains=["noise-brevity-pref"],
                ),
            ],
        ),
        judge=ScenarioJudgeSpec(
            goal="验证八股文场景下技术知识不被误判为 agent procedure。",
            expected_result=(
                "memory_rows 中不应出现任何 memory_type=procedure 且 source_ref 包含"
                " '@post_response' 的条目。"
                "Raft 选主的技术步骤（RequestVote、quorum、term、心跳）不是 agent 的行为规范，"
                "不应被提取为 procedure 或 preference。"
            ),
            rubric=[
                "检查 memory_rows 中是否存在 source_ref 包含 '@post_response' 且"
                " memory_type='procedure' 的条目。",
                "若不存在，则通过（B1 修复生效）。",
                "若存在且 summary 描述的是 Raft 技术流程（如选主、quorum、心跳、RequestVote），则不通过。",
                "若 procedure 描述的是 agent 应做的事（如'帮用户纠错时应先逐条检查'），则可以通过——"
                "这类 agent 行为提取是合理的。",
                "不要因为最终回答内容而判定，只看 memory_rows 的 post_response 条目。",
            ],
        ),
    )


# ---------------------------------------------------------------------------
# B2：Agent 自身承诺不应被提取为 procedure
#
# 场景：用户抱怨 agent 把简单问题搞复杂了，agent 在回复中承诺"下次直接回答"。
# 期望：agent 的承诺话语不被提取为 procedure（"我"指 agent，不是用户）。
# ---------------------------------------------------------------------------

def build_b2_agent_promise_not_extracted() -> ScenarioSpec:
    """
    B2：agent 回复中的自我承诺，不应被提取为 procedure。

    用户行为：抱怨 agent 把简单问题搞复杂，语气直接但不构成操作规范。
    Agent 很可能回复"好的，下次遇到简单问题我直接回答"之类的承诺。
    期望：该承诺不被写入 procedure（因为说话人是 agent，不是用户）。
    """
    return ScenarioSpec(
        id="post_extract_b2_agent_promise_not_extracted",
        message=(
            "这种问题你直接答就行了，别搞那么复杂，"
            "就一个简单的概念解释，你整了一大堆步骤干嘛。"
        ),
        channel="cli",
        chat_id="scenario-post-extract",
        request_time=datetime.fromisoformat("2026-03-26T10:05:00+08:00"),
        memory2_items=[
            *_NOISE_EVENTS,
            *_NOISE_PREFS,
        ],
        assertions=ScenarioAssertions(
            async_wait_timeout_s=12.0,
            async_memory_rows=[
                ScenarioMemoryRowAssertion(
                    status="active",
                    memory_type="preference",
                    summary_keywords=["简洁"],
                    source_ref_contains=["noise-brevity-pref"],
                ),
            ],
        ),
        judge=ScenarioJudgeSpec(
            goal="验证 agent 的自我承诺不被提取为 procedure。",
            expected_result=(
                "【判断依据只有 memory_rows，最终回答内容与本题无关，不得用于判断。】\n"
                "在 memory_rows 中找到所有 source_ref 字段字面包含字符串 '@post_response' 的条目。\n"
                "若数量为 0，passed=true，直接结束，不再看其他任何内容。\n"
                "若数量 >= 1，检查这些条目的 summary：\n"
                "  - summary 含'我下次''我会''以后我'等 agent 第一人称承诺 → passed=false\n"
                "  - summary 是用户侧的要求（如'简单问题应直接回答'）→ passed=true"
            ),
            rubric=[
                "【强制规则】只看 memory_rows，最终回答内容与判断无关，不得引用。",
                "统计 memory_rows 中 source_ref 字面含 '@post_response' 的条目数量。",
                "数量 == 0 → passed=true，立即结束。",
                "数量 >= 1 且 summary 含 agent 第一人称承诺 → passed=false。",
                "数量 >= 1 且 summary 是用户侧要求 → passed=true。",
            ],
        ),
    )


# ---------------------------------------------------------------------------
# B3：session-specific 感悟不应写成 preference
#
# 场景：用户在对话中表达了一个认知感悟，不是对 agent 的行为要求。
# 期望：若提取，应为 event；不应提取为 preference。
# ---------------------------------------------------------------------------

def build_b3_insight_as_event_not_preference() -> ScenarioSpec:
    """
    B3：用户表达认知感悟，不应提取为 preference。

    用户行为：说了一句感悟性的话，表达对某个观点的认可，但没有对 agent 提要求。
    期望：若提取则为 event，不应提取为 preference（"认可某做法的价值"不是 agent 规范）。
    """
    return ScenarioSpec(
        id="post_extract_b3_insight_as_event_not_preference",
        message=(
            "其实定期整理笔记这件事，本质上就是在给自己建一个外部知识库，"
            "相当于把大脑里不可靠的记忆外化成可以检索的文档。"
            "你觉得这个理解对吗？"
        ),
        channel="cli",
        chat_id="scenario-post-extract",
        request_time=datetime.fromisoformat("2026-03-26T10:10:00+08:00"),
        memory2_items=[
            *_NOISE_EVENTS,
            *_NOISE_PREFS,
        ],
        assertions=ScenarioAssertions(
            async_wait_timeout_s=12.0,
            async_memory_rows=[
                ScenarioMemoryRowAssertion(
                    status="active",
                    memory_type="preference",
                    summary_keywords=["简洁"],
                    source_ref_contains=["noise-brevity-pref"],
                ),
            ],
        ),
        judge=ScenarioJudgeSpec(
            goal="验证用户的认知感悟不被 post_response worker 提取为任何记忆条目。",
            expected_result=(
                "memory_rows 中不应出现任何 source_ref 含 '@post_response' 的新条目。"
                "用户只是在表达一个认知感悟，没有对 agent 提出任何要求或稳定偏好信号，"
                "post_response worker 不应提取 procedure 或 preference。"
            ),
            rubric=[
                "检查 memory_rows 中 source_ref 含 '@post_response' 的条目。",
                "若不存在任何 post_response 条目，则通过。",
                "若存在 memory_type='preference' 且 summary 语义为'用户认为笔记/外部知识库有价值'，则不通过。",
                "若存在 memory_type='procedure'，则不通过。",
            ],
        ),
    )


# ---------------------------------------------------------------------------
# B5：单次项目评价不应提取为通用 preference
#
# 场景：用户吐槽某个 GitHub 项目的演示是假的、代码没有对应功能。
# 期望：不提取任何 preference（对一个具体项目的吐槽不是跨对话长期偏好）。
# ---------------------------------------------------------------------------

def build_b5_project_complaint_no_preference() -> ScenarioSpec:
    """
    B5：对单个项目的不满不应提取为通用 preference。

    用户行为：评价某开源项目的演示视频造假，对一个具体项目的单次吐槽。
    期望：不提取任何 preference（绑定了具体项目，不具备跨对话持久性）。
    """
    return ScenarioSpec(
        id="post_extract_b5_project_complaint_no_preference",
        message=(
            "我看了一下 DataVizPro 这个项目，它宣传能实时分析视频流里的人体动作，"
            "但翻了半天代码，根本没有任何 pose estimation 的模型或推理逻辑，"
            "演示视频里的那些效果完全是用数学公式生成的动画假冒的。"
            "这种项目你之前怎么没发现问题就推给我了？"
        ),
        channel="cli",
        chat_id="scenario-post-extract",
        request_time=datetime.fromisoformat("2026-03-26T10:15:00+08:00"),
        memory2_items=[
            *_NOISE_EVENTS,
            *_NOISE_PREFS,
        ],
        assertions=ScenarioAssertions(
            async_wait_timeout_s=12.0,
            async_memory_rows=[
                ScenarioMemoryRowAssertion(
                    status="active",
                    memory_type="preference",
                    summary_keywords=["简洁"],
                    source_ref_contains=["noise-brevity-pref"],
                ),
            ],
        ),
        judge=ScenarioJudgeSpec(
            goal="验证对具体项目的单次吐槽不被提取为通用 preference。",
            expected_result=(
                "memory_rows 中不应出现 source_ref 含 '@post_response' 且"
                " memory_type='preference' 且 summary 绑定了 DataVizPro 这个具体项目的条目。"
                "用户对一个具体项目的单次不满，不构成跨对话持久的内容偏好。"
            ),
            rubric=[
                "检查 memory_rows 中 source_ref 含 '@post_response' 且 memory_type='preference' 的条目。",
                "若存在且 summary 提到 DataVizPro、人体动作分析、演示造假等与本项目强绑定的内容，则不通过。",
                "若存在且 summary 是抽象的方法论（如'推荐项目前应验证其核心功能是否真实可运行'），"
                "则通过——这类泛化是合理的。",
                "若没有 preference 提取，也通过。",
            ],
        ),
    )


# ---------------------------------------------------------------------------
# P1（正向）：纠正 agent 行为 → 正确提取 preference
#
# 场景：用户明确纠正 agent 在讲解流程时遗漏了前置步骤。
# 期望：正确提取 1 条 preference，描述讲解完整性要求。
# ---------------------------------------------------------------------------

def build_p1_correction_extracts_preference() -> ScenarioSpec:
    """
    P1：用户明确纠正 agent 遗漏了讲解前置步骤，应正确提取为 preference。

    用户行为：纠正 agent 只讲了某协议的握手，没有从连接建立讲起。
    期望：提取 1 条 preference，内容要求讲解流程从建立连接开始。
    """
    return ScenarioSpec(
        id="post_extract_p1_correction_extracts_preference",
        message=(
            "你刚才讲 HTTPS 握手只说了 TLS 的部分，但 HTTPS 握手本身应该从 TCP 三次握手开始讲，"
            "TCP 连接建立之后才轮到 TLS。以后讲这类流程性的东西，"
            "前置的连接层步骤不能省，不然逻辑是断的。"
        ),
        channel="cli",
        chat_id="scenario-post-extract",
        request_time=datetime.fromisoformat("2026-03-26T10:20:00+08:00"),
        history=[
            {
                "role": "user",
                "content": "说一下 HTTPS 握手的过程",
                "timestamp": "2026-03-26T10:19:00+08:00",
            },
            {
                "role": "assistant",
                "content": (
                    "HTTPS 握手主要是 TLS 握手：客户端发 ClientHello，"
                    "服务端回 ServerHello 和证书，"
                    "客户端验证证书后协商会话密钥，之后正式加密通信。"
                ),
                "timestamp": "2026-03-26T10:19:10+08:00",
            },
        ],
        memory2_items=[
            *_NOISE_EVENTS,
            *_NOISE_PREFS,
        ],
        assertions=ScenarioAssertions(
            async_wait_timeout_s=15.0,
            async_memory_rows=[
                ScenarioMemoryRowAssertion(
                    status="active",
                    memory_type="procedure",
                    summary_keywords=["完整"],
                    source_ref_contains=["@post_response"],
                ),
            ],
        ),
        judge=ScenarioJudgeSpec(
            goal="验证用户纠正 agent 遗漏步骤后，正确提取为 procedure（用户对 agent 讲解行为的要求）。",
            expected_result=(
                "memory_rows 中应有 1 条 source_ref 含 '@post_response' 且"
                " memory_type='procedure' 的条目，"
                "内容表达讲解流程性内容时应保持完整，不省略前置层。"
            ),
            rubric=[
                "检查是否存在 source_ref 字面包含 '@post_response' 且 memory_type='procedure' 的条目。",
                "若存在且 summary 语义上要求讲解时保持完整、不省略前置步骤，则通过。",
                "若存在但 summary 只是复述了 TLS/TCP 的技术流程本身（而非对 agent 的行为要求），则不通过。",
                "若不存在，则不通过（应该提取但未提取）。",
            ],
        ),
    )


# ---------------------------------------------------------------------------
# P2（正向）：调查方法论 → 提取泛化 preference
#
# 场景：用户明确要求 agent 查资料时不要只看单一来源，要多渠道求证。
# 期望：提取 1 条泛化的调查方法论 preference，不绑定具体项目。
# ---------------------------------------------------------------------------

def build_p2_multichannel_investigation_preference() -> ScenarioSpec:
    """
    P2：用户要求多渠道调查，应提取泛化的调查方法论 preference。

    用户行为：在 agent 只查了 GitHub 后，要求同时去查第三方评价渠道。
    期望：提取 1 条 preference，泛化为"调查时应多渠道"，而非绑定当前项目。
    """
    return ScenarioSpec(
        id="post_extract_p2_multichannel_investigation_preference",
        message=(
            "你别只看 GitHub 上的代码，也去查查第三方的评价，"
            "比如 Hacker News 有没有相关讨论，或者技术论坛、博客里有没有人真实用过的反馈。"
            "只看官方渠道很容易被误导。"
        ),
        channel="cli",
        chat_id="scenario-post-extract",
        request_time=datetime.fromisoformat("2026-03-26T10:25:00+08:00"),
        history=[
            {
                "role": "user",
                "content": "帮我查一下 StreamForge 这个项目靠不靠谱",
                "timestamp": "2026-03-26T10:24:00+08:00",
            },
            {
                "role": "assistant",
                "content": (
                    "我查了一下 GitHub：项目有 1.2k Star，代码库活跃，"
                    "issue 响应也比较及时，整体看起来维护状态不错。"
                ),
                "timestamp": "2026-03-26T10:24:10+08:00",
            },
        ],
        memory2_items=[
            *_NOISE_EVENTS,
            *_NOISE_PREFS,
        ],
        assertions=ScenarioAssertions(
            async_wait_timeout_s=15.0,
            async_memory_rows=[
                ScenarioMemoryRowAssertion(
                    status="active",
                    memory_type="procedure",
                    summary_keywords=["渠道"],
                    source_ref_contains=["@post_response"],
                ),
            ],
        ),
        judge=ScenarioJudgeSpec(
            goal="验证用户要求多渠道调查时，提取泛化方法论 procedure，而非绑定具体项目。",
            expected_result=(
                "memory_rows 中应有 1 条 source_ref 含 '@post_response' 且"
                " memory_type='procedure' 的条目，"
                "其 summary 语义上表达'调查/评估时不能只看单一渠道，应包括第三方评价'。"
                "summary 不应提及 StreamForge 这个具体项目名称。"
            ),
            rubric=[
                "检查是否存在 source_ref 含 '@post_response' 且 memory_type='procedure' 的条目。",
                "若存在且 summary 不含 StreamForge 但包含多渠道/第三方/Hacker News 等方法论关键词，则通过。",
                "若存在但 summary 绑定了 StreamForge 这个具体项目，则不通过（泛化不足）。",
                "若不存在，则不通过（应该提取但未提取）。",
            ],
        ),
    )
