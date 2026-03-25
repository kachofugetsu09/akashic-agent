"""
Memory2 dedup 场景 fixture 构建器。
每个场景都包含噪音条目，覆盖 SKIP / NONE+merge / CREATE 三条路径。
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
# 公共噪音条目（各场景复用）
# ---------------------------------------------------------------------------

_NOISE_EVENTS = [
    ScenarioMemoryItem(
        summary="用户上周买了新的手冲咖啡壶，最近每天早上都在试不同的冲煮参数。",
        memory_type="event",
        extra={"scope_channel": "cli", "scope_chat_id": "scenario-dedup"},
        source_ref="noise-coffee",
        happened_at="2026-03-20T08:00:00+08:00",
    ),
    ScenarioMemoryItem(
        summary="用户计划清明节去苏州游玩，两天一夜，想去拙政园和平江路。",
        memory_type="event",
        extra={"scope_channel": "cli", "scope_chat_id": "scenario-dedup"},
        source_ref="noise-travel",
        happened_at="2026-03-18T11:00:00+08:00",
    ),
    ScenarioMemoryItem(
        summary="用户最近在玩《仁王2》，这周连着打了好几晚，最近在刷最高难度。",
        memory_type="event",
        extra={"scope_channel": "cli", "scope_chat_id": "scenario-dedup"},
        source_ref="noise-game",
        happened_at="2026-03-22T22:00:00+08:00",
    ),
]


# ---------------------------------------------------------------------------
# S1：同义改写偏好 → SKIP（不写入重复记录）
# ---------------------------------------------------------------------------

def build_dedup_skip_synonym_preference() -> ScenarioSpec:
    """
    预置"简洁回复"偏好 + 3 条噪音事件。
    用户再次表达同一偏好（措辞不同），DedupDecider 应判 SKIP，不写入新记录。
    验证点：active 简洁偏好恰好只有 1 条，不因同义改写产生重复。
    """
    return ScenarioSpec(
        id="dedup_skip_synonym_preference",
        message="说话简洁点，别废话连篇，直接说重点就行。",
        channel="cli",
        chat_id="scenario-dedup",
        request_time=datetime.fromisoformat("2026-03-25T10:00:00+08:00"),
        memory2_items=[
            # 目标：与用户本次消息语义高度重叠，应触发 SKIP
            ScenarioMemoryItem(
                summary="用户偏好简洁直接的回复，不需要过多铺垫和废话，直达重点即可。",
                memory_type="preference",
                extra={"scope_channel": "cli", "scope_chat_id": "scenario-dedup"},
                source_ref="seed-brevity-pref",
                happened_at="2026-03-10T09:00:00+08:00",
            ),
            # 噪音：无关偏好（不应干扰 dedup 判断）
            ScenarioMemoryItem(
                summary="用户偏好在早上 9 点前处理重要邮件，不喜欢晚间被打扰。",
                memory_type="preference",
                extra={"scope_channel": "cli", "scope_chat_id": "scenario-dedup"},
                source_ref="noise-schedule-pref",
                happened_at="2026-03-05T08:00:00+08:00",
            ),
            *_NOISE_EVENTS,
        ],
        assertions=ScenarioAssertions(
            async_wait_timeout_s=15.0,
            # 原始简洁偏好应依然 active（SKIP 不会 supersede 它）
            async_memory_rows=[
                ScenarioMemoryRowAssertion(
                    status="active",
                    memory_type="preference",
                    summary_keywords=["简洁"],
                    source_ref_contains=["seed-brevity-pref"],
                ),
            ],
        ),
        judge=ScenarioJudgeSpec(
            goal="验证 dedup SKIP 路径：同义改写不产生重复 preference 记录。",
            expected_result=(
                "memory_rows 中 status=active 且 memory_type=preference 且 summary 语义上表达"
                "【简洁/直接回复】的条目应恰好只有 1 条，不应因为本次用户消息而增加为 2 条或更多。"
            ),
            rubric=[
                "统计 memory_rows 中 status=active、memory_type=preference、"
                "且 summary 明确表达简洁/废话少/直达重点含义的条目数量。",
                "若该数量恰好为 1，则通过（SKIP 生效，未写入重复记录）。",
                "若该数量 >= 2，则不通过（SKIP 未生效，产生了重复）。",
                "不要因为最终回答措辞而直接判定，重点看 memory_rows 中的去重效果。",
            ],
        ),
    )


# ---------------------------------------------------------------------------
# S2：增量过程更新 → NONE + merge（合并到现有 procedure）
# ---------------------------------------------------------------------------

def build_dedup_none_merge_procedure() -> ScenarioSpec:
    """
    预置 Steam MCP 查询流程 + 2 条噪音 procedure + 3 条噪音事件。
    用户补充一个新步骤（区服判断），DedupDecider 应判 NONE + merge，
    将新步骤合并进现有 procedure，而非新建一条。
    验证点：合并后 active procedure 同时包含 steam_mcp 和区服关键字。
    """
    return ScenarioSpec(
        id="dedup_none_merge_procedure",
        message=(
            "查 Steam 信息那个流程还要补一步：先判断下区服（大陆区/港区/美区），"
            "因为不同区价格和游戏可用性不一样，然后再用 steam_mcp 查。"
        ),
        channel="cli",
        chat_id="scenario-dedup",
        request_time=datetime.fromisoformat("2026-03-25T10:05:00+08:00"),
        memory2_items=[
            # 目标：与用户补充内容高度相关，应触发 NONE → merge
            ScenarioMemoryItem(
                summary=(
                    "查询 Steam 游戏信息时，必须先使用 steam_mcp 工具查询游戏详情，"
                    "再用 web_search 补充验证价格和评价信息。"
                ),
                memory_type="procedure",
                extra={
                    "steps": [
                        "使用 steam_mcp 工具查询游戏详情",
                        "使用 web_search 补充验证价格和评价",
                    ],
                    "tool_requirement": "steam_mcp",
                    "scope_channel": "cli",
                    "scope_chat_id": "scenario-dedup",
                },
                source_ref="seed-steam-procedure",
                happened_at="2026-03-15T18:00:00+08:00",
            ),
            # 噪音：无关 procedure（应完全不参与 merge）
            ScenarioMemoryItem(
                summary="查询天气时先用 weather_tool 获取实时数据，再根据需要用 web_search 补充预报。",
                memory_type="procedure",
                extra={
                    "steps": ["使用 weather_tool 查询实时天气"],
                    "tool_requirement": "weather",
                    "scope_channel": "cli",
                    "scope_chat_id": "scenario-dedup",
                },
                source_ref="noise-weather-procedure",
                happened_at="2026-03-12T10:00:00+08:00",
            ),
            ScenarioMemoryItem(
                summary="搜索新闻时先用 web_search 获取最新报道，再筛选可信来源进行摘要。",
                memory_type="procedure",
                extra={
                    "steps": ["使用 web_search 搜索新闻"],
                    "tool_requirement": "web_search",
                    "scope_channel": "cli",
                    "scope_chat_id": "scenario-dedup",
                },
                source_ref="noise-news-procedure",
                happened_at="2026-03-10T15:00:00+08:00",
            ),
            *_NOISE_EVENTS,
        ],
        assertions=ScenarioAssertions(
            async_wait_timeout_s=20.0,
            # 合并后应有包含 steam_mcp 和区服关键字的 active procedure
            async_memory_rows=[
                ScenarioMemoryRowAssertion(
                    status="active",
                    memory_type="procedure",
                    summary_keywords=["steam_mcp", "区服"],
                ),
            ],
        ),
        judge=ScenarioJudgeSpec(
            goal="验证 dedup NONE+merge 路径：新步骤被合并进现有 Steam procedure，不产生独立新记录。",
            expected_result=(
                "合并后应有且仅有 1 条 active 的 Steam 查询 procedure，"
                "该 procedure 同时包含原有的 steam_mcp 步骤和新增的区服判断步骤。"
                "不应出现 2 条独立的 Steam procedure（一条旧的 + 一条只包含区服的新的）。"
            ),
            rubric=[
                "检查 memory_rows 中 status=active、memory_type=procedure、"
                "summary 语义上描述 Steam 查询流程的条目数量。",
                "若该数量为 1 且 summary 同时提及 steam_mcp 和区服，则通过。",
                "若出现 2 条独立 Steam procedure（merge 未生效），则不通过。",
                "若 active Steam procedure 只有区服关键字但缺少 steam_mcp，则不通过（原步骤丢失）。",
                "噪音 procedure（天气/新闻）不应被影响，忽略它们。",
            ],
        ),
    )


# ---------------------------------------------------------------------------
# S3：全新偏好 → CREATE（无相似项，直接写入）
# ---------------------------------------------------------------------------

def build_dedup_create_new_preference() -> ScenarioSpec:
    """
    无中文回复偏好种子，只有无关噪音（含另一条完全不同的偏好）。
    用户说"以后只用中文"，DedupDecider 找不到高相似项，应判 CREATE。
    验证点：active 中文偏好被写入，原有噪音偏好不受影响。
    """
    return ScenarioSpec(
        id="dedup_create_new_preference",
        message="之后跟我说话只用中文，不要夹杂英文，哪怕专有名词也尽量翻译。",
        channel="cli",
        chat_id="scenario-dedup",
        request_time=datetime.fromisoformat("2026-03-25T10:10:00+08:00"),
        memory2_items=[
            # 噪音偏好：与"用中文回复"完全不同，不应触发 SKIP/merge
            ScenarioMemoryItem(
                summary="用户偏好在早上 9 点前处理重要邮件，不喜欢晚间被打扰。",
                memory_type="preference",
                extra={"scope_channel": "cli", "scope_chat_id": "scenario-dedup"},
                source_ref="noise-schedule-pref",
                happened_at="2026-03-05T08:00:00+08:00",
            ),
            ScenarioMemoryItem(
                summary="用户偏好简洁直接的回复，不需要过多铺垫和废话。",
                memory_type="preference",
                extra={"scope_channel": "cli", "scope_chat_id": "scenario-dedup"},
                source_ref="noise-brevity-pref",
                happened_at="2026-03-08T09:00:00+08:00",
            ),
            *_NOISE_EVENTS,
        ],
        assertions=ScenarioAssertions(
            async_wait_timeout_s=15.0,
            # CREATE 后应有 active 的中文偏好
            async_memory_rows=[
                ScenarioMemoryRowAssertion(
                    status="active",
                    memory_type="preference",
                    summary_keywords=["中文"],
                ),
            ],
        ),
        judge=ScenarioJudgeSpec(
            goal="验证 dedup CREATE 路径：全新语言偏好被正确写入，不受噪音影响。",
            expected_result=(
                "memory_rows 中应有一条 status=active 的 preference 记录，"
                "其 summary 明确表达【使用中文/不用英文回复】的含义。"
                "原有的时间管理偏好和简洁偏好不应被影响或 supersede。"
            ),
            rubric=[
                "检查 memory_rows 中是否存在 status=active、memory_type=preference、"
                "且 summary 语义上表达【只用中文/不夹英文】的条目。",
                "若存在，则通过（CREATE 生效）。",
                "若不存在，则不通过（CREATE 未生效或 false positive SKIP）。",
                "同时检查 noise-schedule-pref 和 noise-brevity-pref 是否仍然 active（不应被误 supersede）。",
            ],
        ),
    )
