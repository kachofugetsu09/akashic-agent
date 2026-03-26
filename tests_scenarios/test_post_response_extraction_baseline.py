"""
Post-response memory 提取质量基线测试。

对应文档：docs/memory_extraction_baseline_20260326.md
覆盖范围：B1–B5 错误类型 + P1/P2 正向验证。

运行命令：
    AKASIC_RUN_SCENARIOS=1 pytest -c pytest-scenarios.ini \
        tests_scenarios/test_post_response_extraction_baseline.py -q

单条运行：
    AKASIC_RUN_SCENARIOS=1 pytest -c pytest-scenarios.ini \
        tests_scenarios/test_post_response_extraction_baseline.py \
        -k test_b1_interview_prep_no_procedure -q
"""
from __future__ import annotations

import os

import pytest

from tests_scenarios.post_response_extraction_fixtures import (
    build_b1_interview_prep_no_procedure,
    build_b2_agent_promise_not_extracted,
    build_b3_insight_as_event_not_preference,
    build_b5_project_complaint_no_preference,
    build_p1_correction_extracts_preference,
    build_p2_multichannel_investigation_preference,
)
from tests_scenarios.scenario_runner import ScenarioRunner

_RUN_SCENARIOS = os.getenv("AKASIC_RUN_SCENARIOS") == "1"

_RUNNER = ScenarioRunner()


def _print_post_response_rows(result: object) -> None:
    """诊断辅助：打印所有 post_response 提取条目。"""
    rows = getattr(result, "memory_rows", [])
    pr_rows = [r for r in rows if "@post_response" in str(r.get("source_ref", ""))]
    print(f"\n[diag] post_response rows ({len(pr_rows)}):")
    for r in pr_rows:
        print(f"  [{r.get('memory_type')}] {r.get('summary')}  (src={r.get('source_ref')})")


# ---------------------------------------------------------------------------
# B1：八股文场景 — 技术知识不被误判为 procedure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.scenario_post_response
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_b1_interview_prep_no_procedure() -> None:
    """
    B1：用户背诵 Raft 选主流程（面试八股），post-response worker 不应提取为 procedure。

    这是最常见的误判场景：用户在复习技术知识时，对话内容充满"流程性"描述，
    但这些都是技术事实，不是对 agent 的行为规范要求。

    现象（修复前）：
      用户背诵 Redis Sentinel 选主流程时，worker 将"哨兵通过 Pub/Sub 互相发现"
      提取为 procedure，并在后续面试准备对话中注入该条目，干扰 agent 行为。

    期望（修复后）：
      不提取任何 procedure；若有提取，也只能是 agent 行为规范（如纠错方式），
      而非技术知识本身。
    """
    # 1. 构建场景：用户背诵 Raft 选主，请求 agent 纠错
    spec = build_b1_interview_prep_no_procedure()
    # 2. 运行真实 agent loop + 等待 post-response worker 落库
    result = await _RUNNER.run(spec)
    # 3. judge 验证：检查 memory_rows 中是否存在技术知识被提取为 procedure 的条目
    _print_post_response_rows(result)
    assert result.passed, result.failure_message()


# ---------------------------------------------------------------------------
# B2：Agent 承诺不应被提取为 procedure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.scenario_post_response
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_b2_agent_promise_not_extracted() -> None:
    """
    B2：agent 回复中的自我承诺不应被提取为 procedure。

    现象（修复前）：
      用户抱怨 agent 把简单问题搞复杂，agent 在回复中承诺"下次直接回答"，
      worker 将该承诺提取为 procedure（如"下次这种简单题我就不折腾后台任务了，直接答你"），
      随后该条目被注入到后续技术问答对话中，造成上下文污染。

    期望（修复后）：
      若提取任何条目，summary 不得包含以 agent 第一人称（"我"）表达的自我承诺；
      允许提取用户侧的行为要求（如"简单问题应直接回答"）。
    """
    # 1. 构建场景：用户抱怨 agent 把简单问题复杂化
    spec = build_b2_agent_promise_not_extracted()
    # 2. 运行
    result = await _RUNNER.run(spec)
    # 3. judge 验证：检查 post_response 条目中是否有 agent 第一人称承诺
    _print_post_response_rows(result)
    assert result.passed, result.failure_message()


# ---------------------------------------------------------------------------
# B3：Session-specific 感悟不写成 preference
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.scenario_post_response
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_b3_insight_as_event_not_preference() -> None:
    """
    B3：用户表达认知感悟，不应写成长期 preference。

    现象（修复前）：
      用户说"做笔记就是在给自己建外部知识库"，worker 将其提取为 preference
      "用户认可这种做法的价值"——但这只是用户的一次感悟表达，不是对 agent 的稳定行为要求。

    期望（修复后）：
      不提取 preference；若提取则应为 event（记录用户的一次观点表达）。
    """
    # 1. 构建场景：用户表达关于笔记的认知感悟
    spec = build_b3_insight_as_event_not_preference()
    # 2. 运行
    result = await _RUNNER.run(spec)
    # 3. judge 验证：不应出现 preference 类型的该条目
    _print_post_response_rows(result)
    assert result.passed, result.failure_message()


# ---------------------------------------------------------------------------
# B5：单次项目评价不提取为通用 preference
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.scenario_post_response
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_b5_project_complaint_no_preference() -> None:
    """
    B5：对单个项目的吐槽不应被提取为绑定该项目的通用 preference。

    现象（修复前）：
      用户批评 RuView 项目用数学公式生成动画假冒真实数据，worker 提取了
      "用户对声称能实现人体姿态估计…的项目持怀疑态度"——绑定了具体项目，
      6 个月后该项目已无关，却仍以 preference 形式注入上下文。

    期望（修复后）：
      不提取绑定具体项目的 preference；
      允许提取泛化的方法论（如"推荐项目前应验证其核心功能是否真实可运行"）。
    """
    # 1. 构建场景：用户吐槽 DataVizPro 项目演示造假
    spec = build_b5_project_complaint_no_preference()
    # 2. 运行
    result = await _RUNNER.run(spec)
    # 3. judge 验证：不应出现绑定该项目名称的 preference
    _print_post_response_rows(result)
    assert result.passed, result.failure_message()


# ---------------------------------------------------------------------------
# P1（正向）：纠正 agent 行为 → 正确提取 preference
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.scenario_post_response
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_p1_correction_extracts_preference() -> None:
    """
    P1（正向基线）：用户明确纠正 agent 遗漏步骤，应正确提取为 preference。

    场景：用户指出 agent 讲 HTTPS 握手时遗漏了 TCP 建立连接的步骤，并明确要求
    以后讲这类流程性内容时不能省略前置层。

    期望：提取 1 条 preference，描述"讲解流程时应从底层连接建立开始"。
    这是系统应正确处理的基础场景，用于确保修复 B1–B5 时没有破坏正常提取。
    """
    # 1. 构建场景：用户纠正 agent 讲解 HTTPS 时遗漏了 TCP 层
    spec = build_p1_correction_extracts_preference()
    # 2. 运行
    result = await _RUNNER.run(spec)
    # 3. 硬断言：应有 procedure 条目含连接关键字（来自 post_response）+ judge 语义验证
    _print_post_response_rows(result)
    assert result.passed, result.failure_message()


# ---------------------------------------------------------------------------
# P2（正向）：调查方法论 → 提取泛化 preference
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.scenario_post_response
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_p2_multichannel_investigation_preference() -> None:
    """
    P2（正向基线）：用户要求多渠道调查，应提取泛化方法论 preference。

    场景：agent 只查了 GitHub 就给出结论，用户要求同时查 Hacker News 等第三方渠道。
    用户表达的是一个通用的调查方法论，不是对这个具体项目的吐槽。

    期望：提取 1 条 preference，内容泛化为"调查时应多渠道"；
    summary 中不应出现具体项目名称（StreamForge）。

    这个场景与 B5 形成对照：同样是在项目调查场景，但用户表达的是方法论，
    所以应该提取，而 B5（纯吐槽）则不应提取。
    """
    # 1. 构建场景：用户要求 agent 查资料时不要只看 GitHub
    spec = build_p2_multichannel_investigation_preference()
    # 2. 运行
    result = await _RUNNER.run(spec)
    # 3. 硬断言：应有含"渠道"关键字的 procedure（来自 post_response）+ judge 验证泛化程度
    _print_post_response_rows(result)
    assert result.passed, result.failure_message()
