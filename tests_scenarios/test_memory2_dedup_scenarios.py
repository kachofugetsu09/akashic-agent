"""
Memory2 dedup 端到端场景测试。

每个测试都使用真实 LLM、真实 embedding、真实 SQLite，在隔离 workspace 中运行。
需要设置 AKASIC_RUN_SCENARIOS=1 才会执行。

覆盖三条 dedup 路径：
  - S1: SKIP  — 同义改写偏好，不产生重复记录
  - S2: NONE+merge — 增量更新过程，合并进现有 procedure
  - S3: CREATE — 全新偏好，直接写入
"""
from __future__ import annotations

import os

import pytest

from tests_scenarios.dedup_fixtures import (
    build_dedup_create_new_preference,
    build_dedup_none_merge_procedure,
    build_dedup_skip_synonym_preference,
)
from tests_scenarios.scenario_runner import ScenarioRunner

_RUN_SCENARIOS = os.getenv("AKASIC_RUN_SCENARIOS") == "1"

_DEDUP_RUNNER = ScenarioRunner(
    config_patch={"memory_v2.dedup_enabled": True},
)


@pytest.mark.asyncio
@pytest.mark.scenario_dedup
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_dedup_skip_synonym_preference() -> None:
    """
    S1: 同义改写偏好 → SKIP，不产生重复 active 记录。

    场景：预置"简洁回复"偏好 + 3 条无关噪音事件。
    用户再次表达同一偏好（措辞不同但语义相同）。
    期望：post-response worker 判断 SKIP，原偏好仍 active，不新增重复条目。
    """
    spec = build_dedup_skip_synonym_preference()
    result = await _DEDUP_RUNNER.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.scenario_dedup
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_dedup_none_merge_procedure() -> None:
    """
    S2: 增量过程更新 → NONE+merge，新步骤合并进现有 procedure。

    场景：预置 Steam MCP 查询流程 + 2 条噪音 procedure + 3 条噪音事件。
    用户补充一个新步骤（先判断区服），该步骤与现有流程高度相关。
    期望：DedupDecider 判 NONE → merge，合并后 active procedure 同时含
    steam_mcp 和区服关键字；不产生两条独立的 Steam procedure。
    """
    spec = build_dedup_none_merge_procedure()
    result = await _DEDUP_RUNNER.run(spec)
    assert result.passed, result.failure_message()


@pytest.mark.asyncio
@pytest.mark.scenario_dedup
@pytest.mark.scenario_live
@pytest.mark.skipif(not _RUN_SCENARIOS, reason="设置 AKASIC_RUN_SCENARIOS=1 后再执行真实场景测试")
async def test_dedup_create_new_preference() -> None:
    """
    S3: 全新偏好 → CREATE，无相似项直接写入。

    场景：无中文偏好种子，只有无关噪音偏好（时间管理、简洁）和 3 条噪音事件。
    用户明确要求"以后只用中文"。
    期望：DedupDecider 找不到高相似项，判 CREATE，写入新的中文偏好。
    原噪音偏好不受影响，仍为 active。
    """
    spec = build_dedup_create_new_preference()
    result = await _DEDUP_RUNNER.run(spec)
    assert result.passed, result.failure_message()
