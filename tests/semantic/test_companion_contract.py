from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests_scenarios.contracts.oracles import assert_companion_capacity
from tests_scenarios.contracts.oracles import assert_companion_contract
from tests_scenarios.contracts.oracles import assert_control_replay_contract
from tests_scenarios.contracts.oracles import assert_dashboard_contract
from tests_scenarios.contracts.oracles import assert_external_io_contract
from tests_scenarios.contracts.oracles import assert_content_wake_delivery_contract
from tests_scenarios.contracts.oracles import assert_peer_removed
from tests_scenarios.contracts.oracles import assert_receipt_contract
from tests_scenarios.contracts.oracles import assert_schedule_capacity_contract
from tests_scenarios.contracts.oracles import assert_shell_contract
from tests_scenarios.contracts.oracles import assert_tool_context_contract

ROOT = Path(__file__).resolve().parents[2]


def _gate_module():
    path = ROOT / "docker" / "debug" / "gate.py"
    spec = importlib.util.spec_from_file_location("companion_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_companion_contract_catalog() -> None:
    """所有 Companion scenario 都声明观察字段、失败分类和 mutant。"""
    gate = _gate_module()
    catalog = gate.load_catalog()
    required = {
        "companion_tool_context_contract",
        "companion_external_io_contract",
        "companion_peer_removal_contract",
        "content_wake_delivery_contract",
        "companion_schedule_capacity_contract",
        "companion_mobile_receipt_contract",
        "companion_shell_admission_contract",
        "companion_control_replay_contract",
        "companion_dashboard_value_contract",
    }
    assert required <= catalog.scenarios.keys()
    for scenario_id in required:
        scenario = catalog.scenarios[scenario_id]
        assert scenario.observes
        assert scenario.mutants
    assert catalog.groups["companion_dashboard"].paths == ("private_runtime",)
    assert catalog.groups["companion_peer_removal"].deleted_paths == (
        "agent/peer_agent/**",
    )


def test_tool_context_rejects_origin_override_mutant() -> None:
    with pytest.raises(AssertionError, match="可恢复失败错误结束"):
        assert_companion_contract(
            {
                "failure_semantics": "operation_rejected",
                "runtime_alive": False,
                "committed_result": True,
                "live_event_dropped": False,
            }
        )
    with pytest.raises(AssertionError, match="覆盖 runtime provenance"):
        assert_tool_context_contract({"origin_overridden": True})


def test_external_io_rejects_ownerless_spill_mutant() -> None:
    with pytest.raises(AssertionError, match="物理减少缺少 owner"):
        assert_companion_contract(
            {
                "failure_semantics": "cleanup_degraded",
                "runtime_alive": True,
                "committed_result": True,
                "live_event_dropped": False,
                "physical_reduction": True,
                "physical_reduction_owner": "",
                "recovery_evidence": "",
            }
        )
    with pytest.raises(AssertionError, match="execution owner"):
        assert_external_io_contract({"spill_owner": ""})


def test_peer_surface_removal_rejects_surviving_route_mutant() -> None:
    with pytest.raises(AssertionError, match="可恢复失败错误结束"):
        assert_companion_contract(
            {
                "failure_semantics": "operation_rejected",
                "runtime_alive": False,
                "committed_result": True,
                "live_event_dropped": False,
            }
        )
    with pytest.raises(AssertionError, match="Peer route"):
        assert_peer_removed({"peer_route_registered": True})


def test_content_delivery_rejects_early_source_ack_mutant() -> None:
    with pytest.raises(AssertionError, match="早于真实送达"):
        assert_content_wake_delivery_contract({"source_ack_before_delivery": True})


def test_schedule_capacity_rejects_unbounded_add_mutant() -> None:
    with pytest.raises(AssertionError, match="既有状态"):
        assert_companion_capacity(
            {
                "capacity_rejected": True,
                "existing_state_changed": True,
                "runtime_alive": True,
            }
        )
    with pytest.raises(AssertionError, match="超过默认 10"):
        assert_schedule_capacity_contract(
            {"active_jobs": 11, "operation_accepted": True}
        )


def test_receipt_retention_rejects_valid_delete_mutant() -> None:
    with pytest.raises(AssertionError, match="物理减少缺少 owner"):
        assert_companion_contract(
            {
                "failure_semantics": "cleanup_degraded",
                "runtime_alive": True,
                "committed_result": True,
                "live_event_dropped": False,
                "physical_reduction": True,
                "physical_reduction_owner": "",
                "recovery_evidence": "",
            }
        )
    with pytest.raises(AssertionError, match="有效 receipt"):
        assert_receipt_contract({"valid_receipt_deleted": True})


def test_shell_cleanup_rejects_turn_rewrite_mutant() -> None:
    with pytest.raises(AssertionError, match="已提交结果"):
        assert_companion_contract(
            {
                "failure_semantics": "cleanup_degraded",
                "runtime_alive": True,
                "committed_result": False,
                "live_event_dropped": False,
            }
        )
    with pytest.raises(AssertionError, match="已提交 turn"):
        assert_shell_contract(
            {"status": "failed", "final_response": "ok", "dispatch_count": 1}
        )


def test_control_replay_rejects_live_drop_mutant() -> None:
    with pytest.raises(AssertionError, match="live subscriber"):
        assert_companion_contract(
            {
                "failure_semantics": "cleanup_degraded",
                "runtime_alive": True,
                "committed_result": True,
                "live_event_dropped": True,
            }
        )
    with pytest.raises(AssertionError, match="live subscriber"):
        assert_control_replay_contract({"live_event_dropped": True})


def test_dashboard_rejects_html_sink_mutant() -> None:
    with pytest.raises(AssertionError, match="已提交结果"):
        assert_companion_contract(
            {
                "failure_semantics": "degraded_continuation",
                "runtime_alive": True,
                "committed_result": False,
                "live_event_dropped": False,
            }
        )
    with pytest.raises(AssertionError, match="HTML sink"):
        assert_dashboard_contract(
            {"html_sink": True, "invalid_efficiency_display": "--"}
        )
