from __future__ import annotations

from copy import deepcopy

import pytest

from tests_scenarios.contracts.oracles import (
    assert_recursive_plugin_self_validation,
)


def _complete_observation() -> dict[str, object]:
    return {
        "stable_snapshot": "snapshot-s0",
        "candidate_snapshot": "snapshot-s1",
        "install_returned_latest_ready": True,
        "parent_runtime": "snapshot-s0",
        "ordinary_runtime_during_validation": "snapshot-s0",
        "validation_runtime": "snapshot-s1",
        "validation_finished_before_parent_release": True,
        "candidate_tool_item": {
            "type": "toolCall",
            "name": "candidate_only_tool",
            "status": "success",
        },
        "tool_result_matches_domain_state": True,
        "validation_session_persisted": True,
        "recall_observed": True,
        "semantic_memory_write_set": [],
        "push_send_sequence": 4,
        "parent_terminal_sequence": 7,
        "push_receipt_in_caller_trace": True,
        "push_target_history_delta": 0,
        "recovered_stable_snapshot": "snapshot-s0",
        "recovered_latest_snapshot": "snapshot-s1",
        "recovery_promoted_candidate": False,
        "stable_after_promote": "snapshot-s1",
        "parent_runtime_after_promote": "snapshot-s0",
    }


def test_recursive_plugin_self_validation_accepts_complete_evidence() -> None:
    assert_recursive_plugin_self_validation(_complete_observation())


def _assert_mutant(
    field: str,
    value: object,
    error: str,
) -> None:
    mutant = deepcopy(_complete_observation())
    mutant[field] = value

    with pytest.raises(AssertionError, match=error):
        assert_recursive_plugin_self_validation(mutant)


def test_recursive_plugin_oracle_rejects_stable_misbinding_mutant() -> None:
    _assert_mutant("validation_runtime", "snapshot-s0", "没有绑定 latest")


def test_recursive_plugin_oracle_rejects_global_lock_mutant() -> None:
    _assert_mutant(
        "validation_finished_before_parent_release",
        False,
        "全局锁",
    )


def test_recursive_plugin_oracle_rejects_memory_write_mutant() -> None:
    _assert_mutant(
        "semantic_memory_write_set",
        ["memory:event-1"],
        "写入了语义记忆",
    )


def test_recursive_plugin_oracle_rejects_fake_domain_success_mutant() -> None:
    _assert_mutant("tool_result_matches_domain_state", False, "领域状态")


def test_recursive_plugin_oracle_rejects_message_push_blocking_mutant() -> None:
    _assert_mutant("push_send_sequence", 8, "等待了父 session")


def test_recursive_plugin_oracle_rejects_crash_promotion_mutant() -> None:
    _assert_mutant("recovery_promoted_candidate", True, "自动晋升")


def test_recursive_plugin_oracle_rejects_fake_tool_item() -> None:
    mutant = deepcopy(_complete_observation())
    mutant["candidate_tool_item"] = {
        "type": "assistantMessage",
        "name": "candidate_only_tool",
        "status": "success",
    }

    with pytest.raises(AssertionError, match="真实 completed tool item"):
        assert_recursive_plugin_self_validation(mutant)
