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
        "install_publication_state": "latest_ready",
        "parent_runtime": "snapshot-s0",
        "ordinary_runtime_during_validation": "snapshot-s0",
        "validation_runtime": "snapshot-s1",
        "validation_finished_before_parent_release": True,
        "parent_status_before_promote": "in_progress",
        "parent_terminal_status": "completed",
        "validation_turn": {
            "threadId": "programmatic:validation",
            "status": "completed",
            "metadata": {
                "runtime": "latest",
                "inboundMetadata": {
                    "skip_post_memory": True,
                    "disable_memory_writes": True,
                },
            },
            "items": [
                {
                    "type": "toolCall",
                    "data": {
                        "name": "candidate_only_tool",
                        "status": "success",
                        "resultPreview": '{"domain":"ready","snapshot":"snapshot-s1"}',
                    },
                },
                {
                    "type": "toolCall",
                    "data": {
                        "name": "message_push",
                        "status": "success",
                        "resultPreview": "消息已发送",
                    },
                },
            ],
        },
        "candidate_tool_result": {"domain": "ready", "snapshot": "snapshot-s1"},
        "domain_state": "ready",
        "validation_messages": [
            {"role": "user", "content": "validate"},
            {
                "role": "assistant",
                "content": "done",
                "tool_chain": [{"calls": [{"name": "candidate_only_tool"}]}],
            },
        ],
        "recall_session_keys": ["programmatic:validation"],
        "semantic_memory_write_set": [],
        "push_send_sequence": 4,
        "parent_terminal_sequence": 7,
        "push_target_history_before": [],
        "push_target_history_after": [],
        "stable_before_promote": "snapshot-s0",
        "reload_journal_events_before_promote": ["preparing", "latest_ready"],
        "reload_journal_events_after_promote": [
            "preparing",
            "latest_ready",
            "complete",
        ],
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


def test_recursive_plugin_oracle_rejects_parent_terminal_overflow_mutant() -> None:
    _assert_mutant(
        "parent_terminal_status",
        "failed",
        "terminal 未完整送达",
    )


def test_recursive_plugin_oracle_rejects_memory_write_mutant() -> None:
    _assert_mutant(
        "semantic_memory_write_set",
        ["memory:event-1"],
        "写入了语义记忆",
    )


def test_recursive_plugin_oracle_rejects_fake_domain_success_mutant() -> None:
    _assert_mutant("domain_state", "not-ready", "领域状态")


def test_recursive_plugin_oracle_rejects_message_push_blocking_mutant() -> None:
    _assert_mutant("push_send_sequence", 8, "等待了父 session")


def test_recursive_plugin_oracle_rejects_crash_promotion_mutant() -> None:
    _assert_mutant("recovery_promoted_candidate", True, "自动晋升")


def test_recursive_plugin_oracle_rejects_fake_tool_item() -> None:
    mutant = deepcopy(_complete_observation())
    turn = mutant["validation_turn"]
    assert isinstance(turn, dict)
    items = turn["items"]
    assert isinstance(items, list)
    item = items[0]
    assert isinstance(item, dict)
    item["type"] = "assistantMessage"

    with pytest.raises(AssertionError, match="真实 completed tool item"):
        assert_recursive_plugin_self_validation(mutant)
