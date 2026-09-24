from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from tests.semantic.recursive_plugin_fixture import (
    read_child,
    read_failed_selection,
    read_programmatic,
    read_push,
    read_selection,
)
from tests_scenarios.contracts.oracles import assert_recursive_plugin_self_validation


@pytest.fixture(scope="module")
def real_observation(tmp_path_factory) -> dict[str, object]:
    """Collect one successful observation from each real owner boundary."""

    root = tmp_path_factory.mktemp("recursive-p0")
    patch = pytest.MonkeyPatch()

    async def collect() -> dict[str, object]:
        return {
            "selection": await read_selection(root),
            "failure": await read_failed_selection(root, patch),
            "programmatic": await read_programmatic(root, patch),
            "child": await read_child(root),
            "push": await read_push(root),
        }

    try:
        return asyncio.run(collect())
    finally:
        patch.undo()


def test_recursive_plugin_self_validation_reads_real_owners(real_observation) -> None:
    assert_recursive_plugin_self_validation(real_observation)


@pytest.mark.parametrize(
    ("section", "field", "wrong", "message"),
    [
        ("selection", "after_compile_ref", "wrong-ref", "编译失败"),
        ("failure", "failed_fiber", "active", "未 ACTIVE"),
        ("failure", "recovered_before_retry", True, "误报为 A 已恢复"),
        ("failure", "scope_retained", False, "cleanup owner"),
        ("programmatic", "eligible_learning", "excluded", "学习资格"),
        ("child", "domain_file", "missing", "领域效果"),
        ("child", "parent_terminal", "failed", "父 Session terminal"),
        ("push", "new_target_bodies", (), "独立 MessagePush Output"),
    ],
    ids=["compile-selection", "selected-not-active", "no-false-recovery", "cleanup-owner", "admission", "domain-effect", "parent-terminal", "push-output"],
)
def test_recursive_plugin_oracle_rejects_changed_real_observation(
    real_observation, section: str, field: str, wrong: object, message: str,
) -> None:
    mutant = deepcopy(real_observation)
    mutant[section][field] = wrong
    with pytest.raises(AssertionError, match=message):
        assert_recursive_plugin_self_validation(mutant)
