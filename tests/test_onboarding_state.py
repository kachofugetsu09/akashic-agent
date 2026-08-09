from __future__ import annotations

import json
from pathlib import Path

import pytest

from bootstrap.onboarding_state import (
    advance_onboarding,
    complete_onboarding,
    go_back_onboarding,
    onboarding_state_path,
    read_onboarding_state,
    start_onboarding,
)


def test_skip_is_persisted_as_an_explicit_resumable_decision(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"

    state = start_onboarding(workspace, model_configured=False)
    assert state.step == "model"

    state = advance_onboarding(workspace, expected_step="model")
    assert state.step == "memory"
    state = advance_onboarding(
        workspace,
        expected_step="memory",
        decision="skipped",
    )
    assert state.step == "channel"
    assert state.memory_decision == "skipped"

    resumed = read_onboarding_state(
        workspace,
        model_configured=True,
        memory_configured=True,
        channel_configured=True,
    )
    assert resumed.step == "channel"
    assert resumed.memory_decision == "skipped"
    assert resumed.channel_decision == "pending"

    state = advance_onboarding(
        workspace,
        expected_step="channel",
        decision="configured",
    )
    assert complete_onboarding(workspace).completed is True
    assert onboarding_state_path(workspace).stat().st_mode & 0o777 == 0o600


def test_explicit_state_overrides_legacy_config_inference(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _ = start_onboarding(workspace, model_configured=True)

    state = read_onboarding_state(
        workspace,
        model_configured=True,
        memory_configured=True,
        channel_configured=True,
    )

    assert state.step == "model"
    assert state.completed is False
    assert state.memory_decision == "pending"


def test_back_clears_only_decisions_after_the_destination(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _ = start_onboarding(workspace, model_configured=True)
    _ = advance_onboarding(workspace, expected_step="model")
    _ = advance_onboarding(
        workspace,
        expected_step="memory",
        decision="skipped",
    )
    state = go_back_onboarding(workspace)
    assert state.step == "memory"
    assert state.memory_decision == "pending"

    _ = advance_onboarding(
        workspace,
        expected_step="memory",
        decision="configured",
    )
    _ = advance_onboarding(
        workspace,
        expected_step="channel",
        decision="skipped",
    )
    state = go_back_onboarding(workspace)
    assert state.step == "channel"
    assert state.memory_decision == "configured"
    assert state.channel_decision == "pending"


def test_legacy_workspace_without_state_file_keeps_completed_status(
    tmp_path: Path,
) -> None:
    state = read_onboarding_state(
        tmp_path / "workspace",
        model_configured=True,
        memory_configured=True,
        channel_configured=True,
    )

    assert state.completed is True
    assert state.memory_decision == "configured"
    assert state.channel_decision == "configured"


def test_invalid_or_out_of_order_state_fails_loudly(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _ = start_onboarding(workspace, model_configured=False)
    with pytest.raises(ValueError, match="不能提交 memory"):
        advance_onboarding(
            workspace,
            expected_step="memory",
            decision="skipped",
        )

    onboarding_state_path(workspace).write_text(
        json.dumps(
            {
                "version": 1,
                "step": "done",
                "completed": False,
                "memory_decision": "skipped",
                "channel_decision": "pending",
                "updated_at": "2026-08-09T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="可选步骤尚未完成"):
        read_onboarding_state(
            workspace,
            model_configured=True,
            memory_configured=False,
            channel_configured=False,
        )
