from __future__ import annotations

import pytest

from tests_scenarios.mobile_isolated_gateway import GatewayFaultController


def test_before_challenge_fault_waits_for_pairing_and_triggers_once() -> None:
    controller = GatewayFaultController("stall_before_challenge")

    assert not controller.claim_before_challenge(has_paired_device=False)
    assert controller.claim_before_challenge(has_paired_device=True)
    assert not controller.claim_before_challenge(has_paired_device=True)
    assert not controller.claim_after_auth()


def test_after_auth_fault_triggers_once() -> None:
    controller = GatewayFaultController("stall_after_auth")

    assert not controller.claim_before_challenge(has_paired_device=True)
    assert controller.claim_after_auth()
    assert not controller.claim_after_auth()


def test_fault_controller_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="未知隔离 Gateway 故障模式"):
        GatewayFaultController("drop_everything")
