from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from docker.debug.react_core_fixture import receipt_differences, run_fixture


@pytest.mark.asyncio
async def test_react_core_fixture_records_real_turn_boundary(tmp_path: Path) -> None:
    receipt = await run_fixture(tmp_path)

    assert receipt["scenario"] == "scoped-turn-completes"
    assert receipt["lifecycle"] == [
        "scope.fork",
        "scope.child.release",
        "scope.owner.release",
    ]
    assert receipt["terminal"] == {
        "turnId": receipt["accepted"]["turnId"],  # type: ignore[index]
        "status": "completed",
        "response": "fixture:done",
    }
    assert receipt["state"] == {"turnRows": 1, "statuses": ["completed"]}
    assert receipt["effects"] == []


@pytest.mark.asyncio
async def test_receipt_comparator_normalizes_only_registered_identity(
    tmp_path: Path,
) -> None:
    receipt = await run_fixture(tmp_path / "first")
    replay = await run_fixture(tmp_path / "second")
    assert receipt_differences(receipt, replay) == []

    equivalent = deepcopy(receipt)
    equivalent["accepted"]["turnId"] = "turn:other"  # type: ignore[index]
    equivalent["terminal"]["turnId"] = "turn:other"  # type: ignore[index]

    assert receipt_differences(receipt, equivalent) == []

    mutant = deepcopy(equivalent)
    mutant["providerRequests"][0]["input"] = "mutated"  # type: ignore[index]
    assert receipt_differences(receipt, mutant) == ["$.providerRequests[0].input"]
