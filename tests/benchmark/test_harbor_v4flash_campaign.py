import json
from pathlib import Path

import pytest

from benchmark.harbor_v4flash.campaign import (
    CampaignGateError,
    find_open_concurrency_gate,
    plan_diagnostic_wave,
    task_slug,
    validate_campaign_request,
)


def test_campaign_rejects_more_than_three_concurrent_tasks(tmp_path: Path) -> None:
    tasks = [tmp_path / "one", tmp_path / "two"]
    for task in tasks:
        task.mkdir()

    with pytest.raises(ValueError, match="1 到 3"):
        validate_campaign_request(tasks, 4)


def test_campaign_rejects_duplicate_task_instances(tmp_path: Path) -> None:
    task = tmp_path / "same"
    task.mkdir()

    with pytest.raises(ValueError, match="重复"):
        validate_campaign_request([task, task], 2)


def test_open_gate_requires_completed_stopped_isolated_smoke(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "akasic-bench-v4flash-smoke-one"
    trial.mkdir()
    manifest = trial / "campaign-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "state": "completed",
                "trial_name": "smoke-one",
                "source": {"digest_after": "sha256:source"},
                "online": {"status": "passed"},
                "docker": {"all_stopped": True},
                "concurrency_gate": {"opened": True, "max_concurrent": 3},
            }
        ),
        encoding="utf-8",
    )

    gate = find_open_concurrency_gate(tmp_path)

    assert gate["manifest"] == str(manifest)
    assert gate["source_digest"] == "sha256:source"


def test_open_gate_fails_closed_without_smoke(tmp_path: Path) -> None:
    with pytest.raises(CampaignGateError):
        find_open_concurrency_gate(tmp_path)


def test_task_slug_is_bounded_and_docker_safe(tmp_path: Path) -> None:
    task = tmp_path / ("UPPER_case.with spaces-" + "x" * 80)
    assert task_slug(task) == "upper-case-with-spaces-" + "x" * 25


def test_diagnostic_wave_uses_two_discovery_and_one_validation(
    tmp_path: Path,
) -> None:
    discovery = [tmp_path / f"d{index}" for index in range(4)]
    validation = [tmp_path / f"v{index}" for index in range(2)]

    scheduled, pending = plan_diagnostic_wave(discovery, validation)

    assert [item["mode"] for item in scheduled] == [
        "validation",
        "discovery",
        "discovery",
    ]
    assert [(item["mode"], item["task"].name) for item in pending] == [
        ("discovery", "d2"),
        ("discovery", "d3"),
        ("validation", "v1"),
    ]


def test_diagnostic_wave_lends_empty_validation_slot_to_discovery(
    tmp_path: Path,
) -> None:
    discovery = [tmp_path / f"d{index}" for index in range(4)]

    scheduled, pending = plan_diagnostic_wave(discovery, [])

    assert [item["task"].name for item in scheduled] == ["d0", "d1", "d2"]
    assert [item["task"].name for item in pending] == ["d3"]


def test_diagnostic_wave_never_runs_three_validations(tmp_path: Path) -> None:
    validation = [tmp_path / f"v{index}" for index in range(3)]

    scheduled, pending = plan_diagnostic_wave([], validation)

    assert [item["task"].name for item in scheduled] == ["v0"]
    assert [item["task"].name for item in pending] == ["v1", "v2"]
