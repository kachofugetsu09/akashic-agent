import json
from pathlib import Path

import pytest

from benchmark.harbor_v4flash.campaign import (
    CampaignGateError,
    find_open_concurrency_gate,
    task_slug,
    validate_campaign_request,
)


def test_campaign_accepts_four_concurrent_tasks(tmp_path: Path) -> None:
    tasks = [tmp_path / "one", tmp_path / "two"]
    for task in tasks:
        task.mkdir()

    validate_campaign_request(tasks, 4)


def test_campaign_rejects_more_than_four_concurrent_tasks(tmp_path: Path) -> None:
    tasks = [tmp_path / "one", tmp_path / "two"]
    for task in tasks:
        task.mkdir()

    with pytest.raises(ValueError, match="1 到 4"):
        validate_campaign_request(tasks, 5)


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
                "concurrency_gate": {"opened": True, "max_concurrent": 4},
            }
        ),
        encoding="utf-8",
    )

    gate = find_open_concurrency_gate(
        tmp_path,
        expected_source_digest="sha256:source",
    )

    assert gate["manifest"] == str(manifest)
    assert gate["source_digest"] == "sha256:source"


def test_open_gate_fails_closed_without_smoke(tmp_path: Path) -> None:
    with pytest.raises(CampaignGateError):
        find_open_concurrency_gate(
            tmp_path,
            expected_source_digest="sha256:source",
        )


def test_open_gate_rejects_smoke_from_different_source(tmp_path: Path) -> None:
    trial = tmp_path / "akasic-bench-v4flash-smoke-stale"
    trial.mkdir()
    (trial / "campaign-manifest.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "trial_name": "smoke-stale",
                "source": {"digest_after": "sha256:old-source"},
                "online": {"status": "passed"},
                "docker": {"all_stopped": True},
                "concurrency_gate": {"opened": True, "max_concurrent": 4},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CampaignGateError, match="sha256:new-source"):
        find_open_concurrency_gate(
            tmp_path,
            expected_source_digest="sha256:new-source",
        )


def test_open_gate_rejects_old_three_concurrent_authorization(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "akasic-bench-v4flash-smoke-old-limit"
    trial.mkdir()
    (trial / "campaign-manifest.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "trial_name": "smoke-old-limit",
                "source": {"digest_after": "sha256:source"},
                "online": {"status": "passed"},
                "docker": {"all_stopped": True},
                "concurrency_gate": {"opened": True, "max_concurrent": 3},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CampaignGateError, match="concurrency=4"):
        find_open_concurrency_gate(
            tmp_path,
            expected_source_digest="sha256:source",
        )


def test_task_slug_is_bounded_and_docker_safe(tmp_path: Path) -> None:
    task = tmp_path / ("UPPER_case.with spaces-" + "x" * 80)
    assert task_slug(task) == "upper-case-with-spaces-" + "x" * 25
