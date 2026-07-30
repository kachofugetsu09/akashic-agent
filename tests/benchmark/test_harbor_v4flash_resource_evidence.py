import json
from pathlib import Path

import pytest

from benchmark.harbor_v4flash.resource_evidence import (
    RESOURCE_EVIDENCE_FILENAME,
    load_resource_evidence,
    parse_resource_probe_output,
    resource_probe_command,
)


def test_resource_probe_command_reads_only_fixed_cgroup_memory_files() -> None:
    command = resource_probe_command()

    assert "/sys/fs/cgroup/memory.max" in command
    assert "/sys/fs/cgroup/memory.current" in command
    assert "/sys/fs/cgroup/memory.events" in command
    assert "/proc/" not in command
    assert "journalctl" not in command
    assert "docker" not in command


def test_resource_probe_classifies_cgroup_oom_kill_as_resource_limit() -> None:
    evidence = parse_resource_probe_output("""
cgroup_version=2
@@memory.max
4294967296
@@memory.current
1073741824
@@memory.events
low 0
high 0
max 532
oom 3
oom_kill 1
oom_group_kill 0
@@memory.peak
4294967296
@@memory.events.local
low 0
high 0
max 532
oom 3
oom_kill 1
oom_group_kill 0
""".lstrip())

    assert evidence["status"] == "collected"
    assert evidence["classification"] == "resource_limit"
    memory = evidence["cgroup"]["memory"]  # type: ignore[index]
    assert memory["limit_bytes"] == 4294967296
    assert memory["current_bytes"] == 1073741824
    assert memory["peak_bytes"] == 4294967296
    assert memory["events"]["oom_kill"] == 1


def test_resource_probe_classifies_oom_without_kill_as_resource_limit() -> None:
    evidence = parse_resource_probe_output("""
cgroup_version=2
@@memory.max
max
@@memory.current
2048
@@memory.events
low 0
high 0
max 1
oom 1
oom_kill 0
oom_group_kill 0
""".lstrip())

    assert evidence["classification"] == "resource_limit"
    assert evidence["cgroup"]["memory"]["limit_bytes"] is None  # type: ignore[index]


@pytest.mark.parametrize(
    "output",
    [
        "",
        "cgroup_version=1\n",
        "cgroup_version=2\n@@memory.max\n4096\n",
        (
            "cgroup_version=2\n@@memory.max\n4096\n"
            "@@memory.current\ninvalid\n@@memory.events\noom_kill 1\n"
        ),
        (
            "cgroup_version=2\n@@memory.max\n4096\n"
            "@@memory.current\n1024\n@@memory.events\nnot-valid\n"
        ),
    ],
)
def test_resource_probe_rejects_incomplete_or_malformed_evidence(
    output: str,
) -> None:
    with pytest.raises(ValueError):
        parse_resource_probe_output(output)


def test_missing_or_corrupt_resource_artifact_is_explicit(
    tmp_path: Path,
) -> None:
    path = tmp_path / RESOURCE_EVIDENCE_FILENAME

    missing = load_resource_evidence(path)
    assert missing["status"] == "unavailable"
    assert missing["classification"] == "unknown"

    path.write_text("{not-json", encoding="utf-8")
    corrupt = load_resource_evidence(path)
    assert corrupt["status"] == "collection_failed"
    assert corrupt["classification"] == "unknown"


def test_load_resource_evidence_preserves_valid_failure_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / RESOURCE_EVIDENCE_FILENAME
    path.write_text(
        json.dumps(
            {
                "schema": "akasic.container-resource.v1",
                "status": "collection_failed",
                "classification": "unknown",
                "error": {"type": "RuntimeError", "message": "probe failed"},
            }
        ),
        encoding="utf-8",
    )

    evidence = load_resource_evidence(path)

    assert evidence["status"] == "collection_failed"
    assert evidence["error"]["type"] == "RuntimeError"  # type: ignore[index]
