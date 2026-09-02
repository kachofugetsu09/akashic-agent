import ipaddress
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from benchmark.harbor_v4flash.isolation import (
    IsolationError,
    artifact_digests,
    cleanup_compose_project,
    compose_project_name,
    inspect_compose_project,
    require_storage_capacity,
    reserve_compose_network,
    sha256_file,
    source_tree_digest,
    stop_and_cleanup_compose_project,
    validate_isolation,
)
from benchmark.harbor_v4flash.runtime_volume import RUNTIME_MOUNT_PATH


def _container(
    *,
    source: str,
    ports: dict[str, object] | None = None,
    volume_name: str = "akasic-bench-runtime-v1-fixed",
    volume_rw: bool = False,
) -> dict[str, object]:
    return {
        "id": "container",
        "name": "trial-client-1",
        "image": "task:fixed",
        "status": "running",
        "running": True,
        "project": "akasic-bench-v4flash-smoke__env",
        "mounts": [
            {
                "type": "bind",
                "source": source,
                "destination": "/logs/agent",
                "rw": True,
            },
            {
                "type": "volume",
                "name": volume_name,
                "source": f"/var/lib/docker/volumes/{volume_name}/_data",
                "destination": RUNTIME_MOUNT_PATH,
                "rw": volume_rw,
            },
        ],
        "ports": ports or {},
    }


def test_compose_project_name_matches_harbor_normalization() -> None:
    assert (
        compose_project_name("Akasic-Bench-V4Flash-Smoke.Name__env")
        == "akasic-bench-v4flash-smoke-name__env"
    )


def test_reserve_compose_network_retries_overlapping_subnet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    responses = iter(
        [
            subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr="Pool overlaps with other one on this address space",
            ),
            subprocess.CompletedProcess([], 0, stdout="network-id\n", stderr=""),
        ]
    )

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return next(responses)

    monkeypatch.setattr(subprocess, "run", run)

    network = reserve_compose_network(
        "akasic-bench-v4flash-smoke__env",
        network_pool=ipaddress.IPv4Network("10.240.0.0/29"),
        network_prefix=30,
    )

    assert network["id"] == "network-id"
    assert network["pool"] == "10.240.0.0/29"
    assert commands[0][6] != commands[1][6]


def test_reserve_compose_network_rejects_non_benchmark_owner() -> None:
    with pytest.raises(IsolationError, match="benchmark 前缀"):
        reserve_compose_network("production_default")


def test_storage_capacity_fails_before_new_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = type("Usage", (), {"total": 100, "used": 99, "free": 1})()
    monkeypatch.setattr("shutil.disk_usage", lambda path: usage)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "/var/lib/docker\n", ""
        ),
    )

    with pytest.raises(IsolationError, match="停止调度"):
        require_storage_capacity(
            tmp_path,
            min_runs_free_gib=1,
            min_tmp_free_gib=1,
            min_docker_free_gib=1,
        )


def test_cleanup_only_removes_exact_stopped_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = "akasic-bench-v4flash-smoke__env"
    container = {"id": "container-id", "running": False}
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.isolation.inspect_compose_project",
        lambda name: [container],
    )

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["docker", "network", "inspect"]:
            payload = [{
                "Id": "network-id",
                "Name": "network-name",
                "Labels": {
                    "com.docker.compose.project": project,
                    "akasic.benchmark.managed": "true",
                },
            }]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    result = cleanup_compose_project(
        project,
        expected_containers=[container],
        network={"id": "network-id", "name": "network-name"},
    )

    assert result["status"] == "removed"
    assert ["docker", "container", "rm", "container-id"] in calls
    assert ["docker", "network", "rm", "network-id"] in calls


def test_interruption_cleanup_stops_before_exact_project_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = "akasic-bench-v4flash-smoke__env"
    running = {"id": "container-id", "running": True}
    stopped = {"id": "container-id", "running": False}
    inspections = iter(([running], [stopped]))
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.isolation.inspect_compose_project",
        lambda name: next(inspections),
    )
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.isolation.cleanup_compose_project",
        lambda name, *, expected_containers, network: {
            "status": "removed",
            "container_ids": [expected_containers[0]["id"]],
        },
    )

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "container-id\n", "")

    monkeypatch.setattr(subprocess, "run", run)

    result = stop_and_cleanup_compose_project(
        project,
        network={"id": "network-id", "name": "network-name"},
    )

    assert result["status"] == "removed"
    assert calls == [
        ["docker", "container", "stop", "--time", "10", "container-id"]
    ]


def test_inspect_compose_project_records_immutable_image_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="container-id\n", stderr=""),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    [
                        {
                            "Id": "container-id",
                            "Name": "/trial-main-1",
                            "Image": "sha256:image-id",
                            "Config": {
                                "Image": "task:tag",
                                "Labels": {
                                    "com.docker.compose.project": (
                                        "akasic-bench-v4flash-smoke__env"
                                    )
                                },
                            },
                            "State": {
                                "Status": "exited",
                                "Running": False,
                                "ExitCode": 137,
                                "OOMKilled": False,
                            },
                            "Mounts": [
                                {
                                    "Type": "volume",
                                    "Name": "akasic-bench-runtime-v1-fixed",
                                    "Source": (
                                        "/var/lib/docker/volumes/"
                                        "akasic-bench-runtime-v1-fixed/_data"
                                    ),
                                    "Destination": RUNTIME_MOUNT_PATH,
                                    "RW": False,
                                }
                            ],
                            "HostConfig": {
                                "PortBindings": {},
                                "Memory": 4294967296,
                            },
                        }
                    ]
                ),
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr(subprocess, "run", lambda *_, **__: next(responses))

    containers = inspect_compose_project("akasic-bench-v4flash-smoke__env")

    assert containers[0]["image"] == "task:tag"
    assert containers[0]["image_id"] == "sha256:image-id"
    assert containers[0]["exit_code"] == 137
    assert containers[0]["oom_killed"] is False
    assert containers[0]["memory_limit_bytes"] == 4294967296
    mounts = cast(list[dict[str, object]], containers[0]["mounts"])
    assert mounts[0]["name"] == ("akasic-bench-runtime-v1-fixed")


def test_validate_isolation_accepts_only_trial_bind_mounts(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    logs = trial / "agent"
    logs.mkdir(parents=True)
    project = "akasic-bench-v4flash-smoke__env"

    report = validate_isolation(
        [_container(source=str(logs))],
        project_name=project,
        allowed_bind_root=trial,
        forbidden_host_paths=[tmp_path / "online"],
        allowed_volume_mounts=[("akasic-bench-runtime-v1-fixed", RUNTIME_MOUNT_PATH)],
    )

    assert report["status"] == "passed"
    assert report["checked_bind_mounts"] == 1
    assert report["checked_volume_mounts"] == 1


@pytest.mark.parametrize(
    ("source", "ports"),
    [
        ("/home/huashen/.akashic/workspace", {}),
        ("/var/run/docker.sock", {}),
        ("/tmp/allowed/agent", {"6322/tcp": [{"HostPort": "6322"}]}),
    ],
)
def test_validate_isolation_rejects_host_escape(
    tmp_path: Path,
    source: str,
    ports: dict[str, object],
) -> None:
    allowed = Path("/tmp/allowed")
    project = "akasic-bench-v4flash-smoke__env"

    with pytest.raises(IsolationError):
        validate_isolation(
            [_container(source=source, ports=ports)],
            project_name=project,
            allowed_bind_root=allowed,
            forbidden_host_paths=[Path("/home/huashen/.akashic/workspace")],
            allowed_volume_mounts=[
                ("akasic-bench-runtime-v1-fixed", RUNTIME_MOUNT_PATH)
            ],
        )


@pytest.mark.parametrize(
    ("volume_name", "volume_rw"),
    [
        ("other-volume", False),
        ("akasic-bench-runtime-v1-fixed", True),
    ],
)
def test_validate_isolation_rejects_unapproved_or_writable_volume(
    volume_name: str,
    volume_rw: bool,
) -> None:
    project = "akasic-bench-v4flash-smoke__env"

    with pytest.raises(IsolationError):
        validate_isolation(
            [
                _container(
                    source="/tmp/allowed/agent",
                    volume_name=volume_name,
                    volume_rw=volume_rw,
                )
            ],
            project_name=project,
            allowed_bind_root=Path("/tmp/allowed"),
            forbidden_host_paths=[],
            allowed_volume_mounts=[
                ("akasic-bench-runtime-v1-fixed", RUNTIME_MOUNT_PATH)
            ],
        )


def test_validate_isolation_rejects_missing_runtime_volume() -> None:
    project = "akasic-bench-v4flash-smoke__env"
    container = _container(source="/tmp/allowed/agent")
    mounts = cast(list[dict[str, object]], container["mounts"])
    container["mounts"] = mounts[:1]

    with pytest.raises(IsolationError, match="缺少 allowlist volume"):
        validate_isolation(
            [container],
            project_name=project,
            allowed_bind_root=Path("/tmp/allowed"),
            forbidden_host_paths=[],
            allowed_volume_mounts=[
                ("akasic-bench-runtime-v1-fixed", RUNTIME_MOUNT_PATH)
            ],
        )


def test_artifact_digests_excludes_self_referential_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "campaign-manifest.json"
    trace = tmp_path / "agent" / "trace.jsonl"
    trace.parent.mkdir()
    manifest.write_text('{"state":"prepared"}\n', encoding="utf-8")
    trace.write_text('{"event":"completed"}\n', encoding="utf-8")

    digests = artifact_digests(tmp_path, exclude={manifest})

    assert "campaign-manifest.json" not in digests
    assert digests == {"agent/trace.jsonl": sha256_file(trace)}


def test_source_digest_ignores_gitignored_reports_but_keeps_dirty_overlay(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("reports/\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "tracked.txt", ".gitignore"],
        check=True,
    )
    baseline = source_tree_digest(tmp_path)
    report = tmp_path / "reports" / "gate.json"
    report.parent.mkdir()
    report.write_text("ignored\n", encoding="utf-8")
    assert source_tree_digest(tmp_path) == baseline

    tracked.write_text("dirty\n", encoding="utf-8")
    assert source_tree_digest(tmp_path) != baseline
