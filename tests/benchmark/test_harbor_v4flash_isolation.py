from pathlib import Path

import pytest

from benchmark.harbor_v4flash.isolation import (
    IsolationError,
    compose_project_name,
    validate_isolation,
)


def _container(*, source: str, ports: dict[str, object] | None = None) -> dict[str, object]:
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
            }
        ],
        "ports": ports or {},
    }


def test_compose_project_name_matches_harbor_normalization() -> None:
    assert (
        compose_project_name("Akasic-Bench-V4Flash-Smoke.Name__env")
        == "akasic-bench-v4flash-smoke-name__env"
    )


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
    )

    assert report["status"] == "passed"
    assert report["checked_bind_mounts"] == 1


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
        )
