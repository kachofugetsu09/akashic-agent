from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from agent.workloads.client import UnixWorkloadController
from agent.workloads.controller import WorkloadControllerServer
from agent.workloads.model import (
    WorkloadLease,
    WorkloadStartRequest,
    workload_spec_digest,
)


class _FakeEngine:
    def __init__(self) -> None:
        self.container: dict[str, object] | None = None
        self.create_body: dict[str, object] | None = None
        self.fail_next_start = False
        self.lose_create_response = False
        self.crash_after_delete = False

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
        expected: frozenset[int],
    ) -> object:
        _ = expected
        if method == "GET" and path.startswith("/containers/json"):
            if self.container is None:
                return []
            if "all=0" in path and not self._running():
                return []
            return [{"Id": "container-1"}]
        if method == "POST" and path.startswith("/containers/create"):
            assert body is not None
            self.create_body = body
            name = parse_qs(urlsplit(path).query)["name"][0]
            host = body["HostConfig"]
            assert isinstance(host, dict)
            self.container = {
                "Id": "container-1",
                "Name": name,
                "Config": {
                    "Labels": body["Labels"],
                    "Image": body["Image"],
                    "Cmd": body.get("Cmd"),
                    "User": body["User"],
                    "ExposedPorts": body["ExposedPorts"],
                },
                "HostConfig": dict(host),
                "NetworkSettings": {"Networks": {host["NetworkMode"]: {}}},
                "State": {"Running": False},
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": value.rsplit(":", 2)[0],
                        "Destination": value.rsplit(":", 2)[1],
                        "RW": value.rsplit(":", 2)[2] == "rw",
                    }
                    for value in host["Binds"]
                ],
            }
            if self.lose_create_response:
                self.lose_create_response = False
                raise RuntimeError("create response lost")
            return {"Id": "container-1"}
        if method == "GET" and path.endswith("/json"):
            return self.container
        if method == "POST" and path.endswith("/start"):
            if self.fail_next_start:
                self.fail_next_start = False
                raise RuntimeError("start failed")
            self._state()["Running"] = True
            return None
        if method == "POST" and "/stop?" in path:
            self._state()["Running"] = False
            return None
        if method == "DELETE":
            self.container = None
            if self.crash_after_delete:
                self.crash_after_delete = False
                raise SystemExit("controller crashed after delete")
            return None
        raise AssertionError(f"unexpected Docker call: {method} {path}")

    def _state(self) -> dict[str, object]:
        assert self.container is not None
        state = self.container["State"]
        assert isinstance(state, dict)
        return state

    def _running(self) -> bool:
        return self._state()["Running"] is True


def _request(
    workspace_id: str,
    generation_id: str,
    *,
    mode: str = "formal",
) -> WorkloadStartRequest:
    image = "example.invalid/worker@sha256:" + "b" * 64
    ports = (("gateway", 8080),)
    data = (("state", "/data", True),)
    health = ("gateway", "/health", 30.0)
    limits = (128, 1.0, 64)
    digest = workload_spec_digest(
        plugin_id="fixture",
        workload="worker",
        image=image,
        command=("serve",),
        ports=ports,
        data=data,
        health=health,
        limits=limits,
    )
    return WorkloadStartRequest(
        workspace_id=workspace_id,
        plugin_id="fixture",
        workload="worker",
        mode=mode,  # type: ignore[arg-type]
        transaction_id=generation_id,
        generation_id=generation_id,
        spec_digest=digest,
        image=image,
        command=("serve",),
        ports=ports,
        data=data,
        health=health,
        limits=limits,
    )


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "plugin-data").mkdir(parents=True)
    (workspace / "runtime/plugin-validation").mkdir(parents=True)
    return workspace


@pytest.mark.asyncio
async def test_controller_adopt_moves_the_only_stop_lease(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    socket_path = tmp_path / "run" / "controller.sock"
    server = WorkloadControllerServer(
        workspace=workspace,
        socket_path=socket_path,
        docker_socket=tmp_path / "docker.sock",
        state_path=tmp_path / "state" / "leases.json",
        network="test-network",
        allowed_uid=os.getuid(),
        socket_gid=os.getgid(),
        workload_uid=os.getuid(),
        workload_gid=os.getgid(),
        socket_uid=os.getuid(),
    )
    fake = _FakeEngine()
    server._engine = fake  # pyright: ignore[reportPrivateUsage]
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)
    client = UnixWorkloadController(socket_path, timeout_seconds=5)
    workspace_id = server._workspace_id  # pyright: ignore[reportPrivateUsage]

    try:
        first = await client.start(_request(workspace_id, "fixture:a:1"))
        assert fake.create_body is not None
        assert (
            fake.create_body["User"]
            == f"{workspace.stat().st_uid}:{workspace.stat().st_gid}"
        )
        state_dir = workspace / "plugin-data" / "fixture-builtin" / "state"
        assert (state_dir.stat().st_uid, state_dir.stat().st_gid) == (
            os.getuid(),
            os.getgid(),
        )
        second = await client.start(_request(workspace_id, "fixture:a:2"))

        assert second.adopted_from_generation == "fixture:a:1"
        with pytest.raises(RuntimeError, match="过期|adopt"):
            await client.stop(first.lease)
        stopped = await client.stop(second.lease)
        assert stopped.container_absent
        assert stopped.mounts_released
        assert fake.container is None
        assert await client.stop(second.lease) == stopped

        candidate = await client.start(
            _request(workspace_id, "fixture:candidate:1", mode="candidate")
        )
        cleaned = await client.cleanup_candidates(workspace_id)
        assert tuple(item.lease for item in cleaned) == (candidate.lease,)
        assert fake.container is None
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_controller_removes_a_new_container_when_start_fails(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    server = WorkloadControllerServer(
        workspace=workspace,
        socket_path=tmp_path / "run" / "controller.sock",
        docker_socket=tmp_path / "docker.sock",
        state_path=tmp_path / "state" / "leases.json",
        network="test-network",
        allowed_uid=os.getuid(),
        socket_gid=os.getgid(),
        workload_uid=os.getuid(),
        workload_gid=os.getgid(),
        socket_uid=os.getuid(),
    )
    fake = _FakeEngine()
    fake.fail_next_start = True
    server._engine = fake  # pyright: ignore[reportPrivateUsage]
    request = _request(
        server._workspace_id,  # pyright: ignore[reportPrivateUsage]
        "fixture:failed:1",
    )

    with pytest.raises(RuntimeError, match="start failed"):
        await server._start(request)  # pyright: ignore[reportPrivateUsage]

    assert fake.container is None
    assert server._leases == {}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drift",
    (
        "published_port",
        "extra_network",
        "extra_mount",
        "host_pid",
        "device",
        "restart",
    ),
)
async def test_controller_rejects_actual_container_config_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    workspace = _workspace(tmp_path)
    server = WorkloadControllerServer(
        workspace=workspace,
        socket_path=tmp_path / "run" / "controller.sock",
        docker_socket=tmp_path / "docker.sock",
        state_path=tmp_path / "state" / "leases.json",
        network="test-network",
        allowed_uid=os.getuid(),
        socket_gid=os.getgid(),
        workload_uid=os.getuid(),
        workload_gid=os.getgid(),
        socket_uid=os.getuid(),
    )
    fake = _FakeEngine()
    server._engine = fake  # pyright: ignore[reportPrivateUsage]
    workspace_id = server._workspace_id  # pyright: ignore[reportPrivateUsage]
    await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(workspace_id, "fixture:drift:1")
    )
    assert fake.container is not None
    if drift == "published_port":
        host = fake.container["HostConfig"]
        assert isinstance(host, dict)
        host["PortBindings"] = {"8080/tcp": [{"HostPort": "8080"}]}
    elif drift == "extra_network":
        network_settings = fake.container["NetworkSettings"]
        assert isinstance(network_settings, dict)
        networks = network_settings["Networks"]
        assert isinstance(networks, dict)
        networks["extra"] = {}
    elif drift == "extra_mount":
        mounts = fake.container["Mounts"]
        assert isinstance(mounts, list)
        mounts.append(
            {
                "Type": "volume",
                "Source": "/other",
                "Destination": "/other",
                "RW": True,
            }
        )
    else:
        host = fake.container["HostConfig"]
        assert isinstance(host, dict)
        if drift == "host_pid":
            host["PidMode"] = "host"
        elif drift == "device":
            host["Devices"] = [{"PathOnHost": "/dev/null"}]
        else:
            host["RestartPolicy"] = {"Name": "always", "MaximumRetryCount": 0}

    with pytest.raises(RuntimeError):
        await server._start(  # pyright: ignore[reportPrivateUsage]
            _request(workspace_id, "fixture:drift:2")
        )


@pytest.mark.asyncio
async def test_controller_recovers_when_create_response_is_lost(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    server = WorkloadControllerServer(
        workspace=workspace,
        socket_path=tmp_path / "run" / "controller.sock",
        docker_socket=tmp_path / "docker.sock",
        state_path=tmp_path / "state" / "leases.json",
        network="test-network",
        allowed_uid=os.getuid(),
        socket_gid=os.getgid(),
        workload_uid=os.getuid(),
        workload_gid=os.getgid(),
        socket_uid=os.getuid(),
    )
    fake = _FakeEngine()
    fake.lose_create_response = True
    server._engine = fake  # pyright: ignore[reportPrivateUsage]

    receipt = await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(
            server._workspace_id,  # pyright: ignore[reportPrivateUsage]
            "fixture:lost-response:1",
        )
    )

    assert receipt["lease"]["container_id"] == "container-1"  # type: ignore[index]
    assert fake._running()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_stop_keeps_mount_evidence_across_a_crash_after_delete(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    state_path = tmp_path / "state" / "leases.json"
    arguments = {
        "workspace": workspace,
        "socket_path": tmp_path / "run" / "controller.sock",
        "docker_socket": tmp_path / "docker.sock",
        "state_path": state_path,
        "network": "test-network",
        "allowed_uid": os.getuid(),
        "socket_gid": os.getgid(),
        "workload_uid": os.getuid(),
        "workload_gid": os.getgid(),
        "socket_uid": os.getuid(),
    }
    server = WorkloadControllerServer(**arguments)
    fake = _FakeEngine()
    server._engine = fake  # pyright: ignore[reportPrivateUsage]
    request = _request(
        server._workspace_id,  # pyright: ignore[reportPrivateUsage]
        "fixture:crash:1",
    )
    await server._start(request)  # pyright: ignore[reportPrivateUsage]
    lease = WorkloadLease(
        workspace_id=request.workspace_id,
        plugin_id=request.plugin_id,
        workload=request.workload,
        mode=request.mode,
        transaction_id=request.transaction_id,
        generation_id=request.generation_id,
        container_id="container-1",
        spec_digest=request.spec_digest,
    )
    fake.crash_after_delete = True

    with pytest.raises(SystemExit, match="after delete"):
        await server._stop(lease)  # pyright: ignore[reportPrivateUsage]

    recovered = WorkloadControllerServer(**arguments)
    recovered._engine = fake  # pyright: ignore[reportPrivateUsage]
    receipt = await recovered._stop(lease)  # pyright: ignore[reportPrivateUsage]
    evidence = next(
        iter(recovered._stopped.values())
    )  # pyright: ignore[reportPrivateUsage]
    assert evidence["sources"]
    assert receipt["container_absent"] is True
    assert receipt["mounts_released"] is True
