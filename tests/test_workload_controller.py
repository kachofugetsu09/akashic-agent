from __future__ import annotations

import asyncio
import json
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
        self.fail_next_stop = False
        self.lose_create_response = False
        self.crash_after_delete = False
        self.owner_running: bool | None = None
        self.create_count = 0
        self.delete_count = 0

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
            self.create_count += 1
            self.create_body = body
            name = parse_qs(urlsplit(path).query)["name"][0]
            host = body["HostConfig"]
            assert isinstance(host, dict)
            mounts = host["Mounts"]
            assert isinstance(mounts, list)
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
                        "Source": value["Source"],
                        "Destination": value["Target"],
                        "RW": not value["ReadOnly"],
                    }
                    for value in mounts
                ],
            }
            if self.lose_create_response:
                self.lose_create_response = False
                raise RuntimeError("create response lost")
            return {"Id": "container-1"}
        if method == "GET" and path == "/containers/akashic-core/json":
            if self.owner_running is None:
                return None
            return {"State": {"Running": self.owner_running}}
        if method == "GET" and path.endswith("/json"):
            return self.container
        if method == "POST" and path.endswith("/start"):
            if self.fail_next_start:
                self.fail_next_start = False
                raise RuntimeError("start failed")
            self._state()["Running"] = True
            return None
        if method == "POST" and "/stop?" in path:
            if self.fail_next_stop:
                self.fail_next_stop = False
                raise RuntimeError("stop failed")
            self._state()["Running"] = False
            return None
        if method == "DELETE":
            self.delete_count += 1
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
    loopback: bool = False,
    user_namespaces: bool = False,
    limits: tuple[int, float, int] = (128, 1.0, 64),
) -> WorkloadStartRequest:
    image = "example.invalid/worker@sha256:" + "b" * 64
    ports = (("gateway", 8080),)
    data = (("state", "/data", True),)
    health = ("gateway", "/health", 30.0)
    loopback_ports = (("gateway", 18080),) if loopback else ()
    digest = workload_spec_digest(
        plugin_id="fixture",
        workload="worker",
        image=image,
        command=("serve",),
        ports=ports,
        data=data,
        health=health,
        limits=limits,
        loopback_ports=loopback_ports,
        user_namespaces=user_namespaces,
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
        loopback_ports=loopback_ports,
        user_namespaces=user_namespaces,
    )


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "plugin-data").mkdir(parents=True)
    (workspace / "runtime/plugin-validation").mkdir(parents=True)
    return workspace


@pytest.mark.asyncio
async def test_controller_keeps_each_zero_limit_unlimited(tmp_path: Path) -> None:
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

    await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(
            server._workspace_id,  # pyright: ignore[reportPrivateUsage]
            "fixture:formal:unlimited",
            limits=(0, 0.0, 0),
        )
    )

    assert fake.create_body is not None
    host = fake.create_body["HostConfig"]
    assert isinstance(host, dict)
    assert host["Memory"] == 0
    assert host["NanoCpus"] == 0
    assert host["PidsLimit"] is None


@pytest.mark.asyncio
async def test_controller_uses_structured_mounts_for_colon_paths(
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
    server._engine = fake  # pyright: ignore[reportPrivateUsage]

    await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(
            server._workspace_id, "fixture:candidate:1", mode="candidate"
        )  # pyright: ignore[reportPrivateUsage]
    )

    assert fake.create_body is not None
    host = fake.create_body["HostConfig"]
    assert isinstance(host, dict)
    assert "Binds" not in host
    assert host["Mounts"] == [
        {
            "Type": "bind",
            "Source": str(
                workspace
                / "runtime/plugin-validation/fixture:candidate:1"
                / "workspace/plugin-data/fixture-builtin/state"
            ),
            "Target": "/data",
            "ReadOnly": False,
        }
    ]
    assert host["ExtraHosts"] == ["host.docker.internal:host-gateway"]


@pytest.mark.asyncio
async def test_controller_adds_only_the_user_namespace_seccomp_profile(
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
    server._engine = fake  # pyright: ignore[reportPrivateUsage]

    await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(
            server._workspace_id,  # pyright: ignore[reportPrivateUsage]
            "fixture:candidate:userns",
            mode="candidate",
            user_namespaces=True,
        )
    )

    assert fake.create_body is not None
    host = fake.create_body["HostConfig"]
    assert isinstance(host, dict)
    security = host["SecurityOpt"]
    assert isinstance(security, list)
    assert security[0] == "no-new-privileges"
    assert len(security) == 2
    assert isinstance(security[1], str)
    assert security[1].startswith("seccomp=")
    profile = json.loads(security[1].removeprefix("seccomp="))
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    assert any(item.get("names") == ["unshare"] for item in profile["syscalls"])
    assert any(
        item.get("names") == ["chroot"] and "includes" not in item
        for item in profile["syscalls"]
    )
    assert "unconfined" not in security[1]


@pytest.mark.asyncio
async def test_controller_adopt_moves_the_only_stop_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_chown(path: object, uid: int, gid: int) -> None:
        raise PermissionError(f"unexpected chown: {path} {uid}:{gid}")

    monkeypatch.setattr("agent.workloads.controller.os.chown", reject_chown)
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
async def test_controller_adopts_docker_normalized_writable_mounts(
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
    server._engine = fake  # pyright: ignore[reportPrivateUsage]
    workspace_id = server._workspace_id  # pyright: ignore[reportPrivateUsage]

    await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(workspace_id, "fixture:formal:1")
    )
    assert fake.container is not None
    host = fake.container["HostConfig"]
    assert isinstance(host, dict)
    mounts = host["Mounts"]
    assert isinstance(mounts, list)
    host["Mounts"] = [
        {key: value for key, value in mount.items() if key != "ReadOnly"}
        for mount in mounts
        if isinstance(mount, dict)
    ]

    receipt = await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(workspace_id, "fixture:formal:2")
    )

    assert receipt["adopted_from_generation"] == "fixture:formal:1"
    assert receipt["lease"]["container_id"] == "container-1"  # type: ignore[index]
    assert fake.create_count == 1
    assert fake.delete_count == 0
    saved = next(iter(server._leases.values()))  # pyright: ignore[reportPrivateUsage]
    assert saved["generation_id"] == "fixture:formal:2"


@pytest.mark.asyncio
async def test_controller_replaces_exact_lease_when_declared_spec_changes(
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
    server._engine = fake  # pyright: ignore[reportPrivateUsage]
    workspace_id = server._workspace_id  # pyright: ignore[reportPrivateUsage]

    first = await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(workspace_id, "fixture:formal:1")
    )
    saved_owner = dict(server._leases)  # pyright: ignore[reportPrivateUsage]
    server._leases.clear()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match="spec/owner"):
        await server._start(  # pyright: ignore[reportPrivateUsage]
            _request(
                workspace_id,
                "fixture:unowned:2",
                user_namespaces=True,
            )
        )
    assert fake.create_count == 1
    assert fake.delete_count == 0
    server._leases.update(saved_owner)  # pyright: ignore[reportPrivateUsage]
    second = await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(
            workspace_id,
            "fixture:formal:2",
            user_namespaces=True,
        )
    )

    first_lease = first["lease"]
    second_lease = second["lease"]
    assert isinstance(first_lease, dict) and isinstance(second_lease, dict)
    assert first_lease["spec_digest"] != second_lease["spec_digest"]
    assert second["adopted_from_generation"] is None
    assert fake.create_count == 2
    assert fake.delete_count == 1
    assert len(server._leases) == 1  # pyright: ignore[reportPrivateUsage]
    saved = next(iter(server._leases.values()))  # pyright: ignore[reportPrivateUsage]
    assert saved["generation_id"] == "fixture:formal:2"
    assert fake.create_body is not None
    host = fake.create_body["HostConfig"]
    assert isinstance(host, dict)
    security = host["SecurityOpt"]
    assert isinstance(security, list) and len(security) == 2


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
    ("mode", "expected"),
    (
        ("formal", {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18080"}]}),
        ("candidate", {}),
    ),
)
async def test_controller_publishes_declared_loopback_only_for_formal(
    tmp_path: Path,
    mode: str,
    expected: dict[str, object],
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

    await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(
            server._workspace_id,  # pyright: ignore[reportPrivateUsage]
            f"fixture:{mode}:1",
            mode=mode,
            loopback=True,
        )
    )

    assert fake.create_body is not None
    host = fake.create_body["HostConfig"]
    assert isinstance(host, dict)
    assert host["PortBindings"] == expected


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


@pytest.mark.asyncio
async def test_completed_stop_receipts_are_not_silently_removed(tmp_path: Path) -> None:
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
    request = _request(
        server._workspace_id,  # pyright: ignore[reportPrivateUsage]
        "fixture:retention:1",
    )
    receipt = await server._start(request)  # pyright: ignore[reportPrivateUsage]
    server._stopped.update(  # pyright: ignore[reportPrivateUsage]
        {
            f"old-{index}": {"lease": {}, "sources": [], "complete": True}
            for index in range(1024)
        }
    )

    await server._stop(  # pyright: ignore[reportPrivateUsage]
        WorkloadLease(**receipt["lease"])  # type: ignore[arg-type]
    )

    assert "old-0" in server._stopped  # pyright: ignore[reportPrivateUsage]
    assert len(server._stopped) == 1025  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_controller_stops_workloads_when_its_core_container_stops(
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
        owner_container="akashic-core",
    )
    fake = _FakeEngine()
    fake.owner_running = True
    server._engine = fake  # pyright: ignore[reportPrivateUsage]
    await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(
            server._workspace_id, "fixture:owner-stop:1"
        )  # pyright: ignore[reportPrivateUsage]
    )

    await server._check_owner(1.0)  # pyright: ignore[reportPrivateUsage]
    assert fake.container is not None
    fake.owner_running = False
    await server._check_owner(2.0)  # pyright: ignore[reportPrivateUsage]

    assert fake.container is None
    assert not server._leases  # pyright: ignore[reportPrivateUsage]
    assert (workspace / "plugin-data/fixture-builtin/state").is_dir()


@pytest.mark.asyncio
async def test_owner_cleanup_rechecks_core_after_waiting_for_request_lock(
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
        owner_container="akashic-core",
    )
    fake = _FakeEngine()
    server._engine = fake  # pyright: ignore[reportPrivateUsage]
    await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(
            server._workspace_id, "fixture:owner-restart:1"
        )  # pyright: ignore[reportPrivateUsage]
    )
    server._owner_seen_running = True  # pyright: ignore[reportPrivateUsage]
    first_inspect_done = asyncio.Event()
    inspection = 0

    async def inspect_owner(
        _container: str, *, allow_missing: bool = False
    ) -> dict[str, object] | None:
        nonlocal inspection
        assert allow_missing
        inspection += 1
        if inspection == 1:
            first_inspect_done.set()
            return {"State": {"Running": False}}
        return {"State": {"Running": True}}

    server._inspect = inspect_owner  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    await server._lock.acquire()  # pyright: ignore[reportPrivateUsage]
    cleanup = asyncio.create_task(
        server._check_owner(2.0)  # pyright: ignore[reportPrivateUsage]
    )
    await first_inspect_done.wait()
    server._lock.release()  # pyright: ignore[reportPrivateUsage]
    await cleanup

    assert inspection == 2
    assert fake.container is not None
    assert server._leases  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_owner_cleanup_retries_the_same_exact_lease(tmp_path: Path) -> None:
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
        owner_container="akashic-core",
    )
    fake = _FakeEngine()
    fake.owner_running = True
    server._engine = fake  # pyright: ignore[reportPrivateUsage]
    await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(
            server._workspace_id, "fixture:owner-retry:1"
        )  # pyright: ignore[reportPrivateUsage]
    )
    await server._check_owner(1.0)  # pyright: ignore[reportPrivateUsage]

    fake.owner_running = False
    fake.fail_next_stop = True
    with pytest.raises(ExceptionGroup, match="owner cleanup"):
        await server._check_owner(2.0)  # pyright: ignore[reportPrivateUsage]
    assert fake.container is not None

    await server._check_owner(3.0)  # pyright: ignore[reportPrivateUsage]
    assert fake.container is None
    assert not server._leases  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_controller_cleans_old_leases_if_core_never_starts(
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
        owner_container="akashic-core",
        owner_grace_seconds=5.0,
    )
    fake = _FakeEngine()
    server._engine = fake  # pyright: ignore[reportPrivateUsage]
    await server._start(  # pyright: ignore[reportPrivateUsage]
        _request(
            server._workspace_id, "fixture:owner-missing:1"
        )  # pyright: ignore[reportPrivateUsage]
    )

    await server._check_owner(10.0)  # pyright: ignore[reportPrivateUsage]
    await server._check_owner(14.9)  # pyright: ignore[reportPrivateUsage]
    assert fake.container is not None
    await server._check_owner(15.0)  # pyright: ignore[reportPrivateUsage]

    assert fake.container is None
    assert not server._leases  # pyright: ignore[reportPrivateUsage]


def test_controller_state_save_syncs_file_and_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[int] = []
    monkeypatch.setattr(os, "fsync", synced.append)

    path = tmp_path / "state" / "leases.json"
    WorkloadControllerServer._save_state(  # pyright: ignore[reportPrivateUsage]
        path,
        {"one": {"complete": True}},
    )

    assert path.read_text(encoding="utf-8") == '{"one":{"complete":true}}'
    assert len(synced) == 2
