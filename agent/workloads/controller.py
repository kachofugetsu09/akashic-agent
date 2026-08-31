from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import http.client
import json
import logging
import os
import re
import socket
import struct
import tempfile
from dataclasses import asdict
from functools import cache
from pathlib import Path, PurePosixPath
from collections.abc import Coroutine, Mapping
from typing import Any, TypeVar, cast
from urllib.parse import quote, urlencode, urlsplit

from agent.workloads.model import (
    WorkloadLease,
    WorkloadMode,
    WorkloadStartRequest,
    workload_spec_digest,
)

_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,511}$")
_API_VERSION = "v1.47"
_OWNER_LABEL = "com.akashic.workload"
_HOST_GATEWAY = "host.docker.internal:host-gateway"
_USERNS_SECCOMP_PATH = Path(__file__).with_name("userns-seccomp.json")
_T = TypeVar("_T")
logger = logging.getLogger(__name__)


class _UnixHttpConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path) -> None:
        super().__init__("docker", timeout=60)
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self._socket_path))


class DockerEngine:
    """Call only the small Docker API surface needed by Workloads."""

    def __init__(self, socket_path: Path) -> None:
        self._socket_path = socket_path

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
        expected: frozenset[int],
    ) -> object:
        return await asyncio.to_thread(
            self._request_sync,
            method,
            path,
            body,
            expected,
        )

    def _request_sync(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None,
        expected: frozenset[int],
    ) -> object:
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        connection = _UnixHttpConnection(self._socket_path)
        try:
            connection.request(
                method,
                f"/{_API_VERSION}{path}",
                body=encoded,
                headers=(
                    {} if encoded is None else {"Content-Type": "application/json"}
                ),
            )
            response = connection.getresponse()
            raw = response.read()
            if response.status not in expected:
                detail = raw.decode("utf-8", errors="replace")[:4096]
                raise RuntimeError(
                    f"Docker API {method} {path} -> {response.status}: {detail}"
                )
            if response.status == 404:
                return None
            if not raw:
                return None
            value = json.loads(raw)
            if not isinstance(value, (dict, list)):
                raise RuntimeError("Docker API 返回了无效 JSON")
            return value
        finally:
            connection.close()


class WorkloadControllerServer:
    """Validate requests and own all Docker effects for plugin Workloads."""

    def __init__(
        self,
        *,
        workspace: Path,
        socket_path: Path,
        docker_socket: Path,
        state_path: Path,
        network: str,
        allowed_uid: int,
        socket_gid: int,
        workload_uid: int,
        workload_gid: int,
        socket_uid: int = 0,
        owner_container: str | None = None,
        owner_grace_seconds: float = 10.0,
        owner_poll_seconds: float = 2.0,
    ) -> None:
        self._workspace = workspace.resolve(strict=True)
        self._socket_path = socket_path
        self._engine = DockerEngine(docker_socket)
        self._state_path = state_path
        self._network = _safe_segment(network, "network")
        self._allowed_uid = allowed_uid
        self._socket_gid = socket_gid
        self._socket_uid = socket_uid
        self._owner_container = (
            None
            if owner_container is None
            else _safe_segment(owner_container, "owner container")
        )
        if owner_grace_seconds <= 0 or owner_poll_seconds <= 0:
            raise ValueError("Workload owner grace/poll 必须大于零")
        self._owner_grace_seconds = owner_grace_seconds
        self._owner_poll_seconds = owner_poll_seconds
        self._owner_seen_running = False
        self._owner_missing_since: float | None = None
        if workload_uid <= 0 or workload_gid <= 0:
            raise ValueError("Workload uid/gid 必须是非 root 正整数")
        self._workload_uid = workload_uid
        self._workload_gid = workload_gid
        for relative in (Path("plugin-data"), Path("runtime/plugin-validation")):
            root = (self._workspace / relative).resolve(strict=True)
            if not root.is_relative_to(self._workspace):
                raise ValueError(f"Workload data root 越界: {root}")
            stat = root.stat()
            if (stat.st_uid, stat.st_gid) != (workload_uid, workload_gid):
                raise ValueError(f"Workload data root owner 不匹配: {root}")
        self._workspace_id = hashlib.sha256(
            str(self._workspace).encode("utf-8")
        ).hexdigest()[:16]
        self._lock = asyncio.Lock()
        self._leases = self._load_leases()
        self._stopped_path = self._state_path.with_suffix(".stopped.json")
        self._stopped = self._load_state(self._stopped_path)

    async def serve(self) -> None:
        """Bind the private socket and serve one bounded request per connection."""

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = self._state_path.with_suffix(".lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_handle.close()
            raise RuntimeError("已有 Workload Controller 正在运行") from error
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self._socket_path.exists():
                if not self._socket_path.is_socket():
                    raise RuntimeError(
                        f"Controller socket path 已被普通文件占用: {self._socket_path}"
                    )
                self._socket_path.unlink()
            server = await asyncio.start_unix_server(
                self._handle,
                path=self._socket_path,
            )
            os.chmod(self._socket_path, 0o660)
            socket_stat = self._socket_path.stat()
            if (socket_stat.st_uid, socket_stat.st_gid) != (
                self._socket_uid,
                self._socket_gid,
            ):
                os.chown(self._socket_path, self._socket_uid, self._socket_gid)
            async with server:
                if self._owner_container is None:
                    await server.serve_forever()
                else:
                    async with asyncio.TaskGroup() as tasks:
                        tasks.create_task(
                            server.serve_forever(), name="workload-controller-server"
                        )
                        tasks.create_task(
                            self._watch_owner(), name="workload-owner-watch"
                        )
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()

    async def _watch_owner(self) -> None:
        """Stop owned containers after the deployment owner disappears."""

        loop = asyncio.get_running_loop()
        while True:
            try:
                await self._check_owner(loop.time())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Workload owner check failed; cleanup will retry")
            await asyncio.sleep(self._owner_poll_seconds)

    async def _check_owner(self, now: float) -> None:
        """Apply one owner-liveness observation using the supplied monotonic time."""

        owner = self._owner_container
        if owner is None:
            return
        detail = await self._inspect(owner, allow_missing=True)
        if detail is not None and _running(detail):
            self._owner_seen_running = True
            self._owner_missing_since = None
            return

        if self._owner_missing_since is None:
            self._owner_missing_since = now
        if (
            not self._owner_seen_running
            and now - self._owner_missing_since < self._owner_grace_seconds
        ):
            return

        async with self._lock:
            # Core may have restarted while this watcher waited for a request.
            # Recheck under the same lock that guards start/adopt before cleanup.
            detail = await self._inspect(owner, allow_missing=True)
            if detail is not None and _running(detail):
                self._owner_seen_running = True
                self._owner_missing_since = None
                return
            await self._stop_owned_workloads()
        self._owner_seen_running = False
        self._owner_missing_since = now

    async def _stop_owned_workloads(self) -> None:
        """Strongly stop every exact lease still owned by this Controller."""

        errors: list[Exception] = []
        for raw in tuple(self._leases.values()):
            try:
                await _finish_effect(self._stop(_lease(raw)))
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Workload owner cleanup 失败", errors)

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            self._check_peer(writer)
            raw = await reader.readline()
            if not raw or len(raw) > 1_048_576:
                raise ValueError("Controller 请求为空或过大")
            message = json.loads(raw)
            if not isinstance(message, dict) or message.get("version") != 1:
                raise ValueError("Controller 请求版本无效")
            action = message.get("action")
            body = message.get("body")
            if not isinstance(body, dict):
                raise ValueError("Controller 请求 body 无效")
            async with self._lock:
                if action == "start":
                    result = await self._start(_start_request(body))
                elif action == "stop":
                    if set(body) != {"lease"}:
                        raise ValueError("Controller stop schema 不匹配")
                    result = await self._stop(_lease(body.get("lease")))
                elif action == "cleanup_candidates":
                    if set(body) != {"workspace_id"}:
                        raise ValueError("Controller cleanup schema 不匹配")
                    result = await self._cleanup_candidates(
                        _required_text(body, "workspace_id")
                    )
                else:
                    raise ValueError(f"Controller action 无效: {action}")
            response = {"ok": True, "body": result}
        except Exception as error:
            message = str(error).strip() or type(error).__name__
            response = {"ok": False, "error": message[:4096]}
        writer.write(
            json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    def _check_peer(self, writer: asyncio.StreamWriter) -> None:
        sock = writer.get_extra_info("socket")
        if sock is None or not hasattr(socket, "SO_PEERCRED"):
            raise PermissionError("Controller 无法验证 Unix peer")
        raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        _pid, uid, _gid = struct.unpack("3i", raw)
        if uid != self._allowed_uid:
            raise PermissionError(f"Controller peer uid 未授权: {uid}")

    async def _start(self, request: WorkloadStartRequest) -> dict[str, object]:
        self._check_request(request)
        key = _lease_key(request)
        containers = await self._find(request)
        if len(containers) > 1:
            raise RuntimeError(f"Workload stable key 出现多个容器: {key}")
        adopted_from: str | None = None
        if containers:
            container = containers[0]
            container_id = _text(container, "Id")
            detail = await self._inspect(container_id)
            assert detail is not None
            labels = _labels(detail)
            previous_raw = self._leases.get(key)
            previous = _lease(previous_raw) if previous_raw is not None else None
            if previous is not None and (
                _lease_key(previous) != key or previous.container_id != container_id
            ):
                raise RuntimeError("已保存 Workload lease 与现有容器不一致")
            if previous is not None and previous.spec_digest != request.spec_digest:
                # A new generation may declare a new immutable spec. Replace only
                # the exact Controller-owned lease; unrelated drift still fails.
                self._check_lease_labels(previous, labels)
                _ = await _finish_effect(self._stop(previous))
                containers = []
                container_id = await self._create(request)
            else:
                self._check_existing(request, detail, labels)
                if not _running(detail):
                    await self._start_existing(request, key, container_id)
                adopted_from = (
                    previous.generation_id
                    if previous is not None
                    else labels.get("com.akashic.generation")
                )
        else:
            container_id = await self._create(request)
        lease = WorkloadLease(
            workspace_id=request.workspace_id,
            plugin_id=request.plugin_id,
            workload=request.workload,
            mode=request.mode,
            transaction_id=request.transaction_id,
            generation_id=request.generation_id,
            container_id=container_id,
            spec_digest=request.spec_digest,
        )
        if not containers:
            self._leases[key] = asdict(lease)
            self._save_leases()
            try:
                await _finish_effect(self._start_container(container_id))
            except BaseException as start_error:
                try:
                    await asyncio.shield(self._stop(lease))
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "Workload start 失败且 cleanup 未完成",
                        [start_error, cleanup_error],
                    )
                raise
        self._leases[key] = asdict(lease)
        self._save_leases()
        name = _container_name(request)
        return {
            "lease": asdict(lease),
            "endpoints": [
                {"name": port_name, "url": f"http://{name}:{number}"}
                for port_name, number in request.ports
            ],
            "adopted_from_generation": adopted_from,
        }

    async def _stop(self, lease: WorkloadLease) -> dict[str, object]:
        key = _lease_key(lease)
        stop_key = _stop_key(lease)
        saved = self._leases.get(key)
        if saved != asdict(lease):
            stopped = self._stopped.get(stop_key)
            if not isinstance(stopped, dict) or stopped.get("lease") != asdict(lease):
                raise RuntimeError("Workload stop lease 已过期或未 adopt")
            sources_value = stopped.get("sources")
            if not isinstance(sources_value, list) or any(
                not isinstance(item, str) for item in sources_value
            ):
                raise RuntimeError("Workload completed stop state 损坏")
            absent = await self._inspect(lease.container_id, allow_missing=True) is None
            released = absent and not await self._active_mounts(tuple(sources_value))
            if not absent or not released:
                raise RuntimeError("Workload completed stop 证据已失效")
            return _stop_receipt(lease)
        detail = await self._inspect(lease.container_id, allow_missing=True)
        saved_stop = self._stopped.get(stop_key)
        saved_sources = (
            saved_stop.get("sources") if isinstance(saved_stop, dict) else None
        )
        sources = (
            tuple(cast(list[str], saved_sources))
            if isinstance(saved_sources, list)
            and all(isinstance(item, str) for item in saved_sources)
            else ()
        )
        if detail is not None:
            labels = _labels(detail)
            self._check_lease_labels(lease, labels)
            sources = tuple(
                str(item.get("Source"))
                for item in cast(list[dict[str, object]], detail.get("Mounts", []))
                if item.get("Type") == "bind" and isinstance(item.get("Source"), str)
            )
            self._stopped[stop_key] = {
                "lease": asdict(lease),
                "sources": list(sources),
                "complete": False,
            }
            self._save_state(self._stopped_path, self._stopped)
            if _running(detail):
                await self._engine.request(
                    "POST",
                    f"/containers/{quote(lease.container_id)}/stop?t=30",
                    expected=frozenset({204, 304}),
                )
            await self._engine.request(
                "DELETE",
                f"/containers/{quote(lease.container_id)}?v=1",
                expected=frozenset({204, 404}),
            )
        absent = await self._inspect(lease.container_id, allow_missing=True) is None
        released = absent and not await self._active_mounts(sources)
        if not absent or not released:
            raise RuntimeError("Workload remove 后容器或 mount 仍被占用")
        self._stopped[stop_key] = {
            "lease": asdict(lease),
            "sources": list(sources),
            "complete": True,
        }
        self._save_state(self._stopped_path, self._stopped)
        self._leases.pop(key, None)
        self._save_leases()
        return _stop_receipt(lease)

    async def _cleanup_candidates(self, workspace_id: str) -> dict[str, object]:
        """Remove only candidate containers from an earlier Core boot."""

        if workspace_id != self._workspace_id:
            raise ValueError("Workload cleanup workspace identity 不匹配")
        labels = [
            f"{_OWNER_LABEL}=true",
            f"com.akashic.workspace={workspace_id}",
            "com.akashic.workload.mode=candidate",
        ]
        filters = quote(json.dumps({"label": labels}, separators=(",", ":")))
        value = await self._engine.request(
            "GET",
            f"/containers/json?all=1&filters={filters}",
            expected=frozenset({200}),
        )
        if not isinstance(value, list) or any(
            not isinstance(item, dict) for item in value
        ):
            raise RuntimeError("Docker candidate list 回执无效")
        receipts: list[dict[str, object]] = []
        for item in cast(list[dict[str, object]], value):
            container_id = _text(item, "Id")
            detail = await self._inspect(container_id)
            assert detail is not None
            item_labels = _labels(detail)
            lease = _candidate_lease(workspace_id, container_id, item_labels)
            key = _lease_key(lease)
            saved = self._leases.get(key)
            if saved is not None:
                lease = _lease(saved)
                if lease.container_id != container_id:
                    raise RuntimeError("candidate lease container identity 不一致")
            else:
                self._leases[key] = asdict(lease)
                self._save_leases()
            receipts.append(await self._stop(lease))
        return {"receipts": receipts}

    async def _create(self, request: WorkloadStartRequest) -> str:
        mounts = _docker_mounts(self._data_mounts(request))
        labels = {
            _OWNER_LABEL: "true",
            "com.akashic.workspace": request.workspace_id,
            "com.akashic.plugin": request.plugin_id,
            "com.akashic.workload.name": request.workload,
            "com.akashic.workload.mode": request.mode,
            "com.akashic.transaction": request.transaction_id,
            "com.akashic.generation": request.generation_id,
            "com.akashic.spec": request.spec_digest,
        }
        memory, cpu, pids = request.limits
        docker_pids = None if pids == 0 else pids
        body: dict[str, object] = {
            "Image": request.image,
            "User": f"{self._workload_uid}:{self._workload_gid}",
            "Labels": labels,
            "ExposedPorts": {f"{number}/tcp": {} for _, number in request.ports},
            "HostConfig": {
                "Mounts": mounts,
                "Memory": memory * 1024 * 1024,
                "NanoCpus": int(cpu * 1_000_000_000),
                "PidsLimit": docker_pids,
                "NetworkMode": self._network,
                "ExtraHosts": [_HOST_GATEWAY],
                "PortBindings": _port_bindings(request),
                "PublishAllPorts": False,
                "PidMode": "",
                "IpcMode": "private",
                "UTSMode": "",
                "UsernsMode": "",
                "Devices": [],
                "DeviceRequests": [],
                "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                "AutoRemove": False,
                "Privileged": False,
                "ReadonlyRootfs": False,
                "CapDrop": ["ALL"],
                "SecurityOpt": _security_options(request.user_namespaces),
            },
        }
        if request.command:
            body["Cmd"] = list(request.command)
        try:
            value = await _finish_effect(
                self._engine.request(
                    "POST",
                    "/containers/create?"
                    + urlencode({"name": _container_name(request)}),
                    body=body,
                    expected=frozenset({201}),
                )
            )
        except asyncio.CancelledError:
            containers = await _finish_effect(self._find(request))
            if len(containers) == 1:
                lease = _request_lease(request, _text(containers[0], "Id"))
                self._leases[_lease_key(lease)] = asdict(lease)
                self._save_leases()
                await _finish_effect(self._stop(lease))
            raise
        except Exception:
            containers = await _finish_effect(self._find(request))
            if len(containers) == 1:
                return _text(containers[0], "Id")
            raise
        if not isinstance(value, dict):
            raise RuntimeError("Docker create 回执无效")
        return _text(value, "Id")

    async def _start_existing(
        self,
        request: WorkloadStartRequest,
        key: str,
        container_id: str,
    ) -> None:
        """Start an adopted container and keep cleanup ownership on failure."""

        if key in self._leases:
            await _finish_effect(self._start_container(container_id))
            return
        lease = WorkloadLease(
            workspace_id=request.workspace_id,
            plugin_id=request.plugin_id,
            workload=request.workload,
            mode=request.mode,
            transaction_id=request.transaction_id,
            generation_id=request.generation_id,
            container_id=container_id,
            spec_digest=request.spec_digest,
        )
        self._leases[key] = asdict(lease)
        self._save_leases()
        try:
            await _finish_effect(self._start_container(container_id))
        except BaseException as start_error:
            try:
                await asyncio.shield(self._stop(lease))
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "Workload adopt start 失败且 cleanup 未完成",
                    [start_error, cleanup_error],
                )
            raise

    async def _start_container(self, container_id: str) -> None:
        await self._engine.request(
            "POST",
            f"/containers/{quote(container_id)}/start",
            expected=frozenset({204, 304}),
        )

    async def _find(self, request: WorkloadStartRequest) -> list[dict[str, object]]:
        labels = [
            f"{_OWNER_LABEL}=true",
            f"com.akashic.workspace={request.workspace_id}",
            f"com.akashic.plugin={request.plugin_id}",
            f"com.akashic.workload.name={request.workload}",
            f"com.akashic.workload.mode={request.mode}",
        ]
        if request.mode == "candidate":
            labels.append(f"com.akashic.transaction={request.transaction_id}")
        filters = quote(json.dumps({"label": labels}, separators=(",", ":")))
        value = await self._engine.request(
            "GET",
            f"/containers/json?all=1&filters={filters}",
            expected=frozenset({200}),
        )
        if not isinstance(value, list) or any(
            not isinstance(item, dict) for item in value
        ):
            raise RuntimeError("Docker container list 回执无效")
        return cast(list[dict[str, object]], value)

    async def _inspect(
        self,
        container_id: str,
        *,
        allow_missing: bool = False,
    ) -> dict[str, object] | None:
        expected = frozenset({200, 404}) if allow_missing else frozenset({200})
        value = await self._engine.request(
            "GET",
            f"/containers/{quote(container_id)}/json",
            expected=expected,
        )
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RuntimeError("Docker inspect 回执无效")
        return value

    async def _active_mounts(self, sources: tuple[str, ...]) -> bool:
        if not sources:
            return False
        filters = quote(json.dumps({"label": [f"{_OWNER_LABEL}=true"]}))
        value = await self._engine.request(
            "GET",
            f"/containers/json?all=0&filters={filters}",
            expected=frozenset({200}),
        )
        if not isinstance(value, list):
            raise RuntimeError("Docker running container list 回执无效")
        for item in value:
            if not isinstance(item, dict):
                raise RuntimeError("Docker running container entry 无效")
            detail = await self._inspect(_text(item, "Id"))
            assert detail is not None
            for mount in cast(list[dict[str, object]], detail.get("Mounts", [])):
                if mount.get("Source") in sources:
                    return True
        return False

    def _data_mounts(
        self,
        request: WorkloadStartRequest,
    ) -> tuple[tuple[str, str, bool], ...]:
        data_root = _plugin_data_root(self._workspace, request)
        result: list[tuple[str, str, bool]] = []
        for name, target, writable in request.data:
            source = data_root / name
            _make_safe_dir(
                self._workspace,
                source,
                uid=self._workload_uid,
                gid=self._workload_gid,
            )
            result.append((str(source), target, writable))
        return tuple(result)

    def _check_request(self, request: WorkloadStartRequest) -> None:
        if request.workspace_id != self._workspace_id:
            raise ValueError("Workload workspace identity 不匹配")
        _safe_segment(request.plugin_id, "plugin_id")
        if not _NAME.fullmatch(request.workload):
            raise ValueError("Workload name 无效")
        if request.mode not in {"candidate", "formal"}:
            raise ValueError("Workload mode 无效")
        if not request.transaction_id or not request.generation_id:
            raise ValueError("Workload transaction/generation 不能为空")
        if not _ID.fullmatch(request.transaction_id) or not _ID.fullmatch(
            request.generation_id
        ):
            raise ValueError("Workload transaction/generation 无效")
        if not re.fullmatch(r"[0-9a-f]{64}", request.spec_digest):
            raise ValueError("Workload spec digest 无效")
        if not _IMAGE.fullmatch(request.image):
            raise ValueError("Workload image 必须使用 sha256 digest")
        if not request.command:
            raise ValueError("Workload command 不能为空")
        if not isinstance(request.user_namespaces, bool):
            raise ValueError("Workload user_namespaces 无效")
        expected_digest = workload_spec_digest(
            plugin_id=request.plugin_id,
            workload=request.workload,
            image=request.image,
            command=request.command,
            ports=request.ports,
            data=request.data,
            health=request.health,
            limits=request.limits,
            loopback_ports=request.loopback_ports,
            user_namespaces=request.user_namespaces,
        )
        if request.spec_digest != expected_digest:
            raise ValueError("Workload spec digest 与请求内容不一致")
        if not request.ports or len({name for name, _ in request.ports}) != len(
            request.ports
        ):
            raise ValueError("Workload ports 无效")
        if len({number for _, number in request.ports}) != len(request.ports):
            raise ValueError("Workload port number 重复")
        for name, number in request.ports:
            if not _NAME.fullmatch(name) or not 1 <= number <= 65535:
                raise ValueError("Workload port 无效")
        port_names = {name for name, _ in request.ports}
        if len({name for name, _ in request.loopback_ports}) != len(
            request.loopback_ports
        ):
            raise ValueError("Workload loopback port name 重复")
        if len({number for _, number in request.loopback_ports}) != len(
            request.loopback_ports
        ):
            raise ValueError("Workload loopback host port 重复")
        for name, number in request.loopback_ports:
            if name not in port_names or not 1024 <= number <= 65535:
                raise ValueError("Workload loopback port 无效")
        if len({name for name, _, _ in request.data}) != len(request.data):
            raise ValueError("Workload data name 重复")
        if len({target for _, target, _ in request.data}) != len(request.data):
            raise ValueError("Workload data target 重复")
        for name, target, writable in request.data:
            path = PurePosixPath(target)
            if (
                not _NAME.fullmatch(name)
                or not path.is_absolute()
                or path == PurePosixPath("/")
                or ".." in path.parts
                or not isinstance(writable, bool)
            ):
                raise ValueError("Workload data 无效")
        health_port, health_path, health_timeout = request.health
        if (
            health_port not in {name for name, _ in request.ports}
            or not health_path.startswith("/")
            or health_path.startswith("//")
            or health_path != health_path.strip()
            or "\\" in health_path
            or any(part in {".", ".."} for part in health_path.split("/"))
            or urlsplit(health_path).query
            or urlsplit(health_path).fragment
            or not 0 < health_timeout <= 300
        ):
            raise ValueError("Workload health 无效")
        memory, cpu, pids = request.limits
        if not (
            (memory == 0 or 64 <= memory <= 262_144)
            and (cpu == 0 or 0.1 <= cpu <= 256)
            and (pids == 0 or 16 <= pids <= 1_048_576)
        ):
            raise ValueError("Workload limits 超出允许范围")

    def _check_existing(
        self,
        request: WorkloadStartRequest,
        detail: dict[str, object],
        labels: dict[str, str],
    ) -> None:
        expected = {
            _OWNER_LABEL: "true",
            "com.akashic.workspace": request.workspace_id,
            "com.akashic.plugin": request.plugin_id,
            "com.akashic.workload.name": request.workload,
            "com.akashic.workload.mode": request.mode,
            "com.akashic.spec": request.spec_digest,
        }
        if any(labels.get(key) != value for key, value in expected.items()):
            raise RuntimeError("已存在 Workload 与请求 spec/owner 不一致")
        if (
            request.mode == "candidate"
            and labels.get("com.akashic.transaction") != request.transaction_id
        ):
            raise RuntimeError("candidate Workload transaction 不一致")
        memory, cpu, pids = request.limits
        docker_pids = None if pids == 0 else pids
        config = _object(detail, "Config")
        host = _object(detail, "HostConfig")
        expected_config: dict[str, object] = {
            "Image": request.image,
            "Cmd": list(request.command),
            "User": f"{self._workload_uid}:{self._workload_gid}",
            "ExposedPorts": {f"{number}/tcp": {} for _, number in request.ports},
        }
        expected_host: dict[str, object] = {
            "Memory": memory * 1024 * 1024,
            "NanoCpus": int(cpu * 1_000_000_000),
            "PidsLimit": docker_pids,
            "NetworkMode": self._network,
            "ExtraHosts": [_HOST_GATEWAY],
            "PortBindings": _port_bindings(request),
            "PublishAllPorts": False,
            "PidMode": "",
            "IpcMode": "private",
            "UTSMode": "",
            "UsernsMode": "",
            "Devices": [],
            "DeviceRequests": [],
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "AutoRemove": False,
            "Privileged": False,
            "ReadonlyRootfs": False,
            "CapDrop": ["ALL"],
            "SecurityOpt": _security_options(request.user_namespaces),
        }
        if any(config.get(key) != value for key, value in expected_config.items()):
            raise RuntimeError("已存在 Workload 的 image/command/user/ports 已漂移")
        if any(host.get(key) != value for key, value in expected_host.items()):
            raise RuntimeError(
                "已存在 Workload 的 mount/limits/network/security 已漂移"
            )
        networks = _object(_object(detail, "NetworkSettings"), "Networks")
        if set(networks) != {self._network}:
            raise RuntimeError("已存在 Workload 的实际 network 已漂移")
        mounts = detail.get("Mounts")
        if not isinstance(mounts, list) or _mount_profile(mounts) != {
            (source, target, writable)
            for source, target, writable in self._data_mounts(request)
        }:
            raise RuntimeError("已存在 Workload 的实际 mount 已漂移")

    @staticmethod
    def _check_lease_labels(lease: WorkloadLease, labels: dict[str, str]) -> None:
        expected = {
            _OWNER_LABEL: "true",
            "com.akashic.workspace": lease.workspace_id,
            "com.akashic.plugin": lease.plugin_id,
            "com.akashic.workload.name": lease.workload,
            "com.akashic.workload.mode": lease.mode,
            "com.akashic.spec": lease.spec_digest,
        }
        if any(labels.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Workload stop owner/spec labels 不一致")

    def _load_leases(self) -> dict[str, dict[str, object]]:
        return self._load_state(self._state_path)

    @staticmethod
    def _load_state(path: Path) -> dict[str, dict[str, object]]:
        if not path.exists():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or any(
            not isinstance(item, dict) for item in value.values()
        ):
            raise RuntimeError("Workload Controller lease state 损坏")
        return cast(dict[str, dict[str, object]], value)

    def _save_leases(self) -> None:
        self._save_state(self._state_path, self._leases)

    @staticmethod
    def _save_state(path: Path, value: dict[str, dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
            text=True,
        )
        temp = Path(raw_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
            directory = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temp.unlink(missing_ok=True)


def _start_request(raw: dict[str, object]) -> WorkloadStartRequest:
    expected = {
        "workspace_id",
        "plugin_id",
        "workload",
        "mode",
        "transaction_id",
        "generation_id",
        "spec_digest",
        "image",
        "command",
        "ports",
        "data",
        "health",
        "limits",
        "loopback_ports",
        "user_namespaces",
    }
    if set(raw) != expected:
        raise ValueError(f"Workload start schema 不匹配: {sorted(set(raw) ^ expected)}")
    mode = _required_text(raw, "mode")
    if mode not in {"candidate", "formal"}:
        raise ValueError("Workload mode 无效")
    return WorkloadStartRequest(
        workspace_id=_required_text(raw, "workspace_id"),
        plugin_id=_required_text(raw, "plugin_id"),
        workload=_required_text(raw, "workload"),
        mode=cast(WorkloadMode, mode),
        transaction_id=_required_text(raw, "transaction_id"),
        generation_id=_required_text(raw, "generation_id"),
        spec_digest=_required_text(raw, "spec_digest"),
        image=_required_text(raw, "image"),
        command=_text_tuple(raw.get("command"), "command"),
        ports=_ports(raw.get("ports")),
        data=_data(raw.get("data")),
        health=_health(raw.get("health")),
        limits=_limits(raw.get("limits")),
        loopback_ports=_ports(raw.get("loopback_ports")),
        user_namespaces=_required_bool(raw, "user_namespaces"),
    )


def _security_options(user_namespaces: bool) -> list[str]:
    """Keep Docker defaults unless a workload needs nested user namespaces."""

    options = ["no-new-privileges"]
    if user_namespaces:
        options.append(_userns_seccomp_option())
    return options


@cache
def _userns_seccomp_option() -> str:
    """Load the Core-owned user-namespace exception to Docker's default profile."""

    try:
        profile = json.loads(_USERNS_SECCOMP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Workload user namespace seccomp profile 无法加载"
        ) from error
    if not isinstance(profile, dict):
        raise RuntimeError("Workload user namespace seccomp profile 必须是 JSON object")
    return "seccomp=" + json.dumps(profile, separators=(",", ":"))


def _port_bindings(request: WorkloadStartRequest) -> dict[str, list[dict[str, str]]]:
    """Publish declared formal ports only on the host loopback interface."""

    if request.mode != "formal":
        return {}
    host_ports = dict(request.loopback_ports)
    return {
        f"{number}/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(host_ports[name])}]
        for name, number in request.ports
        if name in host_ports
    }


def _lease(raw: object) -> WorkloadLease:
    if not isinstance(raw, dict):
        raise ValueError("Workload lease 无效")
    expected = {
        "workspace_id",
        "plugin_id",
        "workload",
        "mode",
        "transaction_id",
        "generation_id",
        "container_id",
        "spec_digest",
    }
    if set(raw) != expected:
        raise ValueError(f"Workload lease schema 不匹配: {sorted(set(raw) ^ expected)}")
    mode = _required_text(raw, "mode")
    if mode not in {"candidate", "formal"}:
        raise ValueError("Workload lease mode 无效")
    return WorkloadLease(
        workspace_id=_required_text(raw, "workspace_id"),
        plugin_id=_required_text(raw, "plugin_id"),
        workload=_required_text(raw, "workload"),
        mode=cast(WorkloadMode, mode),
        transaction_id=_required_text(raw, "transaction_id"),
        generation_id=_required_text(raw, "generation_id"),
        container_id=_required_text(raw, "container_id"),
        spec_digest=_required_text(raw, "spec_digest"),
    )


def _candidate_lease(
    workspace_id: str,
    container_id: str,
    labels: dict[str, str],
) -> WorkloadLease:
    expected = {
        _OWNER_LABEL: "true",
        "com.akashic.workspace": workspace_id,
        "com.akashic.workload.mode": "candidate",
    }
    if any(labels.get(key) != value for key, value in expected.items()):
        raise RuntimeError("candidate cleanup labels 不属于请求 workspace")
    return WorkloadLease(
        workspace_id=workspace_id,
        plugin_id=_required_text(labels, "com.akashic.plugin"),
        workload=_required_text(labels, "com.akashic.workload.name"),
        mode="candidate",
        transaction_id=_required_text(labels, "com.akashic.transaction"),
        generation_id=_required_text(labels, "com.akashic.generation"),
        container_id=container_id,
        spec_digest=_required_text(labels, "com.akashic.spec"),
    )


def _ports(raw: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(raw, list):
        raise ValueError("Workload ports 无效")
    result: list[tuple[str, int]] = []
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
        ):
            raise ValueError("Workload port entry 无效")
        result.append((item[0], item[1]))
    return tuple(result)


def _data(raw: object) -> tuple[tuple[str, str, bool], ...]:
    if not isinstance(raw, list):
        raise ValueError("Workload data 无效")
    result: list[tuple[str, str, bool]] = []
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 3
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            or not isinstance(item[2], bool)
        ):
            raise ValueError("Workload data entry 无效")
        result.append((item[0], item[1], item[2]))
    return tuple(result)


def _limits(raw: object) -> tuple[int, float, int]:
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 3
        or not isinstance(raw[0], int)
        or isinstance(raw[0], bool)
        or not isinstance(raw[1], (int, float))
        or isinstance(raw[1], bool)
        or not isinstance(raw[2], int)
        or isinstance(raw[2], bool)
    ):
        raise ValueError("Workload limits 无效")
    return raw[0], float(raw[1]), raw[2]


def _health(raw: object) -> tuple[str, str, float]:
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 3
        or not isinstance(raw[0], str)
        or not isinstance(raw[1], str)
        or not isinstance(raw[2], (int, float))
        or isinstance(raw[2], bool)
    ):
        raise ValueError("Workload health 无效")
    return raw[0], raw[1], float(raw[2])


def _plugin_data_root(workspace: Path, request: WorkloadStartRequest) -> Path:
    if "@" in request.plugin_id:
        name, marketplace = request.plugin_id.rsplit("@", 1)
    else:
        name, marketplace = request.plugin_id, "builtin"
    _safe_segment(name, "plugin name")
    _safe_segment(marketplace, "plugin marketplace")
    dirname = f"{name}-{marketplace}"
    if request.mode == "formal":
        return workspace / "plugin-data" / dirname
    return (
        workspace
        / "runtime"
        / "plugin-validation"
        / request.transaction_id
        / "workspace"
        / "plugin-data"
        / dirname
    )


def _make_safe_dir(workspace: Path, target: Path, *, uid: int, gid: int) -> None:
    try:
        relative = target.relative_to(workspace)
    except ValueError as error:
        raise ValueError("Workload data path 越界") from error
    current = workspace
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Workload data path 穿过 symlink: {current}")
        if not current.exists():
            current.mkdir()
            current_stat = current.stat()
            if (current_stat.st_uid, current_stat.st_gid) != (uid, gid):
                os.chown(current, uid, gid)
        if not current.is_dir():
            raise ValueError(f"Workload data path 不是目录: {current}")
    stat = target.stat()
    if (stat.st_uid, stat.st_gid) != (uid, gid):
        raise ValueError(f"Workload data path owner 不匹配: {target}")


def _container_name(value: WorkloadStartRequest) -> str:
    key = _lease_key(value)
    return "akashic-w-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _lease_key(value: WorkloadStartRequest | WorkloadLease) -> str:
    parts = [value.workspace_id, value.plugin_id, value.workload, value.mode]
    if value.mode == "candidate":
        parts.append(value.transaction_id)
    return "|".join(parts)


def _stop_key(lease: WorkloadLease) -> str:
    encoded = json.dumps(asdict(lease), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stop_receipt(lease: WorkloadLease) -> dict[str, object]:
    return {
        "lease": asdict(lease),
        "container_absent": True,
        "mounts_released": True,
    }


def _request_lease(
    request: WorkloadStartRequest,
    container_id: str,
) -> WorkloadLease:
    return WorkloadLease(
        workspace_id=request.workspace_id,
        plugin_id=request.plugin_id,
        workload=request.workload,
        mode=request.mode,
        transaction_id=request.transaction_id,
        generation_id=request.generation_id,
        container_id=container_id,
        spec_digest=request.spec_digest,
    )


async def _finish_effect(effect: Coroutine[Any, Any, _T]) -> _T:
    """Finish one Docker effect before restoring caller cancellation."""

    task = asyncio.create_task(effect)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                break
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


def _labels(detail: dict[str, object]) -> dict[str, str]:
    config = detail.get("Config")
    if not isinstance(config, dict) or not isinstance(config.get("Labels"), dict):
        raise RuntimeError("Docker inspect 缺少 labels")
    labels = config["Labels"]
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in labels.items()
    ):
        raise RuntimeError("Docker labels 无效")
    return cast(dict[str, str], labels)


def _mount_profile(raw: list[object]) -> set[tuple[str, str, bool]]:
    result: set[tuple[str, str, bool]] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("Docker inspect mount 无效")
        source = item.get("Source")
        target = item.get("Destination")
        writable = item.get("RW")
        if (
            item.get("Type") != "bind"
            or not isinstance(source, str)
            or not isinstance(target, str)
            or not isinstance(writable, bool)
        ):
            raise RuntimeError("Docker inspect 出现未声明 mount")
        result.add((source, target, writable))
    return result


def _docker_mounts(
    data_mounts: tuple[tuple[str, str, bool], ...],
) -> list[dict[str, object]]:
    return [
        {
            "Type": "bind",
            "Source": source,
            "Target": target,
            "ReadOnly": not writable,
        }
        for source, target, writable in data_mounts
    ]


def _running(detail: dict[str, object]) -> bool:
    state = detail.get("State")
    if not isinstance(state, dict) or not isinstance(state.get("Running"), bool):
        raise RuntimeError("Docker inspect 缺少 running state")
    return cast(bool, state["Running"])


def _required_text(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Workload {key} 无效")
    return value


def _required_bool(raw: Mapping[str, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Workload {key} 无效")
    return value


def _object(raw: Mapping[str, object], key: str) -> dict[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict) or any(not isinstance(item, str) for item in value):
        raise RuntimeError(f"Docker inspect {key} 无效")
    return cast(dict[str, object], value)


def _text_tuple(raw: object, label: str) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in raw
    ):
        raise ValueError(f"Workload {label} 无效")
    return tuple(raw)


def _text(raw: dict[str, object], key: str) -> str:
    return _required_text(raw, key)


def _safe_segment(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SEGMENT.fullmatch(value):
        raise ValueError(f"{label} 不是安全路径段: {value!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Akashic Workload Controller")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument(
        "--docker-socket", type=Path, default=Path("/var/run/docker.sock")
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--allowed-uid", type=int, required=True)
    parser.add_argument("--socket-gid", type=int, required=True)
    parser.add_argument("--workload-uid", type=int, required=True)
    parser.add_argument("--workload-gid", type=int, required=True)
    parser.add_argument("--socket-uid", type=int, default=0)
    parser.add_argument("--owner-container")
    parser.add_argument("--owner-grace-seconds", type=float, default=10.0)
    parser.add_argument("--owner-poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    server = WorkloadControllerServer(
        workspace=args.workspace,
        socket_path=args.socket,
        docker_socket=args.docker_socket,
        state_path=args.state,
        network=args.network,
        allowed_uid=args.allowed_uid,
        socket_gid=args.socket_gid,
        workload_uid=args.workload_uid,
        workload_gid=args.workload_gid,
        socket_uid=args.socket_uid,
        owner_container=args.owner_container,
        owner_grace_seconds=args.owner_grace_seconds,
        owner_poll_seconds=args.owner_poll_seconds,
    )
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
