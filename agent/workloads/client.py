from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Protocol, cast

from agent.workloads.model import (
    WorkloadEndpoint,
    WorkloadLease,
    WorkloadStartRequest,
    WorkloadStartReceipt,
    WorkloadStopReceipt,
    WorkloadMode,
)


class WorkloadEffectUnknown(RuntimeError):
    """The connection failed after Controller may have changed Docker state."""


class WorkloadController(Protocol):
    async def start(self, request: WorkloadStartRequest) -> WorkloadStartReceipt: ...

    async def stop(self, lease: WorkloadLease) -> WorkloadStopReceipt: ...

    async def cleanup_candidates(
        self, workspace_id: str
    ) -> tuple[WorkloadStopReceipt, ...]: ...


class UnixWorkloadController:
    """Call the narrow Workload Controller protocol over a Unix socket."""

    def __init__(self, socket_path: Path, *, timeout_seconds: float = 120.0) -> None:
        if not socket_path.is_absolute():
            raise ValueError("Workload Controller socket 必须是绝对路径")
        if timeout_seconds <= 0:
            raise ValueError("Workload Controller timeout 必须大于零")
        self._socket_path = socket_path
        self._timeout_seconds = timeout_seconds

    async def start(self, request: WorkloadStartRequest) -> WorkloadStartReceipt:
        payload = await self._call("start", request.to_dict())
        lease = _lease(payload.get("lease"))
        raw_endpoints = payload.get("endpoints")
        if not isinstance(raw_endpoints, list):
            raise RuntimeError("Workload Controller start 回执缺少 endpoints")
        endpoints = tuple(
            WorkloadEndpoint(
                _text(item, "name"),
                _text(item, "url"),
            )
            for item in raw_endpoints
            if isinstance(item, dict)
        )
        if len(endpoints) != len(raw_endpoints):
            raise RuntimeError("Workload Controller endpoint 回执无效")
        adopted = payload.get("adopted_from_generation")
        if adopted is not None and not isinstance(adopted, str):
            raise RuntimeError("Workload Controller adopt 回执无效")
        return WorkloadStartReceipt(lease, endpoints, adopted)

    async def stop(self, lease: WorkloadLease) -> WorkloadStopReceipt:
        payload = await self._call("stop", {"lease": _lease_dict(lease)})
        return WorkloadStopReceipt(
            lease=_lease(payload.get("lease")),
            container_absent=_bool(payload, "container_absent"),
            mounts_released=_bool(payload, "mounts_released"),
        )

    async def cleanup_candidates(
        self,
        workspace_id: str,
    ) -> tuple[WorkloadStopReceipt, ...]:
        payload = await self._call(
            "cleanup_candidates",
            {"workspace_id": workspace_id},
        )
        raw_items = payload.get("receipts")
        if not isinstance(raw_items, list):
            raise RuntimeError("Workload Controller cleanup 回执无效")
        result: list[WorkloadStopReceipt] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise RuntimeError("Workload Controller cleanup entry 无效")
            result.append(
                WorkloadStopReceipt(
                    lease=_lease(raw.get("lease")),
                    container_absent=_bool(raw, "container_absent"),
                    mounts_released=_bool(raw, "mounts_released"),
                )
            )
        return tuple(result)

    async def _call(
        self,
        action: str,
        body: dict[str, object],
    ) -> dict[str, object]:
        async def exchange() -> dict[str, object]:
            reader, writer = await asyncio.open_unix_connection(self._socket_path)
            try:
                writer.write(
                    json.dumps(
                        {"version": 1, "action": action, "body": body},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
                await writer.drain()
                raw = await reader.readline()
                if not raw or len(raw) > 1_048_576:
                    raise WorkloadEffectUnknown("Workload Controller 回执为空或过大")
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise WorkloadEffectUnknown(
                        "Workload Controller 回执不是 JSON"
                    ) from error
                if not isinstance(result, dict):
                    raise RuntimeError("Workload Controller 回执不是对象")
                if result.get("ok") is not True:
                    error = result.get("error")
                    raise RuntimeError(
                        error
                        if isinstance(error, str) and error
                        else "Workload Controller 失败"
                    )
                payload = result.get("body")
                if not isinstance(payload, dict):
                    raise RuntimeError("Workload Controller 回执缺少 body")
                return payload
            finally:
                writer.close()
                await writer.wait_closed()

        try:
            return await asyncio.wait_for(exchange(), timeout=self._timeout_seconds)
        except WorkloadEffectUnknown:
            raise
        except (TimeoutError, OSError) as error:
            raise WorkloadEffectUnknown("Workload Controller 连接结果未知") from error


def _lease(raw: object) -> WorkloadLease:
    if not isinstance(raw, dict):
        raise RuntimeError("Workload Controller 回执缺少 lease")
    mode = _text(raw, "mode")
    if mode not in {"candidate", "formal"}:
        raise RuntimeError("Workload Controller lease mode 无效")
    return WorkloadLease(
        workspace_id=_text(raw, "workspace_id"),
        plugin_id=_text(raw, "plugin_id"),
        workload=_text(raw, "workload"),
        mode=cast(WorkloadMode, mode),
        transaction_id=_text(raw, "transaction_id"),
        generation_id=_text(raw, "generation_id"),
        container_id=_text(raw, "container_id"),
        spec_digest=_text(raw, "spec_digest"),
    )


def _lease_dict(value: WorkloadLease) -> dict[str, object]:
    return {
        "workspace_id": value.workspace_id,
        "plugin_id": value.plugin_id,
        "workload": value.workload,
        "mode": value.mode,
        "transaction_id": value.transaction_id,
        "generation_id": value.generation_id,
        "container_id": value.container_id,
        "spec_digest": value.spec_digest,
    }


def _text(raw: dict[object, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Workload Controller {key} 无效")
    return value


def _bool(raw: dict[str, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"Workload Controller {key} 无效")
    return value
